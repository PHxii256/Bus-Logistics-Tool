"""
Safety-Aware Bus Optimization — Main Entry Point

Dispatches based on the 'mode' field in the input JSON:
  - generate_routes : Full ALNS optimization from scratch
  - change_location : Insert/move a single student into existing routes
"""

import sys
import os
import json
import time as _t
import hashlib
import shutil
import osmnx as ox
import networkx as nx

from data_loader import (
    load_json, load_mode1_input, load_mode2_input,
    serialize_routes, print_input_summary
)
from detour_engine import (
    calculate_route_distance, calculate_route_time,
    cheapest_insertion, process_detour_request, insert_with_2opt,
    snap_address_to_edge, precalculate_distance_matrix,
    find_safe_nodes_within_radius, find_shortest_path_with_turns,
    get_walk_absolute_max, haversine_walk_distance,
    _MATRIX_CACHE, _MATRIX_CACHE_LENGTH, _path_cache
)
from visualization import create_route_map
from solution_state import ServiceSolution
from alns_engine import ALNSEngine
from entities import Student


# ============================================================================
# RUN HISTORY: Save each run to runs_history/{mode}_{school}_{hash8}/
# ============================================================================

def _input_hash(data: dict) -> str:
    """Stable 8-char hash of the input so same input → same folder."""
    canonical = json.dumps(data, sort_keys=True, ensure_ascii=True)
    return hashlib.md5(canonical.encode()).hexdigest()[:8]


def _slug(text: str) -> str:
    """Turn an arbitrary string into a safe folder-name component."""
    import re
    return re.sub(r'[^a-zA-Z0-9]+', '_', text).strip('_').lower()[:30]


def save_run(input_data: dict, output_data: dict, report_data: dict,
            map_files: dict = None):
    """
    Persist a run to:  runs_history/<hash8>/
      input.json       — exact input passed to the algorithm
      output.json      — serialized routes / result
      report.json      — diagnostics (runtime, buses used, …)
      <map_files>      — one or more html maps (copied if they exist)

    map_files: dict of {dest_filename: src_path}, e.g.
      {'route_map.html': 'route_map.html'}
      {'route_map_old.html': 'route_map_old.html',
       'route_map_new.html': 'route_map_new.html'}

    Same input always maps to the same folder — no duplicates created.
    """
    run_hash = _input_hash(input_data)
    run_dir  = os.path.join('runs_history', run_hash)

    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, 'input.json'),  'w') as f:
        json.dump(input_data,  f, indent=2)
    with open(os.path.join(run_dir, 'output.json'), 'w') as f:
        json.dump(output_data, f, indent=2)
    with open(os.path.join(run_dir, 'report.json'), 'w') as f:
        json.dump(report_data, f, indent=2)

    if map_files:
        for dest_name, src_path in map_files.items():
            if os.path.exists(src_path):
                shutil.copy2(src_path, os.path.join(run_dir, dest_name))

    print(f"  Run saved to '{run_dir}/'")
    return run_dir


# ============================================================================
# GRAPH SETUP (shared by both modes)
# ============================================================================

def setup_graph():
    """Download road network and apply safety/speed tags."""
    print("Downloading road network...")
    G = ox.graph_from_bbox(
        [31.229084, 29.925630, 31.331909, 29.991682],
        network_type='drive'
    )

    print("Applying safety tags (Cairo Factor)...")
    for u, v, k, data in G.edges(keys=True, data=True):
        maxspeed = data.get('maxspeed', 30)
        if isinstance(maxspeed, list):
            try:
                base_speed = float(maxspeed[0])
            except (ValueError, TypeError):
                base_speed = 30
        else:
            try:
                base_speed = float(maxspeed)
            except (ValueError, TypeError):
                base_speed = 30

        highway = data.get('highway', 'unclassified')
        if isinstance(highway, list):
            highway = highway[0]

        if highway in ['primary', 'trunk']:
            data['speed_kph'] = base_speed * 0.8
            data['is_safe_to_cross'] = False
        elif highway in ['secondary', 'tertiary']:
            data['speed_kph'] = base_speed * 0.6
            data['is_safe_to_cross'] = False
        elif highway in ['residential', 'living_street']:
            data['speed_kph'] = base_speed * 0.3
            data['is_safe_to_cross'] = True
        else:
            data['speed_kph'] = base_speed * 0.2
            data['is_safe_to_cross'] = True

        meters_per_min = (data['speed_kph'] * 1000) / 60
        data['travel_time'] = data['length'] / meters_per_min

    print("Adding edge bearings for turn-penalty calculations...")
    G = ox.bearing.add_edge_bearings(G)
    print(f"Graph ready: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges\n")
    return G


# ============================================================================
# MATRIX PRECOMPUTATION (used by generate_routes, optionally by change_location)
# ============================================================================

def precompute_matrix(students, routes, G):
    """Snap all students & route stops, build the distance/time matrix.

    Returns:
        Tuple of (critical_nodes set, student_frontages dict)
    """
    print("[Optimization] Preparing distance matrix...")
    critical_nodes = set()
    student_frontages = {}

    for s in students:
        node_id, _ = snap_address_to_edge(s.coords, G)
        critical_nodes.add(node_id)
        student_frontages[s.id] = node_id
        if s.walk_radius > 0:
            safe_nodes = find_safe_nodes_within_radius(s.coords, G, 500, s.walk_radius)
            for safe_node_id, _ in sorted(safe_nodes, key=lambda x: x[1])[:5]:
                critical_nodes.add(safe_node_id)

    school_node = None
    for route in routes:
        for stop in route.stops:
            # Only add nodes that actually exist in the graph
            # (virtual frontage nodes from a prior session won't be here)
            if stop.node_id in G:
                critical_nodes.add(stop.node_id)
                if school_node is None:
                    school_node = stop.node_id
            else:
                # Re-snap the stop to nearest real node
                nearest = ox.nearest_nodes(G, stop.coords[1], stop.coords[0])
                stop.node_id = nearest
                stop.coords = (G.nodes[nearest]['y'], G.nodes[nearest]['x'])
                critical_nodes.add(nearest)
                if school_node is None:
                    school_node = nearest

    # Phase 1: Build initial matrix
    print(f"[Optimization] {len(critical_nodes)} critical nodes (incl. walk candidates)")
    precalculate_distance_matrix(G, list(critical_nodes))

    # Phase 2: Expand for unreachable frontages
    extra_reachable_nodes = set()
    for s in students:
        fnode = student_frontages[s.id]
        to_school = _MATRIX_CACHE.get((fnode, school_node), float('inf'))
        from_school = _MATRIX_CACHE.get((school_node, fnode), float('inf'))
        if to_school == float('inf') or from_school == float('inf'):
            print(f"[Optimization] {s.id} ({s.school_stage.name}) frontage unreachable, expanding...")
            lat, lon = s.coords
            center_node = ox.nearest_nodes(G, lon, lat)
            max_walk = get_walk_absolute_max(s.walk_radius)
            visited = set()
            bfs_queue = [(center_node, 0)]
            bfs_candidates = []
            while bfs_queue:
                node, dist = bfs_queue.pop(0)
                if node in visited or dist > max_walk:
                    continue
                visited.add(node)
                ts = _MATRIX_CACHE.get((node, school_node), float('inf'))
                fs = _MATRIX_CACHE.get((school_node, node), float('inf'))
                if ts < float('inf') and fs < float('inf'):
                    bfs_candidates.append((node, dist, True))
                elif node not in critical_nodes:
                    bfs_candidates.append((node, dist, False))
                for neighbor in G.successors(node):
                    ed = G.get_edge_data(node, neighbor)
                    if ed:
                        d = ed[0] if 0 in ed else list(ed.values())[0]
                        new_dist = dist + d.get('length', 0)
                        if new_dist <= max_walk:
                            bfs_queue.append((neighbor, new_dist))
                for predecessor in G.predecessors(node):
                    ed = G.get_edge_data(predecessor, node)
                    if ed:
                        d = ed[0] if 0 in ed else list(ed.values())[0]
                        new_dist = dist + d.get('length', 0)
                        if new_dist <= max_walk:
                            bfs_queue.append((predecessor, new_dist))

            bfs_candidates.sort(key=lambda x: x[1])
            found_count = 0
            for node, dist, already_known in bfs_candidates:
                if found_count >= 5:
                    break
                if already_known:
                    extra_reachable_nodes.add(node)
                    found_count += 1
                else:
                    path_to, t_to = find_shortest_path_with_turns(G, school_node, node)
                    path_from, t_from = find_shortest_path_with_turns(G, node, school_node)
                    if t_to < float('inf') and t_from < float('inf'):
                        extra_reachable_nodes.add(node)
                        found_count += 1
            if found_count == 0:
                print(f"  WARNING: No bus-reachable nodes found within {max_walk}m walk!")

    # Targeted precomputation for fallback nodes
    if extra_reachable_nodes:
        new_nodes = extra_reachable_nodes - critical_nodes
        if new_nodes:
            pairs_to_compute = []
            for fn in new_nodes:
                for cn in critical_nodes:
                    pairs_to_compute.append((fn, cn))
                    pairs_to_compute.append((cn, fn))
            print(f"[Optimization] Pre-computing {len(pairs_to_compute)} pairs for {len(new_nodes)} fallback nodes...")
            _start = _t.time()
            for src, dst in pairs_to_compute:
                find_shortest_path_with_turns(G, src, dst)
                cache_key = (src, dst, None)
                if cache_key in _path_cache:
                    path, t = _path_cache[cache_key]
                    if path and t < float('inf'):
                        dist_m = 0.0
                        for pi in range(len(path) - 1):
                            ed = G.get_edge_data(path[pi], path[pi + 1])
                            if ed:
                                d = ed[0] if 0 in ed else list(ed.values())[0]
                                dist_m += d.get('length', 0)
                        _MATRIX_CACHE_LENGTH[(src, dst)] = dist_m
                    else:
                        _MATRIX_CACHE_LENGTH[(src, dst)] = float('inf')
            _elapsed = _t.time() - _start
            print(f"  Done in {_elapsed:.1f}s ({len(pairs_to_compute)} pairs)")
            critical_nodes |= new_nodes

    return critical_nodes, student_frontages


# ============================================================================
# MODE 1: generate_routes — Full ALNS optimization
# ============================================================================

def run_generate_routes(data, G, input_file_path):
    """Run Mode 1: ALNS-based route generation from scratch."""
    _run_start = _t.time()
    students, buses, routes, school_coords, constraints, algo_config = load_mode1_input(data, G)
    print_input_summary(students, buses, routes, school_coords)

    # Pre-compute distance matrix
    precompute_matrix(students, routes, G)

    # Run ALNS
    print(f"\n{'='*80}")
    print("RUNNING GLOBAL OPTIMIZATION (ALNS)")
    print(f"{'='*80}\n")

    initial_sol = ServiceSolution(students, routes, G)
    iterations = algo_config.get('iterations', 60)
    optimizer = ALNSEngine(initial_sol, iterations=iterations)
    _alns_start = _t.time()
    best_sol = optimizer.run()
    _alns_elapsed = _t.time() - _alns_start

    students = best_sol.students
    routes = best_sol.routes

    # Recalculate route metrics
    print(f"\nRecalculating route metrics...")
    for route in routes:
        route.total_distance = calculate_route_distance(route, G)
        route.total_time = calculate_route_time(route, G)
        sc = sum(len(stop.students) for stop in route.stops)
        print(f"  {route.route_id}: {len(route.stops)} stops, {sc} students, "
              f"{route.total_distance:.2f}km, {route.total_time:.2f}min, "
              f"Profit: {route.get_profit_margin():.1%}")

    # Visualization
    routes_with_students = [r for r in routes if r.get_student_count() > 0]
    if routes_with_students:
        create_route_map(G, routes_with_students, all_students=students,
                         school_coords=school_coords, output_file='route_map.html')
        print(f"\nRoute map saved to route_map.html")

    # Collect unserved students
    unserved = [s for s in students if not s.is_served]

    # Serialize to unified route schema
    output = serialize_routes(routes, buses, school_coords, unserved, G)

    output_path = 'output_data.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    served_count = len(students) - len(unserved)
    _total_elapsed = _t.time() - _run_start
    print(f"\nResults saved to '{output_path}'")
    print(f"  Served: {served_count}/{len(students)} students")
    print(f"  Active routes: {len(routes_with_students)}")
    if unserved:
        print(f"  Unserved: {[s.id for s in unserved]}")
    print(f"  Total runtime: {_total_elapsed:.2f}s")

    # ── Save run history ──────────────────────────────────────────────────────
    buses_used = [
        {"id": bus_id, "capacity": cap}
        for bus_id, cap in {
            r.bus.bus_id: r.bus.capacity for r in routes_with_students
        }.items()
    ]
    report = {
        "run_timestamp":             _t.strftime('%Y-%m-%dT%H:%M:%S'),
        "mode":                      "generate_routes",
        "input_file":                input_file_path,
        "total_runtime_seconds":     round(_total_elapsed, 2),
        "optimization_time_seconds": round(_alns_elapsed, 2),
        "students_total":            len(students),
        "students_served":           served_count,
        "students_unserved":         [s.id for s in unserved],
        "routes_created":            len(routes_with_students),
        "buses_used":                buses_used,
        "final_objective":           round(best_sol.calculate_objective(), 2),
        "alns_iterations":           algo_config.get('iterations', 60),
    }
    save_run(data, output, report, map_files={'route_map.html': 'route_map.html'})
    # ─────────────────────────────────────────────────────────────────────────

    return output


# ============================================================================
# MODE 2: change_location — Insert/move a single student
# ============================================================================

def run_change_location(data, G, input_file_path):
    """Run Mode 2: Insert a student at a new location into existing routes."""
    _run_start = _t.time()
    (student_id, new_coords, change_type, valid_from, valid_until,
     algo_config, routes, all_students, buses, school_coords) = load_mode2_input(data, G)

    method = algo_config.get('method', 'cheapest_insertion')
    daily_budget = data.get('constraints', {}).get('daily_detour_budget_minutes', 5)

    print(f"\n{'='*80}")
    print(f"MODE 2: CHANGE LOCATION — Student {student_id}")
    print(f"  Method: {method}")
    print(f"  Change type: {change_type}")
    print(f"  New location: {new_coords}")
    print(f"{'='*80}\n")

    # Find or create the Student object
    target_student = next((s for s in all_students if s.id == student_id), None)

    if target_student and target_student.is_served:
        # Remove student from their current stop first
        if target_student.assigned_stop:
            target_student.assigned_stop.remove_student(target_student)
        print(f"  Removed {student_id} from current assignment")

    if target_student:
        # Update coords
        target_student.coords = new_coords
        target_student.assignment = change_type
        target_student.valid_from = valid_from
        target_student.valid_until = valid_until
    else:
        # New student not in current routes
        from data_loader import school_stage_from_string
        new_loc = data.get('new_location', {})
        target_student = Student(
            id=student_id,
            lat=new_coords[0],
            lon=new_coords[1],
            age=new_loc.get('age', 10),
            school_stage=school_stage_from_string(
                new_loc.get('school_stage', 'ELEMENTARY')
            ),
            fee=new_loc.get('fee', 100),
            assignment=change_type,
            valid_from=valid_from,
            valid_until=valid_until
        )

    # Precompute matrix for the new location + existing route stops
    precompute_matrix([target_student], routes, G)

    # Dispatch to algorithm
    if method == '2opt':
        success, updated_route, message = insert_with_2opt(
            target_student, routes, G,
            detour_type=change_type, daily_detour_budget=daily_budget
        )
    elif method == 'alns':
        # Full ALNS re-optimization with student added to pool
        if target_student not in all_students:
            all_students.append(target_student)
        initial_sol = ServiceSolution(all_students, routes, G)
        iterations = algo_config.get('iterations', 30)
        optimizer = ALNSEngine(initial_sol, iterations=iterations)
        best_sol = optimizer.run()
        routes = best_sol.routes
        all_students = best_sol.students
        target = next((s for s in all_students if s.id == student_id), None)
        success = target is not None and target.is_served
        updated_route = None
        if success:
            updated_route = next(
                (r for r in routes if any(
                    any(st.id == student_id for st in stop.students)
                    for stop in r.stops
                )), None
            )
        message = f"ALNS re-optimization: {'student placed' if success else 'student not placed'}"
    else:
        # Default: cheapest_insertion
        success, updated_route, message = process_detour_request(
            target_student, routes, G,
            detour_type=change_type, daily_detour_budget=daily_budget
        )

    print(f"  Result: {message}")

    # Recalculate route metrics for all routes
    for route in routes:
        route.total_distance = calculate_route_distance(route, G)
        route.total_time = calculate_route_time(route, G)

    # Update route map visualization if insertion succeeded
    # Save the pre-change map as route_map_old.html first
    if os.path.exists('route_map.html'):
        shutil.copy2('route_map.html', 'route_map_old.html')

    if success:
        routes_with_students = [r for r in routes if r.get_student_count() > 0]
        if routes_with_students:
            create_route_map(G, routes_with_students, all_students=all_students,
                             school_coords=school_coords, output_file='route_map_new.html')
            print(f"  Route map saved: route_map_old.html (before) / route_map_new.html (after)")

    # Build output
    if success:
        # Calculate walk distance for the inserted student
        walk_d = 0.0
        if target_student.assigned_stop:
            walk_d = haversine_walk_distance(
                target_student.coords[0], target_student.coords[1],
                target_student.assigned_stop.coords[0], target_student.assigned_stop.coords[1]
            )

        unserved = [s for s in all_students if not s.is_served]
        output = {
            "status": "success",
            "student_id": student_id,
            "change_type": change_type,
            "algorithm_used": method,
            "insertion_cost_minutes": round(updated_route.total_time if updated_route else 0, 2),
            "walk_distance": round(walk_d, 1),
            **serialize_routes(routes, buses, school_coords, unserved, G)
        }
    else:
        output = {
            "status": "failed",
            "student_id": student_id,
            "reason": message
        }

    output_path = 'output_data.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    _total_elapsed = _t.time() - _run_start
    print(f"\nResponse saved to '{output_path}'")
    print(f"  Total runtime: {_total_elapsed:.2f}s")

    # ── Save run history ──────────────────────────────────────────────────────
    active_routes = [r for r in routes if r.get_student_count() > 0]
    buses_used = [
        {"id": bus_id, "capacity": cap}
        for bus_id, cap in {
            r.bus.bus_id: r.bus.capacity for r in active_routes
        }.items()
    ]
    served_count = sum(1 for s in all_students if s.is_served)
    report = {
        "run_timestamp":         _t.strftime('%Y-%m-%dT%H:%M:%S'),
        "mode":                  "change_location",
        "input_file":            input_file_path,
        "total_runtime_seconds": round(_total_elapsed, 2),
        "student_changed":       student_id,
        "change_type":           change_type,
        "algorithm_used":        method,
        "status":                output.get('status'),
        "students_total":        len(all_students),
        "students_served":       served_count,
        "routes_active":         len(active_routes),
        "buses_used":            buses_used,
    }
    save_run(data, output, report, map_files={
        'route_map_old.html': 'route_map_old.html',
        'route_map_new.html': 'route_map_new.html',
    })
    # ─────────────────────────────────────────────────────────────────────────

    return output


# ============================================================================
# MAIN: Parse mode and dispatch
# ============================================================================

def main(input_file='api_requests/generate_routes_input.json'):
    """Main entry point — read input JSON, dispatch to the correct mode."""
    print(f"Loading '{input_file}'...")
    data = load_json(input_file)
    mode = data.get('mode', 'generate_routes')

    G = setup_graph()

    if mode == 'generate_routes':
        return run_generate_routes(data, G, input_file)
    elif mode == 'change_location':
        return run_change_location(data, G, input_file)
    else:
        print(f"ERROR: Unknown mode '{mode}'. Use 'generate_routes' or 'change_location'.")
        sys.exit(1)


if __name__ == '__main__':
    main('api_requests/generate_routes_input.json')