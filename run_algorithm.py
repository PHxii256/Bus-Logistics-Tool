"""
Example script showing how to run the Safety-Aware Bus Optimization algorithm
using data loaded from input_data.json
"""

import osmnx as ox
import networkx as nx
import json
from data_loader import setup_algorithm_inputs, print_input_summary
from detour_engine import (
    calculate_route_distance, calculate_route_time,
    cheapest_insertion, process_detour_request,
    calculate_student_ride_time
)
from visualization import create_route_map
from solution_state import ServiceSolution
from alns_engine import ALNSEngine

# ============================================================================
# SETUP: Download graph and load input data
# ============================================================================

print("Downloading road network...")
G = ox.graph_from_bbox([31.229084, 29.925630, 31.331909, 29.991682], network_type='drive')

# Apply safety/speed tags (Cairo Factor)
print("Applying safety tags...")
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

    # Cairo Factor: Reduce speed based on road type for realism.
    # Prioritize arterial roads (primary/secondary/tertiary) over residential.
    highway = data.get('highway', 'unclassified')
    if isinstance(highway, list): highway = highway[0]

    if highway in ['primary', 'trunk']:
        data['speed_kph'] = base_speed * 0.8  # Main roads are fast
        data['is_safe_to_cross'] = False
    elif highway in ['secondary', 'tertiary']:
        data['speed_kph'] = base_speed * 0.6  # Secondary roads
        data['is_safe_to_cross'] = False
    elif highway in ['residential', 'living_street']:
        data['speed_kph'] = base_speed * 0.3  # Residential roads are slower
        data['is_safe_to_cross'] = True
    else:
        # Others (links, unclassified, service, etc.)
        data['speed_kph'] = base_speed * 0.2
        data['is_safe_to_cross'] = True



    meters_per_min = (data['speed_kph'] * 1000) / 60
    data['travel_time'] = data['length'] / meters_per_min

# Add bearings to edges to allow calculating turn angles later
print("Adding edge bearings for turn-penalty calculations...")
G = ox.bearing.add_edge_bearings(G)

print(f"Graph ready: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges\n")

# Load input data from JSON
print("Loading input data from input_data.json...")
students, buses, routes, school_coords, config = setup_algorithm_inputs('input_data.json', G)

# Print summary
print_input_summary(students, buses, routes, school_coords)

# ============================================================================
# ALGORITHM: GLOBAL OPTIMIZATION (ALNS)
# ============================================================================

print(f"{'='*80}")
print("RUNNING GLOBAL OPTIMIZATION (ALNS)")
print(f"{'='*80}\n")

# Pre-snap students and collect critical nodes for Distance Matrix priming
print("[Optimization] Preparing road network for large-scale routing...")
from detour_engine import snap_address_to_edge, precalculate_distance_matrix, find_safe_nodes_within_radius, _MATRIX_CACHE
critical_nodes = set()
student_frontages = {}
for s in students:
    node_id, _ = snap_address_to_edge(s.coords, G)
    critical_nodes.add(node_id)
    student_frontages[s.id] = node_id
    # Include walk-radius candidate nodes for MIDDLE/HIGH students
    if s.walk_radius > 0:
        safe_nodes = find_safe_nodes_within_radius(s.coords, G, 500, s.walk_radius)
        for safe_node_id, _ in sorted(safe_nodes, key=lambda x: x[1])[:5]:
            critical_nodes.add(safe_node_id)

school_node = None
for route in routes:
    for stop in route.stops:
        critical_nodes.add(stop.node_id)
        if school_node is None:
            school_node = stop.node_id

# Phase 1: Build initial matrix
print(f"[Optimization] {len(critical_nodes)} critical nodes (incl. walk candidates)")
precalculate_distance_matrix(G, list(critical_nodes))

# Phase 2: For students whose frontage is unreachable by bus, find nearby 
# bus-reachable nodes and add them to the matrix
from detour_engine import find_shortest_path_with_turns, _MATRIX_CACHE_LENGTH, _path_cache
extra_reachable_nodes = set()
for s in students:
    fnode = student_frontages[s.id]
    to_school = _MATRIX_CACHE.get((fnode, school_node), float('inf'))
    from_school = _MATRIX_CACHE.get((school_node, fnode), float('inf'))
    if to_school == float('inf') or from_school == float('inf'):
        print(f"[Optimization] {s.id} ({s.school_stage.name}) frontage unreachable, expanding search...")
        lat, lon = s.coords
        center_node = ox.nearest_nodes(G, lon, lat)
        # Absolute max walk based on student stage
        from detour_engine import get_walk_absolute_max
        max_walk = get_walk_absolute_max(s.walk_radius)
        print(f"  Recommended: {s.walk_radius}m, Absolute max: {max_walk}m")
        visited = set()
        bfs_queue = [(center_node, 0)]
        bfs_candidates = []  # (node, walk_dist)
        while bfs_queue:
            node, dist = bfs_queue.pop(0)
            if node in visited or dist > max_walk:
                continue
            visited.add(node)
            # Check if already in matrix and reachable
            ts = _MATRIX_CACHE.get((node, school_node), float('inf'))
            fs = _MATRIX_CACHE.get((school_node, node), float('inf'))
            if ts < float('inf') and fs < float('inf'):
                bfs_candidates.append((node, dist, True))  # already known reachable
            elif node not in critical_nodes:
                bfs_candidates.append((node, dist, False))  # needs A* check
            # Expand bidirectionally (walking ignores one-way)
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
        
        # Sort by walk distance, then check school-reachability for unknown nodes
        bfs_candidates.sort(key=lambda x: x[1])
        found_count = 0
        for node, dist, already_known in bfs_candidates:
            if found_count >= 5:  # Limit to 5 reachable fallback nodes
                break
            if already_known:
                print(f"  Reachable node {node} at {dist:.0f}m walk (already in matrix)")
                extra_reachable_nodes.add(node)
                found_count += 1
            else:
                # Quick A* check: school->node and node->school only
                path_to, t_to = find_shortest_path_with_turns(G, school_node, node)
                path_from, t_from = find_shortest_path_with_turns(G, node, school_node)
                if t_to < float('inf') and t_from < float('inf'):
                    print(f"  Reachable node {node} at {dist:.0f}m walk (school->{t_to:.1f}min, ->school={t_from:.1f}min)")
                    extra_reachable_nodes.add(node)
                    found_count += 1
        if found_count == 0:
            print(f"  WARNING: No bus-reachable nodes found within {max_walk}m walk!")

# Register fallback nodes and precompute ONLY their pairs with existing critical nodes
if extra_reachable_nodes:
    new_nodes = extra_reachable_nodes - critical_nodes
    if new_nodes:
        # Targeted precomputation: only fallback↔critical pairs (not all-to-all again)
        pairs_to_compute = []
        for fn in new_nodes:
            for cn in critical_nodes:
                pairs_to_compute.append((fn, cn))
                pairs_to_compute.append((cn, fn))
        print(f"[Optimization] Pre-computing {len(pairs_to_compute)} pairs for {len(new_nodes)} fallback nodes...")
        import time as _t
        _start = _t.time()
        for src, dst in pairs_to_compute:
            find_shortest_path_with_turns(G, src, dst)
            # Also compute length from path
            cache_key = (src, dst, None)
            if cache_key in _path_cache:
                path, t = _path_cache[cache_key]
                if path and t < float('inf'):
                    dist_m = 0.0
                    for pi in range(len(path) - 1):
                        ed = G.get_edge_data(path[pi], path[pi+1])
                        if ed:
                            d = ed[0] if 0 in ed else list(ed.values())[0]
                            dist_m += d.get('length', 0)
                    _MATRIX_CACHE_LENGTH[(src, dst)] = dist_m
                else:
                    _MATRIX_CACHE_LENGTH[(src, dst)] = float('inf')
        _elapsed = _t.time() - _start
        print(f"  Done in {_elapsed:.1f}s ({len(pairs_to_compute)} pairs)")
        critical_nodes |= new_nodes

# Initialize solution state
initial_sol = ServiceSolution(students, routes, G)

# Run ALNS Optimizer (ALWAYS RUN WITH 60 ITTERATIONS OR LESS)
optimizer = ALNSEngine(initial_sol, iterations=60)
best_sol = optimizer.run()

# Update the main students and routes objects with the best solution found
students = best_sol.students
routes = best_sol.routes

# Recalculate route metrics after assignments
print(f"\nRecalculating route metrics after assignments...")
for route in routes:
    route.total_distance = calculate_route_distance(route, G)
    route.total_time = calculate_route_time(route, G)
    student_count = sum(len(stop.students) for stop in route.stops)
    print(f"{route.route_id}: {len(route.stops)} stops, {student_count} students, {route.total_distance:.2f}km, {route.total_time:.2f}min, Profit margin: {route.get_profit_margin():.1%}")

# ============================================================================
# VISUALIZATION: Create comprehensive route map
# ============================================================================

print(f"\n{'='*80}")
print("GENERATING VISUALIZATIONS")
print(f"{'='*80}\n")

# Filter routes with students
routes_with_students = [r for r in routes if r.get_student_count() > 0]

if routes_with_students:
    create_route_map(G, routes_with_students, all_students=students, school_coords=school_coords, output_file='route_map.html')
    print(f"\nRoute map created with:")
    print(f"  - {len(routes_with_students)} active routes")
    print(f"  - {sum(r.get_student_count() for r in routes_with_students)} students assigned")
    print(f"  - School location marked")
    print(f"\nOpen 'route_map.html' in a web browser to view:")
    print(f"  - Actual road network paths between stops")
    print(f"  - Student home locations")
    print(f"  - Walking paths from homes to bus stops")
    print(f"  - Bus routes with proper street-following")
else:
    print("No routes with students to visualize")

# ============================================================================
# SAVE RESULTS: Export the optimized state to JSON
# ============================================================================

print(f"\n{'='*80}")
print("SAVING RESULTS")
print(f"{'='*80}\n")

# Prepare output data structure
output_data = {
    "school": school_coords,
    "cairo_map": {
        "bounding_box": {
            "north": 31.331909,
            "south": 29.92563,
            "east": 31.331909,
            "west": 29.92563
        },
        "total_nodes": G.number_of_nodes(),
        "total_edges": G.number_of_edges()
    },
    "buses": [
        {
            "id": bus_id,
            "type": b.bus_type,
            "capacity": b.capacity,
            "fixed_cost": b.fixed_cost,
            "variable_cost_per_km": b.var_cost_km
        } for bus_id, b in buses.items()
    ],
    "permanent_routes": [
        {
            "route_id": r.route_id,
            "bus_type": r.bus.bus_type,
            "max_trip_time_minutes": r.route_tmax,
            "stops": [
                {
                    "stop_id": f"{r.route_id}-Stop-{idx}",
                    "latitude": stop.coords[0],
                    "longitude": stop.coords[1],
                    "node_id": stop.node_id,
                    "student_count": len(stop.students),
                    "student_ids": [s.id for s in stop.students]
                } for idx, stop in enumerate(r.stops)
            ],
            "total_distance_km": r.total_distance,
            "total_time_minutes": r.total_time,
            "profit_margin": r.get_profit_margin()
        } for r in routes
    ],
    "assigned_students": [
        {
            "id": s.id,
            "latitude": s.coords[0],
            "longitude": s.coords[1],
            "age": s.age,
            "school_stage": s.school_stage.name,
            "walk_radius_m": s.walk_radius,
            "fee": s.fee,
            "assigned_route": next((r.route_id for r in routes if any(any(st.id == s.id for st in stop.students) for stop in r.stops)), None)
        } for s in students
    ],
    "constraints": config.get('constraints', {})
}

# Save to output_data.json
with open('output_data.json', 'w') as f:
    json.dump(output_data, f, indent=2)

print(f"Results saved to 'output_data.json'")
print(f"  - Permanent students: {len([s for s in output_data['assigned_students'] if s['assigned_route']])}/{len(students)}")
print(f"  - Active routes: {len(routes_with_students)}")

print(f"\n{'='*80}")
print("ALGORITHM EXECUTION COMPLETE")
print(f"{'='*80}")
