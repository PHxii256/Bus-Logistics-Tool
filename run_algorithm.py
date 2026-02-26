"""
Safety-Aware Bus Optimization  Main Entry Point

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
import argparse

from data_loader import (
    load_json, load_mode1_input, load_mode2_input,
    serialize_routes, print_input_summary
)
import detour_engine as _det_eng
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
from entities import Student, School_Stage

# ============================================================================
# DEFAULT WALK LIMITS PER SCHOOL STAGE  (metres)
# ============================================================================
DEFAULT_STAGE_WALK_LIMITS = {
    "KG":         0,
    "ELEMENTARY": 0,
    "MIDDLE":     150,
    "HIGH":       200,
}

# ============================================================================
# RUN HISTORY: Save each run to runs_history/{mode}_{school}_{hash8}/
# ============================================================================

def _input_hash(data: dict) -> str:
    """Stable 8-char hash of the input so same input  same folder."""
    canonical = json.dumps(data, sort_keys=True, ensure_ascii=True)
    return hashlib.md5(canonical.encode()).hexdigest()[:8]

def save_run(input_data: dict, output_data: dict, report_data: dict,
            map_files: dict = None):
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
# GRAPH SETUP
# ============================================================================

_DEFAULT_BBOX = [31.229084, 29.925630, 31.331909, 29.991682]
_ROAD_SPEEDS_CONFIG_PATH = 'road_speeds_config.json'

def _load_road_speeds(override: dict = None) -> dict:
    builtin = {
        'default_speed_kph': 30,
        'road_types': {
            'primary':       {'speed_multiplier': 0.8, 'safe_to_cross': False},
            'trunk':         {'speed_multiplier': 0.8, 'safe_to_cross': False},
            'secondary':     {'speed_multiplier': 0.6, 'safe_to_cross': False},
            'tertiary':      {'speed_multiplier': 0.6, 'safe_to_cross': True},
            'residential':   {'speed_multiplier': 0.3, 'safe_to_cross': True},
            'living_street': {'speed_multiplier': 0.3, 'safe_to_cross': True},
            'default':       {'speed_multiplier': 0.2, 'safe_to_cross': True},
        }
    }
    try:
        with open(_ROAD_SPEEDS_CONFIG_PATH) as f:
            file_cfg = json.load(f)
        builtin.update({k: v for k, v in file_cfg.items() if not k.startswith('_')})
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    if override:
        builtin.update(override)
    return builtin

def setup_graph(meta: dict = None, unconstrained: bool = False):
    graph_cfg    = (meta or {}).get('graph', {})
    bbox         = graph_cfg.get('bbox', _DEFAULT_BBOX)
    
    # Simple hash of bbox for caching
    bbox_hash = hashlib.md5(str(bbox).encode()).hexdigest()[:8]
    cache_dir   = 'cache'
    pkl_file    = os.path.join(cache_dir, f"graph_{bbox_hash}.pkl")
    cache_file  = os.path.join(cache_dir, f"graph_{bbox_hash}.graphml")
    os.makedirs(cache_dir, exist_ok=True)

    if os.path.exists(pkl_file):
        import pickle
        print(f"Loading cached road network (pickle): {pkl_file}")
        with open(pkl_file, 'rb') as fh:
            G = pickle.load(fh)
    elif os.path.exists(cache_file):
        print(f"Loading cached road network: {cache_file}")
        G = ox.load_graphml(cache_file)
        # Save as pickle for faster future loads
        import pickle
        print(f"Saving pickle cache for faster future loads...")
        with open(pkl_file, 'wb') as fh:
            pickle.dump(G, fh, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print("Downloading road network...")
        north, south, east, west = bbox[3], bbox[1], bbox[2], bbox[0]
        # OSMnx 2.0+ expects a single tuple (north, south, east, west)
        G = ox.graph_from_bbox((north, south, east, west), network_type='drive')
        ox.save_graphml(G, cache_file)
        import pickle
        with open(pkl_file, 'wb') as fh:
            pickle.dump(G, fh, protocol=pickle.HIGHEST_PROTOCOL)

    road_cfg     = _load_road_speeds((meta or {}).get('road_speeds'))
    road_types   = road_cfg['road_types']
    default_spd  = road_cfg.get('default_speed_kph', 30)

    print("Applying road speed config...")
    for u, v, k, data in G.edges(keys=True, data=True):
        maxspeed = data.get('maxspeed', default_spd)
        if isinstance(maxspeed, list):
            try:    base_speed = float(maxspeed[0])
            except: base_speed = default_spd
        else:
            try:    base_speed = float(maxspeed)
            except: base_speed = default_spd
        highway = data.get('highway', 'unclassified')
        if isinstance(highway, list): highway = highway[0]
        cfg = road_types.get(highway, road_types.get('default', {'speed_multiplier': 0.2, 'safe_to_cross': True}))
        data['speed_kph']       = base_speed * cfg['speed_multiplier']
        data['is_safe_to_cross'] = True if unconstrained else cfg['safe_to_cross']
        meters_per_min = (data['speed_kph'] * 1000) / 60
        data['travel_time'] = data['length'] / meters_per_min
    print("Adding edge bearings for turn-penalty calculations...")
    G = ox.bearing.add_edge_bearings(G)
    print(f"Graph ready: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges\n")
    return G

# ============================================================================
# MATRIX PRECOMPUTATION
# ============================================================================

def precompute_matrix(students, routes, G, fast_mode=None, G_drive=None):
    """Build the distance matrix for ALNS.

    Parameters
    ----------
    G       : graph used for walking BFS (may be constrained)
    G_drive : graph used for bus driving distances (should always be the
              full unconstrained network).  Falls back to *G* if not given,
              preserving backward-compatibility.
    """
    if G_drive is None:
        G_drive = G
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
            if stop.node_id in G_drive:
                critical_nodes.add(stop.node_id)
                if school_node is None: school_node = stop.node_id
            else:
                nearest = _det_eng.fast_nearest_node(G_drive, stop.coords[1], stop.coords[0])
                stop.node_id = nearest
                stop.coords = (G_drive.nodes[nearest]['y'], G_drive.nodes[nearest]['x'])
                critical_nodes.add(nearest)
                if school_node is None: school_node = nearest
    # Auto-select fast mode for large graphs (>50K nodes) to avoid minutes-long precomputes
    if fast_mode is None:
        fast_mode = G_drive.number_of_nodes() > 50_000
    # Bus distance matrix ALWAYS uses the full driving graph
    precalculate_distance_matrix(G_drive, list(critical_nodes), fast_mode=fast_mode)
    return critical_nodes, student_frontages

# ============================================================================
# MODE 1: generate_routes
# ============================================================================

def run_generate_routes(data, G, input_file_path):
    _run_start = _t.time()
    students, buses, routes, school_coords, constraints, algo_config = load_mode1_input(data, G)
    print_input_summary(students, buses, routes, school_coords)
    precompute_matrix(students, routes, G)
    print(f"\nRUNNING ALNS OPTIMIZATION ({algo_config.get('iterations', 60)} iters)")
    initial_sol = ServiceSolution(students, routes, G)
    optimizer = ALNSEngine(initial_sol, iterations=algo_config.get('iterations', 60))
    _alns_start = _t.time()
    best_sol = optimizer.run()
    _alns_elapsed = _t.time() - _alns_start
    for r in best_sol.routes:
        r.total_distance = calculate_route_distance(r, G)
        r.total_time = calculate_route_time(r, G)
    routes_with_students = [r for r in best_sol.routes if r.get_student_count() > 0]
    if routes_with_students:
        create_route_map(G, routes_with_students, all_students=best_sol.students,
                         school_coords=school_coords, output_file='route_map.html')
    unserved = [s for s in best_sol.students if not s.is_served]
    output = serialize_routes(best_sol.routes, buses, school_coords, unserved, G)
    output["meta"] = {
        "students_served": len(best_sol.students) - len(unserved),
        "total_time_minutes": sum(r.total_time for r in best_sol.routes),
        "objective": round(best_sol.calculate_objective(), 2)
    }
    with open('output_data.json', 'w') as f: json.dump(output, f, indent=2)
    _te = _t.time() - _run_start
    report = {
        "mode": "generate_routes", "input_file": input_file_path,
        "total_runtime_seconds": round(_te, 2), "optimization_time_seconds": round(_alns_elapsed, 2),
        "students_total": len(best_sol.students), "students_served": len(best_sol.students) - len(unserved),
        "routes_created": len(routes_with_students), "final_objective": round(best_sol.calculate_objective(), 2),
    }
    save_run(data, output, report, map_files={'route_map.html': 'route_map.html'})
    return output


# ============================================================================
# CALLABLE API  (used by experiments/comparison/run_comparison.py)
# ============================================================================

def run_algorithm(data: dict, G, iterations: int = None,
                  stage_walk_limits: dict = None, save=False,
                  G_drive=None, time_budget_seconds: float = None):
    """Run ALNS on *data* using graph *G* and return (best_solution, stats_dict, school_coords).

    Parameters
    ----------
    data : dict            – standard input dict  ({"meta": …, "data": …})
    G    : networkx.Graph  – graph for **walking BFS** (may be constrained)
    G_drive : networkx.Graph – graph for **bus driving** distances (should be
                               the full unconstrained network).  Falls back to
                               *G* when not given, preserving backward compat.
    iterations : int       – override ALNS iterations (None = use data["meta"]["algorithm"]["iterations"])
    stage_walk_limits : dict – override walk limits *after* students are created
                               e.g. {"KG": 0, "MIDDLE": 150, "HIGH": 200}
    save : bool            – persist run artefacts to runs_history/

    Returns
    -------
    tuple : (ServiceSolution, stats_dict, school_coords_dict)
    """
    if G_drive is None:
        G_drive = G
    import time as _time
    students, buses, routes, school_coords, constraints, algo_cfg = load_mode1_input(data, G)

    # Apply stage-specific walk limits if provided
    if stage_walk_limits:
        _stage_map = {
            "KG":         School_Stage.KG,
            "ELEMENTARY": School_Stage.ELEMENTARY,
            "MIDDLE":     School_Stage.MIDDLE,
            "HIGH":       School_Stage.HIGH,
        }
        for s in students:
            stage_name = s.school_stage.name
            if stage_name in stage_walk_limits:
                s.walk_radius = stage_walk_limits[stage_name]

    iters  = iterations or algo_cfg.get("iterations", 60)
    budget = time_budget_seconds or algo_cfg.get("time_budget_seconds", None)
    max_cands = algo_cfg.get("max_candidates_per_student", None)
    # Walking BFS uses G (may be constrained); bus routing uses G_drive (unconstrained)
    precompute_matrix(students, routes, G, G_drive=G_drive)

    initial = ServiceSolution(students, routes, G_drive)
    engine  = ALNSEngine(initial, iterations=iters, time_budget_seconds=budget,
                         max_candidates_per_student=max_cands)
    t0      = _time.time()
    best    = engine.run()
    elapsed = _time.time() - t0

    for r in best.routes:
        from detour_engine import (
            calculate_route_time_from_matrix,
            calculate_route_distance_from_matrix,
        )
        t = calculate_route_time_from_matrix(r.stops, G_drive)
        r.total_time = t if t is not None else 0.0
        d = calculate_route_distance_from_matrix(r.stops, G_drive)
        r.total_distance = d if d is not None else 0.0

    served = sum(1 for s in best.students if s.is_served)
    total  = len(best.students)
    active = [r for r in best.routes if r.get_student_count() > 0]
    total_time = sum(r.total_time for r in active)
    total_dist = sum(r.total_distance for r in active)

    stats = {
        "served": served, "total": total,
        "routes": len(active),
        "total_time": round(total_time, 2),
        "total_dist": round(total_dist, 2),
        "objective": round(best.calculate_objective(), 2),
        "runtime": round(elapsed, 2),
    }

    return best, stats, school_coords

# ============================================================================
# MODE 2: change_location
# ============================================================================

def run_change_location(data, G, input_file_path):
    _run_start = _t.time()
    (student_id, new_coords, change_type, valid_from, valid_until,
     algo_config, routes, all_students, buses, school_coords) = load_mode2_input(data, G)
    method = algo_config.get('method', 'cheapest_insertion')
    daily_budget = data.get('constraints', {}).get('daily_detour_budget_minutes', 5)
    target_student = next((s for s in all_students if s.id == student_id), None)
    if target_student and target_student.is_served:
        if target_student.assigned_stop: target_student.assigned_stop.remove_student(target_student)
    if not target_student:
        from data_loader import school_stage_from_string
        new_loc = data.get('new_location', {})
        target_student = Student(id=student_id, lat=new_coords[0], lon=new_coords[1],
            age=new_loc.get('age', 10), school_stage=school_stage_from_string(new_loc.get('school_stage', 'ELEMENTARY')),
            fee=new_loc.get('fee', 100), assignment=change_type, valid_from=valid_from, valid_until=valid_until)
    else:
        target_student.coords = new_coords
        target_student.assignment = change_type
    precompute_matrix([target_student], routes, G)
    if method == '2opt': success, updated_route, message = insert_with_2opt(target_student, routes, G, change_type, daily_budget)
    elif method == 'alns':
        if target_student not in all_students: all_students.append(target_student)
        optimizer = ALNSEngine(ServiceSolution(all_students, routes, G), iterations=algo_config.get('iterations', 30))
        best_sol = optimizer.run()
        routes = best_sol.routes
        target = next((s for s in best_sol.students if s.id == student_id), None)
        success = target and target.is_served
        message = f"ALNS: {'placed' if success else 'failed'}"
        updated_route = next((r for r in routes if any(any(st.id == student_id for st in stp.students) for stp in r.stops)), None)
    else: success, updated_route, message = process_detour_request(target_student, routes, G, change_type, daily_budget)
    for r in routes: r.total_distance = calculate_route_distance(r, G); r.total_time = calculate_route_time(r, G)
    if os.path.exists('route_map.html'): shutil.copy2('route_map.html', 'route_map_old.html')
    if success:
        create_route_map(G, [r for r in routes if r.get_student_count() > 0], all_students=all_students,
                         school_coords=school_coords, output_file='route_map_new.html')
    unserved = [s for s in all_students if not s.is_served]
    output = serialize_routes(routes, buses, school_coords, unserved, G)
    if not success: output = {"status": "failed", "student_id": student_id, "reason": message}
    with open('output_data.json', 'w') as f: json.dump(output, f, indent=2)
    report = {"mode": "change_location", "status": output.get('status', 'success'), "students_total": len(all_students)}
    save_run(data, output, report, map_files={'route_map_old.html': 'route_map_old.html', 'route_map_new.html': 'route_map_new.html'})
    return output

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Safety-Aware Bus Optimization")
    parser.add_argument('input', nargs='?', default='api_requests/generate_routes_input.json', help="Input JSON")
    parser.add_argument('--unconstrained', action='store_true', help="Disable safety constraints")
    parser.add_argument('--iterations', type=int, default=None, help="Override ALNS iters")
    args = parser.parse_args()
    data = load_json(args.input)
    if args.unconstrained:
        if 'data' in data and 'students' in data['data']:
            for s in data['data']['students']: s['walk_radius_override'] = 400
        if 'meta' in data:
            if 'constraints' not in data['meta']: data['meta']['constraints'] = {}
            data['meta']['constraints'].update({"ride_time_multiplier": 999, "floor_minutes": 999, "ceiling_minutes": 999})
    if args.iterations: data['meta'].setdefault('algorithm', {})['iterations'] = args.iterations
    G = setup_graph(data['meta'], unconstrained=args.unconstrained)
    if data['meta']['mode'] == 'generate_routes': run_generate_routes(data, G, args.input)
    else: run_change_location(data, G, args.input)
