"""
Safety-Aware Dynamic Detour Engine for School Bus Optimization

This module provides the core algorithms for:
1. Geocoding student addresses to the road network (snapping to edges)
2. Validating safe pedestrian paths (no arterial crossings)
3. Calculating Cheapest Insertion costs for student assignments
4. Managing temporary and permanent student detour requests
5. Enforcing safety and time constraints
"""

import networkx as nx
import osmnx as ox
import math
import heapq
import time
from shapely.geometry import Point, LineString
from shapely.ops import substring


# ============================================================================
# WALKING DISTANCE AND PENALTY CALCULATIONS
# ============================================================================

# Cached undirected graph for pedestrian walking (ignores one-way, U-turn rules)
_WALK_GRAPH = None
# Cache: (student_node, stop_node) -> walk_distance_meters
_WALK_DIST_CACHE = {}
# Cache: (lat, lon) -> nearest_graph_node for walking
_STUDENT_NODE_CACHE = {}

def _get_walk_graph(graph):
    """Get or create an undirected version of the road graph for walking.
    Pedestrians can walk on any road regardless of direction.
    Cached after first creation."""
    global _WALK_GRAPH
    if _WALK_GRAPH is None:
        _WALK_GRAPH = graph.to_undirected()
    return _WALK_GRAPH


def haversine_walk_distance(lat1, lon1, lat2, lon2):
    """Calculate straight-line distance in meters (for quick estimates only)."""
    R = 6371000  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lon2 - lon1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def walk_distance_on_roads(graph, node_a, node_b):
    """Calculate walking distance along roads (undirected) between two nodes.
    Ignores one-way restrictions and U-turn rules since pedestrians use sidewalks.
    Results are cached for fast repeated lookups during ALNS.
    
    Returns:
        float: distance in meters, or float('inf') if no path exists
    """
    if node_a == node_b:
        return 0.0
    cache_key = (node_a, node_b)
    if cache_key in _WALK_DIST_CACHE:
        return _WALK_DIST_CACHE[cache_key]
    walk_g = _get_walk_graph(graph)
    try:
        dist = nx.shortest_path_length(walk_g, node_a, node_b, weight='length')
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        dist = float('inf')
    _WALK_DIST_CACHE[cache_key] = dist
    _WALK_DIST_CACHE[(node_b, node_a)] = dist  # Symmetric
    return dist


def walk_path_on_roads(graph, node_a, node_b):
    """Find the walking path along roads (undirected) between two nodes.
    Returns the list of node IDs along the path.
    
    Returns:
        list: path node IDs, or empty list if no path exists
    """
    if node_a == node_b:
        return [node_a]
    walk_g = _get_walk_graph(graph)
    try:
        return nx.shortest_path(walk_g, node_a, node_b, weight='length')
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []


def get_walk_absolute_max(walk_radius):
    """Get the absolute maximum walk distance based on student stage.
    - ELEMENTARY/KG (0m recommended): 150m emergency max
    - MIDDLE (100m recommended): 300m absolute max
    - HIGH (200m recommended): 500m absolute max
    """
    if walk_radius == 0:
        return 150
    return min(walk_radius * 3, 500)


def calculate_walk_penalty(student, stop_node, graph):
    """Calculate a soft penalty for walking beyond the recommended radius.
    Uses straight-line (haversine) distance for fast O(1) evaluation during ALNS.
    Road-network walking paths are only computed for visualization (see walk_path_on_roads).
    
    Returns:
        (penalty_minutes, actual_walk_m, is_over_limit)
        - penalty_minutes: float, time penalty to add to objective
        - actual_walk_m: float, straight-line walking distance in meters
        - is_over_limit: bool, True if walk exceeds recommended radius
    """
    student_lat, student_lon = student.coords
    
    try:
        stop_lat = graph.nodes[stop_node]['y']
        stop_lon = graph.nodes[stop_node]['x']
    except (KeyError, TypeError):
        return 0.0, 0.0, False
    
    # Haversine: O(1) math, called thousands of times during ALNS
    actual_walk_m = haversine_walk_distance(student_lat, student_lon, stop_lat, stop_lon)
    
    recommended_radius = student.walk_radius
    absolute_max = get_walk_absolute_max(recommended_radius)
    
    # Within recommended radius - no penalty
    if actual_walk_m <= recommended_radius:
        return 0.0, actual_walk_m, False
    
    # Beyond absolute maximum - reject
    if actual_walk_m > absolute_max:
        return float('inf'), actual_walk_m, True
    
    # Between recommended and absolute max - escalating penalty
    excess_m = actual_walk_m - recommended_radius
    if recommended_radius > 0:
        ratio = excess_m / recommended_radius  # 0.0 to 2.0
    else:
        ratio = excess_m / 50.0  # Normalize against 50m baseline for elementary
    
    # 2 minutes penalty per ratio unit
    penalty_minutes = ratio * 2.0
    return penalty_minutes, actual_walk_m, True


def get_turn_penalty(bearing1, bearing2):
    """Calculate a time penalty based on the difference between two edge bearings.
    
    Args:
        bearing1: Bearing of the incoming edge
        bearing2: Bearing of the outgoing edge
        
    Returns:
        float: Penalty in minutes (Massive for U-turns to make them illegal)
    """
    if bearing1 is None or bearing2 is None:
        return 0
        
    angle_diff = abs(bearing1 - bearing2)
    if angle_diff > 180:
        angle_diff = 360 - angle_diff
        
    # U-turn penalty (very sharp angles) - SET TO MASSIVE TO MAKE ILLEGAL
    if angle_diff > 140:
        return 9999.0
    
    # Normal turn penalty (90 degrees approx)
    if angle_diff > 45:
        return 0.2
        
    return 0.0


def calculate_weighted_path_time(graph, path_nodes):
    """Calculate total travel time including turn penalties for a given path.
    """
    if not path_nodes or len(path_nodes) < 2:
        return 0.0
        
    total_time = 0.0
    prev_bearing = None
    
    for i in range(len(path_nodes) - 1):
        u, v = path_nodes[i], path_nodes[i+1]
        
        # Get edge data (using first available key)
        edge_data = graph.get_edge_data(u, v)
        if not edge_data:
            continue
            
        # Handle Multigraph
        data = edge_data[0] if 0 in edge_data else list(edge_data.values())[0]
        
        # Add travel time
        total_time += data.get('travel_time', 0)
        
        # Add turn penalty if we have a previous edge
        current_bearing = data.get('bearing')
        if prev_bearing is not None:
            total_time += get_turn_penalty(prev_bearing, current_bearing)
            
        prev_bearing = current_bearing
        
    return total_time


# Global cache for shortest paths to speed up iterations
_path_cache = {}
_MATRIX_CACHE = {}       # (source, target) -> travel_time in minutes
_MATRIX_CACHE_LENGTH = {} # (source, target) -> length in meters

def get_heuristic_time(u, v, graph, max_speed_kmh=80):
    """Admissible heuristic for time-based A*: straight line distance / max speed."""
    node_u = graph.nodes[u]
    node_v = graph.nodes[v]
    # Haversine distance in meters
    lat1, lon1 = node_u['y'], node_u['x']
    lat2, lon2 = node_v['y'], node_v['x']
    
    # Simple Haversine approx
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lon2 - lon1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    dist_m = 2 * 6371000 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    # Max speed in meters per minute (80km/h conservative max)
    max_meters_per_min = (max_speed_kmh * 1000) / 60
    return dist_m / max_meters_per_min


def _reconstruct_path(came_from, source, target_state):
    """Reconstruct path from predecessor map."""
    path = []
    state = target_state
    while state is not None:
        node = state[0]
        path.append(node)
        state = came_from.get(state)
    path.reverse()
    return path


def find_shortest_path_with_turns(graph, source, target, weight='travel_time', initial_bearing=None):
    """Find the shortest path while making U-turns effectively illegal.
    
    Uses a custom A* search where the state is (node, incoming_bearing).
    This prevents 180-degree turns and applies minor penalties for 90-degree turns.
    Uses a predecessor map instead of storing full paths on the heap for speed.
    """
    if source == target:
        return [source], 0.0

    # Cache key: (source, target, rounded_bearing)
    cache_key = (source, target, round(initial_bearing, 1) if initial_bearing is not None else None)
    
    # Check _path_cache FIRST (has both path AND time — needed by path-dependent functions)
    if cache_key in _path_cache:
        return _path_cache[cache_key]

    # Fallback: _MATRIX_CACHE has time only (no path). Used for time-only lookups.
    if initial_bearing is None and (source, target) in _MATRIX_CACHE:
        return None, _MATRIX_CACHE[(source, target)]

    distances = {}
    came_from = {}  # state -> parent_state (predecessor map)
    
    initial_brg_key = round(initial_bearing, 2) if initial_bearing is not None else None
    start_state = (source, initial_brg_key)
    
    # Counter for stable heap ordering (avoids comparing tuples with None)
    counter = 0
    
    # (estimated_total_time, counter, actual_time, current_node, prev_bearing_key)
    h = get_heuristic_time(source, target, graph)
    pq = [(h, counter, 0.0, source, initial_bearing)]
    came_from[start_state] = None
    distances[start_state] = 0.0
    
    while pq:
        (est_total, _, current_time, current_node, prev_bearing) = heapq.heappop(pq)
        
        state = (current_node, round(prev_bearing, 2) if prev_bearing is not None else None)
        
        if current_node == target:
            # Reconstruct path from predecessor map
            path = _reconstruct_path(came_from, source, state)
            # Store in both caches
            _path_cache[cache_key] = (path, current_time)
            if initial_bearing is None:
                _MATRIX_CACHE[(source, target)] = current_time
            return path, current_time
            
        if state in distances and distances[state] < current_time:
            continue
        
        if current_node not in graph:
            continue
            
        for neighbor in graph.successors(current_node):
            edge_data_dict = graph.get_edge_data(current_node, neighbor)
            for key, data in edge_data_dict.items():
                cost = data.get(weight, 0)
                current_bearing = data.get('bearing')
                
                penalty = get_turn_penalty(prev_bearing, current_bearing)
                if penalty > 1000:
                    continue
                    
                new_time = current_time + cost + penalty
                bearing_key = round(current_bearing, 2) if current_bearing is not None else None
                new_state = (neighbor, bearing_key)
                
                if new_state not in distances or distances[new_state] > new_time:
                    distances[new_state] = new_time
                    came_from[new_state] = state
                    h_neighbor = get_heuristic_time(neighbor, target, graph)
                    counter += 1
                    heapq.heappush(pq, (new_time + h_neighbor, counter, new_time, neighbor, current_bearing))
    
    # CACHE NEGATIVE RESULTS — prevents re-running expensive exhaustive searches
    _path_cache[cache_key] = (None, float('inf'))
    if initial_bearing is None:
        _MATRIX_CACHE[(source, target)] = float('inf')
    return None, float('inf')


def precalculate_distance_matrix(graph, critical_node_ids):
    """Pre-calculate path times AND distances between all critical nodes.
    Uses A* with 180-turn illegal logic. Fills both _MATRIX_CACHE (time)
    and _MATRIX_CACHE_LENGTH (distance) and _path_cache (path+time).
    """
    total = len(critical_node_ids) * (len(critical_node_ids) - 1)
    print(f"Pre-calculating distance matrix for {len(critical_node_ids)} nodes ({total} pairs)...")
    
    count = 0
    start_time = time.time()
    
    for start_node in critical_node_ids:
        for end_node in critical_node_ids:
            if start_node == end_node:
                continue
            
            # This fills _path_cache and _MATRIX_CACHE with time data
            path, t = find_shortest_path_with_turns(graph, start_node, end_node)
            
            # Also compute length from the path to fill _MATRIX_CACHE_LENGTH
            if path and t < float('inf'):
                dist_m = 0.0
                for pi in range(len(path) - 1):
                    ed = graph.get_edge_data(path[pi], path[pi+1])
                    if ed:
                        d = ed[0] if 0 in ed else list(ed.values())[0]
                        dist_m += d.get('length', 0)
                _MATRIX_CACHE_LENGTH[(start_node, end_node)] = dist_m
            else:
                _MATRIX_CACHE_LENGTH[(start_node, end_node)] = float('inf')
            
            count += 1
            if count % 200 == 0:
                elapsed = time.time() - start_time
                print(f"  Processed {count}/{total} pairs... ({elapsed:.1f}s)")
    
    print(f"Pre-calculation complete. Matrix entries: {len(_MATRIX_CACHE)}, Length entries: {len(_MATRIX_CACHE_LENGTH)}")


def shortest_path_length_with_turns(graph, source, target, weight='travel_time', initial_bearing=None):
    """Fast lookup from Matrix Cache if available, else run A*."""
    if initial_bearing is None:
        if weight == 'length' and (source, target) in _MATRIX_CACHE_LENGTH:
            return _MATRIX_CACHE_LENGTH[(source, target)]
        if weight == 'travel_time' and (source, target) in _MATRIX_CACHE:
            return _MATRIX_CACHE[(source, target)]
        
    _, t = find_shortest_path_with_turns(graph, source, target, weight=weight, initial_bearing=initial_bearing)
    return t


def calculate_route_time_from_matrix(stops, graph=None):
    """Ultra-fast route time using O(1) matrix lookups.
    On cache miss: lazily computes the pair via A* and caches it.
    No need for upfront all-pairs precomputation.
    """
    if len(stops) < 2:
        return 0.0
    total = 0.0
    for i in range(len(stops) - 1):
        pair = (stops[i].node_id, stops[i+1].node_id)
        if pair in _MATRIX_CACHE:
            t = _MATRIX_CACHE[pair]
            if t == float('inf'):
                return 9999.0
            total += t
        elif graph is not None:
            # Lazy compute: run A* once, result is cached for future lookups
            path, t = find_shortest_path_with_turns(graph, pair[0], pair[1])
            if t == float('inf'):
                return 9999.0
            # Also compute length while we have the path
            if path:
                dist_m = 0.0
                for pi in range(len(path) - 1):
                    ed = graph.get_edge_data(path[pi], path[pi+1])
                    if ed:
                        d = ed[0] if 0 in ed else list(ed.values())[0]
                        dist_m += d.get('length', 0)
                _MATRIX_CACHE_LENGTH[pair] = dist_m
            total += t
        else:
            return None  # No graph provided, can't compute
    return total


def calculate_route_distance_from_matrix(stops, graph=None):
    """Ultra-fast route distance using O(1) matrix lookups.
    On cache miss: lazily computes the pair via A* and caches it.
    Returns distance in kilometers.
    """
    if len(stops) < 2:
        return 0.0
    total_m = 0.0
    for i in range(len(stops) - 1):
        pair = (stops[i].node_id, stops[i+1].node_id)
        if pair in _MATRIX_CACHE_LENGTH:
            d = _MATRIX_CACHE_LENGTH[pair]
            if d == float('inf'):
                return 0.0
            total_m += d
        elif graph is not None:
            # Lazy compute: run A* to get path, compute length from it
            path, t = find_shortest_path_with_turns(graph, pair[0], pair[1])
            if path and t < float('inf'):
                dist_m = 0.0
                for pi in range(len(path) - 1):
                    ed = graph.get_edge_data(path[pi], path[pi+1])
                    if ed:
                        dd = ed[0] if 0 in ed else list(ed.values())[0]
                        dist_m += dd.get('length', 0)
                _MATRIX_CACHE_LENGTH[pair] = dist_m
                total_m += dist_m
            else:
                _MATRIX_CACHE_LENGTH[pair] = float('inf')
                return 0.0
        else:
            return None  # No graph provided
    return total_m / 1000.0


def get_bearing_of_path(graph, path):
    """Get the bearing of the last edge in a path."""
    if not path or len(path) < 2:
        return None
    u, v = path[-2], path[-1]
    edge_data = graph.get_edge_data(u, v)
    if not edge_data:
        return None
    data = edge_data[0] if 0 in edge_data else list(edge_data.values())[0]
    return data.get('bearing')


from entities import Stop, Route, Student


# ============================================================================
# GEOCODING & NODE MAPPING: Convert addresses to graph nodes
# ============================================================================

def snap_address_to_edge(coords, graph):
    """Snap student coordinates to the exact point on the nearest road edge by splitting it.
    
    This creates a virtual node in the graph at the projected point, effectively
    forcing the routing algorithm to pass right in front of the house.
    """
    lat, lon = coords
    # Use a unique but consistent INT ID for the virtual node for OSMnx compatibility
    lat_key = int(abs(lat) * 1000000)
    lon_key = int(abs(lon) * 1000000)
    vnode_id = int(f"999{lat_key}{lon_key}")
    
    # If the virtual node already exists, return it
    if vnode_id in graph:
        return (vnode_id, (graph.nodes[vnode_id]['y'], graph.nodes[vnode_id]['x']))

    
    point_geom = Point(lon, lat)
    
    try:
        # Find nearest edge (u, v, key)
        u, v, k = ox.nearest_edges(graph, lon, lat)
        original_data = graph.get_edge_data(u, v, k)
        if original_data is None:
            raise ValueError(f"No edge data found for ({u}, {v}, {k})")
        
        edge_data = original_data.copy()
        
        # Get edge geometry for interpolation
        if 'geometry' in edge_data:
            line = edge_data['geometry']
        else:
            u_node = graph.nodes[u]
            v_node = graph.nodes[v]
            line = LineString([(u_node['x'], u_node['y']), (v_node['x'], v_node['y'])])
            
        # Project house onto edge and find exact intersection point
        projected_dist = line.project(point_geom)
        snapped_point = line.interpolate(projected_dist)
        snapped_coords = (snapped_point.y, snapped_point.x)
        
        # 1. Add the virtual node to the graph
        graph.add_node(vnode_id, x=snapped_coords[1], y=snapped_coords[0], street_count=2)
        
        # 2. Calculate the split ratio for attributes
        line_len = line.length if line.length > 0 else 1.0
        ratio = projected_dist / line_len
        # Clamp ratio to avoid very small edges
        ratio = max(0.01, min(0.99, ratio))
        
        total_len = edge_data.get('length', 1.0)
        
        # 3. Create two new edges by splitting the original
        data1 = edge_data.copy()
        data2 = edge_data.copy()
        
        data1['length'] = total_len * ratio
        data2['length'] = total_len * (1 - ratio)
        
        # Split geometry if it exists
        if 'geometry' in edge_data:
            line = edge_data['geometry']
            # Use shapely.ops.substring for precise splitting
            data1['geometry'] = substring(line, 0, projected_dist)
            data2['geometry'] = substring(line, projected_dist, line.length)
        
        # Approximate travel times
        if 'travel_time' in edge_data:
            data1['travel_time'] = edge_data['travel_time'] * ratio
            data2['travel_time'] = edge_data['travel_time'] * (1 - ratio)
            
        # Add the new edges and remove the original one
        graph.add_edge(u, vnode_id, **data1)
        graph.add_edge(vnode_id, v, **data2)
        
        if graph.has_edge(u, v, k):
            graph.remove_edge(u, v, k)
            
        # 4. Handle the reverse direction (if it exists)
        if graph.has_edge(v, u):
            rev_options = graph.get_edge_data(v, u)
            for rev_k, rev_data_orig in rev_options.items():
                rev_data = rev_data_orig.copy()
                rev_data1 = rev_data.copy()
                rev_data2 = rev_data.copy()
                rev_data1['length'] = rev_data.get('length', 1.0) * (1 - ratio)
                rev_data2['length'] = rev_data.get('length', 1.0) * ratio
                
                if 'travel_time' in rev_data:
                    rev_data1['travel_time'] = rev_data['travel_time'] * (1 - ratio)
                    rev_data2['travel_time'] = rev_data['travel_time'] * ratio

                if 'geometry' in rev_data:
                    rev_line = rev_data['geometry']
                    rev_len = rev_line.length
                    # Split at (rev_len - projected_dist) because rev edge starts at v
                    rev_data1['geometry'] = substring(rev_line, 0, rev_len - projected_dist)
                    rev_data2['geometry'] = substring(rev_line, rev_len - projected_dist, rev_len)
                    
                graph.add_edge(v, vnode_id, **rev_data1)
                graph.add_edge(vnode_id, u, **rev_data2)
                graph.remove_edge(v, u, rev_k)
                break
            
        return (vnode_id, snapped_coords)
        
    except Exception as e:
        # Fallback to nearest node if edge splitting fails
        nearest_id = ox.nearest_nodes(graph, lon, lat)
        return (nearest_id, (graph.nodes[nearest_id]['y'], graph.nodes[nearest_id]['x']))



# Cache for pedestrian-safe nodes
_safe_nodes_cache = {}

def find_safe_nodes_within_radius(coords, graph, radius_meters, walk_distance_limit):
    """Find all nodes reachable via safe pedestrian paths within radius."""
    lat, lon = coords
    cache_key = (lat, lon, walk_distance_limit)
    if cache_key in _safe_nodes_cache:
        return _safe_nodes_cache[cache_key]
    
    # Start from the nearest node
    start_node = ox.nearest_nodes(graph, lon, lat)
    
    safe_nodes = []
    visited = set()
    queue = [(start_node, 0)]  # (node, distance_so_far)
    
    while queue:
        current_node, dist_so_far = queue.pop(0)
        
        if current_node in visited or dist_so_far > walk_distance_limit:
            continue
        visited.add(current_node)
        
        safe_nodes.append((current_node, dist_so_far))
        
        # Explore neighbors
        for neighbor in graph.successors(current_node):
            edge_data = graph[current_node][neighbor]
            
            # Use distance/travel_time as the weight for pedestrian walk
            # (Assuming all simple roads are safe for now, filters can be added)
            is_safe = False
            edge_length = float('inf')
            
            for key, data in edge_data.items():
                if data.get('is_safe_to_cross', True):
                    is_safe = True
                    edge_length = min(edge_length, data.get('length', 0))
            
            if is_safe:
                new_dist = dist_so_far + edge_length
                if new_dist <= walk_distance_limit:
                    queue.append((neighbor, new_dist))
    
    _safe_nodes_cache[cache_key] = safe_nodes
    return safe_nodes


# ============================================================================
# ROUTE ANALYSIS: Calculate distance, time, and safety for routes
# ============================================================================


def calculate_route_distance(route, graph):
    """Calculate total distance of a route in kilometers.
    
    Args:
        route: Route object with ordered stops
        graph: NetworkX road network
        
    Returns:
        float: Total distance in kilometers
    """
    if len(route.stops) < 2:
        return 0.0
    
    # Try fast matrix lookup first (lazy: computes on cache miss)
    fast = calculate_route_distance_from_matrix(route.stops, graph)
    if fast is not None:
        return fast
    
    total_distance_m = 0
    
    # Sum distances between consecutive stops
    for i in range(len(route.stops) - 1):
        from_node = route.stops[i].node_id
        to_node = route.stops[i + 1].node_id
        
        # Check length matrix
        if (from_node, to_node) in _MATRIX_CACHE_LENGTH:
            d = _MATRIX_CACHE_LENGTH[(from_node, to_node)]
            if d < float('inf'):
                total_distance_m += d
            continue
        
        try:
            # Use shortest path in terms of distance (turn-aware)
            dist_val = shortest_path_length_with_turns(
                graph, from_node, to_node, weight='length'
            )
            if dist_val < 1000000:
                total_distance_m += dist_val
        except Exception:
            print(f"Warning: No path between stops {from_node} and {to_node}")
            continue
    
    return total_distance_m / 1000  # Convert to km


def calculate_route_path_and_stats(graph, stops, weight='travel_time'):
    """Calculate the full path and travel time for a sequence of stops.
    
    This function ensures that turns BETWEEN segments (at the stops) are also
    penalized, preventing the bus from doing a 180 at a stop.
    
    Returns:
        tuple: (full_path_nodes, total_time)
    """
    if not stops:
        return [], 0.0
    if len(stops) == 1:
        return [stops[0].node_id], 0.0
        
    full_path = []
    total_time = 0.0
    last_bearing = None
    
    for i in range(len(stops) - 1):
        u_node = stops[i].node_id
        v_node = stops[i+1].node_id
        
        path_segment, segment_time = find_shortest_path_with_turns(
            graph, u_node, v_node, weight=weight, initial_bearing=last_bearing
        )
        
        if not path_segment:
            return None, float('inf')
            
        total_time += segment_time
        
        if not full_path:
            full_path.extend(path_segment)
        else:
            full_path.extend(path_segment[1:])
            
        last_bearing = get_bearing_of_path(graph, path_segment)
        
    return full_path, total_time


def calculate_route_time(route, graph):
    """Calculate total travel time of a route in minutes.
    """
    if len(route.stops) < 2:
        return 0.0
    
    _, total_time = calculate_route_path_and_stats(graph, route.stops)
    return total_time if total_time != float('inf') else 9999.0


def calculate_stops_time(stops, graph):
    """Purely functional travel time calculation for a sequence of Stop objects.
    Does not modify any objects.
    """
    if len(stops) < 2:
        return 0.0
    # Reuses the existing logic that handles turn penalties
    _, total_time = calculate_route_path_and_stats(graph, stops)
    return total_time if total_time != float('inf') else 9999.0


def calculate_student_ride_time(route, graph):
    """Calculate the travel time from the first student boarding to the end (school).
    """
    if len(route.stops) < 2:
        return 0.0
    
    # Find the index of the first stop that has students
    first_student_stop_idx = -1
    for i, stop in enumerate(route.stops):
        if stop.get_student_count() > 0:
            first_student_stop_idx = i
            break
            
    if first_student_stop_idx == -1 or first_student_stop_idx >= len(route.stops) - 1:
        return 0.0
        
    student_stops = route.stops[first_student_stop_idx:]
    _, ride_time = calculate_route_path_and_stats(graph, student_stops)
    return ride_time if ride_time != float('inf') else 9999.0


def calculate_student_ride_time_potential(route, new_stop, insert_position, graph):
    """Calculate what the student ride time WOULD be if a stop were inserted.
    
    Args:
        route: Route object
        new_stop: Stop object to potentially insert
        insert_position: Index where new_stop would be placed
        graph: NetworkX graph
        
    Returns:
        float: Predicted student ride time in minutes
    """
    # Create temporary stop list
    temp_stops = list(route.stops)
    temp_stops.insert(insert_position, new_stop)
    
    # This function is used during search, so we assume the 'new_stop' WILL have a student.
    # So the first student stop is either the current first or the new stop.
    
    first_idx = -1
    for i, stop in enumerate(temp_stops):
        # If it's the new stop, it definitely has a student (the one we are trying to add)
        # If it's an existing stop, check its count
        if stop == new_stop or stop.get_student_count() > 0:
            first_idx = i
            break
            
    if first_idx == -1 or first_idx >= len(temp_stops) - 1:
        return 0.0
        
    ride_time = 0.0
    for i in range(first_idx, len(temp_stops) - 1):
        u = temp_stops[i].node_id
        v = temp_stops[i+1].node_id
        # Fast matrix lookup first
        if (u, v) in _MATRIX_CACHE:
            t = _MATRIX_CACHE[(u, v)]
            if t == float('inf'):
                return 9999.0
            ride_time += t
        else:
            # Fallback to graph search
            path, time_minutes = find_shortest_path_with_turns(graph, u, v)
            if time_minutes < float('inf'):
                ride_time += time_minutes
            else:
                return 9999.0
    return ride_time


def calculate_afternoon_ride_time_potential(route, new_stop, insert_position, graph,
                                            target_stop=None):
    """Compute the PM ride time for a student in the reversed route after a potential insertion.
    
    Afternoon route = morning route with pickup stops reversed:
        [School, Stop_n, ..., Stop_1, School]
    A student's afternoon ride = time from school to their stop in the reversed sequence.
    Students near school in the morning (last pickup) are dropped off FIRST in the afternoon.
    
    Args:
        route: Route object (morning direction)
        new_stop: Stop being inserted
        insert_position: Position in morning sequence
        graph: NetworkX graph
        target_stop: Which stop to measure for; None → new_stop itself
    
    Returns:
        float: Afternoon ride time in minutes
    """
    temp_stops = list(route.stops)
    temp_stops.insert(insert_position, new_stop)

    # Reverse the interior pickup stops; keep school at start/end
    interior = temp_stops[1:-1][::-1]
    afternoon_stops = [temp_stops[0]] + interior + [temp_stops[-1]]

    target = target_stop if target_stop is not None else new_stop

    # Find target in afternoon sequence (match by identity first, then node_id)
    target_idx = -1
    for i, s in enumerate(afternoon_stops):
        if s is target or (target_stop is None and s.node_id == new_stop.node_id):
            target_idx = i
            break

    if target_idx <= 0:
        return 0.0  # school position or not found

    ride_time = 0.0
    for i in range(target_idx):
        u = afternoon_stops[i].node_id
        v = afternoon_stops[i + 1].node_id
        t = _MATRIX_CACHE.get((u, v), None)
        if t is None:
            _, t = find_shortest_path_with_turns(graph, u, v)
        if t == float('inf'):
            return 9999.0
        ride_time += t
    return ride_time
    """Check if all students on the route can reach their stops safely.
    
    A route is safe if every student can reach their assigned stop via
    a pedestrian path that does not cross any arterial roads, and within
    their maximum walk distance.
    
    Args:
        route: Route object
        graph: NetworkX road network
        walk_distance_limits: Dict mapping stop_id to max walk distance
        
    Returns:
        Tuple of (is_safe: bool, unsafe_students: list)
            - is_safe: True if all students can reach stops safely
            - unsafe_students: List of (student, reason) for unsafe assignments
    """
    unsafe_students = []
    
    for stop in route.stops:
        for student in stop.students:
            # Check 1: Student is within walk distance of stop
            lat_s, lon_s = student.coords
            lat_stop, lon_stop = stop.coords
            
            walk_distance = math.sqrt((lat_s - lat_stop)**2 + (lon_s - lon_stop)**2) * 111000  # meters
            
            if walk_distance > student.walk_radius:
                unsafe_students.append((student, f"Beyond walk radius: {walk_distance}m > {student.walk_radius}m"))
                continue
            
            # Check 2: Path from student to stop is safe (details in find_safe_nodes_within_radius)
            safe_nodes = find_safe_nodes_within_radius(
                student.coords, graph, 500, student.walk_radius
            )
            safe_node_ids = [n[0] for n in safe_nodes]
            
            if stop.node_id not in safe_node_ids:
                unsafe_students.append((student, "No safe pedestrian path to stop (arterial crossing required)"))
    
    return len(unsafe_students) == 0, unsafe_students


# ============================================================================
# INSERTION COST CALCULATION: Core Cheapest Insertion logic
# ============================================================================

def calculate_insertion_cost(new_stop, route, insert_position, graph):
    """
    Calculate the time cost of inserting a stop at a specific position.
    Uses incremental delta logic to speed up ALNS by 100x.
    """
    if insert_position < 1 or insert_position >= len(route.stops):
        return None, False, "Insertion must be between existing stops"
    
    # We assume routes have fixed Start (Depot/School) and End (School)
    u_node = route.stops[insert_position - 1].node_id
    v_node = route.stops[insert_position].node_id
    
    # Calculate detour leg 1: u -> new
    # (Optional: Pass incoming bearing from route.stops[i-2] if you want extreme turn precision)
    dt1 = shortest_path_length_with_turns(graph, u_node, new_stop.node_id)
    
    # Calculate detour leg 2: new -> v
    dt2 = shortest_path_length_with_turns(graph, new_stop.node_id, v_node)
    
    # Current distance between u and v
    dt_old = shortest_path_length_with_turns(graph, u_node, v_node)
    
    if dt1 == float('inf') or dt2 == float('inf'):
        return None, False, "No valid path"
        
    delta_time = dt1 + dt2 - dt_old
    return delta_time, True, "Success"


# ============================================================================
# CONSTRAINT VALIDATORS: Check if detour/insertion is allowed
# ============================================================================

def validate_temporary_detour(new_stop, route, delta_time_minutes, daily_budget=5):
    """Validate if a temporary detour request can be accepted.
    
    A temporary detour is a one-time request for a student to get an
    alternate drop-off. The cumulative time cost of this detour plus all
    prior detours today must not exceed the daily budget (5 minutes max).
    
    Args:
        new_stop: Stop object for the temporary detour
        route: Existing Route object
        delta_time_minutes: Time cost of this specific detour (minutes)
        daily_budget: Maximum cumulative detour time per day (minutes, default 5)
        
    Returns:
        Tuple of (valid: bool, remaining_budget: float, reason: str)
    """
    # Check 1: Would adding this detour exceed the daily budget?
    current_used = route.get_current_detour_time()
    total_if_added = current_used + delta_time_minutes
    
    if total_if_added > daily_budget:
        remaining = daily_budget - current_used
        return False, remaining, f"Detour would exceed daily budget: {total_if_added:.2f} > {daily_budget} min (only {remaining:.2f} min remaining)"
    
    # Check 2: Is there a safe pedestrian path to this location?
    # (This would be validated via find_safe_nodes_within_radius in practice)
    # For now, assume safe path exists; validation happens during insertion.
    
    remaining = daily_budget - total_if_added
    return True, remaining, f"Temporary detour accepted ({total_if_added:.2f}/{daily_budget} min used)"


def compute_direct_time(student, school_node, graph):
    """Compute direct drive time from student's home to school (minutes).
    
    Uses the precomputed distance matrix when available, falling back to
    an on-demand A* search.  Result is cached on the student object so
    subsequent calls are O(1).
    
    Args:
        student: Student object (must have .coords)
        school_node: OSM node ID of the school
        graph: NetworkX road network
    
    Returns:
        float: Travel time in minutes (direct, no detours)
    """
    if student.direct_time_to_school is not None:
        return student.direct_time_to_school

    lat, lon = student.coords
    student_node = ox.nearest_nodes(graph, lon, lat)

    # Check precomputed matrix first (fast, no graph search)
    cached = _MATRIX_CACHE.get((student_node, school_node), None)
    if cached is not None and cached < float('inf'):
        student.direct_time_to_school = cached
        return cached

    # Fallback: on-demand A* search
    _, t = find_shortest_path_with_turns(graph, student_node, school_node)
    if t == float('inf'):
        # Last-resort: try nearest reachable node
        t = float('inf')
    student.direct_time_to_school = t
    return t


def compute_afternoon_direct_time(student, school_node, graph):
    """Compute direct drive time from school to student's home (afternoon direction).
    
    Due to one-way streets, this may differ from compute_direct_time.
    Cached on student.direct_time_from_school for O(1) repeat calls.
    """
    if getattr(student, 'direct_time_from_school', None) is not None:
        return student.direct_time_from_school

    lat, lon = student.coords
    student_node = ox.nearest_nodes(graph, lon, lat)

    cached = _MATRIX_CACHE.get((school_node, student_node), None)
    if cached is not None and cached < float('inf'):
        student.direct_time_from_school = cached
        return cached

    _, t = find_shortest_path_with_turns(graph, school_node, student_node)
    student.direct_time_from_school = t
    return t


def compute_student_tmax(student, school_node, G,
                          multiplier=2.5,
                          floor_minutes=45,
                          ceiling_minutes=60):  # changed default 30 → 60
    """
    Tiered per-student ride time cap (morning direction: home -> school):

        T_max(s) = clamp(k * T_direct, floor_minutes, T_direct + ceiling_minutes)

    ceiling_minutes = max EXTRA minutes allowed on top of T_direct.
    The absolute upper bound grows with distance, not a fixed number.

    Default parameters:  k=2.5,  floor=45,  ceiling_extra=60
    ┌─────────────┬──────────────┬──────────────┬─────────────┬──────────────────────┐
    │  T_direct   │  k×T_direct  │  T_d+ceiling │  Effective  │  Which rule binds    │
    ├─────────────┼──────────────┼──────────────┼─────────────┼──────────────────────┤
    │   2  min    │     5  min   │    62  min   │   45  min   │ FLOOR (student nearby)│
    │   5  min    │    12.5 min  │    65  min   │   45  min   │ FLOOR                │
    │  10  min    │    25  min   │    70  min   │   45  min   │ FLOOR                │
    │  18  min    │    45  min   │    78  min   │   45  min   │ FLOOR / RATIO tie    │
    │  20  min    │    50  min   │    80  min   │   50  min   │ RATIO (2.5x binds)   │
    │  25  min    │    62.5 min  │    85  min   │   62.5 min  │ RATIO                │
    │  30  min    │    75  min   │    90  min   │   75  min   │ RATIO                │
    │  40  min    │   100  min   │   100  min   │  100  min   │ RATIO / CEILING tie  │
    │  45  min    │   112.5 min  │   105  min   │  105  min   │ CEILING (+60 binds)  │
    │  60  min    │   150  min   │   120  min   │  120  min   │ CEILING              │
    │  90  min    │   225  min   │   150  min   │  150  min   │ CEILING              │
    └─────────────┴──────────────┴──────────────┴─────────────┴──────────────────────┘

    Breakpoints (where control transfers between rules):
      FLOOR → RATIO  at  T_direct = floor / k = 45 / 2.5 = 18 min
      RATIO → CEILING at  T_direct = ceiling / (k-1) = 60 / 1.5 = 40 min
    """
    t_direct = compute_direct_time(student, school_node, G)

    if t_direct == float('inf'):
        return float('inf')
    if t_direct <= 0:
        return floor_minutes

    raw_cap          = multiplier * t_direct
    absolute_ceiling = t_direct + ceiling_minutes
    personal_tmax    = max(floor_minutes, min(raw_cap, absolute_ceiling))

    # Cache on student for visualization and logging
    student.direct_time_to_school = t_direct
    student.personal_tmax         = personal_tmax
    return personal_tmax


def validate_permanent_student(new_stop, route, insert_position, delta_time_minutes, graph,
                               new_student=None):
    """Validate if a permanent student can be added to the route.
    
    Uses per-student ride-time caps when the student object is available:
        T_ride \u2264 min(k * T_direct,  T_direct + Δmax)
    Falls back to the flat route_tmax when no student object is provided.
    
    Also checks that no existing student on the route has their personal
    cap violated by the insertion (inserting a stop after them increases
    their ride time).
    
    A permanent student assignment is accepted if it doesn't violate any
    ride-time constraint and bus capacity is available.
    
    Args:
        new_stop: Stop object for the new student
        route: Existing Route object
        insert_position: Where the stop is being inserted
        delta_time_minutes: Time cost of this insertion (for info)
        graph: NetworkX road network
        new_student: Optional Student object; enables per-student ride-time caps
        
    Returns:
        Tuple of (valid: bool, student_ride_time: float, reason: str)
    """
    # Check 1: Bus capacity
    if route.get_student_count() >= route.bus.capacity:
        return False, route.total_time, f"Bus at capacity ({route.get_student_count()}/{route.bus.capacity})"

    school_node  = route.stops[-1].node_id
    k            = getattr(route, 'ride_time_multiplier', 2.5)
    floor_min    = getattr(route, 'floor_minutes',        45)
    ceiling_min  = getattr(route, 'ceiling_minutes',      60)  # extra minutes over direct

    # Check 2: New student's personal ride-time cap (bidirectional fairness rule)
    # A route is only rejected for ride time if the constraint is broken in BOTH
    # the morning (home→school) AND the afternoon (school→home reversed route).
    new_student_ride_time = calculate_student_ride_time_potential(route, new_stop, insert_position, graph)

    if new_student is not None:
        morning_cap     = compute_student_tmax(new_student, school_node, graph, k, floor_min, ceiling_min)
        morning_bad     = new_student_ride_time > morning_cap

        if morning_bad:
            # Compute afternoon ride for the same student
            afternoon_ride = calculate_afternoon_ride_time_potential(
                route, new_stop, insert_position, graph)
            t_direct_aft   = compute_afternoon_direct_time(new_student, school_node, graph)
            if t_direct_aft > 0 and t_direct_aft < float('inf'):
                afternoon_cap = max(floor_min, min(k * t_direct_aft, t_direct_aft + ceiling_min))
            else:
                afternoon_cap = morning_cap  # symmetric fallback

            afternoon_bad = afternoon_ride > afternoon_cap

            if afternoon_bad:
                t_direct = compute_direct_time(new_student, school_node, graph)
                return (False, new_student_ride_time,
                        f"Ride cap exceeded in BOTH directions: "
                        f"AM {new_student_ride_time:.1f}>{morning_cap:.1f} min, "
                        f"PM {afternoon_ride:.1f}>{afternoon_cap:.1f} min "
                        f"(direct AM={t_direct:.1f}, PM={t_direct_aft:.1f}, "
                        f"clamp({k}×, {floor_min}, +{ceiling_min}))")
            # Morning bad but afternoon OK — acceptable: student gets a short PM ride
    else:
        # Fallback: flat route_tmax
        if new_student_ride_time > route.route_tmax:
            return (False, new_student_ride_time,
                    f"Student ride time exceeds Tmax: {new_student_ride_time:.1f} > {route.route_tmax} min")

    # Check 3: Existing students — only reject if insertion pushes them over cap in BOTH directions.
    if new_student is not None:
        for stop in route.stops:
            if stop.stop_type == 'school':
                continue
            for existing_student in stop.students:
                t_d = compute_direct_time(existing_student, school_node, graph)
                if t_d == float('inf') or t_d <= 0:
                    continue
                ex_floor   = getattr(existing_student, 'floor_minutes',   floor_min)
                ex_ceiling = getattr(existing_student, 'ceiling_minutes', ceiling_min)
                existing_morning_cap = max(ex_floor, min(k * t_d, t_d + ex_ceiling))

                stop_idx = route.stops.index(stop) if stop in route.stops else -1
                if stop_idx != -1 and stop_idx < insert_position:
                    continue  # new stop inserted after them — morning ride unaffected

                # Build the post-insertion stop list and find this student's position
                temp_stops = list(route.stops)
                temp_stops.insert(insert_position, new_stop)
                student_stop_idx_in_temp = next(
                    (ti for ti, ts in enumerate(temp_stops) if ts is stop), -1)
                if student_stop_idx_in_temp == -1:
                    continue

                # Morning ride check
                morning_ride_check = 0.0
                for si in range(student_stop_idx_in_temp, len(temp_stops) - 1):
                    u = temp_stops[si].node_id
                    v = temp_stops[si + 1].node_id
                    t = _MATRIX_CACHE.get((u, v), None)
                    if t is None:
                        _, t = find_shortest_path_with_turns(graph, u, v)
                    if t == float('inf'):
                        morning_ride_check = float('inf')
                        break
                    morning_ride_check += t

                if morning_ride_check > existing_morning_cap:
                    # Check afternoon before rejecting
                    aft_ride_check = calculate_afternoon_ride_time_potential(
                        route, new_stop, insert_position, graph, target_stop=stop)
                    t_d_aft = compute_afternoon_direct_time(existing_student, school_node, graph)
                    if t_d_aft > 0 and t_d_aft < float('inf'):
                        aft_cap = max(ex_floor, min(k * t_d_aft, t_d_aft + ex_ceiling))
                    else:
                        aft_cap = existing_morning_cap
                    if aft_ride_check > aft_cap:
                        return (False, morning_ride_check,
                                f"Insertion pushes {existing_student.id} over cap in BOTH directions: "
                                f"AM {morning_ride_check:.1f}>{existing_morning_cap:.1f}, "
                                f"PM {aft_ride_check:.1f}>{aft_cap:.1f} min")

    return True, new_student_ride_time, "Permanent student accepted"


# ============================================================================
# CHEAPEST INSERTION ALGORITHM: Find best route and position for student
# ============================================================================

def cheapest_insertion(new_student, existing_routes, graph, detour_type='temporary', 
                       daily_detour_budget=5, student_walk_distance_limit=None):
    """Find the cheapest insertion position across all existing routes.
    
    This is the core algorithm for the Dynamic Detour Engine. It evaluates
    inserting a new student (via a new stop) at every possible position in
    every existing route, returns the minimum-cost option that passes all
    constraints.
    
    Args:
        new_student: Student object to be added
        existing_routes: List of Route objects
        graph: NetworkX road network
        detour_type: 'temporary' or 'permanent'
        daily_detour_budget: Budget for temporary detours (minutes)
        student_walk_distance_limit: Optional override for student walk distance
        
    Returns:
        Tuple of (result_dict or None, reason: str)
            result_dict contains:
                - route: Route object
                - new_stop: Stop object
                - insertion_position: Position in route.stops
                - insertion_cost_minutes: Time delta
                - is_new_stop: Whether a new stop was created or existing used
            reason: Explanation of result
    """
    best_cost = float('inf')
    best_route = None
    best_position = None
    best_stop = None
    best_reason = ""
    
    walk_limit = student_walk_distance_limit or new_student.walk_radius
    
    # Mapping from node_id to specific coordinates (for virtual stops)
    node_coords_mapping = {}
    
    # ALWAYS ensure the absolute frontage (virtual node) is the first candidate
    # This is critical for precisely hitting the front of the house
    frontage_node_id, frontage_coords = snap_address_to_edge(new_student.coords, graph)
    candidate_node_ids = [frontage_node_id]
    node_coords_mapping[frontage_node_id] = frontage_coords
    
    # 2. Find other candidate nodes within walking distance (only if walk_limit > 0)
    if walk_limit > 0:
        safe_nodes = find_safe_nodes_within_radius(new_student.coords, graph, 500, walk_limit)
        for node_id, dist in sorted(safe_nodes, key=lambda x: x[1]):
            if node_id not in candidate_node_ids:
                candidate_node_ids.append(node_id)
    
    # 3. Bus-reachable fallback: If frontage is unreachable by bus (one-way / U-turn),
    #    find nearby graph nodes that ARE bus-reachable and within a reasonable walk.
    #    This handles cases where the student's street is one-way AWAY from school.
    frontage_reachable = (
        _MATRIX_CACHE.get((frontage_node_id, existing_routes[0].stops[0].node_id if existing_routes else None), float('inf')) < float('inf')
        and _MATRIX_CACHE.get((existing_routes[0].stops[0].node_id if existing_routes else None, frontage_node_id), float('inf')) < float('inf')
    ) if existing_routes else True
    
    if not frontage_reachable:
        # Find nearest graph nodes that the bus CAN reach via bidirectional BFS
        # (walking is not constrained by one-way streets)
        lat, lon = new_student.coords
        try:
            center_node = ox.nearest_nodes(graph, lon, lat)
            visited = set()
            bfs_queue = [(center_node, 0)]
            reachable_candidates = []
            max_walk = get_walk_absolute_max(walk_limit)  # Stage-based absolute max
            school_node = existing_routes[0].stops[0].node_id
            while bfs_queue and len(reachable_candidates) < 10:
                node, dist = bfs_queue.pop(0)
                if node in visited or dist > max_walk:
                    continue
                visited.add(node)
                to_school = _MATRIX_CACHE.get((node, school_node), float('inf'))
                from_school = _MATRIX_CACHE.get((school_node, node), float('inf'))
                if to_school < float('inf') and from_school < float('inf'):
                    reachable_candidates.append((node, dist))
                # Expand along out-edges
                for neighbor in graph.successors(node):
                    ed = graph.get_edge_data(node, neighbor)
                    if ed:
                        d = ed[0] if 0 in ed else list(ed.values())[0]
                        new_dist = dist + d.get('length', 0)
                        if new_dist <= max_walk:
                            bfs_queue.append((neighbor, new_dist))
                # Also expand along in-edges (walking is bidirectional)
                for predecessor in graph.predecessors(node):
                    ed = graph.get_edge_data(predecessor, node)
                    if ed:
                        d = ed[0] if 0 in ed else list(ed.values())[0]
                        new_dist = dist + d.get('length', 0)
                        if new_dist <= max_walk:
                            bfs_queue.append((predecessor, new_dist))
            
            for node_id, dist in sorted(reachable_candidates, key=lambda x: x[1]):
                if node_id not in candidate_node_ids:
                    candidate_node_ids.append(node_id)
        except Exception:
            pass
    
    # To keep it efficient, we'll only check up to 8 best candidate nodes
    candidate_node_ids = candidate_node_ids[:8]
    
    # Pre-filter: skip candidates not known to be bus-reachable (avoids cold A* calls)
    if existing_routes:
        school_node = existing_routes[0].stops[0].node_id
        filtered = []
        for cid in candidate_node_ids:
            to_s = _MATRIX_CACHE.get((cid, school_node), None)
            from_s = _MATRIX_CACHE.get((school_node, cid), None)
            if to_s is not None and from_s is not None and to_s < float('inf') and from_s < float('inf'):
                filtered.append(cid)
        candidate_node_ids = filtered if filtered else candidate_node_ids  # Fallback if nothing passes
    
    for route in existing_routes:

        # If route has at least 2 stops (Start and End), strictly insert between them
        if len(route.stops) >= 2:
            start_pos = 1
            end_pos = len(route.stops)
        else:
            start_pos = 0
            end_pos = len(route.stops) + 1
            
        # Try every valid insertion position in this route
        for position in range(start_pos, end_pos):
            
            # Check each candidate node for this position
            for node_id in candidate_node_ids:
                # Check if a stop already exists at this node in this route
                existing_stop = None
                for stop in route.stops:
                    if stop.node_id == node_id:
                        existing_stop = stop
                        break
                
                if existing_stop:
                    eval_stop = existing_stop
                else:
                    # Use virtual snapped coords if available, else use graph node coords
                    if node_id in node_coords_mapping:
                        lat, lon = node_coords_mapping[node_id]
                    else:
                        lat = graph.nodes[node_id]['y']
                        lon = graph.nodes[node_id]['x']
                    
                    eval_stop = Stop(node_id, lat, lon)

                
                # Calculate insertion cost
                delta_time, is_valid, cost_reason = calculate_insertion_cost(
                    eval_stop, route, position, graph
                )
                
                if not is_valid:
                    continue
                
                # Validate constraints based on detour type
                if detour_type == 'temporary':
                    valid, remaining, constraint_reason = validate_temporary_detour(
                        eval_stop, route, delta_time, daily_detour_budget
                    )
                    if not valid:
                        continue
                else:  # permanent
                    valid, new_ride_time, constraint_reason = validate_permanent_student(
                        eval_stop, route, position, delta_time, graph,
                        new_student=new_student
                    )
                    if not valid:
                        continue
                
                # Calculate walk penalty (straight-line distance)
                walk_penalty, walk_m, over_limit = calculate_walk_penalty(
                    new_student, node_id, graph
                )
                if walk_penalty == float('inf'):
                    continue  # Beyond absolute walk maximum
                
                # Total cost = bus insertion time + walk penalty
                total_cost = delta_time + walk_penalty
                
                # Track best option (penalized cost for comparison)
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_route = route
                    best_position = position
                    best_stop = eval_stop
                    best_reason = f"Cost: {delta_time:.2f} min + walk penalty {walk_penalty:.1f} min (walk {walk_m:.0f}m), {constraint_reason}"
    
    # Return result
    if best_route is not None:
        result = {
            'route': best_route,
            'new_stop': best_stop,
            'insertion_position': best_position,
            'insertion_cost_minutes': best_cost,
            'is_new_stop': best_stop not in best_route.stops
        }
        return result, best_reason
    else:
        return None, "No valid insertion found in any existing route"


# ============================================================================
# 2-OPT INTRA-ROUTE OPTIMIZATION
# ============================================================================

def two_opt_improve(route, graph):
    """Apply 2-opt local search to improve a single route's stop ordering.
    
    Repeatedly tries reversing sub-sequences of pickup stops (keeping
    school start/end stops fixed) and keeps each improvement that reduces
    total route time. Runs until no further improvement is found.
    
    Args:
        route: Route object with at least 2 stops (school-start ... school-end)
        graph: NetworkX road graph
        
    Returns:
        float: Total time improvement in minutes (positive = saved)
    """
    if len(route.stops) < 4:
        # Need at least: school-start, 2 pickups, school-end to swap anything
        return 0.0
    
    original_time = calculate_route_time(route, graph)
    improved = True
    
    while improved:
        improved = False
        # Only reverse among interior stops (index 1 .. len-2)
        n = len(route.stops)
        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                # Try reversing stops[i..j]
                new_stops = route.stops[:i] + route.stops[i:j+1][::-1] + route.stops[j+1:]
                
                # Evaluate the new ordering
                old_stops = route.stops
                route.stops = new_stops
                new_time = calculate_route_time(route, graph)
                
                if new_time < original_time - 0.01:
                    # Improvement found — keep it
                    original_time = new_time
                    route.total_time = new_time
                    improved = True
                else:
                    # Revert
                    route.stops = old_stops
    
    final_time = calculate_route_time(route, graph)
    route.total_time = final_time
    route.total_distance = calculate_route_distance(route, graph)
    
    return max(0.0, original_time - final_time)


def insert_with_2opt(new_student, existing_routes, graph, detour_type='temporary',
                     daily_detour_budget=5):
    """Insert student via cheapest insertion, then improve with 2-opt.
    
    Combines cheapest_insertion for initial placement with two_opt_improve
    for local search. This is the recommended fast algorithm for Mode 2
    change_location requests.
    
    Args:
        new_student: Student to insert
        existing_routes: List of Route objects
        graph: NetworkX road graph
        detour_type: 'temporary' or 'permanent'
        daily_detour_budget: Budget for temporary detours
        
    Returns:
        Same as process_detour_request: (success, route_or_none, message)
    """
    if not existing_routes:
        return False, None, "No existing routes available"
    
    # Step 1: Cheapest Insertion
    result, reason = cheapest_insertion(
        new_student, existing_routes, graph, detour_type, daily_detour_budget
    )
    
    if result is None:
        return False, None, f"Insertion failed: {reason}"
    
    route = result['route']
    new_stop = result['new_stop']
    position = result['insertion_position']
    delta_time = result['insertion_cost_minutes']
    
    # Insert the stop
    if result['is_new_stop']:
        route.stops.insert(position, new_stop)
    
    # Set student assignment type
    new_student.assignment = detour_type  # "temporary" or "permanent"
    new_stop.add_student(new_student)
    
    # Step 2: 2-opt improvement on the affected route
    time_saved = two_opt_improve(route, graph)
    
    # Update accounting
    if detour_type == 'temporary':
        route.add_detour_time(max(0, delta_time - time_saved))
    
    route.total_distance = calculate_route_distance(route, graph)
    route.total_time = calculate_route_time(route, graph)
    
    message = (f"2-opt insert: {new_student.id} -> Route {route.route_id}, "
               f"Stop {new_stop.node_id}, Cost +{delta_time:.2f} min, "
               f"2-opt saved {time_saved:.2f} min")
    return True, route, message


# ============================================================================
# MAIN DETOUR REQUEST HANDLER
# ============================================================================

def process_detour_request(student, existing_routes, graph, detour_type='temporary', 
                          daily_detour_budget=5):
    """Process a student's request for a detour/alternate drop-off location.
    
    This is the main entry point for handling dynamic detour requests. It:
    1. Snaps the student to the road network
    2. Finds the cheapest insertion across all routes
    3. Validates safety and time constraints
    4. Updates the route if accepted
    
    Args:
        student: Student object with coords and walk_radius set
        existing_routes: List of Route objects operating today
        graph: NetworkX road network with 'travel_time' and 'is_safe_to_cross' attributes
        detour_type: 'temporary' (one-time request) or 'permanent' (add to system)
        daily_detour_budget: Daily budget for temporary detours (minutes, default 5)
        
    Returns:
        Tuple of (success: bool, route_updated: Route or None, message: str)
    """
    if not existing_routes:
        return False, None, "No existing routes available"
    
    # Run cheapest insertion algorithm
    result, reason = cheapest_insertion(
        student, existing_routes, graph, detour_type, daily_detour_budget
    )
    
    if result is None:
        return False, None, f"Detour rejected: {reason}"
    
    # Unpack result
    route = result['route']
    new_stop = result['new_stop']
    position = result['insertion_position']
    delta_time = result['insertion_cost_minutes']
    
    # Update route
    if result['is_new_stop']:
        route.stops.insert(position, new_stop)
    
    # Set student assignment type
    student.assignment = detour_type  # "temporary" or "permanent"
    new_stop.add_student(student)
    
    # Update accounting
    if detour_type == 'temporary':
        route.add_detour_time(delta_time)
    
    # Recalculate route totals
    route.total_distance = calculate_route_distance(route, graph)
    route.total_time = calculate_route_time(route, graph)
    
    message = f"Detour accepted: {student.id} -> Route {route.route_id}, Stop {new_stop.node_id}, Cost +{delta_time:.2f} min"
    return True, route, message
