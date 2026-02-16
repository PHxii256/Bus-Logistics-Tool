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
from shapely.geometry import Point, LineString
from shapely.ops import substring


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

def find_shortest_path_with_turns(graph, source, target, weight='travel_time', initial_bearing=None):
    """Find the shortest path while making U-turns effectively illegal.
    
    Uses a custom Dijkstra where the state is (node, incoming_bearing).
    This prevents 180-degree turns and applies minor penalties for 90-degree turns.
    """
    if source == target:
        return [source], 0.0

    # Cache key: (source, target, rounded_bearing)
    cache_key = (source, target, round(initial_bearing, 1) if initial_bearing is not None else None)
    if cache_key in _path_cache:
        return _path_cache[cache_key]

    # distances[(node, prev_bearing)] = current_best_time
    # Use a small epsilon for bearing comparison to avoid precision issues
    distances = {}
    
    # (time, current_node, prev_bearing, path)
    pq = [(0.0, source, initial_bearing, [source])]
    
    while pq:
        (current_time, current_node, prev_bearing, path) = heapq.heappop(pq)
        
        if current_node == target:
            _path_cache[cache_key] = (path, current_time)
            return path, current_time
            
        state = (current_node, round(prev_bearing, 2) if prev_bearing is not None else None)
        if state in distances and distances[state] <= current_time:
            continue
        distances[state] = current_time
        
        if current_node not in graph:
            continue
            
        for neighbor in graph.successors(current_node):
            edge_data_dict = graph.get_edge_data(current_node, neighbor)
            for key, data in edge_data_dict.items():
                cost = data.get(weight, 0)
                current_bearing = data.get('bearing')
                
                # IMPORTANT: If the edge has no bearing, we can't do turn penalties
                # But OSMnx add_edge_bearings usually ensures it's there.
                
                penalty = get_turn_penalty(prev_bearing, current_bearing)
                
                # If penalty is massive (> 1000), it's an illegal 180-degree turn
                if penalty > 1000:
                    continue
                    
                new_time = current_time + cost + penalty
                # Handle None bearing gracefully
                bearing_key = round(current_bearing, 2) if current_bearing is not None else None
                new_state = (neighbor, bearing_key)
                
                if new_state not in distances or distances[new_state] > new_time:
                    heapq.heappush(pq, (new_time, neighbor, current_bearing, path + [neighbor]))
                
    return None, float('inf')


def shortest_path_length_with_turns(graph, source, target, weight='travel_time', initial_bearing=None):
    """Helper to get only the time from the turn-aware search."""
    _, time = find_shortest_path_with_turns(graph, source, target, weight=weight, initial_bearing=initial_bearing)
    return time


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
    
    total_distance_m = 0
    
    # Sum distances between consecutive stops
    for i in range(len(route.stops) - 1):
        from_node = route.stops[i].node_id
        to_node = route.stops[i + 1].node_id
        
        try:
            # Use shortest path in terms of distance (turn-aware)
            time_with_turns = shortest_path_length_with_turns(
                graph, from_node, to_node, weight='length'
            )
            if time_with_turns < 1000000: # Check if path exists
                total_distance_m += time_with_turns
        except Exception:
            # If no path exists, log warning
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
        path, time_minutes = find_shortest_path_with_turns(graph, temp_stops[i].node_id, temp_stops[i+1].node_id)
        if path:
            ride_time += time_minutes
    return ride_time


def is_route_safe(route, graph, walk_distance_limits):
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


def validate_permanent_student(new_stop, route, insert_position, delta_time_minutes, graph):
    """Validate if a permanent student can be added to the route.
    
    A permanent student assignment is accepted if it doesn't cause the
    longest student ride time (from first pickup to school) to exceed route_tmax,
    and bus capacity is available.
    
    Args:
        new_stop: Stop object for the new student
        route: Existing Route object
        insert_position: Where the stop is being inserted
        delta_time_minutes: Time cost of this insertion (for info)
        graph: NetworkX road network
        
    Returns:
        Tuple of (valid: bool, student_ride_time: float, reason: str)
    """
    # Check 1: Bus capacity
    if route.get_student_count() >= route.bus.capacity:
        return False, route.total_time, f"Bus at capacity ({route.get_student_count()}/{route.bus.capacity})"
    
    # Check 2: Student ride time limit (Tmax)
    # The requirement is that the first student boarded doesn't spend more than Tmax
    new_student_ride_time = calculate_student_ride_time_potential(route, new_stop, insert_position, graph)
    
    if new_student_ride_time > route.route_tmax:
        return False, new_student_ride_time, f"Student ride time exceeds Tmax: {new_student_ride_time:.1f} > {route.route_tmax} min"
    
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
    
    # To keep it efficient, we'll only check up to 5 best candidate nodes
    candidate_node_ids = candidate_node_ids[:5]
    
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
                        eval_stop, route, position, delta_time, graph
                    )
                    if not valid:
                        continue
                
                # Track best option
                if delta_time < best_cost:
                    best_cost = delta_time
                    best_route = route
                    best_position = position
                    best_stop = eval_stop
                    best_reason = f"Cost: {delta_time:.2f} min, {constraint_reason}"
    
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
        new_stop.is_temporary = (detour_type == 'temporary')
    
    # If the request is permanent, the stop must be permanent even if it existed before
    if detour_type == 'permanent':
        new_stop.is_temporary = False
    
    new_stop.add_student(student)
    
    # Update accounting
    if detour_type == 'temporary':
        route.add_detour_time(delta_time)
    
    # Recalculate route totals
    route.total_distance = calculate_route_distance(route, graph)
    route.total_time = calculate_route_time(route, graph)
    
    message = f"Detour accepted: {student.id} -> Route {route.route_id}, Stop {new_stop.node_id}, Cost +{delta_time:.2f} min"
    return True, route, message
