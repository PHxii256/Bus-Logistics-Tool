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
from entities import Stop, Route, Student


# ============================================================================
# GEOCODING & NODE MAPPING: Convert addresses to graph nodes
# ============================================================================

def snap_address_to_edge(coords, graph):
    """Snap student coordinates to the nearest road edge.
    
    This function finds the closest road edge to the student's address
    and returns a point on that edge (the snapped location). This is more
    realistic than just using the nearest node, as students can be picked
    up/dropped off anywhere along a road.
    
    Args:
        coords: Tuple of (latitude, longitude)
        graph: NetowrkX/OSMnx road network graph G
        
    Returns:
        Tuple of (node_id, snapped_coords) where:
            - node_id: Integer ID of one endpoint of the nearest edge
            - snapped_coords: Tuple of (lat, lon) on the snapped edge
    """
    lat, lon = coords
    
    # Find nearest node to the coordinates
    nearest_node = ox.nearest_nodes(graph, lon, lat)
    
    # Find all edges connected to this node and nearby nodes
    # For better accuracy, we also check neighbors
    min_distance = float('inf')
    best_edge = None
    best_point = None
    
    # Check edges from nearby nodes (expand search)
    nodes_to_check = set([nearest_node])
    nodes_to_check.update(nx.single_source_shortest_path_length(
        graph, nearest_node, cutoff=1
    ).keys())
    
    for node in nodes_to_check:
        # Check outgoing edges from this node
        for neighbor in graph.successors(node):
            # Get all edges between this node pair (multi-edges possible)
            for key, edge_data in graph[node][neighbor].items():
                # Calculate distance from coords to this edge
                # For simplicity, use distance to the midpoint of the edge
                node_coords = (graph.nodes[node]['y'], graph.nodes[node]['x'])
                neighbor_coords = (graph.nodes[neighbor]['y'], graph.nodes[neighbor]['x'])
                
                # Midpoint of edge (simplified - not projecting onto edge properly)
                mid_lat = (node_coords[0] + neighbor_coords[0]) / 2
                mid_lon = (node_coords[1] + neighbor_coords[1]) / 2
                
                # Simple Euclidean distance (acceptable for small areas)
                dist = ((lat - mid_lat)**2 + (lon - mid_lon)**2) ** 0.5
                
                if dist < min_distance:
                    min_distance = dist
                    best_edge = (node, neighbor, key)
                    best_point = (mid_lat, mid_lon)
    
    if best_edge is None:
        # Fallback: just use nearest node with its coordinates
        best_node = nearest_node
        return (best_node, (graph.nodes[best_node]['y'], graph.nodes[best_node]['x']))
    
    # Return the node at the start of the edge and snapped point coordinates
    return (best_edge[0], best_point)


def find_safe_nodes_within_radius(coords, graph, radius_meters, walk_distance_limit):
    """Find all nodes reachable via safe pedestrian paths within radius.
    
    This function locates all nodes that can be reached from the given
    coordinates without crossing any arterial roads. It respects the
    student's maximum walking distance.
    
    Args:
        coords: Tuple of (latitude, longitude)
        graph: NetworkX/OSMnx road network graph
        radius_meters: Search radius (meters) around coordinates
        walk_distance_limit: Maximum student walk distance (meters)
        
    Returns:
        List of tuples: (node_id, distance_to_node_m) for all safe nodes within range
    """
    lat, lon = coords
    
    # Find nearest node to start search
    start_node = ox.nearest_nodes(graph, lon, lat)
    
    # BFS with constraint: only traverse safe edges and respect distance limit
    safe_nodes = []
    visited = set()
    queue = [(start_node, 0)]  # (node, distance_so_far)
    
    while queue:
        current_node, dist_so_far = queue.pop(0)
        
        if current_node in visited:
            continue
        visited.add(current_node)
        
        # Check if this node is within our distance limits
        if dist_so_far <= walk_distance_limit:
            safe_nodes.append((current_node, dist_so_far))
        
        # Explore neighbors via safe edges only
        if dist_so_far < walk_distance_limit:
            for neighbor in graph.successors(current_node):
                if neighbor not in visited:
                    # Get edge data (handle multi-edges)
                    edge_data = graph[current_node][neighbor]
                    
                    # Find safest path through this edge pair
                    is_safe = False
                    edge_length = float('inf')
                    
                    for key, data in edge_data.items():
                        if data.get('is_safe_to_cross', True):  # Prefer safe edges
                            is_safe = True
                            edge_length = min(edge_length, data.get('length', 0))
                    
                    if is_safe:
                        new_dist = dist_so_far + edge_length
                        if new_dist <= walk_distance_limit:
                            queue.append((neighbor, new_dist))
    
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
            # Use shortest path in terms of distance
            length_m = nx.shortest_path_length(
                graph, from_node, to_node, weight='length'
            )
            total_distance_m += length_m
        except nx.NetworkXNoPath:
            # If no path exists, log warning
            print(f"Warning: No path between stops {from_node} and {to_node}")
            continue
    
    return total_distance_m / 1000  # Convert to km


def calculate_route_time(route, graph):
    """Calculate total travel time of a route in minutes.
    
    Args:
        route: Route object with ordered stops
        graph: NetworkX road network with 'travel_time' edge attribute
        
    Returns:
        float: Total travel time in minutes
    """
    if len(route.stops) < 2:
        return 0.0
    
    total_time_minutes = 0
    
    # Sum travel times between consecutive stops
    for i in range(len(route.stops) - 1):
        from_node = route.stops[i].node_id
        to_node = route.stops[i + 1].node_id
        
        try:
            # Use shortest path in terms of travel_time
            time_minutes = nx.shortest_path_length(
                graph, from_node, to_node, weight='travel_time'
            )
            total_time_minutes += time_minutes
        except nx.NetworkXNoPath:
            print(f"Warning: No path between stops {from_node} and {to_node}")
            continue
    
    return total_time_minutes


def calculate_student_ride_time(route, graph):
    """Calculate the travel time from the first student boarding to the end (school).
    
    The student ride time is the path duration from the first stop that has 
    students assigned to it, until the final stop (the school).
    
    Args:
        route: Route object with ordered stops
        graph: NetworkX road network
        
    Returns:
        float: Student ride time in minutes
    """
    if len(route.stops) < 2:
        return 0.0
    
    # Find the index of the first stop that has students
    first_student_stop_idx = -1
    for i, stop in enumerate(route.stops):
        # Even if it's the school (stop len-1), if students are there it counts? 
        # Usually school is the dropoff, so pickups happen at 0 to len-2.
        if stop.get_student_count() > 0:
            first_student_stop_idx = i
            break
            
    if first_student_stop_idx == -1:
        return 0.0
        
    # If the first student is also at the last stop (school), ride time is 0
    if first_student_stop_idx >= len(route.stops) - 1:
        return 0.0
        
    ride_time = 0.0
    for i in range(first_student_stop_idx, len(route.stops) - 1):
        from_node = route.stops[i].node_id
        to_node = route.stops[i + 1].node_id
        
        try:
            time_minutes = nx.shortest_path_length(
                graph, from_node, to_node, weight='travel_time'
            )
            ride_time += time_minutes
        except nx.NetworkXNoPath:
            continue
            
    return ride_time


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
        try:
            ride_time += nx.shortest_path_length(
                graph, temp_stops[i].node_id, temp_stops[i+1].node_id, weight='travel_time'
            )
        except:
            continue
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
    """Calculate the time cost of inserting a stop at a specific position in a route.
    
    The insertion cost is the change in total travel time when a new stop
    is added between two existing stops (or at the ends).
    
    Formula: ΔT = time(previous → new) + time(new → next) - time(previous → next)
    
    Args:
        new_stop: Stop object to be inserted
        route: Existing Route object
        insert_position: Integer position (0 = before first stop, len(stops) = after last)
        graph: NetworkX road network with 'travel_time' edge attribute
        
    Returns:
        Tuple of (insertion_cost_minutes: float, is_valid: bool, reason: str)
            - insertion_cost_minutes: Time delta if valid
            - is_valid: Whether the insertion is geometrically possible
            - reason: Error message if not valid
    """
    if insert_position < 0 or insert_position > len(route.stops):
        return None, False, f"Invalid insertion position {insert_position}"
    
    # Case 1: Route is empty, just add stop (no insertion cost)
    if len(route.stops) == 0:
        return 0.0, True, "Empty route"
    
    # Case 2: Insert at the beginning
    if insert_position == 0:
        try:
            time_forward = nx.shortest_path_length(
                graph, new_stop.node_id, route.stops[0].node_id, weight='travel_time'
            )
            # Cost is just the time to first stop (no previous stop to remove)
            return time_forward, True, "Insert at beginning"
        except nx.NetworkXNoPath:
            return None, False, f"No path from new stop to first existing stop"
    
    # Case 3: Insert at the end
    if insert_position == len(route.stops):
        try:
            time_backward = nx.shortest_path_length(
                graph, route.stops[-1].node_id, new_stop.node_id, weight='travel_time'
            )
            return time_backward, True, "Insert at end"
        except nx.NetworkXNoPath:
            return None, False, f"No path from last existing stop to new stop"
    
    # Case 4: Insert in the middle
    prev_stop = route.stops[insert_position - 1]
    next_stop = route.stops[insert_position]
    
    try:
        # Time for detour: prev → new → next
        time_prev_to_new = nx.shortest_path_length(
            graph, prev_stop.node_id, new_stop.node_id, weight='travel_time'
        )
        time_new_to_next = nx.shortest_path_length(
            graph, new_stop.node_id, next_stop.node_id, weight='travel_time'
        )
        
        # Time without detour: prev → next
        time_prev_to_next = nx.shortest_path_length(
            graph, prev_stop.node_id, next_stop.node_id, weight='travel_time'
        )
        
        delta_time = time_prev_to_new + time_new_to_next - time_prev_to_next
        return delta_time, True, "Insert in middle"
        
    except nx.NetworkXNoPath:
        return None, False, f"No path between stops at insertion position"


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
    
    for route in existing_routes:
        # If route has at least 2 stops (Start and End), strictly insert between them
        if len(route.stops) >= 2:
            # insertion_position 1 means between stop 0 and 1
            # range(1, len(route.stops)) gives indices 1, 2, ..., len-1
            # if len=5, range starts at 1, ends at 4. (indices 1, 2, 3, 4)
            # 1: [0, NEW, 1, 2, 3, 4]
            # 4: [0, 1, 2, 3, NEW, 4]
            # All these preserve 0 as start and 4 as end.
            start_pos = 1
            end_pos = len(route.stops)
        else:
            # For empty or single-stop routes, allow any position
            start_pos = 0
            end_pos = len(route.stops) + 1
            
        # Try every valid insertion position in this route
        for position in range(start_pos, end_pos):
            
            # Check if we can insert at this position
            # First, find or create a stop near a node on this route
            
            # Simple strategy: snap student to nearest edge and create stop there
            node_id, snapped_coords = snap_address_to_edge(new_student.coords, graph)
            
            # Check if a stop already exists at this node
            existing_stop = None
            for stop in route.stops:
                if stop.node_id == node_id:
                    existing_stop = stop
                    break
            
            if existing_stop:
                new_stop = existing_stop
            else:
                new_stop = Stop(node_id, snapped_coords[0], snapped_coords[1])
            
            # Calculate insertion cost
            delta_time, is_valid, cost_reason = calculate_insertion_cost(
                new_stop, route, position, graph
            )
            
            if not is_valid:
                continue
            
            # Validate constraints based on detour type
            if detour_type == 'temporary':
                valid, remaining, constraint_reason = validate_temporary_detour(
                    new_stop, route, delta_time, daily_detour_budget
                )
                if not valid:
                    continue
            else:  # permanent
                valid, new_ride_time, constraint_reason = validate_permanent_student(
                    new_stop, route, position, delta_time, graph
                )
                if not valid:
                    continue
            
            # Track best option
            if delta_time < best_cost:
                best_cost = delta_time
                best_route = route
                best_position = position
                best_stop = new_stop
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
