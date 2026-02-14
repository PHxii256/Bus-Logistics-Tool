"""
Loader for output_data.json format
Reconstructs Route and Student objects from pre-computed algorithm data
"""

import json
from entities import Student, Bus, Route, Stop, School_Stage


def school_stage_from_string(stage_str):
    """Convert string to School_Stage enum."""
    stage_map = {
        'KG': School_Stage.KG,
        'ELEMENTARY': School_Stage.ELEMENTARY,
        'MIDDLE': School_Stage.MIDDLE,
        'HIGH': School_Stage.HIGH
    }
    return stage_map.get(stage_str.upper(), School_Stage.ELEMENTARY)


def load_algorithm_data(json_file):
    """Load output_data.json format.
    
    Args:
        json_file: Path to JSON file
        
    Returns:
        Dict with school, buses, routes, students
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    return data


def reconstruct_routes_and_students(data):
    """Reconstruct Route and Student objects from algorithm data.
    
    Args:
        data: Dict loaded from output_data.json
        
    Returns:
        Tuple of (routes, students, school_coords, buses_dict)
    """
    
    # Extract school coordinates
    school = data.get('school', {})
    school_coords = {
        'latitude': school.get('latitude'),
        'longitude': school.get('longitude'),
        'name': school.get('name', 'School')
    }
    
    # Create buses
    buses_dict = {}
    for bus_data in data.get('buses', []):
        bus = Bus(
            bus_type=bus_data.get('type'),
            capacity=bus_data.get('capacity'),
            fixed_cost=bus_data.get('fixed_cost'),
            var_cost_km=bus_data.get('variable_cost_per_km')
        )
        buses_dict[bus_data.get('id')] = bus
    
    # Create routes with stops
    routes = []
    for route_data in data.get('permanent_routes', []):
        bus = buses_dict.get(route_data.get('bus_id'))
        if not bus:
            print(f"Warning: Bus {route_data.get('bus_id')} not found")
            continue
        
        route = Route(
            bus=bus,
            route_id=route_data.get('route_id'),
            route_tmax=route_data.get('max_trip_time_minutes', 60)
        )
        
        # Add stops to route
        for stop_data in route_data.get('stops', []):
            stop = Stop(
                node_id=stop_data.get('node_id'),
                lat=stop_data.get('latitude'),
                lon=stop_data.get('longitude'),
                stop_id=stop_data.get('stop_id')
            )
            route.stops.append(stop)
        
        route.total_distance = route_data.get('total_distance_km', 0)
        route.total_time = route_data.get('total_time_minutes', 0)
        
        routes.append(route)
    
    # Create students and assign to routes/stops
    students = []
    assignment_map = {}  # route_id -> {stop_id -> student_list}
    
    for student_data in data.get('assigned_students', []):
        student = Student(
            id=student_data.get('id'),
            lat=student_data.get('latitude'),
            lon=student_data.get('longitude'),
            age=student_data.get('age'),
            school_stage=school_stage_from_string(student_data.get('school_stage', 'ELEMENTARY')),
            fee=student_data.get('fee', 100)
        )
        students.append(student)
        
        # Track assignment
        route_id = student_data.get('assigned_route')
        if route_id:
            if route_id not in assignment_map:
                assignment_map[route_id] = []
            assignment_map[route_id].append(student)
    
    # Assign students to stops
    for route in routes:
        route_id = route.route_id
        if route_id in assignment_map:
            route_students = assignment_map[route_id]
            
            # Find best stop to assign each student
            for student in route_students:
                # Find closest stop to student
                best_stop = min(route.stops, 
                              key=lambda s: ((s.coords[0] - student.coords[0])**2 + 
                                           (s.coords[1] - student.coords[1])**2) ** 0.5)
                best_stop.add_student(student)
    
    # Add temporary detour students (if needed for visualization)
    temp_students = []
    for detour_data in data.get('temporary_detour_requests', []):
        student = Student(
            id=detour_data.get('student_id'),
            lat=detour_data.get('latitude'),
            lon=detour_data.get('longitude'),
            age=0,
            school_stage=school_stage_from_string(detour_data.get('school_stage', 'ELEMENTARY')),
            fee=0
        )
        temp_students.append(student)
    
    return routes, students, school_coords, buses_dict, temp_students


def print_data_summary(routes, students, school_coords):
    """Print summary of loaded data."""
    print(f"\n{'='*80}")
    print("ALGORITHM DATA SUMMARY")
    print(f"{'='*80}\n")
    
    print(f"School: {school_coords['name']}")
    print(f"  Location: ({school_coords['latitude']}, {school_coords['longitude']})\n")
    
    total_students = sum(sum(len(stop.students) for stop in route.stops) for route in routes)
    print(f"Students Assigned: {total_students}")
    
    print(f"\nRoutes: {len(routes)}")
    for route in routes:
        student_count = sum(len(stop.students) for stop in route.stops)
        print(f"  {route.route_id}: {len(route.stops)} stops, {student_count} students")
        print(f"    Distance: {route.total_distance:.2f} km, Time: {route.total_time:.2f} min")
        print(f"    Capacity: {route.bus.capacity}, Fixed cost: ${route.bus.fixed_cost}")
        if student_count > 0:
            print(f"    Revenue: ${route.get_revenue():.2f}, Cost: ${route.get_total_cost():.2f}, Margin: {route.get_profit_margin():.1%}")
        for stop_idx, stop in enumerate(route.stops):
            if stop.students:
                student_ids = ", ".join([s.id for s in stop.students])
                print(f"      Stop {stop_idx}: {student_ids}")
    
    print(f"\n{'='*80}\n")
