"""
Data loader for the Safety-Aware Bus Optimization algorithm

This module loads student, bus, and route data from a JSON config file
and sets up the data structures needed to run the algorithm.
"""

import json
from entities import Student, Bus, Route, Stop, School_Stage
import networkx as nx
import osmnx as ox


def load_input_data(json_file):
    """Load input data from JSON file.
    
    Args:
        json_file: Path to JSON input file
        
    Returns:
        Dict with keys: school, buses, students, routes, config
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    return data


def school_stage_from_string(stage_str):
    """Convert string to School_Stage enum.
    
    Args:
        stage_str: String like 'ELEMENTARY', 'MIDDLE', 'HIGH', 'KG'
        
    Returns:
        School_Stage enum value
    """
    stage_map = {
        'KG': School_Stage.KG,
        'ELEMENTARY': School_Stage.ELEMENTARY,
        'MIDDLE': School_Stage.MIDDLE,
        'HIGH': School_Stage.HIGH
    }
    return stage_map.get(stage_str.upper(), School_Stage.ELEMENTARY)


def create_students_from_data(student_data_list):
    """Create Student objects from JSON data.
    
    Args:
        student_data_list: List of student dicts from JSON
        
    Returns:
        List of Student objects
    """
    students = []
    for data in student_data_list:
        student = Student(
            id=data.get('id'),
            lat=data.get('latitude'),
            lon=data.get('longitude'),
            age=data.get('age'),
            school_stage=school_stage_from_string(data.get('school_stage', 'ELEMENTARY')),
            fee=data.get('fee', 100.0)
        )
        students.append(student)
    
    return students


def create_buses_from_data(bus_data_list):
    """Create Bus objects from JSON data.
    
    Args:
        bus_data_list: List of bus dicts from JSON
        
    Returns:
        Dict mapping bus_id to Bus object
    """
    buses = {}
    for data in bus_data_list:
        bus = Bus(
            bus_type=data.get('type'),
            capacity=data.get('capacity'),
            fixed_cost=data.get('fixed_cost'),
            var_cost_km=data.get('var_cost_km')
        )
        buses[data.get('id')] = bus
    
    return buses


def create_routes_from_data(route_data_list, buses_dict, G):
    """Create Route objects with initial stops from JSON data.
    
    Args:
        route_data_list: List of route dicts from JSON
        buses_dict: Dict of bus_id -> Bus object
        G: NetworkX graph for snapping stops to nodes
        
    Returns:
        List of Route objects
    """
    routes = []
    
    for data in route_data_list:
        bus = buses_dict.get(data.get('bus_id'))
        if not bus:
            print(f"Warning: Bus {data.get('bus_id')} not found")
            continue
        
        route = Route(
            bus=bus,
            route_id=data.get('id'),
            route_tmax=data.get('route_tmax', 60)
        )
        
        # Add stops to route by snapping to nearest nodes
        for stop_idx, stop_data in enumerate(data.get('initial_stops', [])):
            lat = stop_data.get('latitude')
            lon = stop_data.get('longitude')
            
            # Snap to nearest node
            try:
                nearest_node = ox.nearest_nodes(G, lon, lat)
                node_lat = G.nodes[nearest_node]['y']
                node_lon = G.nodes[nearest_node]['x']
                
                stop = Stop(
                    node_id=nearest_node,
                    lat=node_lat,
                    lon=node_lon,
                    stop_id=f"{data.get('id')}-Stop-{stop_idx}"
                )
                route.stops.append(stop)
                
            except Exception as e:
                print(f"Warning: Could not snap stop {stop_data.get('description')} for route {data.get('id')}: {e}")
                continue
        
        routes.append(route)
    
    return routes


def setup_algorithm_inputs(json_file, G):
    """Load all data and prepare for algorithm execution.
    
    Args:
        json_file: Path to JSON input file
        G: NetworkX graph (must have travel_time and is_safe_to_cross attributes)
        
    Returns:
        Tuple of (students, buses, routes, school_coords, config)
    """
    # Load JSON
    data = load_input_data(json_file)
    
    # Extract school coordinates
    school = data.get('school', {})
    school_coords = {
        'latitude': school.get('latitude'),
        'longitude': school.get('longitude'),
        'name': school.get('name')
    }
    
    # Create objects
    students = create_students_from_data(data.get('permanent_students', []))
    buses = create_buses_from_data(data.get('buses', []))
    routes = create_routes_from_data(data.get('routes', []), buses, G)
    
    # Extract config
    config = data.get('algorithm_config', {})
    
    print(f"Loaded {len(students)} students")
    print(f"Loaded {len(buses)} buses")
    print(f"Loaded {len(routes)} routes")
    print(f"School: {school_coords['name']} at ({school_coords['latitude']}, {school_coords['longitude']})")
    
    return students, buses, routes, school_coords, config


def print_input_summary(students, buses, routes, school_coords):
    """Print a summary of loaded data.
    
    Args:
        students: List of Student objects
        buses: Dict of bus_id -> Bus objects
        routes: List of Route objects
        school_coords: Dict with school location
    """
    print(f"\n{'='*80}")
    print("INPUT DATA SUMMARY")
    print(f"{'='*80}\n")
    
    print(f"School: {school_coords['name']}")
    print(f"  Location: ({school_coords['latitude']}, {school_coords['longitude']})\n")
    
    print(f"Students: {len(students)}")
    for student in students:
        print(f"  {student.id}: {student.school_stage.name}, walk={student.walk_radius}m, fee=${student.fee}")
    
    print(f"\nBuses: {len(buses)}")
    for bus_id, bus in buses.items():
        print(f"  {bus_id}: {bus.bus_type}, capacity={bus.capacity}, fixed=${bus.fixed_cost}, var=${bus.var_cost_km}/km")
    
    print(f"\nRoutes: {len(routes)}")
    for route in routes:
        print(f"  {route.route_id}: {len(route.stops)} stops, Tmax={route.route_tmax}min, capacity={route.bus.capacity}")
    
    print(f"\n{'='*80}\n")
