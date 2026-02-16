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
# Pre-snap all students to the graph to save time during optimization
print("\n[Optimization] Pre-snapping students to road network...")
from detour_engine import snap_address_to_edge
for s in students:
    snap_address_to_edge(s.coords, G)
print("Pre-snap complete.")
# ============================================================================
# ALGORITHM: Assign permanent students using Cheapest Insertion
# ============================================================================

print(f"{'='*80}")
print("RUNNING GLOBAL OPTIMIZATION (ALNS)")
print(f"{'='*80}\n")

# Initialize solution state
initial_sol = ServiceSolution(students, routes, G)

# Run ALNS Optimizer
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
