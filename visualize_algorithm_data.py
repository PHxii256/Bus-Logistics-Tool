"""
Visualization script for output_data.json

Loads pre-computed route assignments and creates an interactive map showing:
- Routes with actual students assigned
- Bus stops and student groups
- Walking paths from student homes to stops
"""

import pickle
import os
import osmnx as ox
import networkx as nx
from algorithm_data_loader import load_algorithm_data, reconstruct_routes_and_students, print_data_summary
from visualization import create_route_map

# Cache file for the Cairo graph
GRAPH_CACHE = 'cairo_graph.pkl'

print("="*80)
print("SAFETY-AWARE SCHOOL BUS OPTIMIZATION - VISUALIZATION")
print("="*80)

# Load or download graph
print("\n1. Setting up road network graph...")
if os.path.exists(GRAPH_CACHE):
    print(f"   Loading cached graph from {GRAPH_CACHE}...")
    with open(GRAPH_CACHE, 'rb') as f:
        G = pickle.load(f)
    print(f"   ✓ Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
else:
    print("   Downloading Cairo road network (this may take a few minutes)...")
    G = ox.graph_from_bbox([31.229084, 29.925630, 31.331909, 29.991682], network_type='drive')
    
    # Apply Cairo Factor safety/speed tags
    print("   Applying Cairo Factor safety tags...")
    for u, v, k, data in G.edges(keys=True, data=True):
        maxspeed = data.get('maxspeed', 30)
        if isinstance(maxspeed, list):
            try:
                base_speed = float(maxspeed[0])
            except:
                base_speed = 30
        else:
            try:
                base_speed = float(maxspeed)
            except:
                base_speed = 30

        if data['highway'] in ['primary', 'trunk', 'secondary']:
            data['speed_kph'] = base_speed * 0.4
            data['is_safe_to_cross'] = False
        else:
            data['speed_kph'] = base_speed * 0.7
            data['is_safe_to_cross'] = True

        meters_per_min = (data['speed_kph'] * 1000) / 60
        data['travel_time'] = data['length'] / meters_per_min
    
    # Cache for future runs
    print(f"   Caching graph to {GRAPH_CACHE} for faster future runs...")
    with open(GRAPH_CACHE, 'wb') as f:
        pickle.dump(G, f)
    
    print(f"   ✓ Graph ready: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Load algorithm data
print("\n2. Loading algorithm data from output_data.json...")
data = load_algorithm_data('output_data.json')
print(f"   ✓ Data loaded")

# Reconstruct routes and students
print("\n3. Reconstructing routes and students...")
routes, students, school_coords, buses_dict, temp_students = reconstruct_routes_and_students(data)
print(f"   ✓ Routes: {len(routes)}")
print(f"   ✓ Permanent students: {len(students)}")
print(f"   ✓ School: {school_coords['name']} at ({school_coords['latitude']}, {school_coords['longitude']})")

# Print summary
print_data_summary(routes, students, school_coords)

# Create visualization
print("4. Creating comprehensive route map...")
print("   - Drawing actual road network paths")
print("   - Showing student walking routes")
print("   - Marking bus stops and school location")

try:
    create_route_map(G, routes, school_coords=school_coords, output_file='route_map.html')
    print(f"\n✓ SUCCESS: Visualization saved as route_map.html")
    print(f"\nTo view the map:")
    print(f"  - Open route_map.html in your web browser")
    print(f"  - Zoom and pan to explore the routes")
    print(f"  - Click on markers for details")
    print(f"\nWhat the map shows:")
    print(f"  - Blue/Red roads: Road network (Blue=safe residential, Red=arterial)")
    print(f"  - Dashed colored lines: Bus routes following actual streets")
    print(f"  - Colored circles: Bus stops with student count")
    print(f"  - Light blue circles: Student home locations")
    print(f"  - Dotted lines: Walking paths from homes to stops")
    print(f"  - Green marker (top): School location")
    
except Exception as e:
    print(f"\n✗ ERROR creating visualization: {e}")
    import traceback
    traceback.print_exc()

print(f"\n{'='*80}")
