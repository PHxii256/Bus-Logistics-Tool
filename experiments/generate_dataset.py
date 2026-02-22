"""
generate_dataset.py — Synthetic student dataset generator for Cairo graph.

Generates students following a Gaussian Annulus distribution around a school,
filtering out locations on major highways (e.g., the Ring Road).

Usage:
    python generate_dataset.py --n_students 50 --output my_dataset.json
"""

import json
import random
import math
import argparse
import osmnx as ox
import numpy as np

# Default school (Victory College School)
DEFAULT_SCHOOL = {
    "name": "Victory College School",
    "latitude": 29.964406,
    "longitude": 31.270319
}

# Highway types to avoid for student addresses
FORBIDDEN_HIGHWAYS = {'motorway', 'motorway_link', 'trunk', 'trunk_link'}

def gaussian_annulus_sample(center_lat, center_lon, peak_km, sigma_km, min_km, max_km):
    """
    Samples a (lat, lon) following a Gaussian distribution of distances from center.
    """
    while True:
        dist = random.gauss(peak_km, sigma_km)
        if min_km <= dist <= max_km:
            break
            
    angle = random.uniform(0, 2 * math.pi)
    
    # 1 degree lat ≈ 111 km
    d_lat = (dist * math.sin(angle)) / 111.0
    # 1 degree lon ≈ 111 * cos(lat) km
    d_lon = (dist * math.cos(angle)) / (111.0 * math.cos(math.radians(center_lat)))
    
    return center_lat + d_lat, center_lon + d_lon

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic student dataset")
    parser.add_argument('--n_students', type=int, default=40, help="Number of students to generate")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--output', default='synthetic_dataset.json', help="Output JSON path")
    
    # Annulus params
    parser.add_argument('--peak_km', type=float, default=2.0, help="Peak distance from school in km")
    parser.add_argument('--sigma_km', type=float, default=1.0, help="Std dev of distance in km")
    parser.add_argument('--min_km', type=float, default=0.4, help="Min distance in km (walk limit fallback)")
    parser.add_argument('--max_km', type=float, default=5.0, help="Max distance in km")
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    print(f"Generating {args.n_students} students around {DEFAULT_SCHOOL['name']}...")

    # 1. Get graph to find valid nodes
    # Using a circle around the school to speed up
    # School is at 29.964, 31.270
    center_lat, center_lon = DEFAULT_SCHOOL['latitude'], DEFAULT_SCHOOL['longitude']
    print(f"Downloading graph around {DEFAULT_SCHOOL['name']} (5km radius)...")
    
    try:
        # Distance-based query is more reliable across versions
        G = ox.graph_from_point((center_lat, center_lon), dist=5000, network_type='drive', simplify=False)
    except Exception as e:
        print(f"Graph download failed: {e}. Trying bbox fallback...")
        north, south, east, west = 29.99, 29.93, 31.30, 31.24
        try:
            G = ox.graph_from_bbox(bbox=(north, south, east, west), network_type='drive', simplify=False)
        except:
             G = ox.graph_from_bbox(north, south, east, west, network_type='drive', simplify=False)
    
    # 2. Identify forbidden nodes (those only on highways)
    forbidden_nodes = set()
    for u, v, k, data in G.edges(keys=True, data=True):
        h_type = data.get('highway', '')
        if isinstance(h_type, list):
            is_forbidden = any(t in FORBIDDEN_HIGHWAYS for t in h_type)
        else:
            is_forbidden = h_type in FORBIDDEN_HIGHWAYS
            
        if is_forbidden:
            forbidden_nodes.add(u)
            forbidden_nodes.add(v)
            
    # Also find nodes that HAVE at least one non-forbidden edge
    allowed_nodes = set()
    for u, v, k, data in G.edges(keys=True, data=True):
        h_type = data.get('highway', '')
        if isinstance(h_type, list):
            is_forbidden = any(t in FORBIDDEN_HIGHWAYS for t in h_type)
        else:
            is_forbidden = h_type in FORBIDDEN_HIGHWAYS
            
        if not is_forbidden:
            allowed_nodes.add(u)
            allowed_nodes.add(v)
            
    # Nodes that are ONLY on forbidden highways are still forbidden
    final_forbidden = forbidden_nodes - allowed_nodes
    print(f"Found {len(final_forbidden)} nodes on restricted highways.")

    # 3. Sample students
    students = []
    for i in range(args.n_students):
        while True:
            lat, lon = gaussian_annulus_sample(
                DEFAULT_SCHOOL['latitude'], DEFAULT_SCHOOL['longitude'],
                args.peak_km, args.sigma_km, args.min_km, args.max_km
            )
            
            # Snap to nearest node and check if it's forbidden
            nearest_node = ox.nearest_nodes(G, lon, lat)
            if nearest_node not in final_forbidden:
                # Use the actual node coordinates if they're close, or just the sampled ones?
                # Usually snapping students to nodes is cleaner for ALNS.
                node_data = G.nodes[nearest_node]
                s_lat, s_lon = node_data['y'], node_data['x']
                
                # Randomize age and stage
                age = random.randint(5, 17)
                if age <= 6: stage = "KG"
                elif age <= 11: stage = "ELEMENTARY"
                elif age <= 14: stage = "MIDDLE"
                else: stage = "HIGH"
                
                students.append({
                    "id": f"S{i+1:03d}",
                    "latitude": s_lat,
                    "longitude": s_lon,
                    "age": age,
                    "school_stage": stage,
                    "fee": 100.0
                })
                break
                
    # 4. Construct output JSON
    output_data = {
        "meta": {
            "mode": "generate_routes",
            "city": "Cairo",
            "description": f"Synthetic dataset - {args.n_students} students, seed {args.seed}",
            "constraints": {
                "ride_time_multiplier": 2.5,
                "floor_minutes": 45,
                "ceiling_minutes": 60,
                "daily_detour_budget_minutes": 5
            },
            "algorithm": {
                "method": "alns",
                "iterations": 200
            }
        },
        "data": {
            "school": DEFAULT_SCHOOL,
            "buses": [
                {"id": "BUS_1", "type": "Standard", "capacity": 60, "fixed_cost": 50, "var_cost_km": 1.0},
                {"id": "BUS_2", "type": "Standard", "capacity": 60, "fixed_cost": 50, "var_cost_km": 1.0},
                {"id": "BUS_3", "type": "Standard", "capacity": 60, "fixed_cost": 50, "var_cost_km": 1.0},
                {"id": "BUS_4", "type": "Standard", "capacity": 60, "fixed_cost": 50, "var_cost_km": 1.0}
            ],
            "students": students
        }
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
        
    print(f"Successfully generated {len(students)} students and saved to {args.output}")

if __name__ == '__main__':
    main()
