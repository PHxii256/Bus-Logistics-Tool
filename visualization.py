"""
Visualization module for routes, students, and bus stops
Creates an enhanced Folium map showing:
- Road network (colored by safety)
- Bus routes (sequences of stops)
- Student home locations
- Bus stop boarding points
- Walking paths from student homes to stops
"""

import folium
from folium import plugins
import networkx as nx
import osmnx as ox
import math
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from detour_engine import (
    calculate_student_ride_time, 
    calculate_route_path_and_stats,
    find_shortest_path_with_turns,
    walk_distance_on_roads,
    walk_path_on_roads,
    calculate_walk_penalty,
    compute_direct_time,
    compute_student_tmax,
)


def get_address_from_coords(lat, lon):
    """Reverse geocode coordinates to an address string."""
    try:
        geolocator = Nominatim(user_agent="bus_logistics_tool")
        location = geolocator.reverse((lat, lon), language='en')
        return location.address if location else "Address not found"
    except Exception as e:
        print(f"Geocoding error: {e}")
        return "Address lookup error"


def create_route_map(G, routes, students_to_routes=None, all_students=None, school_coords=None, output_file='route_map.html'):
    """Create a comprehensive map showing routes, stops, and students.
    
    Args:
        G: NetworkX road network graph with 'travel_time' and 'is_safe_to_cross' attributes
        routes: List of Route objects
        students_to_routes: Dict mapping student id to route (for finding assignments)
        all_students: List of all Student objects (to identify unserved ones)
        output_file: Output HTML file name
        
    Returns:
        Folium Map object
    """
    
    # Calculate map center
    lats = [G.nodes[node]['y'] for node in G.nodes()]
    lons = [G.nodes[node]['x'] for node in G.nodes()]
    center_lat = (min(lats) + max(lats)) / 2
    center_lon = (min(lons) + max(lons)) / 2
    
    # Create base map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=14,
        tiles='OpenStreetMap'
    )
    
    # Color palette for routes
    route_colors = ['blue', 'red', 'green', 'purple', 'orange', 'darkblue', 'darkred', 'darkgreen']
    
    # Add school location if provided
    if school_coords:
        folium.Marker(
            location=(school_coords.get('latitude', school_coords.get('lat')), 
                     school_coords.get('longitude', school_coords.get('lon'))),
            popup="<b>SCHOOL</b>",
            tooltip="School Location",
            icon=folium.Icon(color='darkgreen', icon='graduation-cap', prefix='fa')
        ).add_to(m)
    
    # Track all students and their stops for visualization
    student_stop_map = {}  # student_id -> stop object
    
    # Add routes, stops, and build student mapping
    for route_idx, route in enumerate(routes):
        route_color = route_colors[route_idx % len(route_colors)]
        
        # Get school coordinates for comparison
        school_lat = school_coords.get('latitude', school_coords.get('lat')) if school_coords else None
        school_lon = school_coords.get('longitude', school_coords.get('lon')) if school_coords else None
        
        # Draw route path connecting stops along actual road network
        if len(route.stops) > 1:
            # Calculate the ENTIRE route path to ensure turns across stops are handled correctly
            # Use travel_time weight to match the engine's choice of path
            full_path_nodes, _ = calculate_route_path_and_stats(G, route.stops, weight='travel_time')
            
            if full_path_nodes:
                # Convert node IDs to coordinates using edge geometries for precise road following
                coord_offset = 0.00003 * (route_idx - 0.5)
                path_coords = []
                
                for i in range(len(full_path_nodes) - 1):
                    u, v = full_path_nodes[i], full_path_nodes[i+1]
                    # Get edge data (handle multigraph keys)
                    data = G.get_edge_data(u, v)
                    if not data: continue
                    
                    # Pick the first available edge (standard for routing)
                    edge_data = data[0] if 0 in data else list(data.values())[0]
                    
                    if 'geometry' in edge_data:
                        # Extract all intermediate points from the road geometry
                        # This prevents the line from "cutting through" buildings
                        for lon, lat in edge_data['geometry'].coords:
                            path_coords.append((lat + coord_offset, lon + coord_offset))
                    else:
                        # Fallback for straight roads
                        path_coords.append((G.nodes[u]['y'] + coord_offset, G.nodes[u]['x'] + coord_offset))
                
                # Add the final node
                last_node = full_path_nodes[-1]
                path_coords.append((G.nodes[last_node]['y'] + coord_offset, G.nodes[last_node]['x'] + coord_offset))
                
                # Draw the full path on the map
                path_polyline = folium.PolyLine(
                    path_coords,
                    color=route_color,
                    weight=5,
                    opacity=0.8,
                    popup=folium.Popup(f"Route {route.route_id} complete path", max_width=450),
                    dash_array='1, 0'
                ).add_to(m)
                
                # Add directional arrows
                folium.plugins.PolyLineTextPath(
                    path_polyline,
                    '          \u27A4          ',
                    repeat=True,
                    offset=6,
                    attributes={'fill': route_color, 'font-weight': 'bold', 'font-size': '24'}
                ).add_to(m)
            else:
                # Fallback to straight lines if path calculation fails
                straight_coords = [(s.coords[0], s.coords[1]) for s in route.stops]
                folium.PolyLine(
                    straight_coords,
                    color='gray',
                    weight=2,
                    opacity=0.5,
                    dash_array='5, 5',
                    popup=f"Route {route.route_id}: Path calculation failed (Check for disconnected road network)"
                ).add_to(m)
        
        # Add stop markers
        for stop_idx, stop in enumerate(route.stops):
            student_count = len(stop.students)
            stop_type = stop.stop_type.capitalize()  # "School" or "Pickup"
            
            # Create student list for popup
            student_list_html = ""
            if stop.students:
                student_list_html = f"<br><b>Stop Type:</b> {stop_type}"
                student_list_html += "<br><b>Students:</b><ul style='margin: 0; padding-left: 15px;'>"
                for s in stop.students:
                    student_list_html += f"<li>{s.id} ({s.school_stage.name})</li>"
                student_list_html += "</ul>"
            else:
                student_list_html = f"<br><b>Stop Type:</b> {stop_type}"
            
            # Check if this is a school stop (start or end)
            is_school_start = (stop_idx == 0)
            is_school_end = (stop_idx == len(route.stops) - 1)
            is_school_stop = False
            
            if school_coords:
                school_lat = school_coords.get('latitude', school_coords.get('lat'))
                school_lon = school_coords.get('longitude', school_coords.get('lon'))
                # Check if stop coords match school coords (within small tolerance)
                if (abs(stop.coords[0] - school_lat) < 0.0005 and 
                    abs(stop.coords[1] - school_lon) < 0.0005):
                    is_school_stop = True
            
            if is_school_stop:
                label = "SCHOOL (Start)" if is_school_start else "SCHOOL (End)"
                if is_school_start and is_school_end: label = "SCHOOL (Start/End)"
                
                # Special styling for school stops: larger marker with label
                folium.CircleMarker(
                    location=(stop.coords[0], stop.coords[1]),
                    radius=12,
                    popup=folium.Popup(f"<b>{label}: {route.route_id}</b><br>Coords: {stop.coords[0]:.6f}, {stop.coords[1]:.6f}<br>Count: {student_count}{student_list_html}", max_width=200),
                    tooltip=f"{route.route_id} {label}",

                    color='darkgreen',
                    fill=True,
                    fillColor='lightgreen',
                    fillOpacity=0.9,
                    weight=3
                ).add_to(m)
                
                # Add a label showing student count at school stop
                folium.Marker(
                    location=(stop.coords[0], stop.coords[1]),
                    popup=folium.Popup(f"<b>{route.route_id}: SCHOOL</b><br>Coords: {stop.coords[0]:.6f}, {stop.coords[1]:.6f}<br>{student_count} students{student_list_html}", max_width=200),
                    icon=folium.Icon(
                        color='green',
                        icon_color='white',
                        prefix='fa',
                        icon='users'
                    )
                ).add_to(m)
            else:
                # Regular stop marker
                folium.CircleMarker(
                    location=(stop.coords[0], stop.coords[1]),
                    radius=8,
                    popup=folium.Popup(f"<b>Stop: {route.route_id}-{stop_idx}</b><br>Coords: {stop.coords[0]:.6f}, {stop.coords[1]:.6f}<br>Count: {student_count}{student_list_html}", max_width=200),
                    tooltip=f"{route.route_id} Stop {stop_idx} ({student_count} students)",

                    color=route_color,
                    fill=True,
                    fillColor=route_color,
                    fillOpacity=0.8,
                    weight=2
                ).add_to(m)
            
            # Record students at this stop
            for student in stop.students:
                student_stop_map[student.id] = (stop, route, route_color)
    
    # Add student home locations and walking paths to stops
    for student_id, (stop, route, route_color) in student_stop_map.items():
        # Find the student object directly from their assigned stop
        student_obj = next((s for s in stop.students if s.id == student_id), None)
        
        if student_obj:
            # Get address from Nominatim
            print(f"Geocoding address for student {student_obj.id}...")
            address_name = get_address_from_coords(student_obj.coords[0], student_obj.coords[1])
            
            # Calculate Individual Ride Time (from assigned stop to school along route)
            ride_time = 0
            stop_idx = -1
            for i, s in enumerate(route.stops):
                if s == stop:
                    stop_idx = i
                    break
            
            if stop_idx != -1:
                # Use turn-aware calculation for student's individual ride
                remaining_stops = route.stops[stop_idx:]
                _, ride_time = calculate_route_path_and_stats(G, remaining_stops, weight='travel_time')
            
            # Calculate Direct Ride Time (from student home node to school)
            direct_time = "N/A"
            direct_time_mins = None
            try:
                student_node = ox.nearest_nodes(G, student_obj.coords[1], student_obj.coords[0])
                school_node = route.stops[-1].node_id
                # Use turn-aware search for direct potential too
                _, direct_time_mins = find_shortest_path_with_turns(G, student_node, school_node, weight='travel_time')
                direct_time = f"{direct_time_mins:.1f} min"
            except:
                pass

            # Per-student ride-time cap and ratio (tiered: clamp(k*T_direct, floor, ceiling))
            ride_cap_html = ""
            if direct_time_mins is not None and direct_time_mins > 0:
                k           = getattr(route, 'ride_time_multiplier', 2.5)
                floor_min   = getattr(route, 'floor_minutes',        45)
                ceiling_min = getattr(route, 'ceiling_minutes',      30)  # extra over direct
                raw_cap          = k * direct_time_mins
                absolute_ceiling = direct_time_mins + ceiling_min
                student_tmax     = max(floor_min, min(raw_cap, absolute_ceiling))

                # Which tier is active?
                if raw_cap <= floor_min:
                    tier_label = f'floor ({floor_min} min)'
                    tier_color = '#888'
                elif raw_cap >= absolute_ceiling:
                    tier_label = f'direct+{ceiling_min} min ({absolute_ceiling:.0f} min)'
                    tier_color = '#c0392b'
                else:
                    tier_label = f'{k}× direct'
                    tier_color = '#2980b9'

                ratio = ride_time / direct_time_mins
                if ratio <= 1.5:
                    ratio_color = 'green'
                elif ratio <= k:
                    ratio_color = 'darkorange'
                else:
                    ratio_color = 'red'

                usage_pct = min(100, int(ride_time / student_tmax * 100)) if student_tmax > 0 else 0
                ride_cap_html = (
                    f'<div style="margin-top:4px; font-size:11px;">'
                    f'  <b>Ride Cap:</b> '
                    f'  <b style="color:{ratio_color};">{ride_time:.1f}</b> / '
                    f'  <b>{student_tmax:.1f} min</b>'
                    f'  &nbsp;<span style="color:{ratio_color};">({ratio:.2f}× direct)</span><br>'
                    f'  <span style="color:{tier_color};">Active tier: {tier_label}</span>'
                    f'  <div style="background:#eee;border-radius:3px;height:6px;margin-top:2px;">'
                    f'    <div style="background:{ratio_color};width:{usage_pct}%;height:6px;border-radius:3px;"></div>'
                    f'  </div>'
                    f'</div>'
                )

            # Calculate walking distance along roads (undirected - ignores one-way)
            # Pedestrians walk on sidewalks/roads without U-turn or direction rules
            student_nearest_node = ox.nearest_nodes(G, student_obj.coords[1], student_obj.coords[0])
            walk_dist_m = walk_distance_on_roads(G, student_nearest_node, stop.node_id)
            if walk_dist_m == float('inf'):
                walk_dist_m = 0.0  # Fallback if graph disconnected
            
            # Calculate walk penalty for popup display
            walk_penalty, _, walk_over_limit = calculate_walk_penalty(
                student_obj, stop.node_id, G
            )

            walk_warning_html = ""
            if student_obj.walk_radius > 0:
                if walk_over_limit and walk_penalty < float('inf'):
                    walk_info = f"{walk_dist_m:.0f}m / {student_obj.walk_radius}m"
                    walk_warning_html = f'<br><b style="color:orange;">Walking beyond limit (+{walk_penalty:.1f} min penalty)</b>'
                elif walk_penalty == float('inf'):
                    walk_info = f"{walk_dist_m:.0f}m / {student_obj.walk_radius}m"
                    walk_warning_html = f'<br><b style="color:red;">Beyond absolute walk maximum!</b>'
                else:
                    walk_info = f"{walk_dist_m:.0f}m / {student_obj.walk_radius}m"
            else:
                walk_info = f"{walk_dist_m:.0f}m (Door-to-Door Required)"
                if walk_dist_m > 20:  # 20m tolerance for snapping to nearest road
                    walk_warning_html = f'<br><b style="color:red;">House far from stop ({walk_dist_m:.0f}m)</b>'

            # Temporary students get a distinct red/orange marker so they stand out
            is_temporary = getattr(student_obj, 'assignment', 'permanent') == 'temporary'
            home_icon = folium.Icon(
                color='orange' if is_temporary else 'blue',
                icon='home',
                prefix='fa'
            )
            assignment_label = (
                f'<br><b style="color:darkorange;">Temporary: {student_obj.valid_from} → {student_obj.valid_until}</b>'
                if is_temporary else ''
            )

            # Marker for student home
            folium.Marker(
                location=student_obj.coords,
                popup=folium.Popup(f"""
                <div style="width: 220px;">
                    <b>Student: {student_obj.id}</b>{assignment_label}<br>
                    Stage: {student_obj.school_stage.name}<br>
                    Home: {student_obj.coords[0]:.6f}, {student_obj.coords[1]:.6f}<br>
                    Address: {address_name}<br>
                    <div style="margin-top:5px; border-top:1px solid #ccc; padding-top:5px;">
                        <b>Routing Details:</b><br>
                        Assigned Route: [ {route.route_id} ]<br>
                        Actual Ride Time: {ride_time:.1f} min<br>
                        Direct Potential: {direct_time}<br>
                        {ride_cap_html}
                        Walk to Stop: {walk_info} {walk_warning_html}
                    </div>
                </div>
                """, max_width=450),
                tooltip=f"{'[TEMP] ' if is_temporary else ''}Home: {student_obj.id} ({student_obj.coords[0]:.5f}, {student_obj.coords[1]:.5f})",
                icon=home_icon
            ).add_to(m)
            
            # Draw walking path along roads (undirected shortest path)
            if walk_dist_m > 5:  # Only draw if meaningful distance
                walk_line_color = 'orange' if walk_over_limit else route_color
                
                # Get road-following walk path
                walk_path_nodes = walk_path_on_roads(G, student_nearest_node, stop.node_id)
                
                if walk_path_nodes and len(walk_path_nodes) >= 2:
                    # Build coordinates following road geometry
                    walk_coords = []
                    for wi in range(len(walk_path_nodes) - 1):
                        u, v = walk_path_nodes[wi], walk_path_nodes[wi + 1]
                        # Try both edge directions (undirected path may reference either)
                        data = G.get_edge_data(u, v) or G.get_edge_data(v, u)
                        if data:
                            edge_data = data[0] if 0 in data else list(data.values())[0]
                            if 'geometry' in edge_data:
                                geom_coords = list(edge_data['geometry'].coords)
                                # Ensure correct direction
                                node_u_x = G.nodes[u]['x'] if u in G.nodes else None
                                if node_u_x is not None and len(geom_coords) > 1:
                                    first_x = geom_coords[0][0]
                                    if abs(first_x - node_u_x) > 0.0001:
                                        geom_coords = list(reversed(geom_coords))
                                for lon, lat in geom_coords:
                                    walk_coords.append((lat, lon))
                            else:
                                walk_coords.append((G.nodes[u]['y'], G.nodes[u]['x']))
                        else:
                            walk_coords.append((G.nodes[u]['y'], G.nodes[u]['x']))
                    # Add final node
                    last_node = walk_path_nodes[-1]
                    walk_coords.append((G.nodes[last_node]['y'], G.nodes[last_node]['x']))
                    
                    folium.PolyLine(
                        locations=walk_coords,
                        color=walk_line_color,
                        weight=2,
                        opacity=0.6,
                        dash_array='8, 6',
                        popup=folium.Popup(
                            f"<div style='width:250px;'>"
                            f"<b>Walking Path</b><br>"
                            f"<b>Student:</b> {student_obj.id} ({student_obj.school_stage.name})<br>"
                            f"<b>Distance:</b> {walk_dist_m:.0f}m (along roads)<br>"
                            f"<b>Walk Limit:</b> {student_obj.walk_radius}m"
                            f"{'<br><b>Penalty:</b> +' + f'{walk_penalty:.1f} min' if walk_over_limit else ''}"
                            f"</div>",
                            max_width=300
                        )
                    ).add_to(m)
                else:
                    # Fallback: straight dashed line if no path found
                    folium.PolyLine(
                        locations=[student_obj.coords, stop.coords],
                        color=walk_line_color,
                        weight=2,
                        opacity=0.4,
                        dash_array='8, 6'
                    ).add_to(m)
    
    # Add unserved students (failed assignments)
    if all_students:
        for student in all_students:
            if not student.is_served:
                print(f"Geocoding address for unassigned student {student.id}...")
                address_name = get_address_from_coords(student.coords[0], student.coords[1])
                failure_msg = getattr(student, 'failure_reason', 'No valid insertion found')
                
                folium.CircleMarker(
                    location=student.coords,
                    radius=7,
                    popup=folium.Popup(f"""
                    <div style="width: 300px;">
                        <b style="color:red;">UNASSIGNED: {student.id}</b><br>
                        Stage: {student.school_stage.name}<br>
                        Address: {address_name}<br>
                        <div style="margin-top:5px; border-top:1px solid #ccc; padding-top:5px;">
                            <b>Reason for Failure:</b><br>
                            <span style="color:darkred;">{failure_msg}</span>
                        </div>
                    </div>
                    """, max_width=400),
                    tooltip=f"UNASSIGNED: {student.id}",
                    color='black',
                    fill=True,
                    fillColor='black',
                    fillOpacity=0.6,
                    weight=3
                ).add_to(m)

    # Calculate total served students and total pool
    total_assigned = 0
    total_pool = len(all_students) if all_students else 0
    
    if all_students:
        total_assigned = sum(1 for s in all_students if s.is_served)
    else:
        # Fallback: sum of students in displayed routes
        total_assigned = sum(sum(len(stop.students) for stop in r.stops) for r in routes)
    
    # Add legend and statistics
    stats_html = ""
    for route_idx, route in enumerate(routes):
        route_color = route_colors[route_idx % len(route_colors)]
        student_count = sum(len(stop.students) for stop in route.stops)
        
        # Calculate student ride time (1st pickup to school)
        ride_time = calculate_student_ride_time(route, G)
        
        stats_html += f"""
        <div style="margin-top: 10px; border-top: 1px solid #ddd; padding-top: 5px;">
            <p style="margin: 0; font-weight: bold; color: {route_color};">Route {route.route_id}</p>
            <p style="margin: 0; font-size: 12px;">
                Stops: {len(route.stops) - 2} | Students: {student_count} / {total_pool}<br>
                Distance: {route.total_distance:.2f} km<br>
                Max Student Ride Time: {ride_time:.1f} min<br>
            </p>
        </div>
        """

    unassigned_count = total_pool - total_assigned
    unassigned_text = f"<br><span style='color:red; font-size:11px;'>({unassigned_count} unassigned)</span>" if unassigned_count > 0 else ""

    legend_html = f'''
    <div style="position: fixed; 
                bottom: 20px; right: 20px; width: 320px; height: auto; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px; border-radius: 5px;">
        <p style="margin: 0; font-weight: bold;">Map Legend</p>
        <div style="font-size: 12px; border-bottom: 1px solid #ccc; padding-bottom: 5px; margin-bottom: 5px;">
            Total Students: {total_pool} | Served: {total_assigned} {unassigned_text}
        </div>
        <p style="margin: 5px 0;">
            <span style="color: darkgreen; font-weight: bold;">●</span> School Location (Start/End)
        </p>
        <p style="margin: 5px 0;">
            <span style="color: blue;">●</span> Student Home (permanent)
        </p>
        <p style="margin: 5px 0;">
            <span style="color: orange;">●</span> Student Home (temporary change)
        </p>
        <p style="margin: 5px 0;">
            <span style="color: red;">●</span> Bus Stop (Frontage/Stop)
        </p>
        <p style="margin: 5px 0;">
            <span style="color: black;">●</span> Unassigned Student (Failed)
        </p>
        <p style="margin: 10px 0 5px 0; font-weight: bold;">Route Statistics</p>
        {stats_html}
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Save map
    m.save(output_file)
    print(f"Route map saved as '{output_file}'")
    
    return m
