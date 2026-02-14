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


def create_route_map(G, routes, students_to_routes=None, school_coords=None, output_file='route_map.html'):
    """Create a comprehensive map showing routes, stops, and students.
    
    Args:
        G: NetworkX road network graph with 'travel_time' and 'is_safe_to_cross' attributes
        routes: List of Route objects
        students_to_routes: Dict mapping student id to route (for finding assignments)
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
            # For each consecutive pair of stops, find shortest path and draw it
            # We draw all segments (0->1, 1->2, ... , N-1->N)
            for stop_idx in range(len(route.stops) - 1):
                from_stop = route.stops[stop_idx]
                to_stop = route.stops[stop_idx + 1]
                
                # Skip ONLY if the stops are the exact same node (zero-length line)
                if from_stop.node_id == to_stop.node_id:
                    continue
                
                try:
                    # Find shortest path between stops
                    shortest_path_nodes = nx.shortest_path(
                        G, 
                        from_stop.node_id, 
                        to_stop.node_id, 
                        weight='length'
                    )
                    
                    # Convert node IDs to coordinates and add a small offset based on route index
                    # This helps to see overlapping routes side-by-side
                    coord_offset = 0.00003 * (route_idx - 0.5)  # Offset by approx 3-5 meters
                    path_coords = [
                        (G.nodes[node]['y'] + coord_offset, G.nodes[node]['x'] + coord_offset) 
                        for node in shortest_path_nodes
                    ]
                    
                    # Draw the actual path on the map
                    path_line = folium.PolyLine(
                        path_coords,
                        color=route_color,
                        weight=5,
                        opacity=0.8,
                        popup=f"Route {route.route_id}: {from_stop.node_id} → {to_stop.node_id}",
                        dash_array='1, 0'  # Solid line for better visibility of overlaps
                    ).add_to(m)
                    
                    # Add directional arrows using PolyLineTextPath
                    # Since lines are already offset by coord_offset, we keep arrows centered on them
                    folium.plugins.PolyLineTextPath(
                        path_line,
                        '          \u27A4          ',
                        repeat=True,
                        offset=6,  # Small offset to center text on line
                        attributes={'fill': route_color, 'font-weight': 'bold', 'font-size': '18px'}
                    ).add_to(m)
                    
                except nx.NetworkXNoPath:
                    # Fallback to straight line if no path exists
                    straight_coords = [(from_stop.coords[0], from_stop.coords[1]), 
                                      (to_stop.coords[0], to_stop.coords[1])]
                    folium.PolyLine(
                        straight_coords,
                        color='gray',
                        weight=2,
                        opacity=0.5,
                        popup=f"Route {route.route_id}: No path found"
                    ).add_to(m)
        
        # Add stop markers
        for stop_idx, stop in enumerate(route.stops):
            student_count = len(stop.students)
            
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
                    popup=f"<b>{label}: {route.route_id}</b><br>Students: {student_count}<br>Node: {stop.node_id}",
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
                    popup=f"<b>{route.route_id}: SCHOOL</b><br>{student_count} students",
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
                    popup=f"<b>Stop: {route.route_id}-{stop_idx}</b><br>Students: {student_count}<br>Node: {stop.node_id}",
                    tooltip=f"{route.route_id} Stop {stop_idx} ({student_count} students)",
                    color=route_color,
                    fill=True,
                    fillColor=route_color,
                    fillOpacity=0.8,
                    weight=2
                ).add_to(m)
            
            # Record students at this stop
            for student in stop.students:
                student_stop_map[student.id] = (stop, route_color)
    
    # Add student home locations and walking paths to stops
    for student_id, (stop, route_color) in student_stop_map.items():
        # Find the student object to get home coordinates
        student_obj = None
        for route in routes:
            for stop_check in route.stops:
                for stud in stop_check.students:
                    if stud.id == student_id:
                        student_obj = stud
                        break
        
        if student_obj:
            # Marker for student home
            folium.Circle(
                location=student_obj.coords,
                radius=20,
                popup=f"<b>Student: {student_obj.id}</b><br>Stage: {student_obj.school_stage.name}<br>Walk distance: {student_obj.walk_radius}m",
                tooltip=f"Home: {student_obj.id}",
                color='lightblue',
                fill=True,
                fillColor='lightblue',
                fillOpacity=0.6,
                weight=2
            ).add_to(m)
            
            # Try to find nearest nodes to student home and stop for walking path
            try:
                student_nearest_node = ox.nearest_nodes(G, student_obj.coords[1], student_obj.coords[0])
                
                # Find walking path (preferring safe roads)
                try:
                    walking_path_nodes = nx.shortest_path(
                        G, 
                        student_nearest_node, 
                        stop.node_id, 
                        weight='length'
                    )
                    
                    # Convert to coordinates
                    walk_coords = [
                        (G.nodes[node]['y'], G.nodes[node]['x']) 
                        for node in walking_path_nodes
                    ]
                    
                    folium.PolyLine(
                        walk_coords,
                        color=route_color,
                        weight=2,
                        opacity=0.4,
                        popup=f"{student_obj.id} walks to {route_color} {stop.node_id}",
                        dash_array='2, 2',
                        dash_offset='5'
                    ).add_to(m)
                    
                except nx.NetworkXNoPath:
                    # Fallback to straight line
                    walk_coords = [student_obj.coords, stop.coords]
                    folium.PolyLine(
                        walk_coords,
                        color=route_color,
                        weight=2,
                        opacity=0.2,
                        popup=f"{student_obj.id} no path to stop"
                    ).add_to(m)
                    
            except Exception as e:
                # If node lookup fails, just draw straight line
                walk_coords = [student_obj.coords, stop.coords]
                folium.PolyLine(
                    walk_coords,
                    color=route_color,
                    weight=2,
                    opacity=0.2
                ).add_to(m)
    
    # Add legend and statistics
    stats_html = ""
    for route_idx, route in enumerate(routes):
        route_color = route_colors[route_idx % len(route_colors)]
        student_count = sum(len(stop.students) for stop in route.stops)
        stats_html += f"""
        <div style="margin-top: 10px; border-top: 1px solid #ddd; padding-top: 5px;">
            <p style="margin: 0; font-weight: bold; color: {route_color};">Route {route.route_id}</p>
            <p style="margin: 0; font-size: 12px;">
                Stops: {len(route.stops)} | Students: {student_count}<br>
                Distance: {route.total_distance:.2f} km<br>
                Time: {route.total_time:.1f} min
            </p>
        </div>
        """

    legend_html = f'''
    <div style="position: fixed; 
                bottom: 20px; right: 20px; width: 320px; height: auto; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px; border-radius: 5px;">
        <p style="margin: 0; font-weight: bold;">Map Legend</p>
        <p style="margin: 5px 0;">
            <span style="color: darkgreen; font-weight: bold;">●</span> School Location (Start/End)
        </p>
        <p style="margin: 5px 0;">
            <span style="color: lightblue;">●</span> Student Home Location
        </p>
        <p style="margin: 5px 0;">
            <span style="color: red;">●</span> Bus Stop / Boarding Point
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
