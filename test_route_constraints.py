"""
Tests for Route Boundary Constraints
Ensures that all bus routes strictly start and end at the school.
"""

import json
import os
import unittest
import pickle
from entities import Route, Stop, Student, Bus, School_Stage
from data_loader import load_input_data, setup_algorithm_inputs

class TestRouteBoundaries(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load graph for data_loader
        graph_cache = 'cairo_graph.pkl'
        if os.path.exists(graph_cache):
            with open(graph_cache, 'rb') as f:
                cls.G = pickle.load(f)
        else:
            cls.G = None

    def setUp(self):
        self.input_file = 'input_data.json'
        # Coordinates of Victory College School
        self.school_coords = (29.964406, 31.270319)
        self.tolerance = 0.0001

    def test_input_data_integrity(self):
        """Verify that input_data.json has school as first and last stop for all routes."""
        if not os.path.exists(self.input_file):
            self.skipTest("input_data.json not found")
        with open(self.input_file, 'r') as f:
            data = json.load(f)
        for route in data['routes']:
            stops = route['initial_stops']
            self.assertGreaterEqual(len(stops), 2)
            start_stop = stops[0]
            end_stop = stops[-1]
            self.assertAlmostEqual(start_stop['latitude'], self.school_coords[0], delta=self.tolerance)
            self.assertAlmostEqual(start_stop['longitude'], self.school_coords[1], delta=self.tolerance)
            self.assertAlmostEqual(end_stop['latitude'], self.school_coords[0], delta=self.tolerance)
            self.assertAlmostEqual(end_stop['longitude'], self.school_coords[1], delta=self.tolerance)

    def test_object_creation_continuity(self):
        """Verify that data_loader correctly preserves school start/end during object construction."""
        if self.G is None:
            self.skipTest("Cairo graph cache not found for snapping")
            
        students, buses, routes, school_info, config = setup_algorithm_inputs(self.input_file, self.G)
        
        for route in routes:
            self.assertGreaterEqual(len(route.stops), 2, f"Route {route.route_id} must have at least 2 stops")
            
            start_stop = route.stops[0]
            end_stop = route.stops[-1]
            
            # Distance check (should be effectively 0 from school coords)
            dist_start = ((start_stop.coords[0] - self.school_coords[0])**2 + (start_stop.coords[1] - self.school_coords[1])**2)**0.5
            dist_end = ((end_stop.coords[0] - self.school_coords[0])**2 + (end_stop.coords[1] - self.school_coords[1])**2)**0.5
            
            self.assertLess(dist_start, self.tolerance, f"Route object {route.route_id} start stop is not at school")
            self.assertLess(dist_end, self.tolerance, f"Route object {route.route_id} end stop is not at school")

if __name__ == '__main__':
    unittest.main()
