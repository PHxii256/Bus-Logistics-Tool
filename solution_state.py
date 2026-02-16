import copy

class ServiceSolution:
    """Represents a complete state of student-to-route assignments."""
    
    def __init__(self, students, routes, graph):
        """
        Args:
            students: List of all Student objects.
            routes: List of active Route objects.
            graph: Reference to the road network (read-only).
        """
        self.students = students
        self.routes = routes
        self.graph = graph
        
    def calculate_objective(self):
        """
        Formula: (count_served * 10000) - sum(route_travel_times)
        """
        served_count = sum(1 for s in self.students if s.is_served)
        total_time = sum(r.total_time for r in self.routes)
        return (served_count * 10000) - total_time
        
    def clone(self):
        """
        Creates a deep clone by rebuilding the student-stop relationships.
        Ensures the new solution's objects do not point back to the old one.
        """
        # 1. Clone students and reset temporary state
        new_students = [copy.copy(s) for s in self.students]
        for s in new_students:
            s.assigned_stop = None
            s.is_served = False
            
        student_map = {s.id: s for s in new_students}
        
        # 2. Clone routes and their internal stops
        new_routes = []
        for old_route in self.routes:
            new_route = copy.copy(old_route)
            new_route.stops = []
            
            for old_stop in old_route.stops:
                new_stop = copy.copy(old_stop)
                new_stop.students = [] # Clear the old reference list
                
                # Re-link corresponding new students to this new stop
                for old_val_student in old_stop.students:
                    if old_val_student.id in student_map:
                        # add_student updates student.is_served and student.assigned_stop
                        new_stop.add_student(student_map[old_val_student.id])
                
                new_route.stops.append(new_stop)
            
            new_routes.append(new_route)
            
        return ServiceSolution(new_students, new_routes, self.graph)

    def __repr__(self):
        served = sum(1 for s in self.students if s.is_served)
        return f"ServiceSolution(served={served}/{len(self.students)}, objective={self.calculate_objective():.2f})"
