import random
import math
import time
import numpy as np
import copy
from detour_engine import (
    cheapest_insertion, 
    calculate_route_time, 
    calculate_route_distance,
    calculate_stops_time,
    calculate_insertion_cost,
    validate_permanent_student,
    snap_address_to_edge
)
from entities import Stop

# ============================================================================
# DESTROY OPERATORS
# ============================================================================

def random_removal(solution, n):
    """Removes n random students from the solution."""
    served_students = [s for s in solution.students if s.is_served]
    n = min(n, len(served_students))
    if n == 0:
        return []
    
    removed = random.sample(served_students, n)
    for student in removed:
        _remove_student_from_solution(solution, student)
    return removed

def worst_cost_removal(solution, n):
    """Removes students who add the most travel time to their routes."""
    served_students = [s for s in solution.students if s.is_served]
    n = min(n, len(served_students))
    if n == 0:
        return []

    removal_candidates = []
    for student in served_students:
        stop = student.assigned_stop
        route = next((r for r in solution.routes if stop in r.stops), None)
        
        if not route:
            continue
            
        old_time = route.total_time
        
        # Calculate time saved if this student is removed
        temp_stops = list(route.stops)
        # Find the correct stop in temp_stops (since they were cloned)
        current_stop = next((s for s in temp_stops if s.node_id == stop.node_id), None)
        
        if current_stop:
            # We want to know the time of the route WITHOUT this specific student
            # If they are the only student at the stop, the stop is removed
            if len(current_stop.students) <= 1:
                temp_stops.remove(current_stop)
            
            new_time = calculate_stops_time(temp_stops, solution.graph)
            delta_time = old_time - new_time
            removal_candidates.append((student, delta_time))
    
    # Sort by time saved (highest first)
    removal_candidates.sort(key=lambda x: x[1], reverse=True)
    removed = [c[0] for c in removal_candidates[:n]]
    
    for student in removed:
        _remove_student_from_solution(solution, student)
    return removed

def _remove_student_from_solution(solution, student):
    """Helper to safely decouple student from stop and route."""
    stop = student.assigned_stop
    if not stop: return
    
    route = next((r for r in solution.routes if stop in r.stops), None)
    if not route: return
        
    stop.remove_student(student)
    
    # If stop becomes empty, remove it from the route entirely
    if len(stop.students) == 0:
        route.stops.remove(stop)
        
    # Update route metrics after removal
    route.total_time = calculate_route_time(route, solution.graph)
    route.total_distance = calculate_route_distance(route, solution.graph)


# ============================================================================
# REPAIR OPERATORS
# ============================================================================

def greedy_repair(solution):
    """Inserts all unassigned students using the cheapest available insertion point."""
    unassigned = [s for s in solution.students if not s.is_served]
    random.shuffle(unassigned) # Shuffle to provide variation across calls
    
    for student in unassigned:
        # Reuse existing logic from detour_engine
        result, _ = cheapest_insertion(student, solution.routes, solution.graph, detour_type='permanent')
        if result:
            _apply_insertion(solution, student, result)

def regret_repair(solution, k=2):
    """Inserts students with the highest 'regret' cost between best and k-best options.
    Optimized to minimize redundant calculations.
    """
    unassigned = [s for s in solution.students if not s.is_served]
    if not unassigned:
        return

    # 1. Pre-calculate frontage nodes to avoid repeated snapping logic
    student_frontages = {}
    for s in unassigned:
        node_id, coords = snap_address_to_edge(s.coords, solution.graph)
        student_frontages[s.id] = (node_id, coords)

    # 2. Initial calculation for all unassigned students
    # student_route_options[student_id][route_id] = list of insertions
    student_route_options = {}
    for s in unassigned:
        student_route_options[s.id] = {}
        for route in solution.routes:
            student_route_options[s.id][route.route_id] = _get_insertions_for_route(
                s, route, solution.graph, student_frontages[s.id]
            )

    while unassigned:
        best_regret = -1
        target_student = None
        target_insertion = None
        
        for student in unassigned:
            # Flatten all valid options across all routes
            all_options = []
            for r_id in student_route_options[student.id]:
                all_options.extend(student_route_options[student.id][r_id])
            
            if not all_options:
                continue
            
            all_options.sort(key=lambda x: x['insertion_cost_minutes'])
            
            # Regret calculation
            if len(all_options) >= k:
                regret = all_options[k-1]['insertion_cost_minutes'] - all_options[0]['insertion_cost_minutes']
            else:
                regret = 2000 - all_options[0]['insertion_cost_minutes']
                
            if regret > best_regret:
                best_regret = regret
                target_student = student
                target_insertion = all_options[0]
        
        if target_student and target_insertion:
            # Apply insertion
            affected_route = target_insertion['route']
            _apply_insertion(solution, target_student, target_insertion)
            
            # Remove from unassigned
            unassigned.remove(target_student)
            
            # Update only the affected route's options for all remaining unassigned students
            for s in unassigned:
                student_route_options[s.id][affected_route.route_id] = _get_insertions_for_route(
                    s, affected_route, solution.graph, student_frontages[s.id]
                )
        else:
            break

def _get_insertions_for_route(student, route, graph, frontage_info):
    """Helper to find all possible valid insertion points for a student in ONE route."""
    options = []
    frontage_node_id, frontage_coords = frontage_info
    
    # Start and end stops are fixed (Depot/School), strictly insert between
    start_pos = 1 if len(route.stops) >= 2 else 0
    end_pos = len(route.stops) if len(route.stops) >= 2 else len(route.stops) + 1
        
    for pos in range(start_pos, end_pos):
        # Check if an existing stop at this node can be reused
        existing_stop = next((s for s in route.stops if s.node_id == frontage_node_id), None)
        eval_stop = existing_stop if existing_stop else Stop(frontage_node_id, frontage_coords[0], frontage_coords[1])
        
        res = calculate_insertion_cost(eval_stop, route, pos, graph)
        if res is None: continue
        
        cost, is_valid, _ = res
        if not is_valid: continue
        
        valid, _, _ = validate_permanent_student(eval_stop, route, pos, cost, graph)
        if valid:
            options.append({
                'route': route,
                'new_stop': eval_stop,
                'insertion_position': pos,
                'insertion_cost_minutes': cost,
                'is_new_stop': existing_stop is None
            })
    return options

def _get_all_valid_insertions(student, routes, graph):
    """Legacy helper (still needed for greedy_repair)"""
    node_id, coords = snap_address_to_edge(student.coords, graph)
    all_options = []
    for route in routes:
        all_options.extend(_get_insertions_for_route(student, route, graph, (node_id, coords)))
    return all_options

def _apply_insertion(solution, student, result):
    """Actually update the route and student state based on insertion search."""
    route = result['route']
    new_stop = result['new_stop']
    
    if result.get('is_new_stop', True) and new_stop not in route.stops:
        route.stops.insert(result['insertion_position'], new_stop)
    
    new_stop.add_student(student)
    route.total_time = calculate_route_time(route, solution.graph)
    route.total_distance = calculate_route_distance(route, solution.graph)


# ============================================================================
# ALNS ENGINE
# ============================================================================

class ALNSEngine:
    def __init__(self, initial_solution, iterations=100, temp=1000, cooling=0.98):
        self.curr_sol = initial_solution.clone()
        self.best_sol = initial_solution.clone()
        self.iterations = iterations
        self.temp = temp
        self.cooling = cooling
        
        self.destroy_ops = [random_removal, worst_cost_removal]
        self.repair_ops = [greedy_repair, regret_repair]
        
        # Weights for operator selection
        self.d_weights = np.ones(len(self.destroy_ops))
        self.r_weights = np.ones(len(self.repair_ops))
        
        # Reward scores
        self.s1 = 30 # New global best
        self.s2 = 15 # Better than current
        self.s3 = 5  # Accepted (Simulated Annealing)
        
    def run(self):
        t = self.temp
        start_time = time.time()
        block_start_time = start_time
        
        print(f"Starting ALNS Optimization with {self.iterations} iterations...")
        print(f"Initial State: {self.curr_sol}")

        for i in range(self.iterations):
            # Selection
            d_idx = self._select_op(self.d_weights)
            r_idx = self._select_op(self.r_weights)
            
            new_sol = self.curr_sol.clone()
            
            # Destroy: Remove between 5% and 25% of students
            n_remove = max(1, int(len(new_sol.students) * random.uniform(0.05, 0.25)))
            self.destroy_ops[d_idx](new_sol, n_remove)
            
            # Repair
            self.repair_ops[r_idx](new_sol)
            
            # Score calculation
            new_obj = new_sol.calculate_objective()
            curr_obj = self.curr_sol.calculate_objective()
            best_obj = self.best_sol.calculate_objective()
            
            reward = 0
            if new_obj > best_obj:
                self.best_sol = new_sol.clone()
                self.curr_sol = new_sol
                reward = self.s1
            elif new_obj > curr_obj:
                self.curr_sol = new_sol
                reward = self.s2
            else:
                # Simulated Annealing acceptance criteria
                # We use (new - old) because we are MAXIMIZING
                diff = new_obj - curr_obj # will be negative
                p = math.exp(diff / t) if t > 0 else 0
                if random.random() < p:
                    self.curr_sol = new_sol
                    reward = self.s3
            
            # Update weights (Adaptive)
            alpha = 0.7
            self.d_weights[d_idx] = alpha * self.d_weights[d_idx] + (1-alpha) * reward
            self.r_weights[r_idx] = alpha * self.r_weights[r_idx] + (1-alpha) * reward
            
            t *= self.cooling
            
            if (i+1) % 10 == 0:
                block_end_time = time.time()
                block_elapsed = block_end_time - block_start_time
                print(f"Iteration {i+1}: Best Obj = {best_obj:.2f}, Temp = {t:.1f}, Last 10 iter: {block_elapsed:.2f}s")
                block_start_time = block_end_time

        total_elapsed = time.time() - start_time
        print(f"Optimization Complete.")
        print(f"Total Time: {total_elapsed:.2f}s")
        print(f"Final State: {self.best_sol}")
        return self.best_sol

    def _select_op(self, weights):
        probs = weights / np.sum(weights)
        return np.random.choice(len(weights), p=probs)
