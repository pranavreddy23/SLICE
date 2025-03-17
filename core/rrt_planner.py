import numpy as np
import random
import math
from typing import List, Tuple, Callable, Dict, Set

class RRTNode:
    """Node class for RRT algorithm"""
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None  # Parent node
        self.cost = 0.0     # Cost from start to this node

class RRTPlanner:
    """RRT (Rapidly-exploring Random Tree) path planner with cost function support"""
    
    def __init__(self, max_iterations=2000, step_size=5.0, goal_sample_rate=0.1, goal_tolerance=5.0):
        """Initialize the RRT planner
        
        Args:
            max_iterations: Maximum number of iterations
            step_size: Distance to extend tree in each step
            goal_sample_rate: Probability of sampling the goal
            goal_tolerance: Distance within which goal is considered reached
        """
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.goal_tolerance = goal_tolerance
        
    def plan(self, environment, start, goal, cost_function=None, track_expanded=False):
        """Plan a path using RRT
        
        Args:
            environment: The grid environment
            start: (x, y) tuple for start position
            goal: (x, y) tuple for goal position
            cost_function: Function that returns cost between adjacent cells
            track_expanded: Whether to track expanded nodes
        
        Returns:
            path: List of (x, y) tuples representing the path
            cost: Total cost of the path
            expanded_nodes: List of expanded nodes (if track_expanded=True)
            max_tree_size: Maximum size of the tree (for memory usage tracking)
        """
        # Initialize tree with start node
        start_node = RRTNode(start[0], start[1])
        nodes = [start_node]
        expanded_nodes = [] if track_expanded else None
        max_tree_size = 1  # Start with 1 node
        
        # Check if start and goal are valid
        if not environment.is_valid(start[0], start[1]) or not environment.is_valid(goal[0], goal[1]):
            print("Start or goal position is invalid")
            return [], float('inf'), expanded_nodes if track_expanded else [], max_tree_size
        
        # Use default cost function if none provided
        if cost_function is None:
            cost_function = lambda x1, y1, x2, y2: math.sqrt((x2-x1)**2 + (y2-y1)**2)
        
        # Main RRT loop
        for i in range(self.max_iterations):
            # Sample random point (with bias toward goal)
            if random.random() < self.goal_sample_rate:
                sample = goal
            else:
                sample = self._random_sample(environment)
            
            # Find nearest node
            nearest_node_idx = self._nearest_node(nodes, sample)
            nearest_node = nodes[nearest_node_idx]
            
            # Track expanded nodes
            if track_expanded:
                expanded_nodes.append((nearest_node.x, nearest_node.y))
            
            # Steer toward sample
            new_node = self._steer(nearest_node, sample, self.step_size)
            
            # Check if new node is valid
            if not self._is_collision_free(environment, nearest_node, new_node):
                continue
            
            # Calculate cost to new node
            new_node.cost = nearest_node.cost + cost_function(
                nearest_node.x, nearest_node.y, new_node.x, new_node.y)
            
            # Add new node to tree
            new_node.parent = nearest_node
            nodes.append(new_node)
            max_tree_size = max(max_tree_size, len(nodes))  # Track maximum tree size
            
            # Check if goal is reached
            dist_to_goal = math.sqrt((new_node.x - goal[0])**2 + (new_node.y - goal[1])**2)
            if dist_to_goal <= self.goal_tolerance:
                # Try to connect directly to goal
                goal_node = RRTNode(goal[0], goal[1])
                if self._is_collision_free(environment, new_node, goal_node):
                    goal_node.parent = new_node
                    goal_node.cost = new_node.cost + cost_function(
                        new_node.x, new_node.y, goal_node.x, goal_node.y)
                    nodes.append(goal_node)
                    max_tree_size = max(max_tree_size, len(nodes))  # Update max tree size
                    
                    # Extract path
                    path = self._extract_path(goal_node)
                    total_cost = goal_node.cost
                    
                    return path, total_cost, expanded_nodes if track_expanded else [], max_tree_size
        
        # If max iterations reached without finding path
        print("Max iterations reached without finding path")
        
        # Try to find the node closest to goal
        closest_node_idx = self._nearest_node(nodes, goal)
        closest_node = nodes[closest_node_idx]
        
        # Extract path to closest node
        path = self._extract_path(closest_node)
        total_cost = closest_node.cost
        
        return path, total_cost, expanded_nodes if track_expanded else [], max_tree_size
    
    def _random_sample(self, environment):
        """Generate a random sample within the environment bounds"""
        x = random.randint(0, environment.width - 1)
        y = random.randint(0, environment.height - 1)
        return (x, y)
    
    def _nearest_node(self, nodes, point):
        """Find the nearest node in the tree to the given point"""
        distances = [(node.x - point[0])**2 + (node.y - point[1])**2 for node in nodes]
        return distances.index(min(distances))
    
    def _steer(self, from_node, to_point, step_size):
        """Steer from a node toward a point with limited step size"""
        dx = to_point[0] - from_node.x
        dy = to_point[1] - from_node.y
        dist = math.sqrt(dx**2 + dy**2)
        
        if dist <= step_size:
            new_x, new_y = to_point
        else:
            theta = math.atan2(dy, dx)
            new_x = from_node.x + step_size * math.cos(theta)
            new_y = from_node.y + step_size * math.sin(theta)
        
        return RRTNode(int(new_x), int(new_y))
    
    def _is_collision_free(self, environment, from_node, to_node):
        """Check if the path between two nodes is collision-free"""
        # Use Bresenham's line algorithm to check for collisions
        x1, y1 = int(from_node.x), int(from_node.y)
        x2, y2 = int(to_node.x), int(to_node.y)
        
        # Check if endpoints are valid
        if not environment.is_valid(x1, y1) or not environment.is_valid(x2, y2):
            return False
        
        # Bresenham's line algorithm
        points = self._bresenham_line(x1, y1, x2, y2)
        for x, y in points:
            if not environment.is_valid(x, y):
                return False
        
        return True
    
    def _bresenham_line(self, x1, y1, x2, y2):
        """Bresenham's line algorithm for collision checking"""
        points = []
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy
        
        while True:
            points.append((x1, y1))
            if x1 == x2 and y1 == y2:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy
        
        return points
    
    def _extract_path(self, end_node):
        """Extract the path from start to end node"""
        path = []
        current = end_node
        
        while current is not None:
            path.append((current.x, current.y))
            current = current.parent
        
        return path[::-1]  # Reverse to get path from start to end

def rrt_with_slice(environment, start, goal, instruction, max_iterations=2000, track_expanded=True):
    """Run RRT with SLICE constraints
    
    Args:
        environment: The grid environment
        start: (x, y) tuple for start position
        goal: (x, y) tuple for goal position
        instruction: Natural language instruction
        max_iterations: Maximum number of iterations
        track_expanded: Whether to track expanded nodes
    
    Returns:
        path: List of (x, y) tuples representing the path
        stats: Dictionary with statistics
    """
    from core.cfgen import CostFunctionGenerator
    
    # Extract constraints from instruction
    cfgen = CostFunctionGenerator()
    constraints = cfgen.extractor.extract_constraints(instruction, environment)
    
    # Create cost function based on constraints
    def cost_function(x1, y1, x2, y2):
        # Base cost (Euclidean distance)
        base_cost = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        # Initialize constraint cost
        constraint_cost = 0
        
        # Apply proximity constraints (stay close to regions)
        for region_name, weight in constraints.proximity.items():
            if region_name in environment.regions:
                points = environment.regions[region_name]
                if points:
                    min_distance = min(abs(x2 - px) + abs(y2 - py) for px, py in points)
                    constraint_cost += 0.1 * min_distance * (weight / 10.0)
        
        # Apply avoidance constraints (stay away from regions)
        for region_name, weight in constraints.avoidance.items():
            if region_name in environment.regions:
                points = environment.regions[region_name]
                if (x2, y2) in points:
                    constraint_cost += 5.0 * (weight / 10.0)  # High penalty for avoided regions
        
        # Apply preference constraints (prefer certain regions)
        preference_regions_exist = False
        in_preferred_region = False
        
        for region_name, weight in constraints.preference.items():
            if region_name in environment.regions:
                preference_regions_exist = True
                points = environment.regions[region_name]
                if (x2, y2) in points:
                    in_preferred_region = True
                    constraint_cost -= 0.8 * (weight / 10.0)  # Reward for preferred regions
        
        # If we have preference regions but aren't in one, add a small penalty
        if preference_regions_exist and not in_preferred_region:
            constraint_cost += 0.5
        
        # Combine base cost and constraint cost
        total_cost = base_cost * (1.0 + constraint_cost)
        
        return max(0.1, total_cost)  # Ensure cost is always positive
    
    # Initialize RRT planner
    planner = RRTPlanner(max_iterations=max_iterations)
    
    # Plan with tracking expanded nodes
    path, cost, expanded_nodes, max_tree_size = planner.plan(
        environment, start, goal, cost_function, track_expanded=True
    )
    
    # Return path and statistics
    stats = {
        "nodes_expanded": len(expanded_nodes),
        "path_cost": cost,
        "max_tree_size": max_tree_size
    }
    
    return path, stats, expanded_nodes