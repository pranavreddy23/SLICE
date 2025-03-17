from queue import PriorityQueue
from typing import Callable, List, Tuple, Dict, Set
import numpy as np
from scipy import ndimage as ndi
from core.env import GridEnvironment
from core.constraintext import ConstraintExtractor, ConstraintSet

class CostFunctionGenerator:
    """Generate cost functions based on natural language instructions"""
    
    def __init__(self):
        """Initialize the cost function generator"""
        self.extractor = ConstraintExtractor()
    
    def generate_cost_function(self, environment, instruction, start=None, goal=None):
        """Generate a cost function based on natural language instruction"""
        # Extract constraints from instruction
        constraints = self.extractor.extract_constraints(instruction, environment)
        
        # If no constraints were extracted, return uniform cost
        if not constraints or (not constraints.proximity and not constraints.avoidance and not constraints.preference):
            return uniform_cost_function(environment)
        
        # Create a weighted cost function based on constraints
        def cost_function(x1, y1, x2, y2):
            # Base cost (Manhattan distance)
            base_cost = abs(x2 - x1) + abs(y2 - y1)
            
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
            # Use a higher weight for constraint cost to make it more influential
            total_cost = base_cost * (1.0 + constraint_cost)
            
            return max(0.1, total_cost)  # Ensure cost is always positive
        
        return cost_function


def heuristic(node: tuple, goal: tuple) -> float:
    """Manhattan distance heuristic for A* search"""
    return abs(node[0] - goal[0]) + abs(node[1] - goal[1])


def a_star_search(environment, start, goal, cost_function, track_expanded=False):
    """
    A* search algorithm for pathfinding
    
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
        max_memory_usage: Maximum number of nodes in memory (open_set + closed_set)
    """
    import heapq
    
    # Initialize data structures
    open_set = []  # Priority queue
    closed_set = set()  # Set of visited nodes
    g_score = {start: 0}  # Cost from start to node
    f_score = {start: manhattan_distance(start, goal)}  # Estimated total cost
    came_from = {}  # Parent pointers
    expanded_nodes = [] if track_expanded else None
    
    # Track memory usage (open_set + closed_set)
    max_memory_usage = 1  # Start with 1 node
    
    # Add start node to open set
    heapq.heappush(open_set, (f_score[start], start))
    
    while open_set:
        # Get node with lowest f_score
        _, current = heapq.heappop(open_set)
        
        # Track expanded nodes if requested
        if track_expanded:
            expanded_nodes.append(current)
        
        # Check if we've reached the goal
        if current == goal:
            # Reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            
            if track_expanded:
                return path, g_score[goal], expanded_nodes, max_memory_usage
            else:
                return path, g_score[goal], max_memory_usage
        
        # Mark as visited
        closed_set.add(current)
        
        # Check neighbors
        x, y = current
        neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
        
        for neighbor in neighbors:
            # Skip if not valid or already visited
            if not environment.is_valid(*neighbor) or neighbor in closed_set:
                continue
            
            # Calculate tentative g_score
            tentative_g = g_score[current] + cost_function(x, y, neighbor[0], neighbor[1])
            
            # If neighbor not in open set or we found a better path
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                # Update path
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + manhattan_distance(neighbor, goal)
                
                # Add to open set if not already there
                if neighbor not in [n for _, n in open_set]:
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
                    
                    # Update max memory usage
                    current_memory = len(closed_set) + len(open_set)
                    max_memory_usage = max(max_memory_usage, current_memory)
    
    # No path found
    if track_expanded:
        return [], float('inf'), expanded_nodes, max_memory_usage
    else:
        return [], float('inf'), max_memory_usage


def manhattan_distance(a, b):
    """Calculate Manhattan distance between two points"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def reconstruct_path(came_from: Dict, current: Tuple[int, int]) -> List[Tuple[int, int]]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    return path[::-1]  # Reverse to get path from start to goal


def uniform_cost_function(environment) -> Callable:
    """Generate a uniform cost function for baseline comparison"""
    def cost_fn(x: int, y: int, prev_x: int = None, prev_y: int = None) -> float:
        return 1.0 if environment.is_valid(x, y) else float('inf')
    return cost_fn


def semantic_heuristic(node, goal, environment, constraints):
    """Enhanced heuristic that incorporates semantic understanding"""
    # Base heuristic (Manhattan distance)
    base_h = abs(node[0] - goal[0]) + abs(node[1] - goal[1])
    
    # Semantic adjustments
    semantic_factor = 1.0
    
    # If we have avoidance constraints, check if there's a clear path through avoided regions
    if 'avoidance' in constraints:
        for region_name, weight in constraints['avoidance'].items():
            if region_name in environment.regions:
                # Check if a straight line between node and goal intersects with avoided region
                if line_intersects_region(node, goal, environment.regions[region_name]):
                    # Increase heuristic to encourage exploration of alternative paths
                    semantic_factor *= (1.0 + 0.2 * weight)
    
    # If we have preference constraints, check if we're in a preferred region
    if 'preference' in constraints:
        for region_name, weight in constraints['preference'].items():
            if region_name in environment.regions:
                if environment.is_in_region(node[0], node[1], region_name):
                    # Decrease heuristic to encourage staying in preferred regions
                    semantic_factor *= (1.0 - 0.1 * weight)
    
    return base_h * semantic_factor

def line_intersects_region(p1, p2, region_points):
    """Check if a line between p1 and p2 intersects with any point in the region"""
    # Simple approximation: check points along the line
    steps = max(abs(p2[0] - p1[0]), abs(p2[1] - p1[1]))
    if steps == 0:
        return False
        
    for i in range(steps + 1):
        t = i / steps
        x = int(p1[0] + t * (p2[0] - p1[0]))
        y = int(p1[1] + t * (p2[1] - p1[1]))
        
        if (x, y) in region_points:
            return True
            
    return False

class AdaptiveCostFunctionGenerator:
    def __init__(self, base_cost: float = 1.0):
        self.base_cost = base_cost
    
    def generate_cost_function(self, constraints: Dict, environment) -> Callable:
        """Generate an adaptive cost function based on extracted constraints"""
        
        # Keep track of visited nodes to adapt costs
        visited_nodes = set()
        
        def cost_function(x: int, y: int, prev_x: int = None, prev_y: int = None) -> float:
            if not environment.is_valid(x, y):
                return float('inf')
            
            # Add current node to visited set
            visited_nodes.add((x, y))
            
            cost = self.base_cost
            
            # Apply standard constraint costs
            # Proximity constraints
            if 'proximity' in constraints:
                for region_name, weight in constraints['proximity'].items():
                    if region_name in environment.regions:
                        min_dist = environment.distance_to_region(x, y, region_name)
                        cost += weight * min_dist
            
            # Avoidance constraints
            if 'avoidance' in constraints:
                for region_name, weight in constraints['avoidance'].items():
                    if region_name in environment.regions:
                        if environment.is_in_region(x, y, region_name):
                            cost += weight * 10  # High penalty for being in avoided region
            
            # Preference constraints
            if 'preference' in constraints:
                for region_name, weight in constraints['preference'].items():
                    if region_name in environment.regions:
                        if not environment.is_in_region(x, y, region_name):
                            cost += weight * 5  # Moderate penalty for not being in preferred region
            
            # Adaptive component: increase cost for revisiting areas with high node density
            local_density = sum(1 for nx, ny in visited_nodes 
                              if abs(nx - x) <= 3 and abs(ny - y) <= 3)
            
            # Apply density penalty (higher in dense areas)
            density_factor = 0.1 * (local_density / 10.0) if local_density > 10 else 0
            cost *= (1.0 + density_factor)
            
            return cost
        
        return cost_function


def semantic_a_star_search(environment, start, goal, cost_function, constraints):
    """A* search with semantic pruning to reduce node expansion"""
    from queue import PriorityQueue
    
    # Statistics tracking
    nodes_expanded = 0
    
    # Initialize
    frontier = PriorityQueue()
    frontier.put((0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}
    
    # Identify regions for pruning
    high_cost_regions = []
    if 'avoidance' in constraints:
        high_cost_regions = [r for r, w in constraints['avoidance'].items() 
                           if r in environment.regions and w > 5.0]
    
    while not frontier.empty():
        _, current = frontier.get()
        nodes_expanded += 1
        
        if current == goal:
            break
        
        for next_node in environment.get_neighbors(*current):
            # Semantic pruning: skip high-cost regions unless necessary
            if high_cost_regions and any(environment.is_in_region(next_node[0], next_node[1], r) 
                                       for r in high_cost_regions):
                # Check if we're near the goal or start
                near_goal = abs(next_node[0] - goal[0]) + abs(next_node[1] - goal[1]) <= 3
                near_start = abs(next_node[0] - start[0]) + abs(next_node[1] - start[1]) <= 3
                
                # Skip this node unless it's near the goal or start
                if not (near_goal or near_start):
                    continue
            
            # Extract coordinates for cost calculation
            curr_x, curr_y = current
            next_x, next_y = next_node
            
            # Calculate cost using the cost function
            new_cost = cost_so_far[current] + cost_function(next_x, next_y, curr_x, curr_y)
            
            if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                cost_so_far[next_node] = new_cost
                # Use semantic heuristic
                priority = new_cost + semantic_heuristic(next_node, goal, environment, constraints)
                frontier.put((priority, next_node))
                came_from[next_node] = current
    
    # Reconstruct path
    if goal not in came_from:
        return None, {"nodes_expanded": nodes_expanded}  # No path found
        
    path = []
    current = goal
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start)
    path.reverse()
    
    # Return path and statistics
    return path, {"nodes_expanded": nodes_expanded}

def identify_semantic_corridors(environment, constraints):
    """Identify corridors based on semantic regions"""
    corridors = []
    
    # Find regions that should be part of corridors
    preferred_regions = []
    if 'preference' in constraints:
        preferred_regions = [r for r, w in constraints['preference'].items() 
                           if r in environment.regions]
    
    # Find regions to avoid
    avoided_regions = []
    if 'avoidance' in constraints:
        avoided_regions = [r for r, w in constraints['avoidance'].items() 
                         if r in environment.regions]
    
    # Create corridors through preferred regions
    for region_name in preferred_regions:
        region_points = environment.regions[region_name]
        
        # Find the centroid of the region
        if region_points:
            x_coords = [x for x, y in region_points]
            y_coords = [y for x, y in region_points]
            centroid = (sum(x_coords) // len(x_coords), sum(y_coords) // len(y_coords))
            
            # Add the centroid as a corridor point
            corridors.append(centroid)
    
    return corridors

def corridor_guided_search(environment, start, goal, constraints, base_planner):
    """Use semantic corridors to guide the search"""
    # Identify corridors
    corridors = identify_semantic_corridors(environment, constraints)
    
    if not corridors:
        # If no corridors identified, fall back to base planner
        return base_planner(environment, start, goal, constraints)
    
    # Add start and goal to waypoints
    waypoints = [start] + corridors + [goal]
    
    # Plan path through waypoints
    full_path = []
    stats = {"nodes_expanded": 0}
    
    for i in range(len(waypoints) - 1):
        segment_start = waypoints[i]
        segment_goal = waypoints[i+1]
        
        # Plan segment
        segment_path, segment_stats = base_planner(environment, segment_start, segment_goal, constraints)
        
        if not segment_path:
            # If any segment fails, fall back to direct planning
            return base_planner(environment, start, goal, constraints)
        
        # Add segment to full path (avoid duplicating points)
        if i > 0:
            segment_path = segment_path[1:]  # Skip first point to avoid duplication
            
        full_path.extend(segment_path)
        stats["nodes_expanded"] += segment_stats.get("nodes_expanded", 0)
    
    return full_path, stats

def identify_corridors(environment):
    """Identify corridors in the environment"""
    corridors = []
    
    # Get grid dimensions
    width, height = environment.width, environment.height
    
    # Create a distance transform
    grid = environment.grid.copy()
    binary_grid = (grid > 0).astype(np.uint8)
    distance = ndi.distance_transform_edt(1 - binary_grid)
    
    # Find narrow passages (where distance to obstacles is small)
    narrow_threshold = 2.0  # Adjust based on your grid scale
    narrow_points = np.where(distance < narrow_threshold)
    narrow_points = [(x, y) for y, x in zip(narrow_points[0], narrow_points[1]) 
                   if environment.is_valid(x, y)]
    
    # Group narrow points into corridors
    visited = set()
    for point in narrow_points:
        if point in visited:
            continue
            
        # Start a new corridor
        corridor = []
        queue = [point]
        visited.add(point)
        
        while queue:
            current = queue.pop(0)
            corridor.append(current)
            
            # Check neighbors
            x, y = current
            neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
            
            for nx, ny in neighbors:
                neighbor = (nx, ny)
                if (neighbor in narrow_points and 
                    neighbor not in visited and 
                    environment.is_valid(nx, ny)):
                    queue.append(neighbor)
                    visited.add(neighbor)
        
        # Add corridor if it's long enough
        if len(corridor) > 5:  # Minimum corridor length
            corridors.append(corridor)
    
    # Add corridors to environment regions
    for i, corridor in enumerate(corridors):
        environment.regions[f"corridor_{i}"] = corridor
    
    return corridors