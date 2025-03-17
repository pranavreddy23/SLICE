import os
import json
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional
from pydantic import BaseModel, Field
from groq import Groq

# Import your existing modules
from core.env import GridEnvironment
from core.cfgen import a_star_search, uniform_cost_function
from utils.viz import visualize_grid, visualize_node_expansion, visualize_paths
from core.constraintext import ConstraintExtractor

# Pydantic model for waypoints
class Waypoints(BaseModel):
    """Pydantic model for LLM-generated waypoints"""
    points: List[List[int]] = Field(
        description="List of waypoint coordinates [x, y] from start to goal"
    )

class LLMAStarPlanner:
    """
    Implementation of LLM A* algorithm as described in the paper.
    Uses LLM to generate waypoints and then guides A* search through these waypoints.
    """
    def __init__(self, model="deepseek-r1-distill-llama-70b", api_key="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"):
        """Initialize with Groq client for Llama 3.1"""
        self.client = Groq(api_key=api_key)
        self.model = model
    
    def get_waypoints(self, instruction: str, environment, include_visualization: bool = False) -> List[Tuple[int, int]]:
        """
        Get waypoints from LLM based on the environment and instruction
        
        Args:
            instruction: Navigation instruction
            environment: Grid environment
            include_visualization: Whether to include environment visualization
            
        Returns:
            List of waypoints [(x1, y1), (x2, y2), ...]
        """
        env_context = self._create_environment_context(environment)
        
        # Create visualization if requested
        base64_image = None
        if include_visualization:
            base64_image = self._create_environment_visualization(environment)
        
        # Construct prompt
        prompt = self._construct_waypoints_prompt(instruction, env_context)
        
        # Call Groq API with or without image
        try:
            if base64_image:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a specialized AI for robotic path planning. Generate waypoints for navigation."
                        },
                        {
                            "role": "user",
                            "content": f"{prompt}\n\n[Image: data:image/jpeg;base64,{base64_image}]"
                        }
                    ],
                    temperature=0.2,
                    max_tokens=1024,
                    top_p=1,
                    stream=False,
                    stop=None,
                )
            else:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a specialized AI for robotic path planning. Generate waypoints for navigation."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.2,
                    max_tokens=1024,
                    top_p=1,
                    stream=False,
                    stop=None,
                )
            
            output = completion.choices[0].message.content
            print("Raw LLM output:")
            print(output)
            print("\n")
            
            # Parse the response
            waypoints = self._parse_waypoints(output, environment)
            return waypoints
            
        except Exception as e:
            print(f"Error calling Groq API: {e}")
            # Return just start and goal on failure
            return [environment.start, environment.goal]
    
    def _create_environment_context(self, environment) -> str:
        """Create a textual description of the environment for the LLM"""
        env_dict = environment.to_dict()
        
        context = f"Grid size: {env_dict['width']}x{env_dict['height']}\n"
        
        # Add start and goal positions
        context += f"Start position: {environment.start}\n"
        context += f"Goal position: {environment.goal}\n"
        
        # Describe regions
        if env_dict['regions']:
            context += "Defined regions:\n"
            for name, coords in env_dict['regions'].items():
                # For all regions, just describe the bounds
                xs = [x for x, y in coords]
                ys = [y for x, y in coords]
                if xs and ys:  # Make sure the lists aren't empty
                    context += f"- {name}: bounded by ({min(xs)},{min(ys)}) to ({max(xs)},{max(ys)})\n"
        
        return context
    
    def _create_environment_visualization(self, environment) -> str:
        """Create a visualization of the environment with start/goal positions"""
        # Reuse the visualization function from ConstraintExtractor
        extractor = ConstraintExtractor()
        return extractor._create_environment_visualization(environment)
    
    def _construct_waypoints_prompt(self, instruction: str, env_context: str) -> str:
        """Construct a prompt for waypoint generation"""
        prompt = f"""
        # Environment
        {env_context}
        
        # Instruction
        "{instruction}"
        
        # Task
        Generate a sequence of waypoints (coordinates) for a robot to navigate from start to goal.
        The waypoints should follow the instruction while avoiding obstacles.
        
        # IMPORTANT: ONLY RETURN A JSON ARRAY OF WAYPOINT COORDINATES:
        [
          [x1, y1],
          [x2, y2],
          ...
        ]
        
        Include the start and goal positions as the first and last waypoints.
        DO NOT include any explanations, thinking, or additional text.
        ONLY return the JSON array of waypoints.
        """
        return prompt
    
    def _parse_waypoints(self, output: str, environment) -> List[Tuple[int, int]]:
        """Parse LLM output to extract waypoints"""
        # Initialize with start and goal as fallback
        waypoints = [environment.start, environment.goal]
        
        try:
            # Find JSON array in the response
            import re
            json_match = re.search(r'\[\s*\[.*\]\s*\]', output, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                points = json.loads(json_str)
                
                # Convert to tuples and validate
                valid_waypoints = []
                for point in points:
                    if len(point) == 2:
                        x, y = point
                        # Check if point is valid (not in obstacle)
                        if 0 <= y < environment.height and 0 <= x < environment.width and environment.grid[y, x] == 0:
                            valid_waypoints.append((x, y))
                
                # Ensure start and goal are included
                if valid_waypoints and valid_waypoints[0] != environment.start:
                    valid_waypoints.insert(0, environment.start)
                if valid_waypoints and valid_waypoints[-1] != environment.goal:
                    valid_waypoints.append(environment.goal)
                
                if valid_waypoints:
                    waypoints = valid_waypoints
            
        except Exception as e:
            print(f"Error parsing waypoints: {e}")
        
        return waypoints
    
    def llm_a_star_search(self, env, start, goal, waypoints, track_expanded=False):
        """
        Implementation of LLM A* search algorithm
        
        Args:
            env: Grid environment
            start: Start position
            goal: Goal position
            waypoints: List of waypoints from LLM
            track_expanded: Whether to track expanded nodes
            
        Returns:
            path: List of coordinates from start to goal
            cost: Total path cost
            expanded: Set of expanded nodes (if track_expanded=True)
            max_memory_usage: Maximum number of nodes in memory
        """
        # Initialize open and closed lists
        open_list = []
        closed_set = set()
        
        # Initialize target waypoint index
        target_idx = 1 if len(waypoints) > 1 else 0
        current_target = waypoints[target_idx]
        
        # Initialize g and f scores
        g_score = {start: 0}
        f_score = {start: self._heuristic(start, goal) + self._heuristic(start, current_target)}
        
        # Initialize parent map for path reconstruction
        parent = {}
        
        # Add start node to open list
        open_list.append(start)
        
        # Track expanded nodes if requested
        expanded = set() if track_expanded else None
        
        # Track memory usage (open_list + closed_set)
        max_memory_usage = 1  # Start with 1 node
        
        while open_list:
            # Get node with lowest f_score
            current = min(open_list, key=lambda x: f_score.get(x, float('inf')))
            
            # Track expanded nodes
            if track_expanded:
                expanded.add(current)
            
            # Check if we reached the goal
            if current == goal:
                return self._reconstruct_path(parent, current), g_score[current], expanded, max_memory_usage
            
            # Check if we reached the current target waypoint
            if current == current_target and current != goal:
                target_idx += 1
                if target_idx < len(waypoints):
                    current_target = waypoints[target_idx]
                    # Recalculate f_scores for all nodes in open list
                    for node in open_list:
                        f_score[node] = g_score[node] + self._heuristic(node, goal) + self._heuristic(node, current_target)
            
            # Remove current from open list and add to closed set
            open_list.remove(current)
            closed_set.add(current)
            
            # Update max memory usage
            current_memory = len(open_list) + len(closed_set)
            max_memory_usage = max(max_memory_usage, current_memory)
            
            # Get neighbors
            neighbors = self._get_neighbors(env, current)
            
            for neighbor in neighbors:
                # Skip if in closed set
                if neighbor in closed_set:
                    continue
                
                # Calculate tentative g_score
                tentative_g = g_score[current] + 1  # Assuming uniform cost
                
                # If neighbor not in open list or better path found
                if neighbor not in open_list or tentative_g < g_score.get(neighbor, float('inf')):
                    # Update parent
                    parent[neighbor] = current
                    
                    # Update g_score
                    g_score[neighbor] = tentative_g
                    
                    # Update f_score with additional term for current target
                    f_score[neighbor] = g_score[neighbor] + self._heuristic(neighbor, goal) + self._heuristic(neighbor, current_target)
                    
                    # Add to open list if not already there
                    if neighbor not in open_list:
                        open_list.append(neighbor)
                        
                        # Update max memory usage after adding to open list
                        current_memory = len(open_list) + len(closed_set)
                        max_memory_usage = max(max_memory_usage, current_memory)
        
        # No path found
        return [], float('inf'), expanded, max_memory_usage
    
    def _heuristic(self, a, b):
        """Manhattan distance heuristic"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def _get_neighbors(self, env, pos):
        """Get valid neighbors of a position"""
        x, y = pos
        neighbors = []
        
        # Check all 4 adjacent cells
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            
            # Check if within bounds and not an obstacle
            if (0 <= nx < env.width and 0 <= ny < env.height and 
                env.grid[ny, nx] == 0):
                neighbors.append((nx, ny))
        
        return neighbors
    
    def _reconstruct_path(self, parent, current):
        """Reconstruct path from parent map"""
        path = [current]
        while current in parent:
            current = parent[current]
            path.append(current)
        
        return path[::-1]  # Reverse to get path from start to goal

def calculate_path_metrics(path, env):
    """Calculate metrics for a path"""
    if not path:
        return {
            "length": 0,
            "expanded_nodes": 0,
            "smoothness": 0,
            "region_coverage": {}
        }
    
    # Calculate path length
    length = 0
    for i in range(1, len(path)):
        dx = path[i][0] - path[i-1][0]
        dy = path[i][1] - path[i-1][1]
        length += np.sqrt(dx*dx + dy*dy)
    
    # Calculate smoothness (average angle between consecutive segments)
    angles = []
    for i in range(1, len(path)-1):
        # Calculate vectors for segments
        v1 = (path[i][0] - path[i-1][0], path[i][1] - path[i-1][1])
        v2 = (path[i+1][0] - path[i][0], path[i+1][1] - path[i][1])
        
        # Calculate angle between vectors
        dot_product = v1[0]*v2[0] + v1[1]*v2[1]
        mag_v1 = np.sqrt(v1[0]**2 + v1[1]**2)
        mag_v2 = np.sqrt(v2[0]**2 + v2[1]**2)
        
        if mag_v1 > 0 and mag_v2 > 0:
            cos_angle = dot_product / (mag_v1 * mag_v2)
            # Clamp to avoid numerical issues
            cos_angle = max(-1, min(1, cos_angle))
            angle = np.arccos(cos_angle)
            angles.append(angle)
    
    # Smoothness is inversely proportional to average angle
    smoothness = 1.0 - (np.mean(angles) / np.pi if angles else 0)
    
    # Calculate region coverage
    region_coverage = {}
    for region_name, region_points in env.regions.items():
        if region_name == "segmentation":
            continue
            
        # Convert region points to set for faster lookup
        region_set = set(region_points)
        
        # Count path points in this region
        points_in_region = sum(1 for p in path if p in region_set)
        
        # Calculate coverage as percentage of path in region
        if len(path) > 0:
            region_coverage[region_name] = points_in_region / len(path)
        else:
            region_coverage[region_name] = 0
    
    return {
        "length": length,
        "expanded_nodes": 0,  # Will be filled later
        "smoothness": smoothness,
        "region_coverage": region_coverage
    }

def visualize_waypoints_and_paths(env, start, goal, waypoints, baseline_path, llm_astar_path, 
                                 baseline_expanded, llm_astar_expanded, scenario_log_dir):
    """
    Create and save detailed visualizations of waypoints and paths
    
    Args:
        env: Environment object
        start: Start position
        goal: Goal position
        waypoints: LLM-generated waypoints
        baseline_path: Baseline A* path
        llm_astar_path: LLM A* path
        baseline_expanded: Nodes expanded by baseline A*
        llm_astar_expanded: Nodes expanded by LLM A*
        scenario_log_dir: Directory to save visualizations
    """
    # 1. Waypoints visualization
    plt.figure(figsize=(12, 10))
    plt.imshow(env.grid, cmap='binary', interpolation='nearest')
    plt.plot(start[0], start[1], 'go', markersize=12, label='Start')
    plt.plot(goal[0], goal[1], 'ro', markersize=12, label='Goal')
    
    # Plot waypoints with stars and numbers
    for i, wp in enumerate(waypoints):
        plt.plot(wp[0], wp[1], 'm*', markersize=15)
        plt.text(wp[0]+0.5, wp[1]+0.5, str(i), color='magenta', fontsize=12, 
                 bbox=dict(facecolor='white', alpha=0.7))
    
    # Connect waypoints with lines
    waypoint_x = [p[0] for p in waypoints]
    waypoint_y = [p[1] for p in waypoints]
    plt.plot(waypoint_x, waypoint_y, 'm--', linewidth=2, label='Waypoints')
    
    plt.title("LLM-Generated Waypoints")
    plt.legend()
    plt.savefig(os.path.join(scenario_log_dir, "visualizations", "waypoints.png"))
    plt.close()
    
    # 2. Path comparison visualization
    plt.figure(figsize=(12, 10))
    plt.imshow(env.grid, cmap='binary', interpolation='nearest')
    plt.plot(start[0], start[1], 'go', markersize=12, label='Start')
    plt.plot(goal[0], goal[1], 'ro', markersize=12, label='Goal')
    
    # Plot waypoints
    for i, wp in enumerate(waypoints):
        plt.plot(wp[0], wp[1], 'm*', markersize=10)
    
    # Plot baseline path
    if baseline_path:
        baseline_x = [p[0] for p in baseline_path]
        baseline_y = [p[1] for p in baseline_path]
        plt.plot(baseline_x, baseline_y, 'b-', linewidth=2, label='Baseline A*')
    
    # Plot LLM A* path
    if llm_astar_path:
        llm_astar_x = [p[0] for p in llm_astar_path]
        llm_astar_y = [p[1] for p in llm_astar_path]
        plt.plot(llm_astar_x, llm_astar_y, 'g-', linewidth=2, label='LLM A*')
    
    plt.title("Path Comparison")
    plt.legend()
    plt.savefig(os.path.join(scenario_log_dir, "visualizations", "path_comparison.png"))
    plt.close()
    
    # 3. Node expansion comparison
    plt.figure(figsize=(12, 10))
    plt.imshow(env.grid, cmap='binary', interpolation='nearest')
    
    # Plot expanded nodes
    baseline_expanded_x = [p[0] for p in baseline_expanded]
    baseline_expanded_y = [p[1] for p in baseline_expanded]
    plt.scatter(baseline_expanded_x, baseline_expanded_y, c='blue', alpha=0.3, s=30, label='Baseline Expanded')
    
    llm_astar_expanded_x = [p[0] for p in llm_astar_expanded]
    llm_astar_expanded_y = [p[1] for p in llm_astar_expanded]
    plt.scatter(llm_astar_expanded_x, llm_astar_expanded_y, c='green', alpha=0.3, s=30, label='LLM A* Expanded')
    
    # Plot start, goal and waypoints
    plt.plot(start[0], start[1], 'go', markersize=12, label='Start')
    plt.plot(goal[0], goal[1], 'ro', markersize=12, label='Goal')
    for wp in waypoints:
        plt.plot(wp[0], wp[1], 'm*', markersize=10)
    
    plt.title("Node Expansion Comparison")
    plt.legend()
    plt.savefig(os.path.join(scenario_log_dir, "visualizations", "node_expansion.png"))
    plt.close()
    
    # 4. Heatmap of expanded nodes
    plt.figure(figsize=(12, 10))
    
    # Create heatmaps
    heatmap = np.zeros((env.height, env.width))
    for x, y in baseline_expanded:
        if 0 <= y < env.height and 0 <= x < env.width:
            heatmap[y, x] += 1
    
    plt.imshow(heatmap, cmap='Blues', interpolation='nearest', alpha=0.7)
    plt.colorbar(label='Expansion Count')
    
    # Plot start, goal and waypoints
    plt.plot(start[0], start[1], 'go', markersize=12, label='Start')
    plt.plot(goal[0], goal[1], 'ro', markersize=12, label='Goal')
    for wp in waypoints:
        plt.plot(wp[0], wp[1], 'm*', markersize=10)
    
    plt.title("Baseline A* Expansion Heatmap")
    plt.legend()
    plt.savefig(os.path.join(scenario_log_dir, "visualizations", "baseline_expansion_heatmap.png"))
    plt.close()
    
    # 5. Heatmap of LLM A* expanded nodes
    plt.figure(figsize=(12, 10))
    
    # Create heatmaps
    heatmap = np.zeros((env.height, env.width))
    for x, y in llm_astar_expanded:
        if 0 <= y < env.height and 0 <= x < env.width:
            heatmap[y, x] += 1
    
    plt.imshow(heatmap, cmap='Greens', interpolation='nearest', alpha=0.7)
    plt.colorbar(label='Expansion Count')
    
    # Plot start, goal and waypoints
    plt.plot(start[0], start[1], 'go', markersize=12, label='Start')
    plt.plot(goal[0], goal[1], 'ro', markersize=12, label='Goal')
    for wp in waypoints:
        plt.plot(wp[0], wp[1], 'm*', markersize=10)
    
    plt.title("LLM A* Expansion Heatmap")
    plt.legend()
    plt.savefig(os.path.join(scenario_log_dir, "visualizations", "llm_astar_expansion_heatmap.png"))
    plt.close()
    
    # 6. Save waypoints to file
    with open(os.path.join(scenario_log_dir, "data", "waypoints.json"), 'w') as f:
        json.dump({
            "waypoints": [list(wp) for wp in waypoints],
            "description": "LLM-generated waypoints for navigation"
        }, f, indent=2)

def visualize_node_expansion_with_waypoints(environment, expanded_nodes_baseline, expanded_nodes_constrained, 
                           baseline_path, constrained_path, start, goal, waypoints, save_path=None):
    """
    Visualize the nodes expanded by baseline and constrained planners, including waypoints
    
    Args:
        environment: The grid environment
        expanded_nodes_baseline: List of (x, y) tuples for nodes expanded by baseline planner
        expanded_nodes_constrained: List of (x, y) tuples for nodes expanded by constrained planner
        baseline_path: List of (x, y) tuples for baseline path
        constrained_path: List of (x, y) tuples for constrained path
        start: (x, y) tuple for start position
        goal: (x, y) tuple for goal position
        waypoints: List of (x, y) tuples for waypoints
        save_path: Path to save the visualization
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Create grid representation
    grid = environment.grid.copy()
    
    # Plot baseline expansion
    ax1.imshow(grid, cmap='binary', interpolation='nearest')
    ax1.set_title(f'Baseline Planner (Nodes Expanded: {len(expanded_nodes_baseline)})')
    
    # Plot expanded nodes
    if expanded_nodes_baseline:
        exp_x, exp_y = zip(*expanded_nodes_baseline)
        ax1.scatter(exp_x, exp_y, c='blue', alpha=0.3, s=10)
    
    # Plot path
    if baseline_path:
        path_x, path_y = zip(*baseline_path)
        ax1.plot(path_x, path_y, 'b-', linewidth=2)
    
    # Plot start and goal
    ax1.plot(start[0], start[1], 'go', markersize=10)
    ax1.plot(goal[0], goal[1], 'ro', markersize=10)
    
    # Plot waypoints on both subplots
    for i, wp in enumerate(waypoints):
        ax1.plot(wp[0], wp[1], 'm*', markersize=12)
        ax1.text(wp[0]+0.5, wp[1]+0.5, str(i), color='magenta', fontsize=10, 
                 bbox=dict(facecolor='white', alpha=0.7))
    
    # Connect waypoints with lines
    if waypoints:
        waypoint_x = [p[0] for p in waypoints]
        waypoint_y = [p[1] for p in waypoints]
        ax1.plot(waypoint_x, waypoint_y, 'm--', linewidth=1.5, alpha=0.7)
    
    # Plot constrained expansion
    ax2.imshow(grid, cmap='binary', interpolation='nearest')
    ax2.set_title(f'LLM A* Planner (Nodes Expanded: {len(expanded_nodes_constrained)})')
    
    # Plot expanded nodes
    if expanded_nodes_constrained:
        exp_x, exp_y = zip(*expanded_nodes_constrained)
        ax2.scatter(exp_x, exp_y, c='green', alpha=0.3, s=10)
    
    # Plot path
    if constrained_path:
        path_x, path_y = zip(*constrained_path)
        ax2.plot(path_x, path_y, 'g-', linewidth=2)
    
    # Plot start and goal
    ax2.plot(start[0], start[1], 'go', markersize=10)
    ax2.plot(goal[0], goal[1], 'ro', markersize=10)
    
    # Plot waypoints on both subplots
    for i, wp in enumerate(waypoints):
        ax2.plot(wp[0], wp[1], 'm*', markersize=12)
        ax2.text(wp[0]+0.5, wp[1]+0.5, str(i), color='magenta', fontsize=10, 
                 bbox=dict(facecolor='white', alpha=0.7))
    
    # Connect waypoints with lines
    if waypoints:
        waypoint_x = [p[0] for p in waypoints]
        waypoint_y = [p[1] for p in waypoints]
        ax2.plot(waypoint_x, waypoint_y, 'm--', linewidth=1.5, alpha=0.7)
    
    # Set axis labels
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def run_llm_astar_experiments(num_scenarios=10):
    """
    Run LLM A* experiments on the dataset scenarios
    
    Args:
        num_scenarios: Number of scenarios to process (default: 10)
    
    Returns:
        Dictionary with experiment results
    """
    # Create timestamp for log directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("ds_logs", "llm_astar", f"experiment_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    
    # Create subdirectories for each scenario
    for i in range(num_scenarios):  # Process specified number of scenarios
        os.makedirs(os.path.join(log_dir, f"scenario_{i}"), exist_ok=True)
        os.makedirs(os.path.join(log_dir, f"scenario_{i}", "maps"), exist_ok=True)
        os.makedirs(os.path.join(log_dir, f"scenario_{i}", "regions"), exist_ok=True)
        os.makedirs(os.path.join(log_dir, f"scenario_{i}", "paths"), exist_ok=True)
        os.makedirs(os.path.join(log_dir, f"scenario_{i}", "metrics"), exist_ok=True)
        os.makedirs(os.path.join(log_dir, f"scenario_{i}", "visualizations"), exist_ok=True)
        os.makedirs(os.path.join(log_dir, f"scenario_{i}", "data"), exist_ok=True)
    
    # Load dataset - fixed path
    dataset_path = "dataset/dataset/dataset_index.json"
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
    
    # Create results dictionary
    results = {
        "timestamp": timestamp,
        "scenarios": []
    }
    
    # Initialize LLM A* planner
    llm_planner = LLMAStarPlanner()
    
    # Process each scenario (up to num_scenarios)
    for scenario_idx, scenario in enumerate(dataset["scenarios"]):
        # Only process the specified number of scenarios
        if scenario_idx >= num_scenarios:
            break
            
        print(f"\nProcessing scenario {scenario_idx}: {scenario['id']}")
        scenario_log_dir = os.path.join(log_dir, f"scenario_{scenario_idx}")
        
        # Find the map for this scenario
        map_id = scenario["map_id"]
        map_info = None
        for m in dataset["maps"]:
            if m["id"] == map_id:
                map_info = m
                break
        
        if not map_info:
            print(f"Map {map_id} not found in dataset, skipping scenario")
            continue
        
        # Fix paths to be relative to the dataset directory
        grid_path = os.path.join("dataset", map_info["grid_path"])
        
        # Load map and regions
        try:
            grid = np.load(grid_path)
            print(f"Loaded grid from {grid_path}")
        except FileNotFoundError:
            # Try alternative path
            alt_grid_path = map_info["grid_path"].replace("./", "dataset/dataset/")
            print(f"Trying alternative path: {alt_grid_path}")
            grid = np.load(alt_grid_path)
        
        regions = {}
        for region_name, region_path in map_info["regions"].items():
            try:
                region_path_full = os.path.join("dataset", region_path)
                regions[region_name] = np.load(region_path_full)
            except FileNotFoundError:
                # Try alternative path
                alt_region_path = region_path.replace("./", "dataset/dataset/")
                print(f"Trying alternative path for region {region_name}: {alt_region_path}")
                regions[region_name] = np.load(alt_region_path)
        
        # Create environment class for compatibility with existing code
        class Environment:
            def __init__(self):
                self.grid = grid
                self.width = grid.shape[1]
                self.height = grid.shape[0]
                self.regions = {}
                
                # Convert region masks to point lists for compatibility
                for region_name, region_mask in regions.items():
                    if region_name != "segmentation":
                        points = []
                        for y in range(region_mask.shape[0]):
                            for x in range(region_mask.shape[1]):
                                if region_mask[y, x]:
                                    points.append((x, y))
                        self.regions[region_name] = points
            
            def is_valid(self, x, y):
                """Check if a position is valid (within bounds and not an obstacle)"""
                # Check bounds
                if x < 0 or x >= self.width or y < 0 or y >= self.height:
                    return False
                
                # Check if obstacle
                if self.grid[y, x] == 1:
                    return False
                
                return True
                
            def to_dict(self):
                """Convert environment to dictionary for API call"""
                return {
                    "width": int(self.width),
                    "height": int(self.height),
                    "start": self.start,
                    "goal": self.goal,
                    "grid_sample": self.grid.tolist()[:10][:10],  # Just a small sample
                    "regions": {k: v[:5] for k, v in self.regions.items()}  # Limited regions
                }
        
        env = Environment()
        
        # Get start and goal - convert to tuples for hashability
        start = tuple(scenario["start"])
        goal = tuple(scenario["goal"])
        
        # Store start/goal in environment for context
        env.start = start
        env.goal = goal
        
        # Get instruction
        instruction = scenario['instruction']
        
        # Run baseline planner with standard A* search
        print("Running baseline planner...")
        baseline_start_time = time.time()
        baseline_path, baseline_cost, baseline_expanded, baseline_max_memory = a_star_search(
            env, start, goal, uniform_cost_function(env), track_expanded=True)
        baseline_end_time = time.time()
        baseline_search_time_ms = (baseline_end_time - baseline_start_time) * 1000  # Convert to milliseconds
        
        # Calculate baseline metrics
        baseline_metrics = calculate_path_metrics(baseline_path, env)
        baseline_metrics["expanded_nodes"] = len(baseline_expanded)
        baseline_metrics["max_memory_usage"] = baseline_max_memory
        # Get waypoints from LLM
        print("Getting waypoints from LLM...")
        waypoints_start_time = time.time()
        waypoints = llm_planner.get_waypoints(instruction, env, include_visualization=True)
        waypoints_end_time = time.time()
        waypoints_time_ms = (waypoints_end_time - waypoints_start_time) * 1000  # Convert to milliseconds
        
        print(f"Generated {len(waypoints)} waypoints: {waypoints}")
        
        # Run LLM A* search
        print("Running LLM A* search...")
        llm_astar_start_time = time.time()
        llm_astar_path, llm_astar_cost, llm_astar_expanded, llm_astar_max_memory = llm_planner.llm_a_star_search(
            env, start, goal, waypoints, track_expanded=True)
        llm_astar_end_time = time.time()
        llm_astar_search_time_ms = (llm_astar_end_time - llm_astar_start_time) * 1000  # Convert to milliseconds
        llm_astar_total_time_ms = waypoints_time_ms + llm_astar_search_time_ms
        
        # Calculate LLM A* metrics
        llm_astar_metrics = calculate_path_metrics(llm_astar_path, env)
        llm_astar_metrics["expanded_nodes"] = len(llm_astar_expanded)
        
        # Save comprehensive data for later analysis
        scenario_data = {
            "scenario_id": scenario['id'],
            "instruction": instruction,
            "environment": {
                "grid_shape": env.grid.shape,
                "start": start,
                "goal": goal,
                "regions": {name: len(points) for name, points in env.regions.items()}
            },
            "annotations": {
                "preference": [],
                "avoidance": [],
                "proximity": []
            },
            "waypoints": [list(p) for p in waypoints],
            "baseline": {
                "path": baseline_path,
                "path_length": baseline_metrics["length"],
                "nodes_expanded": baseline_metrics["expanded_nodes"],
                "search_time_ms": baseline_search_time_ms,
                "region_coverage": baseline_metrics["region_coverage"],
                "smoothness": baseline_metrics["smoothness"],
                "max_memory_usage": baseline_max_memory
            },
            "constrained": {
                "path": llm_astar_path,
                "path_length": llm_astar_metrics["length"],
                "nodes_expanded": llm_astar_metrics["expanded_nodes"],
                "search_time_ms": llm_astar_search_time_ms,
                "region_coverage": llm_astar_metrics["region_coverage"],
                "smoothness": llm_astar_metrics["smoothness"],
                "max_memory_usage": llm_astar_max_memory
            }
        }

        # Try to load annotations if available
        try:
            for constraint_type in ["preference", "avoidance"]:
                annotation_path = f"dataset/dataset/annotations/{scenario['id']}_{constraint_type}.npy"
                if os.path.exists(annotation_path):
                    annotation = np.load(annotation_path)
                    # Find regions that match this annotation
                    for region_name in env.regions:
                        if region_name == "segmentation":
                            continue
                            
                        # Check if region has annotation
                        region_points = env.regions[region_name]
                        region_has_annotation = False
                        for x, y in region_points:
                            if 0 <= y < annotation.shape[0] and 0 <= x < annotation.shape[1] and annotation[y, x] > 0:
                                region_has_annotation = True
                                break
                        
                        if region_has_annotation:
                            scenario_data["annotations"][constraint_type].append(region_name)
        except Exception as e:
            print(f"Error loading annotations: {e}")

        # Save the comprehensive data
        with open(os.path.join(scenario_log_dir, "data", "scenario_data.json"), 'w') as f:
            # Convert any non-serializable objects (like numpy arrays) to lists
            def convert_to_serializable(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, tuple):
                    return list(obj)
                return obj
            
            # Custom JSON encoder to handle non-serializable objects
            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, (np.ndarray, np.integer, np.floating)):
                        return convert_to_serializable(obj)
                    elif isinstance(obj, tuple):
                        return list(obj)
                    return super(NumpyEncoder, self).default(obj)
            
            json.dump(scenario_data, f, indent=2, cls=NumpyEncoder)
        
        # Visualize node expansion with waypoints
        visualize_node_expansion_with_waypoints(
            env,  # environment
            baseline_expanded,  # expanded_nodes_baseline
            llm_astar_expanded,  # expanded_nodes_constrained
            baseline_path,  # baseline_path
            llm_astar_path,  # constrained_path
            start,  # start
            goal,  # goal
            waypoints,  # waypoints
            save_path=os.path.join(scenario_log_dir, "visualizations", "node_expansion_with_waypoints.png")
        )

        # Also keep the original node expansion visualization for comparison
        visualize_node_expansion(
            env,
            baseline_expanded,
            llm_astar_expanded,
            baseline_path,
            llm_astar_path,
            start,
            goal,
            save_path=os.path.join(scenario_log_dir, "visualizations", "node_expansion_comparison.png")
        )

        # And keep the dedicated waypoints visualization
        plt.figure(figsize=(12, 10))
        plt.imshow(env.grid, cmap='binary', interpolation='nearest')
        plt.plot(start[0], start[1], 'go', markersize=12, label='Start')
        plt.plot(goal[0], goal[1], 'ro', markersize=12, label='Goal')

        # Plot waypoints with stars and numbers
        for i, wp in enumerate(waypoints):
            plt.plot(wp[0], wp[1], 'm*', markersize=15)
            plt.text(wp[0]+0.5, wp[1]+0.5, str(i), color='magenta', fontsize=12, 
                     bbox=dict(facecolor='white', alpha=0.7))

        # Connect waypoints with lines
        waypoint_x = [p[0] for p in waypoints]
        waypoint_y = [p[1] for p in waypoints]
        plt.plot(waypoint_x, waypoint_y, 'm--', linewidth=2, label='Waypoints')

        plt.title("LLM-Generated Waypoints")
        plt.legend()
        plt.savefig(os.path.join(scenario_log_dir, "visualizations", "waypoints.png"))
        plt.close()
        
        # Save waypoints to file
        with open(os.path.join(scenario_log_dir, "data", "waypoints.json"), 'w') as f:
            json.dump({
                "waypoints": [list(wp) for wp in waypoints],
                "description": "LLM-generated waypoints for navigation"
            }, f, indent=2)
        
        # Save metrics in the exact same format as DCIP/SLICE
        # Create a dummy constraint dictionary for compatibility
        waypoint_constraint_dict = {
            "preference": {},
            "avoidance": {},
            "proximity": {}
        }

        metrics = {
            "scenario_id": scenario['id'],
            "instruction": instruction,
            "constraint_set": waypoint_constraint_dict,  # Empty constraint set for compatibility
            "baseline": {
                "path_length": len(baseline_path),
                "nodes_expanded": len(baseline_expanded),
                "search_efficiency": len(baseline_path) / len(baseline_expanded) if len(baseline_expanded) > 0 else 0
            },
            "constrained": {  # Use this name to match SLICE format
                "path_length": len(llm_astar_path) if llm_astar_path else 0,
                "nodes_expanded": len(llm_astar_expanded),
                "search_efficiency": len(llm_astar_path) / len(llm_astar_expanded) if len(llm_astar_expanded) > 0 else 0
            },
            "improvement": {
                "path_length": (baseline_metrics["length"] - llm_astar_metrics["length"]) / baseline_metrics["length"] if baseline_metrics["length"] > 0 else 0,
                "smoothness": llm_astar_metrics["smoothness"] - baseline_metrics["smoothness"],
                "nodes_expanded": (baseline_metrics["expanded_nodes"] - llm_astar_metrics["expanded_nodes"]) / baseline_metrics["expanded_nodes"] if baseline_metrics["expanded_nodes"] > 0 else 0
            }
        }
        
        with open(os.path.join(scenario_log_dir, "metrics", "metrics.json"), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Add to results
        results["scenarios"].append(metrics)
        
        print(f"Completed scenario {scenario_idx}")
        print(f"Baseline path length: {baseline_metrics['length']:.2f}, expanded nodes: {baseline_metrics['expanded_nodes']}")
        print(f"LLM A* path length: {llm_astar_metrics['length']:.2f}, expanded nodes: {llm_astar_metrics['expanded_nodes']}")
        print(f"LLM A* waypoints time: {waypoints_time_ms:.2f}ms, search time: {llm_astar_search_time_ms:.2f}ms")
    
    # Save overall results
    with open(os.path.join(log_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nExperiment complete. Results saved to {log_dir}")
    return results

if __name__ == "__main__":
    # Run experiments with fixed number of scenarios
    results = run_llm_astar_experiments(num_scenarios=10)