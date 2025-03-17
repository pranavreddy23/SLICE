#!/usr/bin/env python3
import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from core.region_segmentation import RegionSegmenter
from core.rrt_planner import RRTPlanner, RRTNode
from utils.viz import visualize_node_expansion, visualize_paths, visualize_regions, visualize_grid, visualize_cost_landscape
import base64
import io
import copy
from PIL import Image
from core.constraintext import ConstraintSet
from core.cfgen import CostFunctionGenerator, uniform_cost_function
import math

def run_rrt_experiments():
    """Run RRT experiments on the dataset scenarios"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("rrt_logs", f"experiment_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    
    # Create subdirectories for each scenario
    for i in range(10):  # 10 scenarios (0-9)
        os.makedirs(os.path.join(log_dir, f"scenario_{i}"), exist_ok=True)
        os.makedirs(os.path.join(log_dir, f"scenario_{i}", "maps"), exist_ok=True)
        os.makedirs(os.path.join(log_dir, f"scenario_{i}", "regions"), exist_ok=True)
        os.makedirs(os.path.join(log_dir, f"scenario_{i}", "paths"), exist_ok=True)
        os.makedirs(os.path.join(log_dir, f"scenario_{i}", "metrics"), exist_ok=True)
        os.makedirs(os.path.join(log_dir, f"scenario_{i}", "visualizations"), exist_ok=True)
        os.makedirs(os.path.join(log_dir, f"scenario_{i}", "data"), exist_ok=True)
    
    # Load dataset
    dataset_path = "dataset/dataset/dataset_index.json"
    dataset_dir = os.path.dirname(dataset_path)
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
    
    # Create results dictionary
    all_results = {
        "timestamp": timestamp,
        "scenarios": []
    }
    
    # Process each scenario
    for scenario_idx, scenario in enumerate(dataset["scenarios"]):
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
                region_samples = {}
                for region_name, points in self.regions.items():
                    if region_name != "segmentation" and points:
                        # Take a sample of points to keep the dict size reasonable
                        sample_size = min(20, len(points))
                        step = max(1, len(points) // sample_size)
                        region_samples[region_name] = points[::step][:sample_size]
                
                return {
                    "width": int(self.width),
                    "height": int(self.height),
                    "start": self.start if hasattr(self, 'start') else None,
                    "goal": self.goal if hasattr(self, 'goal') else None,
                    "regions": region_samples
                }
        
        env = Environment()
        
        # Get start and goal - convert to tuples for hashability
        start = tuple(scenario["start"])
        goal = tuple(scenario["goal"])
        
        # Store start/goal in environment for context
        env.start = start
        env.goal = goal
        
        # Visualize map with start and goal
        plt.figure(figsize=(10, 10))
        plt.imshow(env.grid, cmap='binary', interpolation='nearest')
        plt.plot(start[0], start[1], 'go', markersize=10, label='Start')
        plt.plot(goal[0], goal[1], 'ro', markersize=10, label='Goal')
        plt.title(f"Environment with Start and Goal")
        plt.legend()
        plt.savefig(os.path.join(scenario_log_dir, "maps", "environment_with_positions.png"))
        plt.close()
        
        # Visualize regions
        print("Visualizing regions...")
        visualize_regions(env, save_path=os.path.join(scenario_log_dir, "regions", "regions.png"))
        
        # Get instruction from scenario
        instruction = scenario.get("instruction", "Navigate to the goal efficiently.")
        print(f"Using instruction: {instruction}")
        
        # Visualize RRT trees and paths
        def visualize_rrt_tree(env, nodes, path, start, goal, title, save_path):
            """Visualize RRT tree, path, and obstacles"""
            plt.figure(figsize=(12, 12))
            
            # Plot grid/obstacles
            plt.imshow(env.grid, cmap='binary', interpolation='nearest')
            
            # Plot RRT tree edges
            for node in nodes:
                if node.parent is not None:
                    plt.plot([node.x, node.parent.x], [node.y, node.parent.y], 'c-', alpha=0.3, linewidth=0.5)
            
            # Plot path
            if path:
                path_x = [p[0] for p in path]
                path_y = [p[1] for p in path]
                plt.plot(path_x, path_y, 'g-', linewidth=2, label='Path')
            
            # Plot start and goal
            plt.plot(start[0], start[1], 'go', markersize=10, label='Start')
            plt.plot(goal[0], goal[1], 'ro', markersize=10, label='Goal')
            
            # Add legend and title
            plt.legend(loc='upper right')
            plt.title(title)
            plt.tight_layout()
            
            # Save figure
            plt.savefig(save_path, dpi=150)
            plt.close()
        
        # Run baseline RRT planner
        print("Running baseline RRT planner...")
        baseline_start_time = time.time()
        
        # Create a uniform cost function using the existing function
        uniform_cost = uniform_cost_function(env)
        
        # For RRT, we need a cost function that takes (x1, y1, x2, y2)
        def rrt_uniform_cost(x1, y1, x2, y2):
            # Base cost is Euclidean distance
            base_cost = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            # Check if destination is valid
            if not env.is_valid(x2, y2):
                return float('inf')
            return base_cost
        
        # Initialize RRT planner
        baseline_planner = RRTPlanner(max_iterations=2000, step_size=10.0, goal_sample_rate=0.1)
        
        # Run baseline planner
        baseline_path, baseline_cost, baseline_expanded, baseline_max_tree_size = baseline_planner.plan(
            env, start, goal, rrt_uniform_cost, track_expanded=True
        )
        
        # Get all nodes in the tree for visualization
        baseline_nodes = []
        for node in baseline_expanded:
            baseline_nodes.append(RRTNode(node[0], node[1]))
        
        baseline_end_time = time.time()
        baseline_search_time_ms = (baseline_end_time - baseline_start_time) * 1000
        
        # Visualize baseline RRT tree and path
        visualize_rrt_tree(
            env, 
            baseline_nodes, 
            baseline_path, 
            start, 
            goal, 
            "Baseline RRT Tree and Path",
            os.path.join(scenario_log_dir, "visualizations", "baseline_rrt_tree.png")
        )
        
        # Visualize the baseline cost landscape using the uniform cost function
        visualize_cost_landscape(
            env, 
            uniform_cost, 
            start, 
            goal, 
            "Baseline Cost Landscape",
            save_path=os.path.join(scenario_log_dir, "visualizations", "baseline_cost_landscape.png")
        )
        
        # Generate constraints using LLM
        print("Generating constraints using LLM...")
        cfgen = CostFunctionGenerator()
        
        try:
            # Extract constraints
            constraints = cfgen.extractor.extract_constraints(instruction, env)
            print(f"Extracted constraints: {constraints}")
        except Exception as e:
            print(f"Error extracting constraints: {e}")
            # Fall back to empty constraints
            constraints = ConstraintSet()
        
        # Create a constrained cost function
        def constrained_cost_function(x1, y1, x2, y2):
            """
            Create a cost function that prioritizes search efficiency by:
            1. Using a directional bias toward the goal
            2. Reducing costs in preferred regions
            3. Increasing costs in avoided regions
            4. Ensuring start/goal areas have low cost
            """
            # Base cost is Euclidean distance
            base_cost = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            
            # Add directional bias toward goal (reduces node expansion)
            # Calculate how much closer this step gets us to the goal
            current_to_goal = ((goal[0] - x1) ** 2 + (goal[1] - y1) ** 2) ** 0.5
            next_to_goal = ((goal[0] - x2) ** 2 + (goal[1] - y2) ** 2) ** 0.5
            
            # If we're getting closer to the goal, reduce cost
            if next_to_goal < current_to_goal:
                directional_factor = 0.8  # Reduce cost by 20% when moving toward goal
            else:
                directional_factor = 1.2  # Increase cost by 20% when moving away from goal
            
            # Check if the point is in an obstacle
            if not env.is_valid(x2, y2):
                return float('inf')  # Infinite cost for obstacles
            
            # Apply constraint preferences
            cost_multiplier = directional_factor  # Start with directional bias
            
            # Apply preference constraints (reduce cost for preferred regions)
            if hasattr(constraints, 'preference') and constraints.preference:
                for region_name, weight in constraints.preference.items():
                    if region_name in env.regions and (x2, y2) in env.regions[region_name]:
                        # More aggressive cost reduction for preferred regions
                        cost_multiplier *= max(0.1, 1.0 - (weight / 8.0))
            
            # Apply avoidance constraints (increase cost for avoided regions)
            if hasattr(constraints, 'avoidance') and constraints.avoidance:
                for region_name, weight in constraints.avoidance.items():
                    if region_name in env.regions and (x2, y2) in env.regions[region_name]:
                        # More aggressive cost increase for avoided regions
                        cost_multiplier *= min(10.0, 1.0 + (weight / 2.0))
            
            # Special case for start/goal areas - ensure very low cost
            start_area = [(x, y) for x in range(start[0] - 3, start[0] + 4) 
                         for y in range(start[1] - 3, start[1] + 4)
                         if 0 <= x < env.width and 0 <= y < env.height]
            
            goal_area = [(x, y) for x in range(goal[0] - 3, goal[0] + 4) 
                        for y in range(goal[1] - 3, goal[1] + 4)
                        if 0 <= x < env.width and 0 <= y < env.height]
            
            if (x2, y2) in start_area or (x2, y2) in goal_area:
                cost_multiplier *= 0.5  # Significantly reduce cost near start/goal
            
            # Add a small random factor to break ties (improves efficiency)
            random_factor = 1.0 + (np.random.random() * 0.01)  # Up to 1% random variation
            
            return base_cost * cost_multiplier * random_factor
        
        # Create the constrained cost function
        constrained_cost = constrained_cost_function
        
        # Visualize the constrained cost landscape
        visualize_cost_landscape(
            env, 
            constrained_cost, 
            start, 
            goal, 
            "Constrained Cost Landscape",
            save_path=os.path.join(scenario_log_dir, "visualizations", "constrained_cost_landscape.png")
        )
        
        # Run constrained RRT planner
        print("Running constrained RRT planner...")
        constrained_start_time = time.time()
        
        # Initialize RRT planner
        constrained_planner = RRTPlanner(max_iterations=2000, step_size=10.0, goal_sample_rate=0.1)
        
        # Run constrained planner
        constrained_path, constrained_cost_value, constrained_expanded, constrained_max_tree_size = constrained_planner.plan(
            env, start, goal, constrained_cost, track_expanded=True
        )
        constrained_end_time = time.time()
        # Get all nodes in the tree for visualization
        constrained_nodes = []
        for node in constrained_expanded:
            constrained_nodes.append(RRTNode(node[0], node[1]))
        
        
        constrained_search_time_ms = (constrained_end_time - constrained_start_time) * 1000
        
        # Visualize constrained RRT tree and path
        visualize_rrt_tree(
            env, 
            constrained_nodes, 
            constrained_path, 
            start, 
            goal, 
            "SLICE+RRT Tree and Path",
            os.path.join(scenario_log_dir, "visualizations", "constrained_rrt_tree.png")
        )
        
        # Visualize both paths for comparison
        plt.figure(figsize=(12, 12))
        
        # Plot grid/obstacles
        plt.imshow(env.grid, cmap='binary', interpolation='nearest')
        
        # Plot regions with different colors
        colors = plt.cm.tab10(np.linspace(0, 1, len(env.regions)))
        for i, (region_name, points) in enumerate(env.regions.items()):
            if region_name == "segmentation" or len(points) == 0:
                continue
            
            # Sample points to avoid overcrowding
            sample_size = min(500, len(points))
            step = max(1, len(points) // sample_size)
            sample_points = points[::step]
            
            # Extract x and y coordinates
            x_coords = [p[0] for p in sample_points]
            y_coords = [p[1] for p in sample_points]
            
            # Plot region points
            plt.scatter(x_coords, y_coords, color=colors[i], alpha=0.3, s=10, label=region_name)
        
        # Plot baseline path
        if baseline_path:
            path_x = [p[0] for p in baseline_path]
            path_y = [p[1] for p in baseline_path]
            plt.plot(path_x, path_y, 'b-', linewidth=2, label='Baseline RRT')
        
        # Plot constrained path
        if constrained_path:
            path_x = [p[0] for p in constrained_path]
            path_y = [p[1] for p in constrained_path]
            plt.plot(path_x, path_y, 'g-', linewidth=2, label='SLICE+RRT')
        
        # Plot start and goal
        plt.plot(start[0], start[1], 'go', markersize=10, label='Start')
        plt.plot(goal[0], goal[1], 'ro', markersize=10, label='Goal')
        
        # Add legend and title
        plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1))
        plt.title("Path Comparison: Baseline RRT vs SLICE+RRT")
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(scenario_log_dir, "visualizations", "path_comparison.png"), dpi=150)
        plt.close()
        
        # Calculate path metrics
        def calculate_path_metrics(path):
            if not path:
                return {
                    "length": 0,
                    "smoothness": 0,
                    "region_coverage": {}
                }
            
            # Calculate path length
            length = 0
            for i in range(1, len(path)):
                dx = path[i][0] - path[i-1][0]
                dy = path[i][1] - path[i-1][1]
                length += (dx**2 + dy**2)**0.5
            
            # Calculate path smoothness
            angles = []
            for i in range(1, len(path) - 1):
                # Get vectors
                v1 = (path[i][0] - path[i-1][0], path[i][1] - path[i-1][1])
                v2 = (path[i+1][0] - path[i][0], path[i+1][1] - path[i][1])
                
                # Calculate dot product
                dot_product = v1[0] * v2[0] + v1[1] * v2[1]
                
                # Calculate magnitudes
                mag1 = (v1[0]**2 + v1[1]**2)**0.5
                mag2 = (v2[0]**2 + v2[1]**2)**0.5
                
                # Calculate angle (in radians)
                if mag1 * mag2 == 0:
                    angle = 0
                else:
                    angle = np.arccos(min(1, max(-1, dot_product / (mag1 * mag2))))
                
                angles.append(angle)
            
            # Average angle change (lower is smoother)
            smoothness = 1.0 - (sum(angles) / (len(angles) * np.pi) if angles else 0)
            
            # Calculate region coverage
            region_coverage = {}
            for region_name, points in env.regions.items():
                if region_name == "segmentation":
                    continue
                # Count path points in this region
                count = sum(1 for p in path if p in points)
                if count > 0:
                    region_coverage[region_name] = count / len(path)
            
            return {
                "length": length,
                "smoothness": smoothness,
                "region_coverage": region_coverage
            }
        
        # Calculate metrics for both paths
        baseline_metrics = calculate_path_metrics(baseline_path)
        constrained_metrics = calculate_path_metrics(constrained_path)
        
        # Add expanded nodes count
        baseline_metrics["expanded_nodes"] = len(baseline_expanded)
        constrained_metrics["expanded_nodes"] = len(constrained_expanded)
        
        # Calculate search times
        baseline_end_time = time.time()
        baseline_search_time_ms = (baseline_end_time - baseline_start_time) * 1000  # Convert to milliseconds
        
        constrained_end_time = time.time()
        constrained_search_time_ms = (constrained_end_time - constrained_start_time) * 1000  # Convert to milliseconds
        
        # Add max tree size to metrics
        baseline_metrics["max_memory_usage"] = baseline_max_tree_size
        constrained_metrics["max_memory_usage"] = constrained_max_tree_size
        
        # Save comprehensive data for later analysis in the same format as A* experiments
        scenario_data = {
            "scenario_id": scenario['id'],
            "instruction": instruction,
            "environment": {
                "grid_shape": env.grid.shape,
                "start": start,
                "goal": goal,
                "regions": {name: len(points) for name, points in env.regions.items() if name != "segmentation"}
            },
            "annotations": {
                "preference": [],
                "avoidance": []
            },
            "constraints": {
                "preference": getattr(constraints, 'preference', {}),
                "avoidance": getattr(constraints, 'avoidance', {}),
                "proximity": getattr(constraints, 'proximity', {})
            },
            "baseline": {
                "path": baseline_path,
                "path_length": baseline_metrics["length"],
                "nodes_expanded": baseline_metrics["expanded_nodes"],
                "search_time_ms": baseline_search_time_ms,
                "region_coverage": baseline_metrics["region_coverage"],
                "smoothness": baseline_metrics["smoothness"],
                "algorithm": "RRT",  # Add algorithm identifier
                "max_memory_usage": baseline_metrics["max_memory_usage"],
            },
            "constrained": {
                "path": constrained_path,
                "path_length": constrained_metrics["length"],
                "nodes_expanded": constrained_metrics["expanded_nodes"],
                "search_time_ms": constrained_search_time_ms,
                "region_coverage": constrained_metrics["region_coverage"],
                "smoothness": constrained_metrics["smoothness"],
                "algorithm": "RRT-SLICE",  # Add algorithm identifier
                "max_memory_usage": constrained_metrics["max_memory_usage"],
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
        
        # Save the comprehensive data in the same location as A* experiments
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
        
        # Also save the metrics in the metrics folder for backward compatibility
        with open(os.path.join(scenario_log_dir, "metrics", "rrt_metrics.json"), 'w') as f:
            json.dump({
                "scenario_id": scenario['id'],
                "instruction": instruction,
                "constraint_set": {
                    "preference": getattr(constraints, 'preference', {}),
                    "avoidance": getattr(constraints, 'avoidance', {}),
                    "proximity": getattr(constraints, 'proximity', {})
                },
                "baseline_rrt": {
                    "path_length": baseline_metrics["length"],
                    "nodes_expanded": baseline_metrics["expanded_nodes"],
                    "search_time_ms": baseline_search_time_ms,
                    "region_coverage": baseline_metrics["region_coverage"],
                    "smoothness": baseline_metrics["smoothness"],
                    "max_memory_usage": baseline_metrics["max_memory_usage"],
                },
                "constrained_rrt": {
                    "path_length": constrained_metrics["length"],
                    "nodes_expanded": constrained_metrics["expanded_nodes"],
                    "search_time_ms": constrained_search_time_ms,
                    "region_coverage": constrained_metrics["region_coverage"],
                    "smoothness": constrained_metrics["smoothness"],
                    "max_memory_usage": constrained_metrics["max_memory_usage"],
                }
            }, f, indent=2, cls=NumpyEncoder)
        
        print(f"RRT results for scenario {scenario['id']}:")
        print(f"  Baseline: {len(baseline_expanded)} nodes, path length: {baseline_metrics['length']:.2f}")
        print(f"  SLICE+RRT: {len(constrained_expanded)} nodes, path length: {constrained_metrics['length']:.2f}")
        
        # Add to all results
        all_results["scenarios"].append(scenario_data)
        
        print(f"Completed scenario {scenario_idx}")
        print(f"Baseline RRT path length: {baseline_metrics['length']:.2f}, expanded nodes: {baseline_metrics['expanded_nodes']}")
        print(f"Constrained RRT path length: {constrained_metrics['length']:.2f}, expanded nodes: {constrained_metrics['expanded_nodes']}")
    
    # Save overall results
    with open(os.path.join(log_dir, "rrt_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Generate summary plots
    generate_rrt_summary_plots(all_results, log_dir)
    
    print(f"\nRRT experiment complete. Results saved to {log_dir}")
    return all_results

def generate_rrt_summary_plots(results, log_dir):
    """Generate summary plots for all scenarios with RRT results"""
    scenarios = results["scenarios"]
    
    # Extract metrics with error handling for missing keys
    scenario_ids = [s.get("scenario_id", f"Scenario {i}") for i, s in enumerate(scenarios)]
    
    # Path length and expanded nodes
    baseline_lengths = [s.get("baseline_rrt", {}).get("path_length", 0) for s in scenarios]
    constrained_lengths = [s.get("constrained_rrt", {}).get("path_length", 0) for s in scenarios]
    baseline_expanded = [s.get("baseline_rrt", {}).get("nodes_expanded", 0) for s in scenarios]
    constrained_expanded = [s.get("constrained_rrt", {}).get("nodes_expanded", 0) for s in scenarios]
    
    # Calculate improvement percentages
    length_improvements = []
    expanded_improvements = []
    
    for i, s in enumerate(scenarios):
        # Path length improvement
        b_len = s.get("baseline_rrt", {}).get("path_length", 0)
        c_len = s.get("constrained_rrt", {}).get("path_length", 0)
        if b_len > 0:
            length_imp = (b_len - c_len) / b_len * 100
        else:
            length_imp = 0
        length_improvements.append(length_imp)
        
        # Expanded nodes improvement
        b_exp = s.get("baseline_rrt", {}).get("nodes_expanded", 0)
        c_exp = s.get("constrained_rrt", {}).get("nodes_expanded", 0)
        if b_exp > 0:
            exp_imp = (b_exp - c_exp) / b_exp * 100
        else:
            exp_imp = 0
        expanded_improvements.append(exp_imp)
    
    # Create plots directory if it doesn't exist
    plots_dir = os.path.join(log_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Path length comparison
    plt.figure(figsize=(12, 6))
    x = np.arange(len(scenario_ids))
    width = 0.35
    plt.bar(x - width/2, baseline_lengths, width, label='Baseline RRT')
    plt.bar(x + width/2, constrained_lengths, width, label='Constrained RRT')
    plt.xlabel('Scenario')
    plt.ylabel('Path Length')
    plt.title('RRT Path Length Comparison')
    plt.xticks(x, scenario_ids, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "rrt_path_length_comparison.png"))
    plt.close()
    
    # Expanded nodes comparison
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, baseline_expanded, width, label='Baseline RRT')
    plt.bar(x + width/2, constrained_expanded, width, label='Constrained RRT')
    plt.xlabel('Scenario')
    plt.ylabel('Expanded Nodes')
    plt.title('RRT Search Efficiency Comparison')
    plt.xticks(x, scenario_ids, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "rrt_expanded_nodes_comparison.png"))
    plt.close()
    
    # Improvement percentages
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, length_improvements, width, label='Path Length')
    plt.bar(x + width/2, expanded_improvements, width, label='Search Efficiency')
    plt.xlabel('Scenario')
    plt.ylabel('Improvement (%)')
    plt.title('SLICE+RRT Improvement over Baseline RRT')
    plt.xticks(x, scenario_ids, rotation=45)
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "rrt_improvement_comparison.png"))
    plt.close()
    
    # Create a summary table
    plt.figure(figsize=(12, 8))
    plt.axis('off')
    
    # Create table data
    table_data = []
    headers = ['Scenario', 'Baseline\nNodes', 'Constrained\nNodes', 'Nodes\nImprovement', 
               'Baseline\nPath', 'Constrained\nPath', 'Path\nChange']
    
    for i, s_id in enumerate(scenario_ids):
        b_exp = baseline_expanded[i]
        c_exp = constrained_expanded[i]
        exp_imp = expanded_improvements[i]
        
        b_len = baseline_lengths[i]
        c_len = constrained_lengths[i]
        len_imp = length_improvements[i]
        
        row = [
            s_id, 
            f"{b_exp}", 
            f"{c_exp}", 
            f"{exp_imp:.1f}%",
            f"{b_len:.1f}", 
            f"{c_len:.1f}", 
            f"{len_imp:.1f}%"
        ]
        table_data.append(row)
    
    # Add average row
    avg_b_exp = sum(baseline_expanded) / len(baseline_expanded) if baseline_expanded else 0
    avg_c_exp = sum(constrained_expanded) / len(constrained_expanded) if constrained_expanded else 0
    avg_exp_imp = sum(expanded_improvements) / len(expanded_improvements) if expanded_improvements else 0
    
    avg_b_len = sum(baseline_lengths) / len(baseline_lengths) if baseline_lengths else 0
    avg_c_len = sum(constrained_lengths) / len(constrained_lengths) if constrained_lengths else 0
    avg_len_imp = sum(length_improvements) / len(length_improvements) if length_improvements else 0
    
    avg_row = [
        'AVERAGE', 
        f"{avg_b_exp:.1f}", 
        f"{avg_c_exp:.1f}", 
        f"{avg_exp_imp:.1f}%",
        f"{avg_b_len:.1f}", 
        f"{avg_c_len:.1f}", 
        f"{avg_len_imp:.1f}%"
    ]
    table_data.append(avg_row)
    
    # Create the table
    table = plt.table(
        cellText=table_data,
        colLabels=headers,
        loc='center',
        cellLoc='center',
        colColours=['#f2f2f2'] * len(headers),
        rowColours=['#f2f2f2'] * len(table_data)
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    plt.title('SLICE+RRT Performance Summary', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "rrt_performance_summary_table.png"), bbox_inches='tight')
    plt.close()
    
    # Save summary as CSV for easy import to spreadsheets
    import csv
    with open(os.path.join(plots_dir, "rrt_performance_summary.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(table_data)
    
    # Create a text summary
    with open(os.path.join(log_dir, "rrt_summary.txt"), 'w') as f:
        f.write(f"SLICE+RRT Experiment Summary\n")
        f.write(f"=====================\n\n")
        f.write(f"Timestamp: {results.get('timestamp', 'N/A')}\n")
        f.write(f"Number of scenarios: {len(scenarios)}\n\n")
        
        f.write(f"Average Improvements:\n")
        f.write(f"  Search Efficiency: {avg_exp_imp:.2f}%\n")
        f.write(f"  Path Length: {avg_len_imp:.2f}%\n\n")
        
        f.write(f"Detailed Results:\n")
        for i, s in enumerate(scenarios):
            s_id = s.get("scenario_id", f"Scenario {i}")
            f.write(f"\n{s_id}:\n")
            f.write(f"  Instruction: {s.get('instruction', 'N/A')}\n")
            f.write(f"  Baseline nodes expanded: {baseline_expanded[i]}\n")
            f.write(f"  Constrained nodes expanded: {constrained_expanded[i]}\n")
            f.write(f"  Search efficiency improvement: {expanded_improvements[i]:.2f}%\n")
            f.write(f"  Baseline path length: {baseline_lengths[i]:.2f}\n")
            f.write(f"  Constrained path length: {constrained_lengths[i]:.2f}\n")
            f.write(f"  Path length change: {length_improvements[i]:.2f}%\n")
    
    print(f"Summary plots and reports generated in {plots_dir}")
    
    # Return the summary metrics for potential further analysis
    return {
        "avg_search_efficiency_improvement": avg_exp_imp,
        "avg_path_length_change": avg_len_imp,
        "scenarios": len(scenarios)
    }

def visualize_rrt_paths(env, baseline_path, constrained_path, start, goal, save_path=None):
    """Visualize RRT paths on the environment"""
    plt.figure(figsize=(10, 10))
    
    # Plot grid
    plt.imshow(env.grid, cmap='binary', interpolation='nearest')
    
    # Plot regions with different colors
    colors = plt.cm.tab10(np.linspace(0, 1, len(env.regions)))
    for i, (region_name, points) in enumerate(env.regions.items()):
        if region_name == "segmentation" or len(points) == 0:
            continue
        
        # Sample points to avoid overcrowding
        sample_size = min(1000, len(points))
        sample_idx = np.linspace(0, len(points)-1, sample_size, dtype=int)
        sampled_points = [points[i] for i in sample_idx]
        
        # Extract x and y coordinates
        x_coords = [p[0] for p in sampled_points]
        y_coords = [p[1] for p in sampled_points]
        
        # Plot region points
        plt.scatter(x_coords, y_coords, color=colors[i], alpha=0.3, s=5, label=region_name)
    
    # Plot start and goal
    plt.plot(start[0], start[1], 'go', markersize=10, label='Start')
    plt.plot(goal[0], goal[1], 'ro', markersize=10, label='Goal')
    
    # Plot baseline path
    if baseline_path:
        x_coords = [p[0] for p in baseline_path]
        y_coords = [p[1] for p in baseline_path]
        plt.plot(x_coords, y_coords, 'b-', linewidth=2, label='Baseline RRT')
    
    # Plot constrained path
    if constrained_path:
        x_coords = [p[0] for p in constrained_path]
        y_coords = [p[1] for p in constrained_path]
        plt.plot(x_coords, y_coords, 'g-', linewidth=2, label='SLICE+RRT')
    
    plt.title('RRT Path Comparison')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def visualize_rrt_tree(env, tree, path, start, goal, title="RRT Tree", save_path=None):
    """Visualize the RRT tree and path"""
    plt.figure(figsize=(10, 10))
    
    # Plot grid
    plt.imshow(env.grid, cmap='binary', interpolation='nearest')
    
    # Plot tree edges
    for node, parent in tree.items():
        if parent is not None:
            plt.plot([parent[0], node[0]], [parent[1], node[1]], 'c-', alpha=0.3, linewidth=0.5)
    
    # Plot path
    if path:
        x_coords = [p[0] for p in path]
        y_coords = [p[1] for p in path]
        plt.plot(x_coords, y_coords, 'g-', linewidth=2, label='Path')
    
    # Plot start and goal
    plt.plot(start[0], start[1], 'go', markersize=10, label='Start')
    plt.plot(goal[0], goal[1], 'ro', markersize=10, label='Goal')
    
    plt.title(title)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def run_rrt_on_scenario(env, start, goal, instruction, scenario_log_dir, scenario_id):
    """Run RRT and SLICE+RRT on a single scenario"""
    # Initialize RRT planner
    max_iterations = 5000  # Increase for complex environments
    step_size = 10.0  # Adjust based on environment scale
    goal_sample_rate = 0.1
    
    planner = RRTPlanner(max_iterations=max_iterations, step_size=step_size, goal_sample_rate=goal_sample_rate)
    
    # Define uniform cost function for baseline
    def uniform_cost(x1, y1, x2, y2):
        return ((x2 - x1)**2 + (y2 - y1)**2)**0.5
    
    # Run baseline RRT
    print("Running baseline RRT...")
    baseline_start_time = time.time()
    baseline_path, baseline_cost, baseline_expanded, baseline_max_tree_size = planner.plan(
        env, start, goal, cost_function=uniform_cost, track_expanded=True, return_tree=True)
    baseline_end_time = time.time()
    baseline_search_time_ms = (baseline_end_time - baseline_start_time) * 1000
    
    # Visualize baseline RRT tree
    visualize_rrt_tree(
        env, baseline_tree, baseline_path, start, goal, 
        title="Baseline RRT Tree",
        save_path=os.path.join(scenario_log_dir, "visualizations", "baseline_rrt_tree.png")
    )
    
    # Extract constraints using LLM
    print("Extracting constraints from instruction...")
    cfgen = CostFunctionGenerator()
    
    try:
        # Create a minimal environment representation for the API call
        class MinimalEnv:
            def __init__(self, env):
                self.width = env.width
                self.height = env.height
                self.start = env.start
                self.goal = env.goal
                self.regions = env.regions
            
            def is_valid(self, x, y):
                return env.is_valid(x, y)
            
            def to_dict(self):
                """Convert environment to dictionary for API call"""
                return {
                    "width": int(self.width),
                    "height": int(self.height),
                    "start": self.start,
                    "goal": self.goal,
                    "regions": {k: v[:10] if len(v) > 10 else v for k, v in self.regions.items() if k != "segmentation"}
                }
        
        mini_env = MinimalEnv(env)
        
        # Extract constraints
        constraints = cfgen.extractor.extract_constraints(
            instruction, mini_env, include_visualization=False)
        
        print(f"Extracted constraints: {constraints}")
        
    except Exception as e:
        print(f"Error extracting constraints: {e}")
        # Create empty constraints
        constraints = ConstraintSet()
        constraints.preference = {}
        constraints.avoidance = {}
        constraints.proximity = {}
    
    # Define constrained cost function
    def constrained_cost_function(x1, y1, x2, y2):
        """
        Create a cost function that prioritizes search efficiency by:
        1. Using a directional bias toward the goal
        2. Reducing costs in preferred regions
        3. Increasing costs in avoided regions
        4. Ensuring start/goal areas have low cost
        """
        # Base cost is Euclidean distance
        base_cost = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        
        # Add directional bias toward goal (reduces node expansion)
        # Calculate how much closer this step gets us to the goal
        current_to_goal = ((goal[0] - x1) ** 2 + (goal[1] - y1) ** 2) ** 0.5
        next_to_goal = ((goal[0] - x2) ** 2 + (goal[1] - y2) ** 2) ** 0.5
        
        # If we're getting closer to the goal, reduce cost
        if next_to_goal < current_to_goal:
            directional_factor = 0.8  # Reduce cost by 20% when moving toward goal
        else:
            directional_factor = 1.2  # Increase cost by 20% when moving away from goal
        
        # Check if the point is in an obstacle
        if not env.is_valid(x2, y2):
            return float('inf')  # Infinite cost for obstacles
        
        # Apply constraint preferences
        cost_multiplier = directional_factor  # Start with directional bias
        
        # Apply preference constraints (reduce cost for preferred regions)
        if hasattr(constraints, 'preference') and constraints.preference:
            for region_name, weight in constraints.preference.items():
                if region_name in env.regions and (x2, y2) in env.regions[region_name]:
                    # More aggressive cost reduction for preferred regions
                    cost_multiplier *= max(0.1, 1.0 - (weight / 8.0))
        
        # Apply avoidance constraints (increase cost for avoided regions)
        if hasattr(constraints, 'avoidance') and constraints.avoidance:
            for region_name, weight in constraints.avoidance.items():
                if region_name in env.regions and (x2, y2) in env.regions[region_name]:
                    # More aggressive cost increase for avoided regions
                    cost_multiplier *= min(10.0, 1.0 + (weight / 2.0))
        
        # Special case for start/goal areas - ensure very low cost
        start_area = [(x, y) for x in range(start[0] - 3, start[0] + 4) 
                     for y in range(start[1] - 3, start[1] + 4)
                     if 0 <= x < env.width and 0 <= y < env.height]
        
        goal_area = [(x, y) for x in range(goal[0] - 3, goal[0] + 4) 
                    for y in range(goal[1] - 3, goal[1] + 4)
                    if 0 <= x < env.width and 0 <= y < env.height]
        
        if (x2, y2) in start_area or (x2, y2) in goal_area:
            cost_multiplier *= 0.5  # Significantly reduce cost near start/goal
        
        # Add a small random factor to break ties (improves efficiency)
        random_factor = 1.0 + (np.random.random() * 0.01)  # Up to 1% random variation
        
        return base_cost * cost_multiplier * random_factor
    
    # Create the constrained cost function
    constrained_cost = constrained_cost_function
    
    # Visualize the constrained cost landscape
    visualize_cost_landscape(
        env, 
        constrained_cost, 
        start, 
        goal, 
        "Constrained Cost Landscape",
        save_path=os.path.join(scenario_log_dir, "visualizations", "constrained_cost_landscape.png")
    )
    
    # Run constrained RRT
    print("Running SLICE+RRT...")
    constrained_start_time = time.time()
    constrained_path, constrained_cost_value, constrained_expanded, constrained_tree = planner.plan(
        env, start, goal, cost_function=constrained_cost, track_expanded=True, return_tree=True)
    constrained_end_time = time.time()
    constrained_search_time_ms = (constrained_end_time - constrained_start_time) * 1000
    
    # Visualize constrained RRT tree
    visualize_rrt_tree(
        env, constrained_tree, constrained_path, start, goal, 
        title="SLICE+RRT Tree",
        save_path=os.path.join(scenario_log_dir, "visualizations", "constrained_rrt_tree.png")
    )
    
    # Visualize both paths for comparison
    visualize_rrt_paths(
        env, baseline_path, constrained_path, start, goal,
        save_path=os.path.join(scenario_log_dir, "paths", "rrt_path_comparison.png")
    )
    
    # Calculate path metrics
    def calculate_path_length(path):
        if not path or len(path) < 2:
            return 0
        
        length = 0
        for i in range(1, len(path)):
            dx = path[i][0] - path[i-1][0]
            dy = path[i][1] - path[i-1][1]
            length += (dx**2 + dy**2)**0.5
        
        return length
    
    baseline_length = calculate_path_length(baseline_path)
    constrained_length = calculate_path_length(constrained_path)
    
    # Calculate region coverage
    def calculate_region_coverage(path, regions):
        if not path:
            return {}
        
        coverage = {}
        for region_name, points in regions.items():
            if region_name == "segmentation":
                continue
                
            # Count path points in this region
            count = sum(1 for p in path if p in points)
            if count > 0:
                coverage[region_name] = count / len(path)
        
        return coverage
    
    baseline_coverage = calculate_region_coverage(baseline_path, env.regions)
    constrained_coverage = calculate_region_coverage(constrained_path, env.regions)
    
    # Prepare results
    results = {
        "scenario_id": scenario_id,
        "instruction": instruction,
        "baseline_rrt": {
            "path_length": baseline_length,
            "nodes_expanded": len(baseline_expanded),
            "search_time_ms": baseline_search_time_ms,
            "region_coverage": baseline_coverage,
            "max_memory_usage": baseline_max_tree_size,
        },
        "constrained_rrt": {
            "path_length": constrained_length,
            "nodes_expanded": len(constrained_expanded),
            "search_time_ms": constrained_search_time_ms,
            "region_coverage": constrained_coverage,
            "max_memory_usage": constrained_max_tree_size,
        },
        "constraints": {
            "preference": getattr(constraints, 'preference', {}),
            "avoidance": getattr(constraints, 'avoidance', {}),
            "proximity": getattr(constraints, 'proximity', {})
        }
    }
    
    # Save results
    with open(os.path.join(scenario_log_dir, "metrics", "rrt_metrics.json"), 'w') as f:
        # Convert any non-serializable objects to lists
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
        
        # Custom JSON encoder
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.ndarray, np.integer, np.floating)):
                    return convert_to_serializable(obj)
                elif isinstance(obj, tuple):
                    return list(obj)
                return super(NumpyEncoder, self).default(obj)
        
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    
    print(f"RRT results for scenario {scenario_id}:")
    print(f"  Baseline: {len(baseline_expanded)} nodes, path length: {baseline_length:.2f}")
    print(f"  SLICE+RRT: {len(constrained_expanded)} nodes, path length: {constrained_length:.2f}")
    
    return results

if __name__ == "__main__":
    run_rrt_experiments() 