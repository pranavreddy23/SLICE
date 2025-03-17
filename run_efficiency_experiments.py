#!/usr/bin/env python3
import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import copy
from experiments.run_experiments import ExperimentRunner
from core.region_segmentation import RegionSegmenter
from core.cfgen import a_star_search, uniform_cost_function, CostFunctionGenerator
from utils.map_generator import ChallengeMapGenerator, generate_random_positions
from utils.viz import visualize_node_expansion, visualize_paths, visualize_regions, visualize_grid, visualize_cost_landscape
from PIL import Image
import io
import base64

def run_efficiency_experiments():
    """Run experiments focused on search efficiency"""
    # Create timestamp for log directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("logs", f"experiment_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    
    # Create subdirectories
    os.makedirs(os.path.join(log_dir, "maps"), exist_ok=True)
    os.makedirs(os.path.join(log_dir, "regions"), exist_ok=True)
    os.makedirs(os.path.join(log_dir, "paths"), exist_ok=True)
    os.makedirs(os.path.join(log_dir, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(log_dir, "visualizations"), exist_ok=True)
    
    # Create a single environment and run the experiment
    print("\nCreating environment and running experiment...")
    
    # Generate environment - increase size for more challenging scenarios
    width, height = 80, 80
    print(f"Generating cluttered environment of size {width}x{height}...")
    env = ChallengeMapGenerator.generate_cluttered(width, height, obstacle_density=0.3)
    
    # Visualize the raw map first
    plt.figure(figsize=(10, 10))
    plt.imshow(env.grid, cmap='binary', interpolation='nearest')
    plt.title(f"Cluttered Environment ({width}x{height})")
    plt.savefig(os.path.join(log_dir, "maps", "environment.png"))
    plt.show()
    
    # Generate random start/goal positions with greater distance
    print("Generating random start/goal positions...")
    start, goal = generate_random_positions(env, min_distance=max(width, height) // 2)
    print(f"Start: {start}, Goal: {goal}")
    
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
    plt.savefig(os.path.join(log_dir, "maps", "environment_with_positions.png"))
    plt.show()
    
    # Segment regions
    print("Segmenting regions...")
    segmenter = RegionSegmenter(method="watershed")
    env.regions = segmenter.segment_regions(env)
    
    # Visualize regions
    print("Visualizing regions...")
    visualize_regions(env, save_path=os.path.join(log_dir, "regions", "regions.png"))
    
    # Create uniform cost function for baseline
    print("Creating uniform cost function for baseline...")
    uniform_cost = uniform_cost_function(env)
    
    # After segmenting regions, create a copy of the environment for the constrained version
    constrained_env = copy.deepcopy(env)

    # Run baseline planner with standard A* search on original environment
    print("Running baseline planner on original environment...")
    baseline_path, baseline_cost, baseline_expanded = a_star_search(
        env, start, goal, uniform_cost, track_expanded=True)
    
    # Choose an instruction focused on efficiency
    instruction = "Navigate to the goal using the shortest and most direct path while avoiding obstacles."
    print(f"Using instruction: {instruction}")
    
    # Generate constraints with better error handling for API limits
    print("Generating constraints...")
    cfgen = CostFunctionGenerator()
    try:
        # Create a minimal environment representation for the API call
        class MinimalEnv:
            def __init__(self, env):
                self.width = env.width
                self.height = env.height
                self.start = env.start
                self.goal = env.goal
                
                # Create a simplified grid (downsampled)
                scale_factor = 4  # Reduce resolution by 4x
                self.grid = np.zeros((env.height // scale_factor, env.width // scale_factor))
                for y in range(0, env.height, scale_factor):
                    for x in range(0, env.width, scale_factor):
                        # If any cell in this block is an obstacle, mark the downsampled cell as obstacle
                        block = env.grid[y:min(y+scale_factor, env.height), x:min(x+scale_factor, env.width)]
                        if np.any(block == 1):
                            self.grid[y // scale_factor, x // scale_factor] = 1
                
                # Create simplified regions (max 10 points per region)
                self.regions = {}
                for region_name, points in env.regions.items():
                    if len(points) > 0:
                        # Take every Nth point to get at most 10 points
                        n = max(1, len(points) // 10)
                        self.regions[region_name] = points[::n][:10]
            
            def to_dict(self):
                """Convert environment to dictionary for API call"""
                return {
                    "width": self.width,
                    "height": self.height,
                    "start": self.start,
                    "goal": self.goal,
                    "grid_sample": self.grid.tolist()[:10][:10],  # Just a small sample
                    "regions": {k: v[:5] for k, v in self.regions.items()}  # Limited regions
                }
        
        # Create minimal environment
        mini_env = MinimalEnv(env)
        
        # After segmenting regions, create a visual representation for LLM
        print("Creating visual representation for LLM...")
        env_image = create_environment_image(env, start, goal)
        
        # Save the image for reference
        with open(os.path.join(log_dir, "visualizations", "environment_for_llm.png"), "wb") as f:
            f.write(base64.b64decode(env_image))
        
        # When extracting constraints, include the image
        try:
            # Modify the extractor to use the image
            # This assumes the extractor has a method to handle images or can be modified
            print("Extracting constraints with visual context...")
            
            # First API call - identify and label regions with visual context
            region_instruction = (
                "Look at the image showing the environment. The green dot is the start position, "
                "and the red dot is the goal position. Colored areas represent different regions. "
                "Identify and label these regions as open areas, narrow passages, or cluttered spaces."
            )
            
            # Check if the extractor has an image-based method
            if hasattr(cfgen.extractor, 'extract_constraints_with_image'):
                region_constraints = cfgen.extractor.extract_constraints_with_image(
                    region_instruction, mini_env, env_image, include_visualization=False)
            else:
                # Fall back to regular method but include image description
                region_constraints = cfgen.extractor.extract_constraints(
                    region_instruction + " [Image of environment with start, goal, and regions provided]", 
                    mini_env, include_visualization=False)
            
            # Second API call - extract navigation constraints with visual context
            navigation_instruction = (
                f"{instruction} Look at the image showing the environment. The green dot is the start position, "
                "and the red dot is the goal position. Colored areas represent different regions. "
                "Ensure the path from start to goal remains clear and accessible."
            )
            
            if hasattr(cfgen.extractor, 'extract_constraints_with_image'):
                constraints = cfgen.extractor.extract_constraints_with_image(
                    navigation_instruction, mini_env, env_image, include_visualization=False)
            else:
                constraints = cfgen.extractor.extract_constraints(
                    navigation_instruction + " [Image of environment with start, goal, and regions provided]", 
                    mini_env, include_visualization=False)
            
            print(f"Extracted constraints: {constraints}")
            
        except Exception as e:
            print(f"Error extracting constraints: {e}")
            # Create default constraints that prioritize efficiency
            from core.constraintext import ConstraintSet
            constraints = ConstraintSet()
            
            # Create meaningful default regions if they don't exist
            if not env.regions or len(env.regions) <= 1:
                print("Creating default regions...")
                # Simple region segmentation based on obstacle density
                open_regions = []
                narrow_regions = []
                cluttered_regions = []
                
                # Sample points and classify them
                for y in range(0, env.height, 4):
                    for x in range(0, env.width, 4):
                        if env.grid[y, x] == 1:  # Skip obstacles
                            continue
                            
                        # Count obstacles in neighborhood
                        obstacle_count = 0
                        for dy in range(-3, 4):
                            for dx in range(-3, 4):
                                ny, nx = y + dy, x + dx
                                if 0 <= ny < env.height and 0 <= nx < env.width and env.grid[ny, nx] == 1:
                                    obstacle_count += 1
                        
                        # Classify based on obstacle density
                        if obstacle_count <= 2:
                            open_regions.append((x, y))
                        elif obstacle_count <= 6:
                            narrow_regions.append((x, y))
                        else:
                            cluttered_regions.append((x, y))
                
                # Add regions to environment
                if open_regions:
                    env.regions["open"] = open_regions
                    constraints.preference["open"] = 8
                if narrow_regions:
                    env.regions["narrow"] = narrow_regions
                    constraints.preference["narrow"] = 3
                if cluttered_regions:
                    env.regions["cluttered"] = cluttered_regions
                    constraints.avoidance["cluttered"] = 7
            else:
                # Add default constraints based on existing regions
                if "open" in env.regions:
                    constraints.preference["open"] = 8
                if "narrow" in env.regions:
                    constraints.preference["narrow"] = 3
                if "cluttered" in env.regions:
                    constraints.avoidance["cluttered"] = 7
            
            print(f"Using default constraints: {constraints}")
        
        # After extracting constraints, ensure start/goal areas remain clear
        print("Ensuring start/goal areas remain accessible...")

        # Define a safety radius around start and goal
        safety_radius = 3

        # First, ensure start and goal positions are clear
        constrained_env.grid[start[1], start[0]] = 0  # Clear start position
        constrained_env.grid[goal[1], goal[0]] = 0    # Clear goal position

        # Clear a small area around start and goal
        for dx in range(-safety_radius, safety_radius + 1):
            for dy in range(-safety_radius, safety_radius + 1):
                sx, sy = start[0] + dx, start[1] + dy
                gx, gy = goal[0] + dx, goal[1] + dy
                
                if 0 <= sx < constrained_env.width and 0 <= sy < constrained_env.height:
                    constrained_env.grid[sy, sx] = 0  # Clear around start
                
                if 0 <= gx < constrained_env.width and 0 <= gy < constrained_env.height:
                    constrained_env.grid[gy, gx] = 0  # Clear around goal

        # Create a modified version of the constraints that preserves start/goal accessibility
        def ensure_start_goal_accessible(constraints, env, start, goal):
            """Modify constraints to ensure start and goal areas remain accessible"""
            # Create a copy of the constraints
            from copy import deepcopy
            modified_constraints = deepcopy(constraints)
            
            # Remove any avoidance constraints that might affect start/goal areas
            if hasattr(modified_constraints, 'avoidance') and modified_constraints.avoidance:
                for region_name, weight in list(modified_constraints.avoidance.items()):
                    if region_name in env.regions:
                        # Check if start or goal is in this region
                        start_in_region = any(abs(x - start[0]) <= safety_radius and abs(y - start[1]) <= safety_radius 
                                             for x, y in env.regions[region_name])
                        goal_in_region = any(abs(x - goal[0]) <= safety_radius and abs(y - goal[1]) <= safety_radius 
                                            for x, y in env.regions[region_name])
                        
                        if start_in_region or goal_in_region:
                            print(f"Removing avoidance constraint for region {region_name} to preserve start/goal access")
                            del modified_constraints.avoidance[region_name]
            
            # Add preference constraints for start/goal areas if not already present
            if not hasattr(modified_constraints, 'preference'):
                modified_constraints.preference = {}
            
            # Add start/goal areas as preferred regions
            modified_constraints.preference["start_area"] = 10  # Maximum preference
            modified_constraints.preference["goal_area"] = 10   # Maximum preference
            
            # Create start/goal area regions if they don't exist
            if "start_area" not in env.regions:
                env.regions["start_area"] = [(x, y) for x in range(start[0] - safety_radius, start[0] + safety_radius + 1)
                                            for y in range(start[1] - safety_radius, start[1] + safety_radius + 1)
                                            if 0 <= x < env.width and 0 <= y < env.height]
            
            if "goal_area" not in env.regions:
                env.regions["goal_area"] = [(x, y) for x in range(goal[0] - safety_radius, goal[0] + safety_radius + 1)
                                           for y in range(goal[1] - safety_radius, goal[1] + safety_radius + 1)
                                           if 0 <= x < env.width and 0 <= y < env.height]
            
            return modified_constraints

        # Apply the modification to ensure start/goal accessibility
        constraints = ensure_start_goal_accessible(constraints, env, start, goal)
        print(f"Modified constraints to ensure accessibility: {constraints}")

        # When modifying the environment based on constraints, ensure we don't override original obstacles
        print("Creating proper overlay of constraints on original map...")

        # Reset the constrained environment to match the original
        constrained_env.grid = env.grid.copy()

        # First, ensure start and goal positions are clear (these are the only exceptions)
        constrained_env.grid[start[1], start[0]] = 0  # Clear start position
        constrained_env.grid[goal[1], goal[0]] = 0    # Clear goal position

        # Clear a small area around start and goal
        for dx in range(-safety_radius, safety_radius + 1):
            for dy in range(-safety_radius, safety_radius + 1):
                sx, sy = start[0] + dx, start[1] + dy
                gx, gy = goal[0] + dx, goal[1] + dy
                
                if 0 <= sx < constrained_env.width and 0 <= sy < constrained_env.height:
                    constrained_env.grid[sy, sx] = 0  # Clear around start
                
                if 0 <= gx < constrained_env.width and 0 <= gy < constrained_env.height:
                    constrained_env.grid[gy, gx] = 0  # Clear around goal

        # Instead of modifying the grid directly, create a cost function that respects both
        # the original obstacles and the constraints
        def constrained_cost_function(x1, y1, x2, y2):
            # Base cost is just the Manhattan distance
            base_cost = abs(x2 - x1) + abs(y2 - y1)
            
            # Check if the point is in an obstacle in the ORIGINAL map
            if 0 <= y2 < env.height and 0 <= x2 < env.width and env.grid[y2, x2] == 1:
                return float('inf')  # Infinite cost for original obstacles
            
            # Apply constraint preferences
            cost_multiplier = 1.0
            
            # Apply preference constraints (reduce cost for preferred regions)
            if hasattr(constraints, 'preference') and constraints.preference:
                for region_name, weight in constraints.preference.items():
                    if region_name in env.regions and (x2, y2) in env.regions[region_name]:
                        cost_multiplier *= max(0.2, 1.0 - (weight / 10.0))  # Reduce cost based on preference weight
            
            # Apply avoidance constraints (increase cost for avoided regions)
            if hasattr(constraints, 'avoidance') and constraints.avoidance:
                for region_name, weight in constraints.avoidance.items():
                    if region_name in env.regions and (x2, y2) in env.regions[region_name]:
                        cost_multiplier *= min(5.0, 1.0 + (weight / 5.0))  # Increase cost based on avoidance weight
            
            return base_cost * cost_multiplier

        # Visualize the baseline cost landscape
        visualize_cost_landscape(
            env, 
            uniform_cost, 
            start, 
            goal, 
            "Baseline Cost Landscape",
            save_path=os.path.join(log_dir, "visualizations", "baseline_cost_landscape.png")
        )

        # Visualize the constrained cost landscape
        visualize_cost_landscape(
            constrained_env, 
            constrained_cost_function, 
            start, 
            goal, 
            "Constrained Cost Landscape",
            save_path=os.path.join(log_dir, "visualizations", "constrained_cost_landscape.png")
        )

        # Run the constrained planner with the constrained cost function
        print("Running constrained planner with constrained cost function...")
        constrained_path, constrained_cost, constrained_expanded = a_star_search(
            env, start, goal, constrained_cost_function, track_expanded=True)
        
        # Calculate path smoothness
        def calculate_path_smoothness(path):
            if not path or len(path) < 3:
                return 1.0  # Perfect smoothness for short paths
            
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
            avg_angle = sum(angles) / len(angles) if angles else 0
            
            # Normalize to [0, 1] where 1 is perfectly smooth
            smoothness = 1 - (avg_angle / np.pi)
            return smoothness
        
        baseline_smoothness = calculate_path_smoothness(baseline_path)
        constrained_smoothness = calculate_path_smoothness(constrained_path)
        
        # Calculate constraint compliance
        def calculate_constraint_compliance(path, constraints):
            if not path or not constraints:
                return {
                    "proximity": 0.0,
                    "avoidance": 0.0,
                    "preference": 0.0,
                    "overall": 0.0
                }
            
            # Initialize compliance metrics
            proximity_compliance = 0.0
            avoidance_compliance = 0.0
            preference_compliance = 0.0
            
            # Count compliant points
            total_points = len(path)
            proximity_points = 0
            avoidance_points = 0
            preference_points = 0
            
            # Check each point in the path
            for x, y in path:
                # Check proximity constraints
                if hasattr(constraints, 'proximity') and constraints.proximity:
                    is_compliant = True
                    for region_name, distance in constraints.proximity.items():
                        if region_name in env.regions:
                            region_points = env.regions[region_name]
                            if region_points:
                                min_dist = min(abs(x - rx) + abs(y - ry) for rx, ry in region_points)
                                if min_dist > distance:
                                    is_compliant = False
                                    break
                    if is_compliant:
                        proximity_points += 1
                else:
                    proximity_points = total_points  # No constraints means full compliance
                
                # Check avoidance constraints
                if hasattr(constraints, 'avoidance') and constraints.avoidance:
                    is_compliant = True
                    for region_name, _ in constraints.avoidance.items():
                        if region_name in env.regions:
                            if (x, y) in env.regions[region_name]:
                                is_compliant = False
                                break
                    if is_compliant:
                        avoidance_points += 1
                else:
                    avoidance_points = total_points
                
                # Check preference constraints
                if hasattr(constraints, 'preference') and constraints.preference:
                    is_compliant = False
                    for region_name, _ in constraints.preference.items():
                        if region_name in env.regions:
                            if (x, y) in env.regions[region_name]:
                                is_compliant = True
                                break
                    if is_compliant:
                        preference_points += 1
                else:
                    preference_points = total_points
            
            # Calculate compliance ratios
            proximity_compliance = proximity_points / total_points if total_points > 0 else 0.0
            avoidance_compliance = avoidance_points / total_points if total_points > 0 else 0.0
            preference_compliance = preference_points / total_points if total_points > 0 else 0.0
            
            # Calculate overall compliance (average of all types)
            overall_compliance = (proximity_compliance + avoidance_compliance + preference_compliance) / 3
            
            return {
                "proximity": proximity_compliance,
                "avoidance": avoidance_compliance,
                "preference": preference_compliance,
                "overall": overall_compliance
            }
        
        baseline_compliance = calculate_constraint_compliance(baseline_path, constraints)
        constrained_compliance = calculate_constraint_compliance(constrained_path, constraints)
        
        # Visualize paths side by side
        print("Visualizing paths side by side...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Baseline path
        ax1.imshow(env.grid, cmap='binary', interpolation='nearest')
        if baseline_path:
            path_x, path_y = zip(*baseline_path)
            ax1.plot(path_x, path_y, 'b-', linewidth=2)
        ax1.plot(start[0], start[1], 'go', markersize=10)
        ax1.plot(goal[0], goal[1], 'ro', markersize=10)
        ax1.set_title(f"Baseline Path\nNodes Expanded: {len(baseline_expanded)}")
        
        # Constrained path
        ax2.imshow(env.grid, cmap='binary', interpolation='nearest')
        if constrained_path:
            path_x, path_y = zip(*constrained_path)
            ax2.plot(path_x, path_y, 'g-', linewidth=2)
        ax2.plot(start[0], start[1], 'go', markersize=10)
        ax2.plot(goal[0], goal[1], 'ro', markersize=10)
        ax2.set_title(f"Constrained Path\nNodes Expanded: {len(constrained_expanded)}")
        
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, "paths", "paths_side_by_side.png"))
        plt.show()
        
        # Visualize node expansion
        print("Visualizing node expansion...")
        visualize_node_expansion(
            env,
            baseline_expanded,
            constrained_expanded,
            baseline_path,
            constrained_path,
            start,
            goal,
            save_path=os.path.join(log_dir, "visualizations", "node_expansion.png")
        )
        plt.show()
        
        # Print statistics
        print("\nBaseline planner:")
        print(f"  Path length: {len(baseline_path)}")
        print(f"  Path smoothness: {baseline_smoothness:.4f}")
        print(f"  Nodes expanded: {len(baseline_expanded)}")
        print(f"  Search efficiency: {len(baseline_path) / len(baseline_expanded) if len(baseline_expanded) > 0 else 0:.4f}")
        print(f"  Constraint compliance: {baseline_compliance['overall']:.4f}")
        
        print("\nConstrained planner:")
        print(f"  Path length: {len(constrained_path)}")
        print(f"  Path smoothness: {constrained_smoothness:.4f}")
        print(f"  Nodes expanded: {len(constrained_expanded)}")
        print(f"  Search efficiency: {len(constrained_path) / len(constrained_expanded) if len(constrained_expanded) > 0 else 0:.4f}")
        print(f"  Constraint compliance: {constrained_compliance['overall']:.4f}")
        
        # Calculate improvement metrics
        path_length_ratio = len(baseline_path) / len(constrained_path) if len(constrained_path) > 0 else 0
        nodes_expanded_ratio = len(baseline_expanded) / len(constrained_expanded) if len(constrained_expanded) > 0 else 0
        search_efficiency_ratio = ((len(constrained_path) / len(constrained_expanded)) / 
                                  (len(baseline_path) / len(baseline_expanded))) if len(baseline_expanded) > 0 and len(constrained_expanded) > 0 else 0
        compliance_improvement = constrained_compliance['overall'] - baseline_compliance['overall']
        
        print("\nImprovement:")
        print(f"  Path length ratio: {path_length_ratio:.2f}x")
        print(f"  Nodes expanded ratio: {nodes_expanded_ratio:.2f}x")
        print(f"  Search efficiency ratio: {search_efficiency_ratio:.2f}x")
        print(f"  Compliance improvement: {compliance_improvement:.4f}")
        
        # Save metrics
        metrics = {
            "constrained": {
                "success": True,
                "time": 0.0,
                "path_length": len(constrained_path),
                "path_smoothness": constrained_smoothness,
                "nodes_expanded": len(constrained_expanded),
                "search_efficiency": len(constrained_path) / len(constrained_expanded) if len(constrained_expanded) > 0 else 0,
                "compliance": constrained_compliance
            },
            "baseline": {
                "success": True,
                "time": 0.0,
                "path_length": len(baseline_path),
                "path_smoothness": baseline_smoothness,
                "nodes_expanded": len(baseline_expanded),
                "search_efficiency": len(baseline_path) / len(baseline_expanded) if len(baseline_expanded) > 0 else 0,
                "compliance": baseline_compliance
            },
            "improvement": {
                "time_ratio": 0.0,
                "path_length_ratio": path_length_ratio,
                "nodes_expanded_ratio": nodes_expanded_ratio,
                "search_efficiency_ratio": search_efficiency_ratio,
                "compliance_improvement": compliance_improvement
            }
        }
        
        with open(os.path.join(log_dir, "metrics", "metrics.json"), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\nExperiment completed. Results saved to {log_dir}")

    except Exception as e:
        print(f"Error running experiment: {e}")

def create_environment_image(env, start, goal):
    """Create a visual representation of the environment with start, goal, and regions"""
    plt.figure(figsize=(8, 8))
    
    # Plot the grid
    plt.imshow(env.grid, cmap='binary', interpolation='nearest')
    
    # Plot start and goal
    plt.plot(start[0], start[1], 'go', markersize=10, label='Start')
    plt.plot(goal[0], goal[1], 'ro', markersize=10, label='Goal')
    
    # Plot regions with different colors and transparency
    colors = ['blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow']
    for i, (region_name, points) in enumerate(env.regions.items()):
        if points:
            color = colors[i % len(colors)]
            x, y = zip(*points)
            plt.scatter(x, y, c=color, alpha=0.3, s=5, label=f'Region: {region_name}')
    
    plt.title("Environment with Start, Goal, and Regions")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
    
    # Save to bytes
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close()
    buf.seek(0)
    
    # Convert to base64 for API
    img = Image.open(buf)
    # Resize to reduce size
    img = img.resize((400, 400), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    
    return base64.b64encode(buf.read()).decode('utf-8')

if __name__ == "__main__":
    run_efficiency_experiments()