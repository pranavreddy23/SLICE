import os
import json
import yaml
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from core.env import GridEnvironment
from core.constraintext import ConstraintExtractor
from core.cfgen import CostFunctionGenerator, a_star_search, uniform_cost_function, AdaptiveCostFunctionGenerator, semantic_a_star_search, corridor_guided_search
from utils.viz import visualize_grid, visualize_comparison, visualize_regions, visualize_paths, visualize_node_expansion
from utils.metrics import PlannerMetrics
from utils.map_generator import ChallengeMapGenerator, generate_random_positions
from core.cfgen import identify_corridors
from core.region_segmentation import RegionSegmenter

class ExperimentRunner:
    """Runs experiments on constraint-based planning"""
    
    def __init__(self, config_path=None, config=None):
        """Initialize with optional configuration"""
        if config:
            self.config = config
        else:
            self.config = self._load_config(config_path)
        
        self.extractor = ConstraintExtractor()
        self.cost_generator = CostFunctionGenerator()
        
        # Create log directories
        self.log_dir = self._create_log_dirs()
    
    def _load_config(self, config_path):
        """Load configuration from YAML file"""
        default_config = {
            "experiments": {
                "num_runs": 5,
                "map_sizes": [(30, 30), (50, 50), (80, 80)],
                "map_types": ["maze", "cluttered", "narrow_passages", "office"],
                "instructions": [
                    "Navigate to the goal efficiently while avoiding narrow passages.",
                    "Reach the goal quickly but stay away from obstacles.",
                    "Find a path to the goal that prefers open areas.",
                    "Go to the goal while staying close to the center region."
                ]
            },
            "visualization": {
                "save_maps": True,
                "save_regions": True,
                "save_paths": True,
                "show_plots": False
            },
            "metrics": {
                "record_nodes_expanded": True,
                "record_time": True,
                "record_path_properties": True
            }
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                # Merge configs
                for key, value in user_config.items():
                    if key in default_config and isinstance(value, dict):
                        default_config[key].update(value)
                    else:
                        default_config[key] = value
        
        return default_config
    
    def _create_log_dirs(self):
        """Create log directories for this experiment run"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join("logs", f"experiment_{timestamp}")
        
        # Create main log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Create subdirectories
        subdirs = ["maps", "regions", "paths", "metrics", "visualizations"]
        for subdir in subdirs:
            os.makedirs(os.path.join(log_dir, subdir), exist_ok=True)
        
        return log_dir
    
    def run_single_experiment(self, environment, start, goal, instruction, experiment_id):
        """Run a single experiment with baseline and constrained planners"""
        print(f"\nRunning experiment {experiment_id}")
        print(f"Instruction: {instruction}")
        
        # Initialize metrics
        metrics = {
            "baseline": {},
            "constrained": {},
            "improvement": {},
            "metadata": {}
        }
        
        # Create uniform cost function (baseline)
        uniform_cost = uniform_cost_function(environment)
        
        # Run baseline planner
        print("Running baseline planner...")
        baseline_start_time = time.time()
        baseline_path, baseline_cost, baseline_expanded = a_star_search(
            environment, start, goal, uniform_cost, track_expanded=True)
        baseline_time = time.time() - baseline_start_time
        
        # Record baseline metrics
        metrics["baseline"]["path_length"] = len(baseline_path)
        metrics["baseline"]["path_cost"] = baseline_cost
        metrics["baseline"]["nodes_expanded"] = len(baseline_expanded)
        metrics["baseline"]["time"] = baseline_time
        metrics["baseline"]["search_efficiency"] = len(baseline_path) / len(baseline_expanded) if len(baseline_expanded) > 0 else 0
        
        # Generate constrained cost function
        print("Generating constrained cost function...")
        constrained_cost = self.cost_generator.generate_cost_function(
            environment, instruction, start, goal)
        
        # Extract constraints for compliance calculation
        # This is the key change - we need to extract constraints separately
        constraints = self.cost_generator.extractor.extract_constraints(instruction, environment)
        print("Extracted constraints:", constraints)
        
        # Run constrained planner
        print("Running constrained planner...")
        constrained_start_time = time.time()
        constrained_path, constrained_cost, constrained_expanded = a_star_search(
            environment, start, goal, constrained_cost, track_expanded=True)
        constrained_time = time.time() - constrained_start_time
        
        # Record constrained metrics
        metrics["constrained"]["path_length"] = len(constrained_path)
        metrics["constrained"]["path_cost"] = constrained_cost
        metrics["constrained"]["nodes_expanded"] = len(constrained_expanded)
        metrics["constrained"]["time"] = constrained_time
        metrics["constrained"]["search_efficiency"] = len(constrained_path) / len(constrained_expanded) if len(constrained_expanded) > 0 else 0
        
        # Calculate constraint compliance
        compliance = self._calculate_constraint_compliance(
            environment, constrained_path, constraints)
        metrics["constrained"]["constraint_compliance"] = compliance
        
        # Calculate improvement metrics
        metrics["improvement"]["path_length_ratio"] = len(baseline_path) / len(constrained_path) if len(constrained_path) > 0 else 0
        metrics["improvement"]["nodes_expanded_ratio"] = len(baseline_expanded) / len(constrained_expanded) if len(constrained_expanded) > 0 else 0
        metrics["improvement"]["time_ratio"] = baseline_time / constrained_time if constrained_time > 0 else 0
        metrics["improvement"]["search_efficiency_ratio"] = metrics["constrained"]["search_efficiency"] / metrics["baseline"]["search_efficiency"] if metrics["baseline"]["search_efficiency"] > 0 else 0
        
        # Print summary
        print("\nExperiment results:")
        print(f"Baseline path length: {len(baseline_path)}, nodes expanded: {len(baseline_expanded)}")
        print(f"Constrained path length: {len(constrained_path)}, nodes expanded: {len(constrained_expanded)}")
        print(f"Improvement in nodes expanded: {metrics['improvement']['nodes_expanded_ratio']:.2f}x")
        print(f"Improvement in search efficiency: {metrics['improvement']['search_efficiency_ratio']:.2f}x")
        print(f"Constraint compliance: {compliance:.2f}")
        
        # Visualize paths and expanded nodes
        if self.config.get("visualization", {}).get("save_paths", False):
            os.makedirs(os.path.join(self.log_dir, "visualizations"), exist_ok=True)
            
            # Visualize paths
            visualize_paths(
                environment, 
                baseline_path, 
                constrained_path, 
                start, 
                goal,
                save_path=os.path.join(self.log_dir, "visualizations", f"paths_{experiment_id}.png")
            )
            
            # Visualize node expansion
            visualize_node_expansion(
                environment,
                baseline_expanded,
                constrained_expanded,
                baseline_path,
                constrained_path,
                start,
                goal,
                save_path=os.path.join(self.log_dir, "visualizations", f"expansion_{experiment_id}.png")
            )
        
        return metrics
    
    def run_experiment_suite(self):
        """Run a suite of experiments based on configuration"""
        print("Starting experiment suite...\n")
        
        # Initialize results list
        results = []
        
        # Get experiment parameters
        num_runs = self.config["experiments"].get("num_runs", 1)
        map_sizes = self.config["experiments"].get("map_sizes", [[40, 40]])
        map_types = self.config["experiments"].get("map_types", ["cluttered"])
        instructions = self.config["experiments"].get("instructions", ["Navigate to the goal."])
        randomize_maps = self.config["experiments"].get("randomize_maps", False)
        randomize_positions = self.config["experiments"].get("randomize_positions", False)
        
        # Run experiments
        experiment_id = 1
        
        for map_size in map_sizes:
            for map_type in map_types:
                # Generate environment
                width, height = map_size
                print(f"\nGenerating {map_type} environment of size {width}x{height}...")
                
                if map_type == "cluttered":
                    env = ChallengeMapGenerator.generate_cluttered(width, height)
                elif map_type == "maze":
                    env = ChallengeMapGenerator.generate_maze(width, height)
                elif map_type == "narrow_passages":
                    env = ChallengeMapGenerator.generate_narrow_passages(width, height)
                elif map_type == "office":
                    env = ChallengeMapGenerator.generate_office(width, height)
                else:
                    print(f"Unknown map type: {map_type}")
                    continue
                
                # Segment regions
                region_method = self.config["semantic_approaches"].get("region_segmentation", "watershed")
                segmenter = RegionSegmenter(method=region_method)
                env.regions = segmenter.segment_regions(env)
                
                # Visualize environment if requested
                if self.config["visualization"].get("save_maps", False):
                    os.makedirs(os.path.join(self.log_dir, "maps"), exist_ok=True)
                    plt.figure(figsize=(10, 10))
                    plt.imshow(env.grid, cmap='binary', interpolation='nearest')
                    plt.title(f"{map_type.capitalize()} Environment ({width}x{height})")
                    plt.savefig(os.path.join(self.log_dir, "maps", f"{map_type}_{width}x{height}.png"))
                    plt.close()
                
                # Visualize regions if requested
                if self.config["visualization"].get("save_regions", False):
                    os.makedirs(os.path.join(self.log_dir, "regions"), exist_ok=True)
                    from utils.viz import visualize_regions  # Import here to avoid circular imports
                    visualize_regions(env, save_path=os.path.join(self.log_dir, "regions", f"{map_type}_{width}x{height}_regions.png"))
                
                # Generate start/goal positions
                print("Generating random start/goal positions...")
                start, goal = generate_random_positions(env, min_distance=max(width, height) // 3)
                
                # If we couldn't find valid positions, use default positions
                if start is None or goal is None:
                    print("Using default positions...")
                    start = (1, 1)
                    goal = (width - 2, height - 2)
                
                print(f"Start: {start}, Goal: {goal}")
                
                # Run experiments for each instruction
                for instruction in instructions:
                    for run in range(1, num_runs + 1):
                        # Regenerate environment if requested
                        if randomize_maps and run > 1:
                            if map_type == "cluttered":
                                env = ChallengeMapGenerator.generate_cluttered(width, height)
                            elif map_type == "maze":
                                env = ChallengeMapGenerator.generate_maze(width, height)
                            elif map_type == "narrow_passages":
                                env = ChallengeMapGenerator.generate_narrow_passages(width, height)
                            elif map_type == "office":
                                env = ChallengeMapGenerator.generate_office(width, height)
                            
                            # Segment regions
                            env.regions = segmenter.segment_regions(env)
                        
                        # Regenerate start/goal positions if requested
                        if randomize_positions and run > 1:
                            start, goal = generate_random_positions(env, min_distance=max(width, height) // 3)
                            
                            # If we couldn't find valid positions, use default positions
                            if start is None or goal is None:
                                print("Using default positions...")
                                start = (1, 1)
                                goal = (width - 2, height - 2)
                        
                        # Run experiment
                        result = self.run_single_experiment(env, start, goal, instruction, experiment_id)
                        
                        # Add metadata
                        result["metadata"] = {
                            "experiment_id": experiment_id,
                            "map_type": map_type,
                            "map_size": map_size,
                            "instruction": instruction,
                            "run": run
                        }
                        
                        # Add result to list
                        results.append(result)
                        
                        # Increment experiment ID
                        experiment_id += 1
        
        # Save results
        self._save_results(results)
        
        # Generate summary statistics
        if results:  # Only generate summary if we have results
            self._generate_summary_statistics(results)
        
        return results
    
    def _find_challenging_positions(self, environment):
        """Find challenging start and goal positions"""
        # Get free cells
        free_cells = []
        for y in range(environment.height):
            for x in range(environment.width):
                if environment.grid[y, x] == 0:
                    free_cells.append((x, y))
        
        if len(free_cells) < 2:
            # Fallback if not enough free cells
            return (0, 0), (environment.width-1, environment.height-1)
        
        # Try to find distant points
        max_distance = 0
        start, goal = free_cells[0], free_cells[1]
        
        # Sample a subset of cells for efficiency
        sample_size = min(100, len(free_cells))
        sampled_cells = np.random.choice(len(free_cells), sample_size, replace=False)
        
        for i in sampled_cells:
            for j in sampled_cells:
                if i == j:
                    continue
                
                p1 = free_cells[i]
                p2 = free_cells[j]
                
                # Manhattan distance
                distance = abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
                
                if distance > max_distance:
                    max_distance = distance
                    start, goal = p1, p2
        
        return start, goal
    
    def _generate_summary_statistics(self, results):
        """Generate summary statistics from experiment results"""
        print("\nGenerating summary statistics...")
        
        # Initialize summary statistics
        summary = {
            "baseline": {
                "avg_path_length": 0,
                "avg_nodes_expanded": 0,
                "avg_time": 0,
                "avg_search_efficiency": 0
            },
            "constrained": {
                "avg_path_length": 0,
                "avg_nodes_expanded": 0,
                "avg_time": 0,
                "avg_search_efficiency": 0,
                "avg_constraint_compliance": 0
            },
            "improvement": {
                "avg_path_length_ratio": 0,
                "avg_nodes_expanded_ratio": 0,
                "avg_time_ratio": 0,
                "avg_search_efficiency_ratio": 0
            }
        }
        
        # Calculate averages
        for result in results:
            # Skip results without metrics
            if "baseline" not in result or "constrained" not in result:
                continue
            
            # Baseline metrics
            baseline = result.get("baseline", {})
            summary["baseline"]["avg_path_length"] += baseline.get("path_length", 0)
            summary["baseline"]["avg_nodes_expanded"] += baseline.get("nodes_expanded", 0)
            summary["baseline"]["avg_time"] += baseline.get("time", 0)
            summary["baseline"]["avg_search_efficiency"] += baseline.get("search_efficiency", 0)
            
            # Constrained metrics
            constrained = result.get("constrained", {})
            summary["constrained"]["avg_path_length"] += constrained.get("path_length", 0)
            summary["constrained"]["avg_nodes_expanded"] += constrained.get("nodes_expanded", 0)
            summary["constrained"]["avg_time"] += constrained.get("time", 0)
            summary["constrained"]["avg_search_efficiency"] += constrained.get("search_efficiency", 0)
            summary["constrained"]["avg_constraint_compliance"] += constrained.get("constraint_compliance", 0)
            
            # Improvement metrics
            improvement = result.get("improvement", {})
            summary["improvement"]["avg_path_length_ratio"] += improvement.get("path_length_ratio", 0)
            summary["improvement"]["avg_nodes_expanded_ratio"] += improvement.get("nodes_expanded_ratio", 0)
            summary["improvement"]["avg_time_ratio"] += improvement.get("time_ratio", 0)
            summary["improvement"]["avg_search_efficiency_ratio"] += improvement.get("search_efficiency_ratio", 0)
        
        # Calculate averages
        num_results = len(results)
        if num_results > 0:
            for category in summary:
                for metric in summary[category]:
                    summary[category][metric] /= num_results
        
        # Print summary
        print("\nSummary Statistics:")
        print(f"Number of experiments: {num_results}")
        print("\nBaseline Planner:")
        print(f"  Average path length: {summary['baseline']['avg_path_length']:.2f}")
        print(f"  Average nodes expanded: {summary['baseline']['avg_nodes_expanded']:.2f}")
        print(f"  Average time: {summary['baseline']['avg_time']:.4f} seconds")
        print(f"  Average search efficiency: {summary['baseline']['avg_search_efficiency']:.4f}")
        
        print("\nConstrained Planner:")
        print(f"  Average path length: {summary['constrained']['avg_path_length']:.2f}")
        print(f"  Average nodes expanded: {summary['constrained']['avg_nodes_expanded']:.2f}")
        print(f"  Average time: {summary['constrained']['avg_time']:.4f} seconds")
        print(f"  Average search efficiency: {summary['constrained']['avg_search_efficiency']:.4f}")
        print(f"  Average constraint compliance: {summary['constrained']['avg_constraint_compliance']:.2f}")
        
        print("\nImprovement:")
        print(f"  Average path length ratio: {summary['improvement']['avg_path_length_ratio']:.2f}x")
        print(f"  Average nodes expanded ratio: {summary['improvement']['avg_nodes_expanded_ratio']:.2f}x")
        print(f"  Average time ratio: {summary['improvement']['avg_time_ratio']:.2f}x")
        print(f"  Average search efficiency ratio: {summary['improvement']['avg_search_efficiency_ratio']:.2f}x")
        
        # Save summary to file
        summary_path = os.path.join(self.log_dir, "summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nSummary statistics saved to {summary_path}")

    def _calculate_constraint_compliance(self, environment, path, constraints):
        """
        Calculate how well a path complies with the given constraints
        
        Args:
            environment: The grid environment
            path: List of (x, y) tuples representing the path
            constraints: ConstraintSet object with extracted constraints
        
        Returns:
            compliance: Float between 0 and 1 indicating compliance
        """
        if not path:
            return 0.0
        
        # If constraints is a string (error case), return 0
        if isinstance(constraints, str):
            print("Warning: constraints is a string, not a ConstraintSet object")
            return 0.0
        
        # Initialize compliance metrics
        total_points = len(path)
        compliant_points = 0
        
        # Check each point in the path
        for x, y in path:
            point_compliant = True
            
            # Check avoidance constraints
            if hasattr(constraints, 'avoidance'):
                for region_name in constraints.avoidance:
                    if region_name in environment.regions:
                        if (x, y) in environment.regions[region_name]:
                            point_compliant = False
                            break
            
            # Check preference constraints
            if hasattr(constraints, 'preference') and constraints.preference:
                in_preferred_region = False
                for region_name in constraints.preference:
                    if region_name in environment.regions:
                        if (x, y) in environment.regions[region_name]:
                            in_preferred_region = True
                            break
                
                if not in_preferred_region:
                    point_compliant = False
            
            # Check proximity constraints
            if hasattr(constraints, 'proximity'):
                for region_name in constraints.proximity:
                    if region_name in environment.regions:
                        points = environment.regions[region_name]
                        if points:
                            min_distance = min(abs(x - px) + abs(y - py) for px, py in points)
                            if min_distance > 5:  # Arbitrary threshold
                                point_compliant = False
                                break
            
            if point_compliant:
                compliant_points += 1
        
        return compliant_points / total_points

    def _save_results(self, results):
        """Save experiment results to a file"""
        results_path = os.path.join(self.log_dir, "results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {results_path}")
