#!/usr/bin/env python3
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from experiments.run_experiments import ExperimentRunner
from core.env import GridEnvironment
from core.cfgen import a_star_search, uniform_cost_function
from utils.map_generator import ChallengeMapGenerator, generate_random_positions
from core.region_segmentation import RegionSegmenter
from experiments.visualization import visualize_node_expansion

def analyze_node_expansion():
    """Analyze node expansion patterns for different planners"""
    print("Analyzing node expansion patterns...")
    
    # Create output directory
    output_dir = "analysis/node_expansion"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate different map types
    map_types = ["maze", "cluttered", "narrow_passages", "office"]
    map_size = [40, 40]
    width, height = map_size
    
    # Instructions to test
    instructions = [
        "Navigate to the goal efficiently while avoiding narrow passages.",
        "Find a path to the goal that prefers open areas.",
        "Go to the goal while staying close to the center region.",
        "Navigate to the goal while keeping a safe distance from all walls.",
        "Find the shortest path to the goal that avoids going through tight spaces.",
        "Reach the goal by following the borders of the environment.",
        "Navigate to the goal by staying in wide corridors whenever possible.",
        "Find a path that avoids the center of the environment."
    ]
    
    # Initialize cost function generator
    from core.cfgen import CostFunctionGenerator
    cost_generator = CostFunctionGenerator()
    
    # Run analysis for each map type
    for map_type in map_types:
        print(f"\nAnalyzing {map_type} environment...")
        
        # Generate environment
        if map_type == "maze":
            env = ChallengeMapGenerator.generate_maze(width, height)
        elif map_type == "cluttered":
            env = ChallengeMapGenerator.generate_cluttered(width, height)
        elif map_type == "narrow_passages":
            env = ChallengeMapGenerator.generate_narrow_passages(width, height)
        elif map_type == "office":
            env = ChallengeMapGenerator.generate_office(width, height)
        
        # Enhance region detection
        segmenter = RegionSegmenter(method="watershed")
        env.regions = segmenter.segment_regions(env)
        
        # Generate random start/goal positions
        start, goal = generate_random_positions(env, min_distance=max(width, height) // 3)
        
        # If we couldn't find valid positions, use default positions
        if start is None or goal is None:
            print("Using default positions...")
            start = (1, 1)
            goal = (width - 2, height - 2)
        
        print(f"Start: {start}, Goal: {goal}")
        
        # Create uniform cost function
        uniform_cost = uniform_cost_function(env)
        
        # Run baseline planner
        print("Running baseline planner...")
        baseline_path, baseline_cost, baseline_expanded = a_star_search(
            env, start, goal, uniform_cost, track_expanded=True)
        
        # Run analysis for each instruction
        for i, instruction in enumerate(instructions):
            print(f"\nInstruction: {instruction}")
            
            # Generate constrained cost function
            constrained_cost = cost_generator.generate_cost_function(
                env, instruction, start, goal)
            
            # Run constrained planner
            print("Running constrained planner...")
            constrained_path, constrained_cost, constrained_expanded = a_star_search(
                env, start, goal, constrained_cost, track_expanded=True)
            
            # Calculate metrics
            baseline_success = len(baseline_path) > 0
            constrained_success = len(constrained_path) > 0
            
            baseline_efficiency = len(baseline_path) / len(baseline_expanded) if baseline_success and len(baseline_expanded) > 0 else 0
            constrained_efficiency = len(constrained_path) / len(constrained_expanded) if constrained_success and len(constrained_expanded) > 0 else 0
            
            print(f"Baseline: {len(baseline_expanded)} nodes, efficiency: {baseline_efficiency:.6f}")
            print(f"Constrained: {len(constrained_expanded)} nodes, efficiency: {constrained_efficiency:.6f}")
            
            # Visualize node expansion
            visualize_node_expansion(
                env,
                baseline_expanded,
                constrained_expanded,
                baseline_path,
                constrained_path,
                start,
                goal,
                save_path=os.path.join(output_dir, f"{map_type}_instruction{i+1}_expansion.png")
            )
            
            # Save metrics
            metrics = {
                "map_type": map_type,
                "instruction": instruction,
                "baseline": {
                    "nodes_expanded": len(baseline_expanded),
                    "path_length": len(baseline_path),
                    "efficiency": baseline_efficiency
                },
                "constrained": {
                    "nodes_expanded": len(constrained_expanded),
                    "path_length": len(constrained_path),
                    "efficiency": constrained_efficiency
                },
                "comparison": {
                    "node_ratio": len(baseline_expanded) / len(constrained_expanded) if len(constrained_expanded) > 0 else 0,
                    "efficiency_ratio": constrained_efficiency / baseline_efficiency if baseline_efficiency > 0 else 0
                }
            }
            
            with open(os.path.join(output_dir, f"{map_type}_instruction{i+1}_metrics.json"), 'w') as f:
                json.dump(metrics, f, indent=2)
    
    print("\nAnalysis complete. Results saved to", output_dir)

if __name__ == "__main__":
    analyze_node_expansion() 