#!/usr/bin/env python3
import argparse
import os
from experiments.run_experiments import ExperimentRunner

def main():
    parser = argparse.ArgumentParser(description='Run constraint-based planning experiments')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--single', action='store_true', help='Run a single experiment instead of the full suite')
    parser.add_argument('--map-type', type=str, choices=['maze', 'cluttered', 'narrow_passages', 'office'],
                       help='Map type for single experiment')
    parser.add_argument('--size', type=int, nargs=2, help='Map size (width height) for single experiment')
    parser.add_argument('--instruction', type=str, help='Instruction for single experiment')
    
    args = parser.parse_args()
    
    # Create experiment runner
    runner = ExperimentRunner(config_path=args.config)
    
    if args.single:
        # Run a single experiment
        from utils.map_generator import ChallengeMapGenerator
        
        # Default values
        map_type = args.map_type or 'maze'
        width, height = args.size or (30, 30)
        instruction = args.instruction or "Navigate to the goal efficiently while avoiding narrow passages."
        
        # Generate environment
        if map_type == "maze":
            env = ChallengeMapGenerator.generate_maze(width, height)
        elif map_type == "cluttered":
            env = ChallengeMapGenerator.generate_cluttered_environment(width, height)
        elif map_type == "narrow_passages":
            env = ChallengeMapGenerator.generate_narrow_passages(width, height)
        elif map_type == "office":
            env = ChallengeMapGenerator.generate_office_like(width, height)
        
        # Find start and goal
        start, goal = runner._find_challenging_positions(env)
        
        # Run experiment
        runner.run_single_experiment(env, start, goal, instruction, "single")
    else:
        # Run full experiment suite
        runner.run_experiment_suite()

if __name__ == "__main__":
    main()