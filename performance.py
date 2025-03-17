import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
import seaborn as sns

def calculate_performance_metrics(log_dir):
    """
    Calculate performance metrics from stored experiment data
    
    Args:
        log_dir: Directory containing experiment logs (e.g., ds_logs/experiment_TIMESTAMP)
    
    Returns:
        DataFrame with calculated metrics for all scenarios
    """
    # Find all scenario data files
    scenario_files = []
    
    # Each experiment has scenario_0 through scenario_9 folders
    for i in range(10):  # Assuming 10 scenarios (0-9)
        scenario_dir = os.path.join(log_dir, f"scenario_{i}")
        data_file = os.path.join(scenario_dir, "data", "scenario_data.json")
        
        if os.path.exists(data_file):
            scenario_files.append(data_file)
    
    print(f"Found {len(scenario_files)} scenario data files")
    
    # Load and process each scenario
    metrics = []
    
    for file_path in scenario_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Extract scenario info
            scenario_id = data.get("scenario_id", "unknown")
            instruction = data.get("instruction", "")
            
            # Check if this is LLM A* data (might be missing some fields)
            is_llm_astar = "waypoints" in data and "constrained" in data
            
            # Calculate core metrics
            
            # 1. Search Efficiency Metrics
            baseline_nodes = data["baseline"]["nodes_expanded"]
            constrained_nodes = data["constrained"]["nodes_expanded"]
            nodes_reduction = ((baseline_nodes - constrained_nodes) / baseline_nodes * 100) if baseline_nodes > 0 else 0
            
            # For LLM A* data, search_time_ms might be named differently
            if "search_time_ms" in data["baseline"]:
                baseline_time = data["baseline"]["search_time_ms"]
            else:
                baseline_time = 0
                
            if "search_time_ms" in data["constrained"]:
                constrained_time = data["constrained"]["search_time_ms"]
            else:
                constrained_time = 0
                
            time_reduction = ((baseline_time - constrained_time) / baseline_time * 100) if baseline_time > 0 else 0
            
            # 2. Path Quality Metrics
            baseline_path_length = data["baseline"]["path_length"]
            constrained_path_length = data["constrained"]["path_length"]
            path_length_ratio = constrained_path_length / baseline_path_length if baseline_path_length > 0 else 0
            
            # 3. Instruction Compliance Metrics
            # Get the path
            constrained_path = data["constrained"]["path"]
            baseline_path = data["baseline"]["path"]
            
            # For LLM A* data, we might not have region annotations
            if "annotations" in data:
                # Get preferred regions from annotations
                preferred_regions = data["annotations"].get("preference", [])
                avoided_regions = data["annotations"].get("avoidance", [])
            else:
                preferred_regions = []
                avoided_regions = []
            
            # Calculate region coverage for constrained path
            if "region_coverage" in data["constrained"]:
                constrained_region_coverage = data["constrained"]["region_coverage"]
                baseline_region_coverage = data["baseline"]["region_coverage"]
            else:
                # For LLM A* data, we might not have region coverage
                constrained_region_coverage = {}
                baseline_region_coverage = {}
            
            # Count points in different zone types
            # For baseline path
            baseline_points_in_preferred = 0
            baseline_points_in_avoided = 0
            baseline_total_points = len(baseline_path)
            
            for region, coverage in baseline_region_coverage.items():
                points_in_region = int(coverage * baseline_total_points)
                if region in preferred_regions:
                    baseline_points_in_preferred += points_in_region
                if region in avoided_regions:
                    baseline_points_in_avoided += points_in_region
            
            # For constrained path
            constrained_points_in_preferred = 0
            constrained_points_in_avoided = 0
            constrained_total_points = len(constrained_path)
            
            for region, coverage in constrained_region_coverage.items():
                points_in_region = int(coverage * constrained_total_points)
                if region in preferred_regions:
                    constrained_points_in_preferred += points_in_region
                if region in avoided_regions:
                    constrained_points_in_avoided += points_in_region
            
            # Calculate percentages
            baseline_pct_in_preferred = (baseline_points_in_preferred / baseline_total_points * 100) if baseline_total_points > 0 else 0
            baseline_pct_in_avoided = (baseline_points_in_avoided / baseline_total_points * 100) if baseline_total_points > 0 else 0
            
            constrained_pct_in_preferred = (constrained_points_in_preferred / constrained_total_points * 100) if constrained_total_points > 0 else 0
            constrained_pct_in_avoided = (constrained_points_in_avoided / constrained_total_points * 100) if constrained_total_points > 0 else 0
            
            # Calculate improvement metrics
            preferred_improvement_pct = constrained_pct_in_preferred - baseline_pct_in_preferred
            avoided_reduction_pct = baseline_pct_in_avoided - constrained_pct_in_avoided
            
            # Calculate annotation coverage for preferred regions
            if preferred_regions:
                preferred_coverage = sum(constrained_region_coverage.get(region, 0) 
                                        for region in preferred_regions)
                preferred_coverage /= len(preferred_regions) if preferred_regions else 1
                
                baseline_preferred_coverage = sum(baseline_region_coverage.get(region, 0) 
                                                for region in preferred_regions)
                baseline_preferred_coverage /= len(preferred_regions) if preferred_regions else 1
            else:
                preferred_coverage = 0
                baseline_preferred_coverage = 0
            
            # Calculate annotation coverage for avoided regions (lower is better)
            if avoided_regions:
                avoided_coverage = sum(constrained_region_coverage.get(region, 0) 
                                      for region in avoided_regions)
                avoided_coverage /= len(avoided_regions) if avoided_regions else 1
                avoided_coverage = 1 - avoided_coverage  # Convert to avoidance score (higher is better)
                
                baseline_avoided_coverage = sum(baseline_region_coverage.get(region, 0) 
                                              for region in avoided_regions)
                baseline_avoided_coverage /= len(avoided_regions) if avoided_regions else 1
                baseline_avoided_coverage = 1 - baseline_avoided_coverage
            else:
                avoided_coverage = 1  # Perfect avoidance if no regions to avoid
                baseline_avoided_coverage = 1
            
            # Combined annotation coverage score (weighted average of preference and avoidance)
            if preferred_regions or avoided_regions:
                annotation_coverage = (preferred_coverage + avoided_coverage) / 2
                baseline_annotation_coverage = (baseline_preferred_coverage + baseline_avoided_coverage) / 2
            else:
                annotation_coverage = 0
                baseline_annotation_coverage = 0
            
            # Calculate annotation coverage improvement
            annotation_improvement = annotation_coverage - baseline_annotation_coverage
            
            # Store metrics for this scenario
            scenario_metrics = {
                "scenario_id": scenario_id,
                "instruction": instruction,
                "baseline_nodes": baseline_nodes,
                "constrained_nodes": constrained_nodes,
                "nodes_reduction_pct": nodes_reduction,
                "baseline_time_ms": baseline_time,
                "constrained_time_ms": constrained_time,
                "time_reduction_pct": time_reduction,
                "baseline_path_length": baseline_path_length,
                "constrained_path_length": constrained_path_length,
                "path_length_ratio": path_length_ratio,
                "baseline_annotation_coverage": baseline_annotation_coverage,
                "constrained_annotation_coverage": annotation_coverage,
                "annotation_improvement": annotation_improvement,
                "preferred_regions": preferred_regions,
                "avoided_regions": avoided_regions,
                # New detailed metrics
                "baseline_points_in_preferred": baseline_points_in_preferred,
                "baseline_pct_in_preferred": baseline_pct_in_preferred,
                "constrained_points_in_preferred": constrained_points_in_preferred,
                "constrained_pct_in_preferred": constrained_pct_in_preferred,
                "preferred_improvement_pct": preferred_improvement_pct,
                "baseline_points_in_avoided": baseline_points_in_avoided,
                "baseline_pct_in_avoided": baseline_pct_in_avoided,
                "constrained_points_in_avoided": constrained_points_in_avoided,
                "constrained_pct_in_avoided": constrained_pct_in_avoided,
                "avoided_reduction_pct": avoided_reduction_pct
            }
            
            metrics.append(scenario_metrics)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Convert to DataFrame for easier analysis
    metrics_df = pd.DataFrame(metrics)
    
    return metrics_df

def generate_performance_visualizations(metrics_df, output_dir):
    """
    Generate visualizations of performance metrics
    
    Args:
        metrics_df: DataFrame with calculated metrics
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    sns.set(style="whitegrid")
    plt.rcParams.update({'font.size': 12})
    
    # 1. Nodes Reduction Bar Chart
    plt.figure(figsize=(12, 6))
    scenarios = metrics_df["scenario_id"].tolist()
    nodes_reduction = metrics_df["nodes_reduction_pct"].tolist()
    
    # Sort by reduction percentage
    sorted_indices = np.argsort(nodes_reduction)
    sorted_scenarios = [scenarios[i] for i in sorted_indices]
    sorted_reductions = [nodes_reduction[i] for i in sorted_indices]
    
    plt.bar(range(len(sorted_scenarios)), sorted_reductions, color='skyblue')
    plt.axhline(y=np.mean(nodes_reduction), color='r', linestyle='--', label=f'Mean: {np.mean(nodes_reduction):.1f}%')
    plt.xlabel('Scenario')
    plt.ylabel('Nodes Reduction (%)')
    plt.title('Search Efficiency: Nodes Expanded Reduction')
    plt.xticks(range(len(sorted_scenarios)), sorted_scenarios, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "nodes_reduction.png"))
    plt.close()
    
    # 2. Time Reduction Bar Chart
    plt.figure(figsize=(12, 6))
    time_reduction = metrics_df["time_reduction_pct"].tolist()
    
    # Sort by reduction percentage
    sorted_indices = np.argsort(time_reduction)
    sorted_scenarios = [scenarios[i] for i in sorted_indices]
    sorted_reductions = [time_reduction[i] for i in sorted_indices]
    
    plt.bar(range(len(sorted_scenarios)), sorted_reductions, color='lightgreen')
    plt.axhline(y=np.mean(time_reduction), color='r', linestyle='--', label=f'Mean: {np.mean(time_reduction):.1f}%')
    plt.xlabel('Scenario')
    plt.ylabel('Time Reduction (%)')
    plt.title('Search Efficiency: Execution Time Reduction')
    plt.xticks(range(len(sorted_scenarios)), sorted_scenarios, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "time_reduction.png"))
    plt.close()
    
    # 3. Annotation Coverage Comparison
    plt.figure(figsize=(12, 6))
    baseline_coverage = metrics_df["baseline_annotation_coverage"].tolist()
    constrained_coverage = metrics_df["constrained_annotation_coverage"].tolist()
    
    # Sort by improvement
    improvements = np.array(constrained_coverage) - np.array(baseline_coverage)
    sorted_indices = np.argsort(improvements)
    sorted_scenarios = [scenarios[i] for i in sorted_indices]
    sorted_baseline = [baseline_coverage[i] for i in sorted_indices]
    sorted_constrained = [constrained_coverage[i] for i in sorted_indices]
    
    x = np.arange(len(sorted_scenarios))
    width = 0.35
    
    plt.bar(x - width/2, sorted_baseline, width, label='Baseline', color='lightcoral')
    plt.bar(x + width/2, sorted_constrained, width, label='Constrained', color='lightblue')
    
    plt.xlabel('Scenario')
    plt.ylabel('Annotation Coverage (0-1)')
    plt.title('Instruction Compliance: Annotation Coverage')
    plt.xticks(x, sorted_scenarios, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "annotation_coverage.png"))
    plt.close()
    
    # 4. Path Length Ratio
    plt.figure(figsize=(12, 6))
    path_ratios = metrics_df["path_length_ratio"].tolist()
    
    # Sort by ratio
    sorted_indices = np.argsort(path_ratios)
    sorted_scenarios = [scenarios[i] for i in sorted_indices]
    sorted_ratios = [path_ratios[i] for i in sorted_indices]
    
    plt.bar(range(len(sorted_scenarios)), sorted_ratios, color='plum')
    plt.axhline(y=1.0, color='r', linestyle='--', label='Baseline (ratio=1.0)')
    plt.axhline(y=np.mean(path_ratios), color='g', linestyle='--', label=f'Mean: {np.mean(path_ratios):.2f}')
    plt.xlabel('Scenario')
    plt.ylabel('Path Length Ratio (Constrained/Baseline)')
    plt.title('Path Quality: Length Ratio')
    plt.xticks(range(len(sorted_scenarios)), sorted_scenarios, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "path_length_ratio.png"))
    plt.close()
    
    # 5. Summary Scatter Plot: Nodes Reduction vs Annotation Improvement
    plt.figure(figsize=(10, 8))
    plt.scatter(metrics_df["nodes_reduction_pct"], 
                metrics_df["annotation_improvement"],
                s=80, alpha=0.7)
    
    # Add scenario labels
    for i, scenario in enumerate(scenarios):
        plt.annotate(scenario, 
                    (metrics_df["nodes_reduction_pct"].iloc[i], 
                     metrics_df["annotation_improvement"].iloc[i]),
                    xytext=(5, 5), textcoords='offset points')
    
    plt.axhline(y=0, color='r', linestyle='--')
    plt.axvline(x=0, color='r', linestyle='--')
    plt.xlabel('Nodes Reduction (%)')
    plt.ylabel('Annotation Coverage Improvement')
    plt.title('Efficiency vs. Instruction Compliance')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "efficiency_vs_compliance.png"))
    plt.close()
    
    # 6. Summary Table
    plt.figure(figsize=(12, 8))
    plt.axis('off')
    
    # Create table data
    table_data = []
    headers = ['Scenario', 'Nodes\nReduction (%)', 'Time\nReduction (%)', 
               'Annotation\nCoverage', 'Path Length\nRatio']
    
    for i, row in metrics_df.iterrows():
        table_data.append([
            row["scenario_id"],
            f"{row['nodes_reduction_pct']:.1f}%",
            f"{row['time_reduction_pct']:.1f}%",
            f"{row['constrained_annotation_coverage']:.2f}",
            f"{row['path_length_ratio']:.2f}"
        ])
    
    # Add average row
    avg_row = [
        'AVERAGE',
        f"{metrics_df['nodes_reduction_pct'].mean():.1f}%",
        f"{metrics_df['time_reduction_pct'].mean():.1f}%",
        f"{metrics_df['constrained_annotation_coverage'].mean():.2f}",
        f"{metrics_df['path_length_ratio'].mean():.2f}"
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
    
    plt.title('DCIP Performance Summary', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "performance_summary_table.png"), bbox_inches='tight')
    plt.close()
    
    # Save metrics as CSV
    metrics_df.to_csv(os.path.join(output_dir, "performance_metrics.csv"), index=False)
    
    # Generate text summary
    with open(os.path.join(output_dir, "performance_summary.txt"), 'w') as f:
        f.write("DCIP Performance Summary\n")
        f.write("=======================\n\n")
        
        f.write(f"Number of scenarios analyzed: {len(metrics_df)}\n\n")
        
        f.write("Search Efficiency:\n")
        f.write(f"  Average nodes reduction: {metrics_df['nodes_reduction_pct'].mean():.1f}%\n")
        f.write(f"  Average time reduction: {metrics_df['time_reduction_pct'].mean():.1f}%\n\n")
        
        f.write("Instruction Compliance:\n")
        f.write(f"  Average annotation coverage: {metrics_df['constrained_annotation_coverage'].mean():.2f}\n")
        f.write(f"  Average improvement over baseline: {metrics_df['annotation_improvement'].mean():.2f}\n\n")
        
        f.write("Detailed Region Coverage:\n")
        f.write(f"  Preferred regions - Baseline: {metrics_df['baseline_pct_in_preferred'].mean():.1f}%\n")
        f.write(f"  Preferred regions - Constrained: {metrics_df['constrained_pct_in_preferred'].mean():.1f}%\n")
        f.write(f"  Preferred regions - Improvement: {metrics_df['preferred_improvement_pct'].mean():.1f}%\n\n")
        
        f.write(f"  Avoided regions - Baseline: {metrics_df['baseline_pct_in_avoided'].mean():.1f}%\n")
        f.write(f"  Avoided regions - Constrained: {metrics_df['constrained_pct_in_avoided'].mean():.1f}%\n")
        f.write(f"  Avoided regions - Reduction: {metrics_df['avoided_reduction_pct'].mean():.1f}%\n\n")
        
        f.write("Path Quality:\n")
        f.write(f"  Average path length ratio: {metrics_df['path_length_ratio'].mean():.2f}\n")
        
        # Count scenarios with improved metrics
        nodes_improved = sum(1 for x in metrics_df['nodes_reduction_pct'] if x > 0)
        time_improved = sum(1 for x in metrics_df['time_reduction_pct'] if x > 0)
        annotation_improved = sum(1 for x in metrics_df['annotation_improvement'] if x > 0)
        preferred_improved = sum(1 for x in metrics_df['preferred_improvement_pct'] if x > 0)
        avoided_improved = sum(1 for x in metrics_df['avoided_reduction_pct'] if x > 0)
        
        f.write("\nImprovement Statistics:\n")
        f.write(f"  Scenarios with reduced nodes: {nodes_improved}/{len(metrics_df)} ({nodes_improved/len(metrics_df)*100:.1f}%)\n")
        f.write(f"  Scenarios with reduced time: {time_improved}/{len(metrics_df)} ({time_improved/len(metrics_df)*100:.1f}%)\n")
        f.write(f"  Scenarios with improved annotation coverage: {annotation_improved}/{len(metrics_df)} ({annotation_improved/len(metrics_df)*100:.1f}%)\n")
        f.write(f"  Scenarios with improved preferred region coverage: {preferred_improved}/{len(metrics_df)} ({preferred_improved/len(metrics_df)*100:.1f}%)\n")
        f.write(f"  Scenarios with reduced avoided region coverage: {avoided_improved}/{len(metrics_df)} ({avoided_improved/len(metrics_df)*100:.1f}%)\n")
    
    print(f"Performance visualizations saved to {output_dir}")
    return

def analyze_path_on_annotations(scenario_data, output_dir):
    """
    Analyze and visualize paths in relation to annotated regions
    
    Args:
        scenario_data: Loaded scenario data from JSON
        output_dir: Directory to save visualizations
    """
    # Extract data
    scenario_id = scenario_data["scenario_id"]
    baseline_path = scenario_data["baseline"]["path"]
    constrained_path = scenario_data["constrained"]["path"]
    preferred_regions = scenario_data["annotations"]["preference"]
    avoided_regions = scenario_data["annotations"]["avoidance"]
    
    # Create a visualization
    plt.figure(figsize=(12, 10))
    
    # We need to reconstruct the grid and regions
    # This would require loading the original grid or using the stored data
    # For now, let's create a simplified visualization
    
    # Plot paths
    baseline_x = [p[0] for p in baseline_path]
    baseline_y = [p[1] for p in baseline_path]
    constrained_x = [p[0] for p in constrained_path]
    constrained_y = [p[1] for p in constrained_path]
    
    plt.plot(baseline_x, baseline_y, 'b-', linewidth=2, label='Baseline Path')
    plt.plot(constrained_x, constrained_y, 'g-', linewidth=2, label='Constrained Path')
    
    # Calculate and display metrics
    # Count points in preferred regions
    baseline_in_preferred = 0
    constrained_in_preferred = 0
    
    # Count points in avoided regions
    baseline_in_avoided = 0
    constrained_in_avoided = 0
    
    # This requires region definitions which we'd need to load
    # For now, we can use the pre-calculated region coverage
    baseline_region_coverage = scenario_data["baseline"]["region_coverage"]
    constrained_region_coverage = scenario_data["constrained"]["region_coverage"]
    
    # Calculate metrics
    preferred_improvement = 0
    if preferred_regions:
        baseline_preferred = sum(baseline_region_coverage.get(r, 0) for r in preferred_regions) / len(preferred_regions)
        constrained_preferred = sum(constrained_region_coverage.get(r, 0) for r in preferred_regions) / len(preferred_regions)
        preferred_improvement = constrained_preferred - baseline_preferred
    
    avoidance_improvement = 0
    if avoided_regions:
        baseline_avoided = sum(baseline_region_coverage.get(r, 0) for r in avoided_regions) / len(avoided_regions)
        constrained_avoided = sum(constrained_region_coverage.get(r, 0) for r in avoided_regions) / len(avoided_regions)
        # For avoidance, lower is better
        avoidance_improvement = baseline_avoided - constrained_avoided
    
    # Add metrics to the plot
    plt.title(f"Path Analysis - Scenario {scenario_id}")
    plt.figtext(0.02, 0.02, 
                f"Preferred regions: {', '.join(preferred_regions)}\n"
                f"Avoided regions: {', '.join(avoided_regions)}\n"
                f"Preference improvement: {preferred_improvement:.2f}\n"
                f"Avoidance improvement: {avoidance_improvement:.2f}",
                fontsize=10)
    
    plt.legend()
    plt.tight_layout()
    
    # Save the visualization
    os.makedirs(os.path.join(output_dir, "path_analysis"), exist_ok=True)
    plt.savefig(os.path.join(output_dir, "path_analysis", f"scenario_{scenario_id}_path_analysis.png"))
    plt.close()
    
    return {
        "preferred_improvement": preferred_improvement,
        "avoidance_improvement": avoidance_improvement
    }

if __name__ == "__main__":
    # Find the most recent experiment directory in DCIP/rrt_logs/
    log_dirs = sorted(glob("rrt_logs/experiment_*"))
    # log_dirs = sorted(glob("ds_logs/slice/experiment_*"))
    if not log_dirs:
        print("No experiment logs found. Run experiments first.")
        exit(1)
    
    latest_log_dir = log_dirs[-1]
    print(f"Analyzing most recent experiment: {latest_log_dir}")
    
    # Calculate metrics
    metrics_df = calculate_performance_metrics(latest_log_dir)
    
    # Generate visualizations
    output_dir = os.path.join(latest_log_dir, "performance_analysis")
    generate_performance_visualizations(metrics_df, output_dir)
    
    print(f"Analysis complete. Results saved to {output_dir}")