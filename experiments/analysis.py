import os
import json
import numpy as np
import matplotlib.pyplot as plt

def analyze_search_efficiency(log_dir):
    """Analyze search efficiency across experiments"""
    # Collect metrics from all experiments
    metrics_files = [f for f in os.listdir(os.path.join(log_dir, "metrics")) 
                   if f.endswith(".json")]
    
    efficiency_data = {
        "constrained": [],
        "baseline": [],
        "improvement_ratio": []
    }
    
    nodes_data = {
        "constrained": [],
        "baseline": [],
        "reduction_ratio": []
    }
    
    # Store metadata for each experiment
    metadata = []
    
    for metrics_file in metrics_files:
        with open(os.path.join(log_dir, "metrics", metrics_file), 'r') as f:
            metrics = json.load(f)
        
        if "constrained" in metrics and "baseline" in metrics:
            c_metrics = metrics["constrained"]
            b_metrics = metrics["baseline"]
            
            if c_metrics["success"] and b_metrics["success"]:
                # Search efficiency
                efficiency_data["constrained"].append(c_metrics["search_efficiency"])
                efficiency_data["baseline"].append(b_metrics["search_efficiency"])
                
                if b_metrics["search_efficiency"] > 0:
                    ratio = c_metrics["search_efficiency"] / b_metrics["search_efficiency"]
                    efficiency_data["improvement_ratio"].append(ratio)
                
                # Nodes expanded
                nodes_data["constrained"].append(c_metrics["nodes_expanded"])
                nodes_data["baseline"].append(b_metrics["nodes_expanded"])
                
                if b_metrics["nodes_expanded"] > 0:
                    ratio = b_metrics["nodes_expanded"] / c_metrics["nodes_expanded"]
                    nodes_data["reduction_ratio"].append(ratio)
                
                # Store metadata
                if "metadata" in metrics:
                    metadata.append(metrics["metadata"])
                else:
                    metadata.append({})
    
    # Create visualizations
    plt.figure(figsize=(15, 12))
    
    # Plot efficiency comparison
    plt.subplot(2, 2, 1)
    plt.boxplot([efficiency_data["baseline"], efficiency_data["constrained"]], 
               labels=["Baseline", "Constrained"])
    plt.title("Search Efficiency Comparison")
    plt.ylabel("Efficiency (path length / nodes expanded)")
    
    # Plot nodes expanded comparison
    plt.subplot(2, 2, 2)
    plt.boxplot([nodes_data["baseline"], nodes_data["constrained"]], 
               labels=["Baseline", "Constrained"])
    plt.title("Nodes Expanded Comparison")
    plt.ylabel("Number of Nodes")
    plt.yscale('log')  # Log scale for better visualization
    
    # Plot improvement ratios
    plt.subplot(2, 2, 3)
    plt.hist(efficiency_data["improvement_ratio"], bins=10)
    plt.axvline(x=1.0, color='r', linestyle='-')
    plt.title("Search Efficiency Improvement Ratio")
    plt.xlabel("Ratio (>1 means improvement)")
    
    # Plot node reduction ratios
    plt.subplot(2, 2, 4)
    plt.hist(nodes_data["reduction_ratio"], bins=10)
    plt.axvline(x=1.0, color='r', linestyle='-')
    plt.title("Node Expansion Reduction Ratio")
    plt.xlabel("Ratio (>1 means reduction)")
    
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "search_efficiency_analysis.png"))
    
    # Analyze by map type
    map_types = set(m.get("map_type", "unknown") for m in metadata)
    
    if len(map_types) > 1:  # Only if we have multiple map types
        plt.figure(figsize=(15, 10))
        
        # Prepare data by map type
        map_type_data = {}
        
        for map_type in map_types:
            if map_type == "unknown":
                continue
                
            map_type_data[map_type] = {
                "efficiency_improvement": [],
                "nodes_reduction": []
            }
        
        for i, m in enumerate(metadata):
            map_type = m.get("map_type", "unknown")
            if map_type != "unknown" and i < len(efficiency_data["improvement_ratio"]):
                map_type_data[map_type]["efficiency_improvement"].append(
                    efficiency_data["improvement_ratio"][i])
                
            if map_type != "unknown" and i < len(nodes_data["reduction_ratio"]):
                map_type_data[map_type]["nodes_reduction"].append(
                    nodes_data["reduction_ratio"][i])
        
        # Plot efficiency improvement by map type
        plt.subplot(1, 2, 1)
        boxplot_data = [map_type_data[mt]["efficiency_improvement"] for mt in map_type_data]
        plt.boxplot(boxplot_data, labels=list(map_type_data.keys()))
        plt.axhline(y=1.0, color='r', linestyle='-')
        plt.title("Search Efficiency Improvement by Map Type")
        plt.ylabel("Improvement Ratio")
        
        # Plot node reduction by map type
        plt.subplot(1, 2, 2)
        boxplot_data = [map_type_data[mt]["nodes_reduction"] for mt in map_type_data]
        plt.boxplot(boxplot_data, labels=list(map_type_data.keys()))
        plt.axhline(y=1.0, color='r', linestyle='-')
        plt.title("Node Expansion Reduction by Map Type")
        plt.ylabel("Reduction Ratio")
        
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, "map_type_analysis.png"))
    
    # Print summary statistics
    print("\nSearch Efficiency Analysis:")
    print(f"Average baseline efficiency: {np.mean(efficiency_data['baseline']):.6f}")
    print(f"Average constrained efficiency: {np.mean(efficiency_data['constrained']):.6f}")
    print(f"Average improvement ratio: {np.mean(efficiency_data['improvement_ratio']):.2f}x")
    
    print("\nNodes Expanded Analysis:")
    print(f"Average baseline nodes: {np.mean(nodes_data['baseline']):.1f}")
    print(f"Average constrained nodes: {np.mean(nodes_data['constrained']):.1f}")
    print(f"Average reduction ratio: {np.mean(nodes_data['reduction_ratio']):.2f}x")
    
    # Print map type analysis
    if len(map_types) > 1:
        print("\nMap Type Analysis:")
        for map_type in map_type_data:
            print(f"\n{map_type.capitalize()}:")
            print(f"  Efficiency improvement: {np.mean(map_type_data[map_type]['efficiency_improvement']):.2f}x")
            print(f"  Node reduction: {np.mean(map_type_data[map_type]['nodes_reduction']):.2f}x")
    
    return {
        "efficiency": efficiency_data,
        "nodes": nodes_data,
        "metadata": metadata
    }