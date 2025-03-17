import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Dict, Optional
import os
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter

def visualize_grid(environment, path: Optional[List[Tuple[int, int]]] = None, 
                  start: Optional[Tuple[int, int]] = None, 
                  goal: Optional[Tuple[int, int]] = None,
                  constraints: Optional[Dict] = None,
                  title: str = "Grid Environment",
                  save_path: Optional[str] = None):
    """Visualize grid environment with path and constraints"""
    plt.figure(figsize=(10, 10))
    
    # Create a colored grid for visualization
    vis_grid = np.zeros((environment.height, environment.width, 3))
    
    # Set obstacles to black
    for y in range(environment.height):
        for x in range(environment.width):
            if environment.grid[y, x] == 1:
                vis_grid[y, x] = [0, 0, 0]  # Black for obstacles
            else:
                vis_grid[y, x] = [1, 1, 1]  # White for free space
    
    # Color regions based on constraints
    if constraints:
        # Avoidance regions in light red
        if 'avoidance' in constraints:
            for region, _ in constraints['avoidance'].items():
                if region in environment.regions:
                    for x, y in environment.regions[region]:
                        if 0 <= x < environment.width and 0 <= y < environment.height:
                            if environment.grid[y, x] == 0:  # Only color free cells
                                vis_grid[y, x] = [1, 0.8, 0.8]  # Light red
        
        # Preference regions in light green
        if 'preference' in constraints:
            for region, _ in constraints['preference'].items():
                if region in environment.regions:
                    for x, y in environment.regions[region]:
                        if 0 <= x < environment.width and 0 <= y < environment.height:
                            if environment.grid[y, x] == 0:  # Only color free cells
                                vis_grid[y, x] = [0.8, 1, 0.8]  # Light green
        
        # Proximity regions in light blue
        if 'proximity' in constraints:
            for region, _ in constraints['proximity'].items():
                if region in environment.regions:
                    for x, y in environment.regions[region]:
                        if 0 <= x < environment.width and 0 <= y < environment.height:
                            if environment.grid[y, x] == 0:  # Only color free cells
                                vis_grid[y, x] = [0.8, 0.8, 1]  # Light blue
    
    plt.imshow(vis_grid)
    
    # Plot path
    if path:
        path_x = [p[0] for p in path]
        path_y = [p[1] for p in path]
        plt.plot(path_x, path_y, 'g-', linewidth=2)
    
    # Plot start and goal
    if start:
        plt.plot(start[0], start[1], 'bs', markersize=10)
    if goal:
        plt.plot(goal[0], goal[1], 'rs', markersize=10)
    
    plt.grid(True)
    plt.title(title)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def visualize_comparison(environment, constraint_path, baseline_path, 
                        start, goal, constraints, save_path=None):
    """Visualize comparison between constraint-based and baseline paths"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Create visualization grid
    vis_grid = np.zeros((environment.height, environment.width, 3))
    
    # Set obstacles to black
    for y in range(environment.height):
        for x in range(environment.width):
            if environment.grid[y, x] == 1:
                vis_grid[y, x] = [0, 0, 0]  # Black for obstacles
            else:
                vis_grid[y, x] = [1, 1, 1]  # White for free space
    
    # Color regions based on constraints
    if constraints:
        # Avoidance regions in light red
        if 'avoidance' in constraints:
            for region, _ in constraints['avoidance'].items():
                if region in environment.regions:
                    for x, y in environment.regions[region]:
                        if 0 <= x < environment.width and 0 <= y < environment.height:
                            if environment.grid[y, x] == 0:  # Only color free cells
                                vis_grid[y, x] = [1, 0.8, 0.8]  # Light red
    
    # Plot constraint-based path
    ax1.imshow(vis_grid)
    if constraint_path:
        path_x = [p[0] for p in constraint_path]
        path_y = [p[1] for p in constraint_path]
        ax1.plot(path_x, path_y, 'g-', linewidth=2)
    if start:
        ax1.plot(start[0], start[1], 'bs', markersize=10)
    if goal:
        ax1.plot(goal[0], goal[1], 'rs', markersize=10)
    ax1.grid(True)
    ax1.set_title("Constraint-Based Path")
    
    # Plot baseline path
    ax2.imshow(vis_grid)
    if baseline_path:
        path_x = [p[0] for p in baseline_path]
        path_y = [p[1] for p in baseline_path]
        ax2.plot(path_x, path_y, 'b-', linewidth=2)
    if start:
        ax2.plot(start[0], start[1], 'bs', markersize=10)
    if goal:
        ax2.plot(goal[0], goal[1], 'rs', markersize=10)
    ax2.grid(True)
    ax2.set_title("Baseline Path (Uniform Cost)")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def visualize_regions(environment, save_path=None):
    """
    Visualize the regions in the environment
    
    Args:
        environment: The grid environment
        save_path: Path to save the visualization
    """
    plt.figure(figsize=(10, 10))
    
    # Create a colormap for regions
    cmap = plt.cm.get_cmap('tab20', len(environment.regions) + 1)
    
    # Create a region map
    region_map = np.zeros_like(environment.grid)
    
    # Fill in regions with different colors
    for i, (region_name, points) in enumerate(environment.regions.items(), 1):
        for x, y in points:
            region_map[y, x] = i
    
    # Plot grid
    plt.imshow(environment.grid, cmap='binary', interpolation='nearest')
    
    # Plot regions with transparency
    masked_region_map = np.ma.masked_where(region_map == 0, region_map)
    plt.imshow(masked_region_map, cmap=cmap, alpha=0.5, interpolation='nearest')
    
    # Add region labels
    for i, region_name in enumerate(environment.regions.keys(), 1):
        points = environment.regions[region_name]
        if points:
            # Find center of region
            x_coords, y_coords = zip(*points)
            center_x = sum(x_coords) / len(x_coords)
            center_y = sum(y_coords) / len(y_coords)
            
            # Add label
            plt.text(center_x, center_y, region_name, 
                     color='black', fontsize=8, ha='center', va='center',
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    # Set title
    plt.title('Environment Regions')
    
    # Save or show
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def visualize_paths(environment, baseline_path, constrained_path, start, goal, save_path=None):
    """
    Visualize baseline and constrained paths
    
    Args:
        environment: The grid environment
        baseline_path: List of (x, y) tuples for baseline path
        constrained_path: List of (x, y) tuples for constrained path
        start: (x, y) tuple for start position
        goal: (x, y) tuple for goal position
        save_path: Path to save the visualization
    """
    plt.figure(figsize=(10, 10))
    
    # Create grid representation
    grid = environment.grid.copy()
    
    # Plot grid
    plt.imshow(grid, cmap='binary', interpolation='nearest')
    
    # Plot paths
    if baseline_path:
        path_x, path_y = zip(*baseline_path)
        plt.plot(path_x, path_y, 'b-', linewidth=2, label='Baseline Path')
    
    if constrained_path:
        path_x, path_y = zip(*constrained_path)
        plt.plot(path_x, path_y, 'g-', linewidth=2, label='Constrained Path')
    
    # Plot start and goal
    plt.plot(start[0], start[1], 'go', markersize=10, label='Start')
    plt.plot(goal[0], goal[1], 'ro', markersize=10, label='Goal')
    
    # Add legend
    plt.legend()
    
    # Set title
    plt.title('Path Comparison')
    
    # Save or show
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def visualize_node_expansion(environment, expanded_nodes_baseline, expanded_nodes_constrained, 
                           baseline_path, constrained_path, start, goal, save_path=None):
    """
    Visualize the nodes expanded by baseline and constrained planners
    
    Args:
        environment: The grid environment
        expanded_nodes_baseline: List of (x, y) tuples for nodes expanded by baseline planner
        expanded_nodes_constrained: List of (x, y) tuples for nodes expanded by constrained planner
        baseline_path: List of (x, y) tuples for baseline path
        constrained_path: List of (x, y) tuples for constrained path
        start: (x, y) tuple for start position
        goal: (x, y) tuple for goal position
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
    
    # Plot constrained expansion
    ax2.imshow(grid, cmap='binary', interpolation='nearest')
    ax2.set_title(f'Constrained Planner (Nodes Expanded: {len(expanded_nodes_constrained)})')
    
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

def visualize_cost_landscape(environment, cost_function, start, goal, instruction, save_path=None):
    """
    Visualize the cost landscape generated by the cost function
    
    Args:
        environment: The grid environment
        cost_function: Function that returns cost between adjacent cells
        start: (x, y) tuple for start position
        goal: (x, y) tuple for goal position
        instruction: Natural language instruction
        save_path: Path to save the visualization
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create grid representation
    grid = environment.grid.copy()
    
    # Calculate cost from start to each cell
    width, height = environment.width, environment.height
    cost_map = np.zeros((height, width))
    
    # Fill with infinity for obstacles
    cost_map.fill(np.inf)
    
    # Calculate cost for each valid cell
    for y in range(height):
        for x in range(width):
            if environment.is_valid(x, y):
                # Use cost from start to this cell
                cost_map[y, x] = cost_function(start[0], start[1], x, y)
    
    # Create custom colormap for costs
    cost_cmap = LinearSegmentedColormap.from_list('cost', 
                                                ['darkgreen', 'green', 'yellow', 'orange', 'red'], 
                                                N=256)
    
    # Plot grid
    ax.imshow(grid, cmap='binary', interpolation='nearest')
    
    # Plot cost landscape
    # Mask out obstacles
    masked_cost_map = np.ma.masked_where(np.isinf(cost_map), cost_map)
    
    # Normalize costs for better visualization
    vmin = np.min(masked_cost_map)
    vmax = np.percentile(masked_cost_map.compressed(), 95)  # Use 95th percentile to avoid outliers
    
    im = ax.imshow(masked_cost_map, cmap=cost_cmap, alpha=0.7, 
                 interpolation='nearest', vmin=vmin, vmax=vmax)
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Cost')
    
    # Plot start and goal
    ax.plot(start[0], start[1], 'go', markersize=10, label='Start')
    ax.plot(goal[0], goal[1], 'ro', markersize=10, label='Goal')
    
    # Set title and labels
    ax.set_title(f'Cost Landscape for Instruction:\n"{instruction}"')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    # Add legend
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()