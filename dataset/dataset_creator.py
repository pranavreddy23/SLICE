# dataset_creator.py
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from scipy import ndimage

import numpy as np
import cv2
from skimage.segmentation import watershed, slic, quickshift
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
from typing import Dict, List, Tuple
class DatasetCreator:
    def __init__(self, output_dir="./dataset"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/maps", exist_ok=True)
        os.makedirs(f"{output_dir}/annotations", exist_ok=True)
        os.makedirs(f"{output_dir}/visualizations", exist_ok=True)
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/maps", exist_ok=True)
        os.makedirs(f"{output_dir}/annotations", exist_ok=True)
        os.makedirs(f"{output_dir}/visualizations", exist_ok=True)
        
    def create_base_map(self, map_type, width=50, height=30, seed=None):
        """Create a base map of specified type"""
        if seed is not None:
            np.random.seed(seed)
            
        grid = np.zeros((height, width))
        
        # Add outer walls
        grid[0, :] = 1
        grid[-1, :] = 1
        grid[:, 0] = 1
        grid[:, -1] = 1
        
        if map_type == "office":
            # Add rooms and corridors
            # Horizontal walls
            grid[10, 5:45] = 1
            grid[20, 5:20] = 1
            grid[20, 30:45] = 1
            
            # Vertical walls
            grid[10:20, 20] = 1
            grid[10:30, 30] = 1
            
        elif map_type == "warehouse":
            # Add shelving units
            for i in range(5, width-5, 10):
                for j in range(5, height-5, 15):
                    grid[j:j+10, i:i+3] = 1
                    
        elif map_type == "outdoor":
            # Add buildings
            for _ in range(5):
                x = np.random.randint(5, width-10)
                y = np.random.randint(5, height-10)
                w = np.random.randint(5, 10)
                h = np.random.randint(5, 10)
                grid[y:y+h, x:x+w] = 1
        
        return grid
    
    def add_clutter(self, grid, clutter_level=0.05, seed=None):
        """Add random clutter to the map"""
        if seed is not None:
            np.random.seed(seed)
            
        height, width = grid.shape
        cluttered_grid = grid.copy()
        
        # Calculate number of clutter points
        free_cells = np.sum(grid == 0)
        num_clutter = int(free_cells * clutter_level)
        
        # Add random obstacles
        for _ in range(num_clutter):
            while True:
                x = np.random.randint(1, width-1)
                y = np.random.randint(1, height-1)
                if cluttered_grid[y, x] == 0:
                    cluttered_grid[y, x] = 1
                    break
        
        return cluttered_grid
    
    def identify_regions(self, grid):
        """Segment the environment into meaningful regions with more granularity"""
        # Convert grid to binary image (0 for free space, 1 for obstacles)
        binary_image = grid.copy()
        
        # Apply distance transform
        distance = ndimage.distance_transform_edt(1 - binary_image)
        
        # Create more regions by using SLIC superpixel segmentation
        from skimage.segmentation import slic
        
        # Create a 3-channel image for SLIC (using distance transform)
        image = np.dstack([distance, distance, distance])
        
        # Apply SLIC with more segments
        mask = (1 - binary_image).astype(bool)
        regions = slic(image, n_segments=30, compactness=5, sigma=1, 
                      mask=mask, start_label=1)
        
        # Create region dictionary
        region_dict = {}
        
        # Get unique region labels
        unique_regions = np.unique(regions)
        
        # Skip background (0)
        for region_id in unique_regions:
            if region_id == 0:
                continue
            
            # Get region mask
            region_mask = (regions == region_id)
            
            # Get points in this region
            points = []
            for y in range(region_mask.shape[0]):
                for x in range(region_mask.shape[1]):
                    if region_mask[y, x] and binary_image[y, x] == 0:  # Only include free space
                        points.append((x, y))
            
            if points:
                # Generate a meaningful name for the region
                region_name = self._generate_region_name(points, region_mask.shape)
                region_dict[region_name] = region_mask
        
        # Add the segmentation map
        region_dict["segmentation"] = regions
        
        return region_dict

    def _generate_region_name(self, points, shape):
        """Generate a meaningful name for a region based on its characteristics"""
        if not points:
            return "empty_region"
        
        # Calculate centroid
        x_coords = [x for x, y in points]
        y_coords = [y for x, y in points]
        
        centroid_x = sum(x_coords) / len(x_coords)
        centroid_y = sum(y_coords) / len(y_coords)
        
        # Determine position (top, bottom, left, right, center)
        height, width = shape
        position = []
        
        if centroid_y < height * 0.33:
            position.append("top")
        elif centroid_y > height * 0.66:
            position.append("bottom")
        else:
            position.append("middle")
        
        if centroid_x < width * 0.33:
            position.append("left")
        elif centroid_x > width * 0.66:
            position.append("right")
        else:
            position.append("center")
        
        # Determine size
        area = len(points)
        total_area = width * height
        
        if area < total_area * 0.05:
            size = "small"
        elif area < total_area * 0.15:
            size = "medium"
        else:
            size = "large"
        
        # Determine shape (rough approximation)
        x_range = max(x_coords) - min(x_coords)
        y_range = max(y_coords) - min(y_coords)
        
        if x_range > y_range * 2:
            shape = "horizontal"
        elif y_range > x_range * 2:
            shape = "vertical"
        elif abs(x_range - y_range) < min(x_range, y_range) * 0.2:
            shape = "square"
        else:
            shape = "irregular"
        
        # Check if it's a corridor
        if (shape in ["horizontal", "vertical"]) and (size in ["small", "medium"]):
            shape = "corridor"
        
        # Check if it's a corner
        corner_threshold = min(width, height) * 0.1
        is_corner = False
        
        # Check if region is near corners of the grid
        corners = [
            (0, 0),                # Top-left
            (0, height-1),         # Bottom-left
            (width-1, 0),          # Top-right
            (width-1, height-1)    # Bottom-right
        ]
        
        for corner in corners:
            corner_x, corner_y = corner
            for x, y in points:
                if abs(x - corner_x) < corner_threshold and abs(y - corner_y) < corner_threshold:
                    is_corner = True
                    if corner == (0, 0):
                        position = ["top", "left"]
                    elif corner == (0, height-1):
                        position = ["bottom", "left"]
                    elif corner == (width-1, 0):
                        position = ["top", "right"]
                    else:
                        position = ["bottom", "right"]
                    break
            if is_corner:
                break
        
        if is_corner:
            shape = "corner"
        
        # Combine characteristics into a name
        name = f"{size}_{shape}_{position[0]}_{position[1]}"
        
        return name
    
    def generate_start_goal_pairs(self, grid, num_pairs=2, seed=None):
        """Generate random start-goal pairs"""
        if seed is not None:
            np.random.seed(seed)
            
        height, width = grid.shape
        pairs = []
        
        for _ in range(num_pairs):
            # Find start position
            while True:
                start_x = np.random.randint(1, width-1)
                start_y = np.random.randint(1, height-1)
                if grid[start_y, start_x] == 0:
                    start = (start_x, start_y)
                    break
            
            # Find goal position (at least 20 cells away)
            while True:
                goal_x = np.random.randint(1, width-1)
                goal_y = np.random.randint(1, height-1)
                if grid[goal_y, goal_x] == 0:
                    goal = (goal_x, goal_y)
                    # Check distance
                    if abs(goal_x - start_x) + abs(goal_y - start_y) >= 20:
                        break
            
            pairs.append((start, goal))
        
        return pairs
    
    def visualize_map_with_regions_and_path(self, grid, regions, start, goal, path=None, save_path=None):
        """Visualize map with clearly labeled regions"""
        plt.figure(figsize=(12, 10))
        
        # Create a colored segmentation map
        segmentation = regions["segmentation"]
        
        # Create a colormap with distinct colors
        from matplotlib.colors import ListedColormap
        import matplotlib.colors as mcolors
        
        # Get a list of distinct colors
        colors = list(mcolors.TABLEAU_COLORS.values())
        # Add more colors if needed
        colors.extend(list(mcolors.CSS4_COLORS.values())[:20])
        
        # Create a colormap with enough colors for all regions
        num_regions = np.max(segmentation)
        if num_regions > 0:
            # Add black for obstacles (value 0)
            region_colors = ['black'] + colors[:num_regions]
            region_cmap = ListedColormap(region_colors)
        else:
            region_cmap = 'viridis'
        
        # Plot the segmentation
        plt.imshow(segmentation, cmap=region_cmap, interpolation='nearest')
        
        # Add start and goal markers
        plt.plot(start[0], start[1], 'go', markersize=12, label='Start')
        plt.plot(goal[0], goal[1], 'ro', markersize=12, label='Goal')
        
        # Add path if provided
        if path:
            path_x = [p[0] for p in path]
            path_y = [p[1] for p in path]
            plt.plot(path_x, path_y, 'b-', linewidth=2, label='Path')
        
        # Add region labels
        for region_name, region_mask in regions.items():
            if region_name != "segmentation":
                # Find centroid of region for label placement
                y_indices, x_indices = np.where(region_mask)
                if len(y_indices) > 0:
                    centroid_y = int(np.mean(y_indices))
                    centroid_x = int(np.mean(x_indices))
                    
                    # Add text label
                    plt.text(centroid_x, centroid_y, region_name, 
                             color='white', fontsize=8, ha='center', va='center',
                             bbox=dict(facecolor='black', alpha=0.5, pad=1))
        
        plt.title('Environment Regions')
        plt.legend(loc='upper right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            plt.close()
        else:
            plt.show()
    
    def create_dataset(self, num_maps=5, pairs_per_map=2):
        """Create a complete dataset with maps, regions, and start-goal pairs"""
        dataset = {
            "maps": [],
            "scenarios": []
        }
        
        # Define map configurations
        map_configs = [
            {"type": "office", "id": 0, "seed": 42},
            {"type": "office", "id": 1, "seed": 43},
            {"type": "warehouse", "id": 0, "seed": 44},
            {"type": "warehouse", "id": 1, "seed": 45},
            {"type": "outdoor", "id": 0, "seed": 46}
        ]
        
        scenario_id = 0
        
        for config in map_configs[:num_maps]:
            map_type = config["type"]
            map_id = config["id"]
            seed = config["seed"]
            map_name = f"{map_type}_{map_id}"
            
            print(f"Creating map: {map_name}")
            
            # Create base map
            base_grid = self.create_base_map(map_type, seed=seed)
            
            # Add clutter
            cluttered_grid = self.add_clutter(base_grid, seed=seed+100)
            
            # Generate start-goal pairs
            pairs = self.generate_start_goal_pairs(cluttered_grid, num_pairs=pairs_per_map, seed=seed+200)
            
            # Identify regions
            regions = self.identify_regions(cluttered_grid)
            
            # Save map
            map_path = f"{self.output_dir}/maps/{map_name}.npy"
            np.save(map_path, cluttered_grid)
            
            # Save regions
            region_paths = {}
            for region_name, region_grid in regions.items():
                region_path = f"{self.output_dir}/maps/{map_name}_{region_name}.npy"
                np.save(region_path, region_grid)
                region_paths[region_name] = region_path
            
            # Add map to dataset
            map_info = {
                "id": map_name,
                "type": map_type,
                "grid_path": map_path,
                "regions": region_paths
            }
            
            dataset["maps"].append(map_info)
            
            # Create scenarios for each start-goal pair
            for pair_idx, (start, goal) in enumerate(pairs):
                scenario_name = f"scenario_{scenario_id}"
                
                # Create visualization for this specific start-goal pair
                vis_path = f"{self.output_dir}/visualizations/{scenario_name}.png"
                self.visualize_map_with_regions_and_path(
                    cluttered_grid, regions, start, goal, save_path=vis_path
                )
                
                # Create scenario
                scenario = {
                    "id": scenario_id,
                    "map_id": map_name,
                    "start": start,
                    "goal": goal,
                    "vis_path": vis_path,
                    "instruction": "",  # To be filled during annotation
                    "constraint_set": {
                        "proximity": {},
                        "avoidance": {},
                        "preference": {}
                    },
                    "has_annotation": False
                }
                
                # Initialize empty annotations for each constraint type
                for constraint_type in ["proximity", "avoidance", "preference"]:
                    annotation_path = f"{self.output_dir}/annotations/{scenario_id}_{constraint_type}.npy"
                    empty_annotation = np.zeros_like(cluttered_grid)
                    np.save(annotation_path, empty_annotation)
                
                dataset["scenarios"].append(scenario)
                scenario_id += 1
        
        # Save dataset index
        with open(f"{self.output_dir}/dataset_index.json", "w") as f:
            json.dump(dataset, f, indent=2)
        
        print(f"Created {len(dataset['maps'])} maps with {len(dataset['scenarios'])} scenarios")
        
        return dataset