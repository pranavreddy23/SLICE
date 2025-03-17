import numpy as np
import cv2
from skimage.segmentation import watershed, slic, quickshift
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
from typing import Dict, List, Tuple

class RegionSegmenter:
    """Advanced region segmentation for grid environments"""
    
    def __init__(self, method="watershed"):
        """Initialize with segmentation method"""
        self.method = method
        
    def segment_regions(self, environment):
        """Segment the environment into meaningful regions"""
        # Convert grid to image format
        grid = environment.grid.copy()
        
        # Create a binary image (0 for free space, 1 for obstacles)
        binary_image = (grid > 0).astype(np.uint8)
        
        # Apply distance transform
        distance = ndi.distance_transform_edt(1 - binary_image)
        
        if self.method == "watershed":
            # Apply watershed segmentation
            regions = self._watershed_segmentation(distance, binary_image)
        elif self.method == "slic":
            # Apply SLIC superpixel segmentation
            regions = self._slic_segmentation(distance, binary_image)
        elif self.method == "quickshift":
            # Apply quickshift segmentation
            regions = self._quickshift_segmentation(distance, binary_image)
        else:
            # Default to watershed
            regions = self._watershed_segmentation(distance, binary_image)
        
        # Convert segmentation to region dictionary
        region_dict = self._create_region_dict(regions, binary_image)
        
        return region_dict
    
    def _watershed_segmentation(self, distance, binary_image):
        """Apply watershed segmentation"""
        # Find local maxima
        coordinates = peak_local_max(distance, min_distance=10, labels=1-binary_image)
        
        # Create a mask of local maxima
        local_max = np.zeros_like(distance, dtype=bool)
        local_max[tuple(coordinates.T)] = True
        
        # Create markers for watershed
        markers = ndi.label(local_max)[0]
        
        # Apply watershed
        regions = watershed(-distance, markers, mask=1-binary_image)
        
        return regions
    
    def _slic_segmentation(self, distance, binary_image):
        """Apply SLIC superpixel segmentation"""
        # Create a 3-channel image for SLIC
        image = np.dstack([distance, distance, distance])
        
        # Apply SLIC
        regions = slic(image, n_segments=20, compactness=10, sigma=1, 
                     mask=1-binary_image)
        
        return regions
    
    def _quickshift_segmentation(self, distance, binary_image):
        """Apply quickshift segmentation"""
        # Create a 3-channel image for quickshift
        image = np.dstack([distance, distance, distance])
        
        # Apply quickshift
        regions = quickshift(image, kernel_size=3, max_dist=6, ratio=0.5)
        
        # Apply mask
        regions = regions * (1 - binary_image)
        
        return regions
    
    def _create_region_dict(self, regions, binary_image):
        """Convert segmentation to region dictionary"""
        region_dict = {}
        
        # Get unique region labels
        unique_regions = np.unique(regions)
        
        # Skip background (0)
        for region_id in unique_regions:
            if region_id == 0:
                continue
                
            # Get points in this region
            region_mask = (regions == region_id)
            points = []
            
            for y in range(region_mask.shape[0]):
                for x in range(region_mask.shape[1]):
                    if region_mask[y, x] and binary_image[y, x] == 0:  # Only include free space
                        points.append((x, y))
            
            if points:
                # Generate a meaningful name for the region
                region_name = self._generate_region_name(points, region_mask.shape)
                region_dict[region_name] = points
        
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