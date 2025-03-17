import numpy as np
import matplotlib.pyplot as plt

class GridEnvironment:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width))  # 0: free, 1: obstacle
        self.regions = {}  # Named regions for semantic mapping
    
    def add_obstacle(self, x, y):
        self.grid[y, x] = 1
    
    def add_obstacles_from_array(self, obstacle_array):
        """Add obstacles from a 2D binary array"""
        assert obstacle_array.shape == self.grid.shape, "Array dimensions must match grid"
        self.grid = obstacle_array
    
    def define_region(self, name, coords):
        """Define a named region in the grid (e.g., 'center', 'north_corridor')"""
        self.regions[name] = coords
    
    def define_region_from_bounds(self, name, x_min, y_min, x_max, y_max):
        """Define a rectangular region from bounds"""
        coords = [(x, y) for x in range(x_min, x_max+1) 
                         for y in range(y_min, y_max+1)]
        self.regions[name] = coords
    
    def is_valid(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height and self.grid[y, x] == 0
    
    def get_neighbors(self, x, y):
        """Get valid neighboring cells"""
        neighbors = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # 4-connected grid
            nx, ny = x + dx, y + dy
            if self.is_valid(nx, ny):
                neighbors.append((nx, ny))
        return neighbors
    
    def distance_to_region(self, x, y, region_name):
        """Calculate the minimum distance from a point to a region"""
        if region_name not in self.regions:
            return float('inf')
        
        min_dist = float('inf')
        for rx, ry in self.regions[region_name]:
            dist = abs(x - rx) + abs(y - ry)  # Manhattan distance
            min_dist = min(min_dist, dist)
        
        return min_dist
    
    def is_in_region(self, x, y, region_name):
        """Check if a point is in a named region"""
        return (x, y) in self.regions.get(region_name, set())
    
    def to_dict(self):
        """Convert environment to dictionary for LLM context"""
        return {
            "width": self.width,
            "height": self.height,
            "obstacles": [(x, y) for y in range(self.height) 
                                 for x in range(self.width) if self.grid[y, x] == 1],
            "regions": self.regions
        }
    
    def create_hierarchical_regions(self, region_size=10):
        """Create hierarchical regions for large grids"""
        # Create quadrants
        w, h = self.width, self.height
        self.define_region_from_bounds("northwest", 0, 0, w//2-1, h//2-1)
        self.define_region_from_bounds("northeast", w//2, 0, w-1, h//2-1)
        self.define_region_from_bounds("southwest", 0, h//2, w//2-1, h-1)
        self.define_region_from_bounds("southeast", w//2, h//2, w-1, h-1)
        
        # Create center region
        center_w, center_h = w//4, h//4
        self.define_region_from_bounds("center", 
                                      w//2 - center_w, h//2 - center_h,
                                      w//2 + center_w, h//2 + center_h)
        
        # Create border region
        self.regions["border"] = [(x, y) for x in range(w) for y in range(h)
                                if x == 0 or y == 0 or x == w-1 or y == h-1]

    def identify_borders(self):
        """Identify border regions in the environment"""
        border_points = []
        
        # Check each valid cell
        for x in range(self.width):
            for y in range(self.height):
                if not self.is_valid(x, y):
                    continue
                    
                # Check if it's adjacent to an obstacle
                neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
                for nx, ny in neighbors:
                    if not (0 <= nx < self.width and 0 <= ny < self.height) or not self.is_valid(nx, ny):
                        border_points.append((x, y))
                        break
        
        # Add to regions
        self.regions["border"] = border_points
        
        # Further classify borders
        self._classify_borders()
        
        return border_points

    def _classify_borders(self):
        """Classify borders into more specific regions"""
        if "border" not in self.regions:
            return
        
        # Initialize border subregions
        self.regions["border_top"] = []
        self.regions["border_bottom"] = []
        self.regions["border_left"] = []
        self.regions["border_right"] = []
        self.regions["corner_topleft"] = []
        self.regions["corner_topright"] = []
        self.regions["corner_bottomleft"] = []
        self.regions["corner_bottomright"] = []
        
        # Define border thresholds
        top_threshold = self.height * 0.2
        bottom_threshold = self.height * 0.8
        left_threshold = self.width * 0.2
        right_threshold = self.width * 0.8
        
        # Classify each border point
        for x, y in self.regions["border"]:
            # Check corners first
            if y < top_threshold and x < left_threshold:
                self.regions["corner_topleft"].append((x, y))
            elif y < top_threshold and x > right_threshold:
                self.regions["corner_topright"].append((x, y))
            elif y > bottom_threshold and x < left_threshold:
                self.regions["corner_bottomleft"].append((x, y))
            elif y > bottom_threshold and x > right_threshold:
                self.regions["corner_bottomright"].append((x, y))
            # Then check borders
            elif y < top_threshold:
                self.regions["border_top"].append((x, y))
            elif y > bottom_threshold:
                self.regions["border_bottom"].append((x, y))
            elif x < left_threshold:
                self.regions["border_left"].append((x, y))
            elif x > right_threshold:
                self.regions["border_right"].append((x, y))