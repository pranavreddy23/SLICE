import numpy as np
import random
from core.env import GridEnvironment
from core.cfgen import a_star_search, uniform_cost_function
from scipy import ndimage as ndi

class ChallengeMapGenerator:
    """Generates challenging environments for testing"""
    
    @staticmethod
    def generate_maze(width, height, complexity=0.5, density=0.5):
        """Generate a maze-like environment with a guaranteed path"""
        # Initialize grid with obstacles
        shape = (height, width)
        grid = np.zeros(shape, dtype=np.int8)
        
        # Fill borders
        grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = 1
        
        # Create maze pattern
        complexity = int(complexity * (5 * (shape[0] + shape[1])))
        density = int(density * ((shape[0] // 2) * (shape[1] // 2)))
        
        # Make random walls
        for _ in range(density):
            # Pick a random point with even coordinates
            x = random.randint(0, (shape[1] // 2) - 1) * 2
            y = random.randint(0, (shape[0] // 2) - 1) * 2
            
            # Ensure we're within bounds
            x = min(x, shape[1] - 2)
            y = min(y, shape[0] - 2)
            
            grid[y, x] = 1
            
            for _ in range(complexity):
                neighbors = []
                if x > 1:
                    neighbors.append((y, x - 2))
                if x < shape[1] - 3:
                    neighbors.append((y, x + 2))
                if y > 1:
                    neighbors.append((y - 2, x))
                if y < shape[0] - 3:
                    neighbors.append((y + 2, x))
                
                if not neighbors:
                    break
                    
                y_next, x_next = neighbors[random.randint(0, len(neighbors) - 1)]
                
                # Double-check bounds
                if 0 <= y_next < shape[0] and 0 <= x_next < shape[1]:
                    if grid[y_next, x_next] == 0:
                        # Calculate the wall coordinates between current and next cell
                        wall_y = y + (y_next - y) // 2
                        wall_x = x + (x_next - x) // 2
                        
                        # Ensure wall coordinates are within bounds
                        if 0 <= wall_y < shape[0] and 0 <= wall_x < shape[1]:
                            grid[wall_y, wall_x] = 1
                            grid[y_next, x_next] = 1
                            x, y = x_next, y_next
        
        # Ensure start and goal areas are clear
        start_area = 3
        grid[:start_area, :start_area] = 0
        grid[-start_area:, -start_area:] = 0
        
        # Ensure a path exists from start to goal
        start = (1, 1)
        goal = (width-2, height-2)
        
        # Create a temporary environment to check path
        temp_env = GridEnvironment(width, height)
        temp_env.add_obstacles_from_array(grid)
        
        # Check if a path exists
        uniform_cost = uniform_cost_function(temp_env)
        path, _ = a_star_search(temp_env, start, goal, uniform_cost)
        
        # If no path exists, create a simple path
        if not path:
            print("No path found in generated maze. Creating a simple path...")
            # Create a simple path by clearing cells
            x, y = start
            while x < goal[0] or y < goal[1]:
                if x < goal[0]:
                    x += 1
                elif y < goal[1]:
                    y += 1
                grid[y, x] = 0  # Clear the cell
        
        # Create environment
        env = GridEnvironment(width, height)
        env.add_obstacles_from_array(grid)
        
        return env
    
    @staticmethod
    def generate_cluttered(width, height, obstacle_density=0.3, 
                                      min_obstacle_size=1, max_obstacle_size=5):
        """Generate a cluttered environment with various obstacle sizes"""
        grid = np.zeros((height, width), dtype=np.int8)
        
        # Add random obstacles
        num_obstacles = int(width * height * obstacle_density / 
                           ((min_obstacle_size + max_obstacle_size) / 2) ** 2)
        
        for _ in range(num_obstacles):
            # Random obstacle size
            obs_width = random.randint(min_obstacle_size, max_obstacle_size)
            obs_height = random.randint(min_obstacle_size, max_obstacle_size)
            
            # Random position
            x = random.randint(0, width - obs_width - 1)
            y = random.randint(0, height - obs_height - 1)
            
            # Add obstacle
            grid[y:y+obs_height, x:x+obs_width] = 1
        
        # Ensure start and goal areas are clear
        start_area = 3
        grid[:start_area, :start_area] = 0
        grid[-start_area:, -start_area:] = 0
        
        # Create environment
        env = GridEnvironment(width, height)
        env.add_obstacles_from_array(grid)
        
        return env
    
    @staticmethod
    def generate_narrow_passages(width, height):
        """Generate an environment with narrow passages"""
        # Initialize grid with free space
        grid = np.zeros((height, width), dtype=np.uint8)
        
        # Add outer walls
        grid[0, :] = 1
        grid[height-1, :] = 1
        grid[:, 0] = 1
        grid[:, width-1] = 1
        
        # Add horizontal walls with narrow passages
        num_h_walls = height // 10
        for i in range(1, num_h_walls + 1):
            y = i * (height // (num_h_walls + 1))
            grid[y, :] = 1
            
            # Add 1-2 narrow passages
            num_passages = np.random.randint(1, 3)
            for _ in range(num_passages):
                passage_width = np.random.randint(1, 3)
                passage_x = np.random.randint(1, width - passage_width - 1)
                grid[y, passage_x:passage_x+passage_width] = 0
        
        # Add vertical walls with narrow passages
        num_v_walls = width // 10
        for i in range(1, num_v_walls + 1):
            x = i * (width // (num_v_walls + 1))
            grid[:, x] = 1
            
            # Add 1-2 narrow passages
            num_passages = np.random.randint(1, 3)
            for _ in range(num_passages):
                passage_height = np.random.randint(1, 3)
                passage_y = np.random.randint(1, height - passage_height - 1)
                grid[passage_y:passage_y+passage_height, x] = 0
        
        # Create environment
        env = GridEnvironment(width, height)
        env.add_obstacles_from_array(grid)
        
        return env
    
    @staticmethod
    def generate_office_like(width, height, room_density=0.6, door_width=2):
        """Generate an office-like environment with rooms and corridors"""
        grid = np.ones((height, width), dtype=np.int8)  # Start with all walls
        
        # Create main corridors
        corridor_width = 3
        h_corridor_y = height // 2
        v_corridor_x = width // 2
        
        # Horizontal corridor
        grid[h_corridor_y-corridor_width//2:h_corridor_y+corridor_width//2+1, :] = 0
        
        # Vertical corridor
        grid[:, v_corridor_x-corridor_width//2:v_corridor_x+corridor_width//2+1] = 0
        
        # Create rooms
        room_min_size = 6
        room_max_size = 12
        
        # Divide into quadrants
        quadrants = [
            (0, 0, v_corridor_x-corridor_width//2, h_corridor_y-corridor_width//2),
            (v_corridor_x+corridor_width//2, 0, width, h_corridor_y-corridor_width//2),
            (0, h_corridor_y+corridor_width//2, v_corridor_x-corridor_width//2, height),
            (v_corridor_x+corridor_width//2, h_corridor_y+corridor_width//2, width, height)
        ]
        
        for q_x1, q_y1, q_x2, q_y2 in quadrants:
            # Skip if quadrant is too small
            if q_x2 - q_x1 < room_min_size or q_y2 - q_y1 < room_min_size:
                continue
                
            # Number of rooms in this quadrant
            q_width = q_x2 - q_x1
            q_height = q_y2 - q_y1
            q_area = q_width * q_height
            
            num_rooms = int(q_area * room_density / (room_min_size * room_min_size))
            
            for _ in range(num_rooms):
                # Room size
                room_width = random.randint(room_min_size, min(room_max_size, q_width-2))
                room_height = random.randint(room_min_size, min(room_max_size, q_height-2))
                
                # Room position
                room_x = random.randint(q_x1+1, q_x2-room_width-1)
                room_y = random.randint(q_y1+1, q_y2-room_height-1)
                
                # Create room (empty space)
                grid[room_y:room_y+room_height, room_x:room_x+room_width] = 0
                
                # Add walls back
                grid[room_y, room_x:room_x+room_width] = 1
                grid[room_y+room_height-1, room_x:room_x+room_width] = 1
                grid[room_y:room_y+room_height, room_x] = 1
                grid[room_y:room_y+room_height, room_x+room_width-1] = 1
                
                # Add door
                wall_options = [
                    (room_y, room_x + random.randint(1, room_width-2), 0, 1),  # Top wall
                    (room_y+room_height-1, room_x + random.randint(1, room_width-2), 0, 1),  # Bottom wall
                    (room_y + random.randint(1, room_height-2), room_x, 1, 0),  # Left wall
                    (room_y + random.randint(1, room_height-2), room_x+room_width-1, 1, 0)   # Right wall
                ]
                
                door_y, door_x, dy, dx = random.choice(wall_options)
                
                # Create door
                for i in range(door_width):
                    door_pos_y = door_y + i * dy
                    door_pos_x = door_x + i * dx
                    if 0 <= door_pos_y < height and 0 <= door_pos_x < width:
                        grid[door_pos_y, door_pos_x] = 0
        
        # Invert grid (0=free, 1=obstacle)
        env = GridEnvironment(width, height)
        env.add_obstacles_from_array(grid)
        
        return env
    
    @staticmethod
    def generate_office(width, height):
        """Generate an office-like environment with rooms and corridors"""
        # Initialize grid with free space
        grid = np.zeros((height, width), dtype=np.uint8)
        
        # Add outer walls
        grid[0, :] = 1
        grid[height-1, :] = 1
        grid[:, 0] = 1
        grid[:, width-1] = 1
        
        # Add rooms
        num_rooms = max(3, width // 15)
        min_room_size = 5
        max_room_size = 10
        
        for _ in range(num_rooms):
            # Random room size
            room_width = np.random.randint(min_room_size, max_room_size)
            room_height = np.random.randint(min_room_size, max_room_size)
            
            # Random room position
            x = np.random.randint(2, width - room_width - 2)
            y = np.random.randint(2, height - room_height - 2)
            
            # Add room walls
            grid[y, x:x+room_width] = 1
            grid[y+room_height, x:x+room_width] = 1
            grid[y:y+room_height, x] = 1
            grid[y:y+room_height, x+room_width] = 1
            
            # Add door
            door_wall = np.random.randint(0, 4)
            if door_wall == 0:  # Top wall
                door_pos = np.random.randint(x+1, x+room_width-1)
                grid[y, door_pos] = 0
            elif door_wall == 1:  # Bottom wall
                door_pos = np.random.randint(x+1, x+room_width-1)
                grid[y+room_height, door_pos] = 0
            elif door_wall == 2:  # Left wall
                door_pos = np.random.randint(y+1, y+room_height-1)
                grid[door_pos, x] = 0
            else:  # Right wall
                door_pos = np.random.randint(y+1, y+room_height-1)
                grid[door_pos, x+room_width] = 0
        
        # Add corridors
        num_corridors = max(2, width // 20)
        corridor_width = 1
        
        for _ in range(num_corridors):
            # Horizontal corridor
            y = np.random.randint(5, height - 5)
            grid[y-corridor_width:y+corridor_width+1, :] = 0
            
            # Vertical corridor
            x = np.random.randint(5, width - 5)
            grid[:, x-corridor_width:x+corridor_width+1] = 0
        
        # Create environment
        env = GridEnvironment(width, height)
        env.add_obstacles_from_array(grid)
        
        return env
    
    @staticmethod
    def generate_challenge_set():
        """Generate a set of challenging environments"""
        environments = {
            "maze_small": ChallengeMapGenerator.generate_maze(30, 30, complexity=0.7, density=0.7),
            "maze_large": ChallengeMapGenerator.generate_maze(50, 50, complexity=0.8, density=0.75),
            "cluttered_small": ChallengeMapGenerator.generate_cluttered_environment(30, 30, obstacle_density=0.3),
            "cluttered_large": ChallengeMapGenerator.generate_cluttered_environment(50, 50, obstacle_density=0.4),
            "narrow_passages": ChallengeMapGenerator.generate_narrow_passages(40, 40, num_passages=5, passage_width=2),
            "office_small": ChallengeMapGenerator.generate_office_like(40, 40, room_density=0.6),
            "office_large": ChallengeMapGenerator.generate_office_like(60, 60, room_density=0.7)
        }
        
        return environments

def generate_random_positions(environment, min_distance=20):
    """Generate random start and goal positions with minimum distance"""
    width, height = environment.width, environment.height
    valid_positions = []
    
    # Find all valid positions
    for x in range(width):
        for y in range(height):
            if environment.is_valid(x, y):
                valid_positions.append((x, y))
    
    if len(valid_positions) < 2:
        return None, None  # Not enough valid positions
    
    # Try to find positions with minimum distance
    max_attempts = 100
    for _ in range(max_attempts):
        # Randomly select two positions
        start_idx = np.random.randint(0, len(valid_positions))
        goal_idx = np.random.randint(0, len(valid_positions))
        
        if start_idx == goal_idx:
            continue
            
        start = valid_positions[start_idx]
        goal = valid_positions[goal_idx]
        
        # Calculate Manhattan distance
        distance = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
        
        if distance >= min_distance:
            return start, goal
    
    # If we couldn't find positions with minimum distance, return any two valid positions
    if len(valid_positions) >= 2:
        indices = np.random.choice(len(valid_positions), 2, replace=False)
        return valid_positions[indices[0]], valid_positions[indices[1]]
    else:
        return None, None