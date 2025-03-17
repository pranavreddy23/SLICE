# annotation_tool.py
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from matplotlib.widgets import Button, RadioButtons, TextBox, CheckButtons

class AnnotationTool:
    def __init__(self, dataset_path="./dataset/dataset_index.json"):
        with open(dataset_path, "r") as f:
            self.dataset = json.load(f)
        
        self.output_dir = os.path.dirname(dataset_path)
        os.makedirs(f"{self.output_dir}/annotations", exist_ok=True)
        
        self.current_scenario_idx = 0
        self.current_constraint_type = "proximity"  # proximity, avoidance, or preference
        self.brush_size = 3
        self.selected_regions = {}  # Dictionary to store selected regions for each constraint type
        
        self.load_current_scenario()
        self.setup_ui()
    
    def load_current_scenario(self):
        """Load the current scenario and associated map data"""
        self.scenario = self.dataset["scenarios"][self.current_scenario_idx]
        
        # Find the map for this scenario
        map_id = self.scenario["map_id"]
        map_info = None
        for m in self.dataset["maps"]:
            if m["id"] == map_id:
                map_info = m
                break
        
        if not map_info:
            raise ValueError(f"Map {map_id} not found in dataset")
        
        # Load map and regions
        self.grid = np.load(map_info["grid_path"])
        self.regions = {}
        for region_name, region_path in map_info["regions"].items():
            self.regions[region_name] = np.load(region_path)
        
        # Get start and goal
        self.start = self.scenario["start"]
        self.goal = self.scenario["goal"]
        
        # Load or initialize instruction
        self.instruction = self.scenario.get("instruction", "")
        
        # Load or initialize constraint set
        self.constraint_set = self.scenario.get("constraint_set", {
            "proximity": {},
            "avoidance": {},
            "preference": {}
        })
        
        # Initialize selected regions for each constraint type
        self.selected_regions = {
            "proximity": set(self.constraint_set.get("proximity", {}).keys()),
            "avoidance": set(self.constraint_set.get("avoidance", {}).keys()),
            "preference": set(self.constraint_set.get("preference", {}).keys())
        }
        
        # Load or initialize annotations
        self.annotations = {}
        for constraint_type in ["proximity", "avoidance", "preference"]:
            annotation_path = f"{self.output_dir}/annotations/{self.scenario['id']}_{constraint_type}.npy"
            if os.path.exists(annotation_path):
                self.annotations[constraint_type] = np.load(annotation_path)
            else:
                self.annotations[constraint_type] = np.zeros_like(self.grid)
    
    def setup_ui(self):
        """Set up the user interface"""
        self.fig = plt.figure(figsize=(15, 10))
        
        # Main plot for annotation
        self.ax_main = plt.subplot2grid((3, 4), (0, 0), colspan=2, rowspan=2)
        
        # Region visualization plots
        self.ax_regions = plt.subplot2grid((3, 4), (0, 2), colspan=2, rowspan=2)
        
        # Instruction and controls
        self.ax_instruction = plt.subplot2grid((3, 4), (2, 0), colspan=2)
        self.ax_controls = plt.subplot2grid((3, 4), (2, 2), colspan=2)
        
        # Remove axis ticks
        for ax in [self.ax_instruction, self.ax_controls]:
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Add text box for instruction
        self.ax_instruction.set_title("Instruction")
        self.text_box = TextBox(self.ax_instruction, "", initial=self.instruction)
        self.text_box.on_submit(self.update_instruction)
        
        # Add buttons for navigation and saving
        button_width = 0.15
        button_height = 0.05
        button_spacing = 0.02
        
        # Navigation buttons
        self.btn_prev = Button(plt.axes([0.05, 0.05, button_width, button_height]), 'Previous')
        self.btn_prev.on_clicked(self.prev_scenario)
        
        self.btn_next = Button(plt.axes([0.05 + button_width + button_spacing, 0.05, 
                                        button_width, button_height]), 'Next')
        self.btn_next.on_clicked(self.next_scenario)
        
        # Save button
        self.btn_save = Button(plt.axes([0.05 + 2 * (button_width + button_spacing), 0.05, 
                                        button_width, button_height]), 'Save')
        self.btn_save.on_clicked(self.save_annotation)
        
        # Clear button
        self.btn_clear = Button(plt.axes([0.05 + 3 * (button_width + button_spacing), 0.05, 
                                         button_width, button_height]), 'Clear')
        self.btn_clear.on_clicked(self.clear_annotation)
        
        # Constraint type selector
        self.radio_constraint = RadioButtons(
            plt.axes([0.7, 0.05, 0.25, 0.15]), 
            ('proximity', 'avoidance', 'preference'),
            active=0
        )
        self.radio_constraint.on_clicked(self.set_constraint_type)
        
        # Region list with checkboxes
        self.ax_regions_list = plt.axes([0.7, 0.25, 0.25, 0.6])
        self.ax_regions_list.set_title("Regions")
        self.ax_regions_list.set_xticks([])
        self.ax_regions_list.set_yticks([])
        
        # Create list of region names (excluding "segmentation")
        self.region_names = [name for name in self.regions.keys() if name != "segmentation"]
        
        # Create checkboxes for regions
        self.checkboxes = CheckButtons(
            self.ax_regions_list, 
            self.region_names,
            [name in self.selected_regions[self.current_constraint_type] for name in self.region_names]
        )
        self.checkboxes.on_clicked(self.toggle_region)
        
        # Add text for constraint set
        self.regions_list_text = self.ax_controls.text(
            0.05, 0.5, self.get_regions_list_text(),
            verticalalignment='center'
        )
        
        # Connect mouse events
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        
        self.is_drawing = False
        
        # Update display
        self.update_display()
    
    def update_display(self):
        """Update the display with current map, regions, and annotations"""
        # Clear axes
        self.ax_main.clear()
        self.ax_regions.clear()
        
        # Show grid on main plot
        self.ax_main.imshow(self.grid, cmap='binary')
        
        # Show current annotation as overlay
        current_annotation = self.annotations[self.current_constraint_type]
        self.ax_main.imshow(current_annotation, cmap='plasma', alpha=0.5)
        
        # Show start and goal
        self.ax_main.plot(self.start[0], self.start[1], 'go', markersize=10, label='Start')
        self.ax_main.plot(self.goal[0], self.goal[1], 'ro', markersize=10, label='Goal')
        
        # Show segmentation map
        if "segmentation" in self.regions:
            # Create a colormap with distinct colors
            from matplotlib.colors import ListedColormap
            import matplotlib.colors as mcolors
            
            # Get a list of distinct colors
            colors = list(mcolors.TABLEAU_COLORS.values())
            # Add more colors if needed
            colors.extend(list(mcolors.CSS4_COLORS.values())[:20])
            
            # Create a colormap with enough colors for all regions
            segmentation = self.regions["segmentation"]
            num_regions = np.max(segmentation)
            if num_regions > 0:
                # Add black for obstacles (value 0)
                region_colors = ['black'] + colors[:num_regions]
                region_cmap = ListedColormap(region_colors)
            else:
                region_cmap = 'viridis'
            
            # Plot the segmentation
            self.ax_regions.imshow(segmentation, cmap=region_cmap, interpolation='nearest')
            
            # Add start and goal markers
            self.ax_regions.plot(self.start[0], self.start[1], 'go', markersize=10, label='Start')
            self.ax_regions.plot(self.goal[0], self.goal[1], 'ro', markersize=10, label='Goal')
            
            # Add region labels
            for region_name in self.region_names:
                region_mask = self.regions[region_name]
                # Find centroid of region for label placement
                y_indices, x_indices = np.where(region_mask)
                if len(y_indices) > 0:
                    centroid_y = int(np.mean(y_indices))
                    centroid_x = int(np.mean(x_indices))
                    
                    # Add text label
                    self.ax_regions.text(centroid_x, centroid_y, region_name, 
                                     color='white', fontsize=8, ha='center', va='center',
                                     bbox=dict(facecolor='black', alpha=0.5, pad=1))
        
        # Set titles
        scenario_id = self.scenario["id"]
        map_id = self.scenario["map_id"]
        self.ax_main.set_title(f"Scenario {scenario_id} (Map: {map_id}) - {self.current_constraint_type.capitalize()}")
        self.ax_regions.set_title("Region Segmentation")
        
        # Add legend
        self.ax_main.legend()
        
        # Update instruction text box
        if hasattr(self, 'text_box'):
            self.text_box.set_val(self.instruction)
        
        # Update regions list text
        if hasattr(self, 'regions_list_text'):
            self.regions_list_text.set_text(self.get_regions_list_text())
        
        # Update checkboxes - recreate them with current state
        if hasattr(self, 'checkboxes'):
            # Remove old checkboxes
            self.ax_regions_list.clear()
            self.ax_regions_list.set_title("Regions")
            self.ax_regions_list.set_xticks([])
            self.ax_regions_list.set_yticks([])
            
            # Create new checkboxes with current state
            self.checkboxes = CheckButtons(
                self.ax_regions_list, 
                self.region_names,
                [name in self.selected_regions[self.current_constraint_type] for name in self.region_names]
            )
            self.checkboxes.on_clicked(self.toggle_region)
        
        self.fig.canvas.draw_idle()
    
    def get_regions_list_text(self):
        """Get text representation of the constraint set"""
        text = f"Constraint Set:\n\n"
        
        for constraint_type in ["proximity", "avoidance", "preference"]:
            text += f"{constraint_type.capitalize()}:\n"
            if constraint_type in self.constraint_set and self.constraint_set[constraint_type]:
                for region, weight in self.constraint_set[constraint_type].items():
                    text += f"  - {region} ({weight})\n"
            else:
                text += "  (none)\n"
            text += "\n"
        
        return text
    
    def update_instruction(self, text):
        """Update the instruction text"""
        self.instruction = text
        self.scenario["instruction"] = text
    
    def set_constraint_type(self, label):
        """Set the current constraint type"""
        self.current_constraint_type = label
        self.update_display()
    
    def toggle_region(self, label):
        """Toggle a region in the current constraint type"""
        if label in self.selected_regions[self.current_constraint_type]:
            self.selected_regions[self.current_constraint_type].remove(label)
            if label in self.constraint_set[self.current_constraint_type]:
                del self.constraint_set[self.current_constraint_type][label]
        else:
            self.selected_regions[self.current_constraint_type].add(label)
            # Default weight is 1.0
            self.constraint_set[self.current_constraint_type][label] = 1.0
            
            # Update annotation based on selected region
            region_mask = self.regions[label]
            self.annotations[self.current_constraint_type][region_mask] = 1
        
        self.update_display()
    
    def on_click(self, event):
        """Handle mouse click event"""
        if event.inaxes == self.ax_main and event.button == 1:  # Left click
            self.is_drawing = True
            self.draw_annotation(event.xdata, event.ydata)
    
    def on_motion(self, event):
        """Handle mouse motion event"""
        if self.is_drawing and event.inaxes == self.ax_main:
            self.draw_annotation(event.xdata, event.ydata)
    
    def on_release(self, event):
        """Handle mouse release event"""
        if event.button == 1:  # Left click
            self.is_drawing = False
    
    def draw_annotation(self, x, y):
        """Draw annotation at the given position"""
        if x is None or y is None:
            return
        
        x, y = int(x), int(y)
        
        # Check bounds
        if x < 0 or x >= self.grid.shape[1] or y < 0 or y >= self.grid.shape[0]:
            return
        
        # Don't annotate obstacles
        if self.grid[y, x] == 1:
            return
        
        # Draw a circle around the point
        y_indices, x_indices = np.ogrid[-self.brush_size:self.brush_size+1, -self.brush_size:self.brush_size+1]
        mask = x_indices**2 + y_indices**2 <= self.brush_size**2
        
        # Apply mask within bounds
        for dy in range(-self.brush_size, self.brush_size+1):
            for dx in range(-self.brush_size, self.brush_size+1):
                if dy**2 + dx**2 <= self.brush_size**2:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < self.grid.shape[0] and 0 <= nx < self.grid.shape[1]:
                        if self.grid[ny, nx] == 0:  # Only annotate free space
                            self.annotations[self.current_constraint_type][ny, nx] = 1
        
        self.update_display()
    
    def prev_scenario(self, event):
        """Go to previous scenario"""
        self.save_annotation(None)
        self.current_scenario_idx = (self.current_scenario_idx - 1) % len(self.dataset["scenarios"])
        self.load_current_scenario()
        self.update_display()
    
    def next_scenario(self, event):
        """Go to next scenario"""
        self.save_annotation(None)
        self.current_scenario_idx = (self.current_scenario_idx + 1) % len(self.dataset["scenarios"])
        self.load_current_scenario()
        self.update_display()
    
    def save_annotation(self, event):
        """Save annotations for the current scenario"""
        # Save annotations for each constraint type
        for constraint_type in ["proximity", "avoidance", "preference"]:
            annotation_path = f"{self.output_dir}/annotations/{self.scenario['id']}_{constraint_type}.npy"
            np.save(annotation_path, self.annotations[constraint_type])
        
        # Update scenario info
        self.scenario["instruction"] = self.instruction
        self.scenario["constraint_set"] = self.constraint_set
        self.scenario["has_annotation"] = True
        
        # Save updated dataset
        with open(f"{self.output_dir}/dataset_index.json", "w") as f:
            json.dump(self.dataset, f, indent=2)
        
        print(f"Saved annotations for scenario {self.scenario['id']}")
    
    def clear_annotation(self, event):
        """Clear annotations for the current constraint type"""
        self.annotations[self.current_constraint_type] = np.zeros_like(self.grid)
        self.selected_regions[self.current_constraint_type] = set()
        self.constraint_set[self.current_constraint_type] = {}
        self.update_display()