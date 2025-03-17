import base64
from typing import Dict, List, Any, Optional, Tuple
from pydantic import BaseModel, Field
from groq import Groq
import io
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import json
from scipy import ndimage
from skimage import measure, segmentation, morphology

class ConstraintSet(BaseModel):
    """Pydantic model for structured constraint output"""
    proximity: Dict[str, float] = Field(default_factory=dict, description="Regions to stay close to")
    avoidance: Dict[str, float] = Field(default_factory=dict, description="Regions to avoid")
    preference: Dict[str, float] = Field(default_factory=dict, description="Regions to prefer")

class ConstraintExtractor:
    def __init__(self, model="deepseek-r1-distill-llama-70b", api_key=""):
        """Initialize with Groq client for Llama 3.1"""
        self.client = Groq(api_key=api_key)
        self.model = model
    
    def extract_constraints(self, instruction: str, environment, 
                           include_visualization: bool = False) -> ConstraintSet:
        """Extract formal constraints from natural language instruction"""
        env_context = self._create_environment_context(environment)
        
        # Create visualization if requested - set to False by default now
        base64_image = None
        if include_visualization:
            base64_image = self._create_environment_visualization(environment)
        
        # Construct prompt
        prompt = self._construct_prompt(instruction, env_context)
        
        # Call Groq API with or without image
        try:
            if base64_image:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a specialized AI for robotic path planning. Extract formal constraints from instructions."
                        },
                        {
                            "role": "user",
                            "content": f"{prompt}\n\n[Image: data:image/jpeg;base64,{base64_image}]"
                        }
                    ],
                    temperature=0.2,
                    max_tokens=1024,
                    top_p=1,
                    stream=False,
                    stop=None,
                )
            else:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a specialized AI for robotic path planning. Extract formal constraints from instructions."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.2,
                    max_tokens=1024,
                    top_p=1,
                    stream=False,
                    stop=None,
                )
            
            output = completion.choices[0].message.content
            print("Raw LLM output:")
            print(output)
            print("\n")
            
            # Parse the response using Pydantic directly
            constraints = self._parse_with_pydantic(output)
            return constraints
            
        except Exception as e:
            print(f"Error calling Groq API: {e}")
            # Return empty constraints on failure
            return ConstraintSet()
    
    def _create_environment_context(self, environment) -> str:
        """Create a textual description of the environment for the LLM"""
        env_dict = environment.to_dict()
        
        context = f"Grid size: {env_dict['width']}x{env_dict['height']}\n"
        
        # Add start and goal positions if available
        if hasattr(environment, 'start') and hasattr(environment, 'goal'):
            context += f"Start position: {environment.start}\n"
            context += f"Goal position: {environment.goal}\n"
        
        # Describe regions more concisely
        if env_dict['regions']:
            context += "Defined regions:\n"
            for name, coords in env_dict['regions'].items():
                # For all regions, just describe the bounds
                xs = [x for x, y in coords]
                ys = [y for x, y in coords]
                if xs and ys:  # Make sure the lists aren't empty
                    context += f"- {name}: bounded by ({min(xs)},{min(ys)}) to ({max(xs)},{max(ys)})\n"
        
        return context
    
    def _create_environment_visualization(self, environment) -> str:
        """Create a smaller visualization of the environment with start/goal positions"""
        plt.figure(figsize=(5, 5))  # Smaller figure size
        
        # Create a colored grid for visualization
        vis_grid = np.zeros((environment.height, environment.width, 3))
        
        # Set obstacles to black
        for y in range(environment.height):
            for x in range(environment.width):
                if environment.grid[y, x] == 1:
                    vis_grid[y, x] = [0, 0, 0]  # Black for obstacles
                else:
                    vis_grid[y, x] = [1, 1, 1]  # White for free space
        
        # Color regions with different colors (simplified)
        colors = {
            'center': [1, 0.8, 0.8],  # Light red
            'walls': [0.8, 0.8, 1],   # Light blue
            'north': [0.8, 1, 0.8],   # Light green
            'south': [1, 1, 0.8],     # Light yellow
        }
        
        for region_name, coords in environment.regions.items():
            color = colors.get(region_name, [0.9, 0.9, 0.9])  # Default gray
            for x, y in coords:
                if 0 <= x < environment.width and 0 <= y < environment.height:
                    if environment.grid[y, x] == 0:  # Only color free cells
                        vis_grid[y, x] = color
        
        plt.imshow(vis_grid)
        
        # Add start and goal positions if available
        if hasattr(environment, 'start') and hasattr(environment, 'goal'):
            plt.plot(environment.start[0], environment.start[1], 'go', markersize=10, label='Start')
            plt.plot(environment.goal[0], environment.goal[1], 'ro', markersize=10, label='Goal')
            plt.legend(loc='upper right', fontsize='small')
        
        plt.grid(False)  # Turn off grid to reduce image size
        plt.title("Grid with Regions")
        
        # Save to bytes buffer with higher compression
        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg', dpi=72, quality=50, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        
        # Convert to base64
        img_bytes = buf.getvalue()
        base64_image = base64.b64encode(img_bytes).decode('utf-8')
        
        return base64_image
    
    def _construct_prompt(self, instruction: str, env_context: str) -> str:
        """Construct a prompt that strictly enforces JSON output format"""
        prompt = f"""
        # Environment
        {env_context}
        
        # Instruction
        "{instruction}"
        
        # Task
        Extract constraints from this instruction for robot navigation from start to goal.
        Consider the start and goal positions when determining which regions to prefer or avoid.
        
        # IMPORTANT: ONLY RETURN A JSON OBJECT WITH EXACTLY THIS FORMAT:
        {{
          "proximity": {{"region_name": weight, ...}},
          "avoidance": {{"region_name": weight, ...}},
          "preference": {{"region_name": weight, ...}}
        }}
        
        Where:
        - "region_name" is one of the regions defined in the environment
        - weight is a number between 1-10 indicating importance
        
        DO NOT include any explanations, thinking, or additional text.
        ONLY return the JSON object.
        """
        return prompt
    
    def _parse_with_pydantic(self, output: str) -> ConstraintSet:
        """Parse LLM output directly with Pydantic"""
        # Initialize empty constraint dictionaries
        proximity = {}
        avoidance = {}
        preference = {}
        
        # First try to parse as JSON
        try:
            # Find JSON object in the response
            json_start = output.find('{')
            json_end = output.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = output[json_start:json_end]
                data = json.loads(json_str)
                
                # Extract constraints from JSON
                if 'proximity' in data and isinstance(data['proximity'], dict):
                    proximity = data['proximity']
                if 'avoidance' in data and isinstance(data['avoidance'], dict):
                    avoidance = data['avoidance']
                if 'preference' in data and isinstance(data['preference'], dict):
                    preference = data['preference']
                    
                return ConstraintSet(
                    proximity=proximity,
                    avoidance=avoidance,
                    preference=preference
                )
        except Exception as e:
            print(f"JSON parsing failed: {e}, falling back to line-by-line parsing")
        
        # If JSON parsing fails, fall back to line-by-line parsing
        lines = output.strip().split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Identify which section we're in
            if line.startswith('proximity:'):
                current_section = 'proximity'
                content = line[len('proximity:'):].strip()
                if content and content != '{}':
                    self._parse_section_content(content, proximity)
                    
            elif line.startswith('avoidance:'):
                current_section = 'avoidance'
                content = line[len('avoidance:'):].strip()
                if content and content != '{}':
                    self._parse_section_content(content, avoidance)
                    
            elif line.startswith('preference:'):
                current_section = 'preference'
                content = line[len('preference:'):].strip()
                if content and content != '{}':
                    self._parse_section_content(content, preference)
                    
            # If we're in a section and the line contains key-value pairs
            elif current_section and ':' in line:
                self._parse_section_content(line, locals()[current_section])
        
        # Create and return the Pydantic model
        return ConstraintSet(
            proximity=proximity,
            avoidance=avoidance,
            preference=preference
        )
    
    def _parse_section_content(self, content: str, target_dict: Dict[str, float]):
        """Parse content of a section into the target dictionary"""
        # Remove curly braces if present
        content = content.strip('{}')
        if not content:
            return
            
        # Split by commas for multiple items
        items = content.split(',')
        for item in items:
            if ':' in item:
                key, value = item.split(':', 1)
                key = key.strip()
                try:
                    value = float(value.strip())
                    target_dict[key] = value
                except ValueError:
                    # If we can't convert to float, use default weight
                    target_dict[key] = 5.0

    def identify_regions(self, environment) -> Dict[str, List[Tuple[int, int]]]:
        """Automatically identify meaningful regions in the environment"""
        # Extract topological features
        regions_data = self._extract_topological_features(environment)
        
        # Create visualization with candidate regions
        base64_image = self._create_region_visualization(environment, regions_data)
        
        # Use LLM to name and describe regions
        named_regions = self._name_regions_with_llm(environment, regions_data, base64_image)
        
        # Convert to environment regions format
        environment_regions = {}
        
        # Store region descriptions for later use
        if not hasattr(environment, 'region_descriptions'):
            environment.region_descriptions = {}
        
        for region_name, region_info in named_regions.items():
            # Get all coordinates in this region
            coords = []
            for y in range(region_info['min_y'], region_info['max_y'] + 1):
                for x in range(region_info['min_x'], region_info['max_x'] + 1):
                    if 0 <= x < environment.width and 0 <= y < environment.height:
                        # Check if this point is in the region and is not an obstacle
                        region_id = regions_data['region_map'][y, x]
                        if region_id > 0 and environment.grid[y, x] == 0:
                            coords.append((x, y))
            
            if coords:  # Only add if there are valid coordinates
                environment_regions[region_name] = coords
                environment.region_descriptions[region_name] = region_info.get('description', '')
        
        return environment_regions
    
    def _extract_topological_features(self, environment) -> Dict:
        """Extract topological features using computer vision techniques"""
        # Create a binary grid (0 for obstacles, 1 for free space)
        binary_grid = (environment.grid == 0).astype(np.uint8)
        
        # Step 1: Identify open areas using distance transform
        dist_transform = ndimage.distance_transform_edt(binary_grid)
        
        # Step 2: Find local maxima in distance transform (centers of open areas)
        local_max = morphology.local_maxima(dist_transform)
        markers = measure.label(local_max)
        
        # Step 3: Use watershed segmentation to find regions
        regions = segmentation.watershed(-dist_transform, markers, mask=binary_grid)
        
        # Step 4: Calculate region properties
        region_props = []
        for i in range(1, np.max(regions) + 1):
            mask = regions == i
            if np.sum(mask) > 0:
                # Get region coordinates
                ys, xs = np.where(mask)
                
                # Calculate region properties
                area = len(xs)
                min_x, max_x = np.min(xs), np.max(xs)
                min_y, max_y = np.min(ys), np.max(ys)
                
                # Calculate average distance to obstacles
                avg_dist = np.mean(dist_transform[mask])
                
                # Determine region type based on shape and distance
                if avg_dist > 3:
                    region_type = "open_area"
                elif max(max_x - min_x, max_y - min_y) > 3 * min(max_x - min_x, max_y - min_y):
                    region_type = "corridor"
                else:
                    region_type = "junction"
                
                region_props.append({
                    'id': i,
                    'area': area,
                    'min_x': min_x,
                    'max_x': max_x,
                    'min_y': min_y,
                    'max_y': max_y,
                    'avg_dist': avg_dist,
                    'type': region_type
                })
        
        return {
            'region_map': regions,
            'region_properties': region_props,
            'distance_transform': dist_transform
        }
    
    def _create_region_visualization(self, environment, regions_data) -> str:
        """Create a smaller visualization of the environment with identified regions"""
        plt.figure(figsize=(5, 5), dpi=72)  # Smaller figure with lower DPI
        
        # Create a simplified visualization
        region_map = regions_data['region_map']
        
        # Use a discrete colormap with fewer colors
        cmap = plt.cm.get_cmap('tab10', np.max(region_map) + 1)
        plt.imshow(region_map, cmap=cmap)
        
        # Add minimal labels
        for prop in regions_data['region_properties']:
            region_id = prop['id']
            center_y = (prop['min_y'] + prop['max_y']) // 2
            center_x = (prop['min_x'] + prop['max_x']) // 2
            plt.text(center_x, center_y, str(region_id), 
                     fontsize=8, ha='center', va='center', 
                     color='white', bbox=dict(facecolor='black', alpha=0.5, pad=0))
        
        plt.title("Regions", fontsize=10)
        plt.axis('off')  # Turn off axis to reduce image size
        
        # Save with high compression
        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg', dpi=72, quality=50, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        
        # Convert to base64
        img_bytes = buf.getvalue()
        base64_image = base64.b64encode(img_bytes).decode('utf-8')
        
        return base64_image
    
    def _name_regions_with_llm(self, environment, regions_data, base64_image) -> Dict:
        """Use LLM to name and describe the identified regions"""
        # Create a description of the regions
        region_desc = ""
        for prop in regions_data['region_properties']:
            region_type = prop['type']
            area = prop['area']
            bounds = f"({prop['min_x']},{prop['min_y']}) to ({prop['max_x']},{prop['max_y']})"
            avg_dist = prop['avg_dist']
            
            region_desc += f"Region {prop['id']}: Type={region_type}, Area={area}, Bounds={bounds}, AvgDistToObstacles={avg_dist:.2f}\n"
        
        # Construct prompt
        prompt = f"""
        # Environment Analysis
        
        You are analyzing a grid environment for robot navigation.
        
        # Grid Information
        Size: {environment.width}x{environment.height}
        
        # Automatically Identified Regions
        {region_desc}
        
        # Task
        For each numbered region in the image, provide:
        1. A descriptive name (e.g., "north_corridor", "central_junction", "open_area_1")
        2. A brief description of what this region represents
        
        # Output Format
        For each region, provide:
        
        Region {{'id'}}: {{'descriptive_name'}}
        Description: {{'brief description'}}
        
        DO NOT include any explanations outside this format.
        """
        
        # Call Groq API with image
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a specialized AI for environmental analysis and region identification."
                    },
                    {
                        "role": "user",
                        "content": f"{prompt}\n\n[Image: data:image/jpeg;base64,{base64_image}]"
                    }
                ],
                temperature=0.2,
                max_tokens=1024,
                top_p=1,
                stream=False,
                stop=None,
            )
            
            output = completion.choices[0].message.content
            print("Raw region naming output:")
            print(output)
            
            # Parse the output
            return self._parse_region_naming_improved(output, regions_data)
            
        except Exception as e:
            print(f"Error in region naming: {e}")
            return {}
    
    def _parse_region_naming_improved(self, output: str, regions_data) -> Dict:
        """Improved parser for region naming output that handles various LLM response formats"""
        named_regions = {}
        
        # Extract region information from the output
        lines = output.strip().split('\n')
        current_region_id = None
        current_name = None
        current_description = None
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Check if this is a region header
            if line.startswith("Region ") and ":" in line:
                # Save previous region if exists
                if current_region_id is not None and current_name is not None and current_name != "{descriptive_name}":
                    # Find the region properties
                    for prop in regions_data['region_properties']:
                        if prop['id'] == current_region_id:
                            named_regions[current_name] = {
                                'min_x': prop['min_x'],
                                'max_x': prop['max_x'],
                                'min_y': prop['min_y'],
                                'max_y': prop['max_y'],
                                'description': current_description or ""
                            }
                            break
                
                # Parse new region
                parts = line.split(":", 1)
                region_part = parts[0].strip()
                name_part = parts[1].strip() if len(parts) > 1 else ""
                
                # Skip if the name is a placeholder
                if name_part == "{descriptive_name}":
                    # Try to find the real name in the next line
                    if i + 1 < len(lines) and not lines[i+1].startswith("Description:"):
                        name_part = lines[i+1].strip()
                
                # Extract region ID
                try:
                    region_id_str = region_part.split(" ")[1]
                    if region_id_str.endswith(":"):
                        region_id_str = region_id_str[:-1]
                    current_region_id = int(region_id_str)
                    current_name = name_part
                    current_description = None
                except (IndexError, ValueError):
                    current_region_id = None
                    current_name = None
            
            # Check for description
            elif (line.startswith("Description:") or "description:" in line.lower()) and current_region_id is not None:
                desc_part = line.split(":", 1)[1].strip()
                # Skip if the description is a placeholder
                if desc_part != "{brief description}":
                    current_description = desc_part
                # If it is a placeholder, try to find the real description in the next line
                elif i + 1 < len(lines):
                    current_description = lines[i+1].strip()
        
        # Add the last region
        if current_region_id is not None and current_name is not None and current_name != "{descriptive_name}":
            for prop in regions_data['region_properties']:
                if prop['id'] == current_region_id:
                    named_regions[current_name] = {
                        'min_x': prop['min_x'],
                        'max_x': prop['max_x'],
                        'min_y': prop['min_y'],
                        'max_y': prop['max_y'],
                        'description': current_description or ""
                    }
                    break
        
        # If no regions were parsed correctly, create a fallback with generic names
        if not named_regions:
            print("Warning: Could not parse region names properly, using generic names")
            for prop in regions_data['region_properties']:
                region_id = prop['id']
                region_type = prop['type']
                generic_name = f"{region_type}_{region_id}"
                
                named_regions[generic_name] = {
                    'min_x': prop['min_x'],
                    'max_x': prop['max_x'],
                    'min_y': prop['min_y'],
                    'max_y': prop['max_y'],
                    'description': f"A {region_type} region"
                }
        
        return named_regions

    def identify_regions_text_only(self, environment) -> Dict[str, List[Tuple[int, int]]]:
        """Identify regions using text-only descriptions (no images)"""
        # Extract topological features
        regions_data = self._extract_topological_features(environment)
        
        # Create a text description of the regions
        region_descriptions = []
        for prop in regions_data['region_properties']:
            region_id = prop['id']
            region_type = prop['type']
            min_x, max_x = prop['min_x'], prop['max_x']
            min_y, max_y = prop['min_y'], prop['max_y']
            area = prop['area']
            avg_dist = prop['avg_dist']
            
            desc = f"Region {region_id}:\n"
            desc += f"  Type: {region_type}\n"
            desc += f"  Bounds: ({min_x},{min_y}) to ({max_x},{max_y})\n"
            desc += f"  Size: {area} cells\n"
            desc += f"  Avg distance to obstacles: {avg_dist:.2f}\n"
            
            region_descriptions.append(desc)
        
        region_text = "\n".join(region_descriptions)
        
        # Construct prompt for region naming
        prompt = f"""
        # Environment Analysis Task
        
        You are given a grid environment for robot navigation.
        
        # Environment Description
        Grid size: {environment.width}x{environment.height}
        
        # Identified Regions
        {region_text}
        
        # Task
        For each region, provide:
        1. A descriptive name based on its type and location
        2. A brief description of its navigational significance
        
        # Output Format
        For each region, respond with:
        
        Region {{id}}: {{descriptive_name}}
        Description: {{brief description}}
        
        IMPORTANT: DO NOT include any thinking, explanations, or additional text.
        DO NOT use placeholders like {{descriptive_name}} or {{brief description}}.
        Replace these with actual names and descriptions.
        """
        
        # Call Groq API without image
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a specialized AI for environmental analysis and region identification. Always provide direct answers without thinking out loud or including placeholders."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.2,
                max_tokens=1024,
                top_p=1,
                stream=False,
                stop=None,
            )
            
            output = completion.choices[0].message.content
            print("Raw region naming output:")
            print(output)
            
            # Filter out thinking process if present
            if "<think>" in output and "</think>" in output:
                think_start = output.find("<think>")
                think_end = output.find("</think>") + len("</think>")
                output = output[:think_start] + output[think_end:]
            
            # Parse the output
            named_regions = self._parse_region_naming_improved(output, regions_data)
            
            # Convert to environment regions format
            environment_regions = {}
            
            # Store region descriptions for later use
            if not hasattr(environment, 'region_descriptions'):
                environment.region_descriptions = {}
            
            for region_name, region_info in named_regions.items():
                # Get all coordinates in this region
                coords = []
                for y in range(region_info['min_y'], region_info['max_y'] + 1):
                    for x in range(region_info['min_x'], region_info['max_x'] + 1):
                        if 0 <= x < environment.width and 0 <= y < environment.height:
                            # Check if this point is in the region and is not an obstacle
                            region_id = regions_data['region_map'][y, x]
                            if region_id > 0 and environment.grid[y, x] == 0:
                                coords.append((x, y))
            
            if coords:  # Only add if there are valid coordinates
                environment_regions[region_name] = coords
                environment.region_descriptions[region_name] = region_info.get('description', '')
            
            return environment_regions
            
        except Exception as e:
            print(f"Error in text-only region naming: {e}")
            return {}