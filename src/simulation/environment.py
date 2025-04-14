import os
import json
import numpy as np
from scipy.spatial.transform import Rotation

from .object_types import ObjectType
from .scenario_loader import load_scenario
from .autonomous_logic import get_autonomous_logic

class Environment:
    def __init__(self, scenario='basic_intersection'):
        """Initialize the simulation environment.
        
        Args:
            scenario (str): Name of the scenario to load
        """
        self.time = 0.0
        self.time_step = 0.1  # simulation step in seconds
        
        # Thêm các thông số vật lý chung
        self.max_acceleration = 3.0  # m/s^2
        self.max_deceleration = 5.0  # m/s^2
        
        # Load scenario
        self.scenario_data = load_scenario(scenario)
        
        # Initialize autonomous driving logic (moved before road network)
        self.autonomous_logic = get_autonomous_logic('lane_keeping')
        
        # Initialize objects in the simulation
        self.objects = []
        self._init_objects()
        
        # Initialize road network (now after autonomous_logic is created)
        self.road_network = self._init_road_network()
        
        # Point cloud data for the static environment (road, buildings, etc.)
        self.static_point_cloud = self._load_static_point_cloud()
        
    def _init_objects(self):
        """Initialize objects from the scenario data."""
        for obj_data in self.scenario_data.get('objects', []):
            obj_type = ObjectType[obj_data['type'].upper()]
            
            # Ensure all arrays are float64 for consistent calculations
            position = np.array(obj_data.get('position', [0, 0, 0]), dtype=np.float64)
            rotation = np.array(obj_data.get('rotation', [0, 0, 0]), dtype=np.float64)
            velocity = np.array(obj_data.get('velocity', [0, 0, 0]), dtype=np.float64)
            dimensions = np.array(obj_data.get('dimensions', [4, 2, 1.5]), dtype=np.float64)
            is_autonomous = obj_data.get('autonomous', False)
            
            self.objects.append({
                'id': len(self.objects),
                'type': obj_type,
                'position': position,
                'rotation': rotation,
                'velocity': velocity,
                'dimensions': dimensions,
                'color': obj_type.get_color(),
                'autonomous': is_autonomous
            })
    
    def _init_road_network(self):
        """Initialize the road network from scenario data."""
        road_network = self.scenario_data.get('road_network', {})
        
        # Extract road data to store in autonomous_logic
        if road_network and 'roads' in road_network and len(road_network['roads']) > 0:
            # Get the first road (main road)
            first_road = road_network['roads'][0]
            
            # Store this road data for use in visualization and physics
            if hasattr(self, 'autonomous_logic') and self.autonomous_logic:
                if 'start' in first_road and 'end' in first_road:
                    # Store road endpoints
                    self.autonomous_logic.road_start = np.array(first_road['start'])
                    self.autonomous_logic.road_end = np.array(first_road['end'])
                    
                    # Calculate road length
                    road_vec = self.autonomous_logic.road_end - self.autonomous_logic.road_start
                    self.autonomous_logic.road_length = np.linalg.norm(road_vec)
                    
                    # Set curve parameters based on road endpoints
                    if abs(first_road['end'][1] - first_road['start'][1]) > 0.1:
                        # This is a curved road
                        self.autonomous_logic.curve_start = 0
                        # Calculate curve intensity
                        self.autonomous_logic.curve_intensity = abs(first_road['end'][1] - first_road['start'][1]) / self.autonomous_logic.road_length
                    else:
                        # Straight road
                        self.autonomous_logic.curve_start = self.autonomous_logic.road_length
                        self.autonomous_logic.curve_intensity = 0
                    
                    # Update road width
                    if 'width' in first_road:
                        self.autonomous_logic.road_width = first_road['width']
        
        return road_network
    
    def _load_static_point_cloud(self):
        """Load or generate a static point cloud for the environment representing the road and surroundings."""
        # Generate the centerline points from the road data
        points = []
        colors = []
        
        # Define parameters - use try/except to handle potential missing attributes
        try:
            road_width = self.autonomous_logic.road_width
        except AttributeError:
            road_width = 16.0  # Increased from 10.0 to accommodate 4 lanes
            
        try:
            road_length = self.autonomous_logic.road_length
        except AttributeError:
            road_length = 300.0  # Default road length in meters
        
        # Environment parameters
        env_range = 50.0
        lamp_spacing = 40.0  # Increased spacing between lampposts
        lamp_height = 5.0
        lamp_color = [0.3, 0.3, 0.3]
        lamp_light_color = [1.0, 1.0, 0.8]
        
        # Parameters for sidewalks
        sidewalk_width = 2.0
        sidewalk_height = 0.2
        left_sidewalk_color = [0.7, 0.7, 0.7]
        right_sidewalk_color = [0.7, 0.7, 0.7]
        
        # Define tree colors as numpy array directly
        tree_colors = np.array([
            [0.1, 0.5, 0.1],  # Dark green
            [0.2, 0.6, 0.2],  # Medium green
            [0.3, 0.7, 0.3],  # Light green
            [0.5, 0.4, 0.1],  # Autumn brown
            [0.6, 0.5, 0.2]   # Autumn yellow
        ])
        
        # Define building colors as numpy array directly
        building_colors = np.array([
            [0.7, 0.7, 0.7],  # Light gray
            [0.8, 0.8, 0.8],  # White
            [0.6, 0.6, 0.5],  # Beige
            [0.5, 0.5, 0.6],  # Light blue
            [0.8, 0.7, 0.6]   # Light brown
        ])

        # Additional environmental colors
        water_color = [0.0, 0.3, 0.8]  # Blue for water
        grass_color = [0.2, 0.6, 0.1]  # Green for grass
        bush_color = [0.1, 0.4, 0.1]   # Dark green for bushes
        flower_colors = np.array([
            [0.9, 0.1, 0.1],  # Red
            [1.0, 0.6, 0.0],  # Orange
            [1.0, 1.0, 0.0],  # Yellow
            [0.8, 0.1, 0.8],  # Purple
            [1.0, 0.5, 0.7]   # Pink
        ])
        sign_pole_color = [0.4, 0.4, 0.4]  # Gray for sign poles
        sign_face_color = [1.0, 1.0, 1.0]  # White for sign face
        
        # Generate the centerline points
        # For lane-keeping scenario, generate a curved road
        # Natural curve parameters - use parameters from autonomous_logic instead of hardcoded values
        curve_start = self.autonomous_logic.curve_start
        curve_intensity = self.autonomous_logic.curve_intensity
        resolution = 2.0  # Resolution of the centerline points (meters)
        
        # Generate the center line path with a natural curve
        num_points = int(road_length / resolution)
        centerline_points = []
        centerline_tangents = []
        
        # Check if autonomous_logic has stored road endpoints for a curved road
        has_stored_road_endpoints = hasattr(self.autonomous_logic, 'road_start') and hasattr(self.autonomous_logic, 'road_end')
        
        # Debug output - check why road isn't curved
        print(f"\n===== ROAD VISUALIZATION DEBUG =====")
        print(f"Curve intensity: {curve_intensity}")
        print(f"Has stored road endpoints: {has_stored_road_endpoints}")
        if has_stored_road_endpoints:
            print(f"Road start: {self.autonomous_logic.road_start}")
            print(f"Road end: {self.autonomous_logic.road_end}")
        print(f"Check condition: {has_stored_road_endpoints and curve_intensity > 0.01}")
        print(f"====================================\n")
        
        # Force use stored road endpoints for curved road if curve_intensity is significant
        if has_stored_road_endpoints and curve_intensity > 0.001:  # Lowered threshold from 0.01 to 0.001
            # Use the same linear interpolation as in autonomous_logic's calculate_lane_center
            start = self.autonomous_logic.road_start
            end = self.autonomous_logic.road_end
            
            print(f"Generating curved road from endpoints: {start} to {end}")
            
            for i in range(num_points):
                # Parametric position along the road (0 to 1)
                t = i / (num_points - 1)
                # X coordinate from start to end
                x = start[0] + t * (end[0] - start[0])
                
                # Y coordinate - linear interpolation between start and end
                y = start[1] + t * (end[1] - start[1])
                
                # Calculate tangent (road direction)
                tangent_x = (end[0] - start[0]) / np.linalg.norm(end - start)
                tangent_y = (end[1] - start[1]) / np.linalg.norm(end - start)
                
                # Normalize tangent
                tangent_norm = np.sqrt(tangent_x**2 + tangent_y**2)
                if tangent_norm > 0.001:
                    tangent_x /= tangent_norm
                    tangent_y /= tangent_norm
                
                centerline_points.append((x, y))
                centerline_tangents.append((tangent_x, tangent_y))
                
                # Debug output for the first and last few points
                if i < 5 or i > num_points - 5:
                    print(f"Point {i}: ({x}, {y}) - Tangent: ({tangent_x}, {tangent_y})")
        else:
            # Original curve generation (fallback)
            print(f"Using fallback curve generation method")
            for i in range(num_points):
                # Parametric position along the road (0 to 1)
                t = i / (num_points - 1)
                # X coordinate - always increases along the road
                x = -road_length/2 + t * road_length
                
                # Y coordinate - starts at 0, then gradually curves right
                if x < -road_length/2 + curve_start:
                    # Straight section
                    y = 0
                    # Tangent is pointing along x-axis
                    tangent_x, tangent_y = 1.0, 0.0
                else:
                    # Curved section - use linear interpolation for smooth transition
                    curve_t = (x - (-road_length/2 + curve_start)) / (road_length - curve_start)
                    # Linear curve matching autonomous_logic's implementation
                    y = curve_intensity * curve_t * road_length
                    # Tangent direction
                    tangent_x = 1.0  # Always moving forward in x
                    tangent_y = curve_intensity  # Constant derivative for linear function
                    # Normalize tangent
                    norm = np.sqrt(tangent_x**2 + tangent_y**2)
                    tangent_x /= norm
                    tangent_y /= norm
                
                centerline_points.append((x, y))
                centerline_tangents.append((tangent_x, tangent_y))
        
        # Create terrain variations - add some rolling hills
        def get_terrain_height(x, y):
            # Use simplex noise for natural-looking terrain
            # Scale parameters to control the size and height of terrain features
            terrain_scale = 0.01
            terrain_height = 0.0
            
            # Basic flat terrain with small undulations
            if abs(y) > road_width/2 + sidewalk_width + 5.0:
                seed = 42  # Fixed seed for reproducibility
                np.random.seed(seed)
                
                # Add some randomness based on position
                freq1 = 0.01
                freq2 = 0.02
                amp1 = 0.5
                amp2 = 0.3
                
                # Simple noise function (since simplex not readily available)
                noise1 = np.sin(x * freq1) * np.cos(y * freq1) * amp1
                noise2 = np.sin(x * freq2 + 0.5) * np.cos(y * freq2 + 0.3) * amp2
                
                terrain_height = noise1 + noise2
            
            # Create a depression for the water puddle in the middle of the road
            water_center_x = road_length/4
            water_center_y = 0  # Middle of the road
            water_radius = 5.0
            
            # Distance from water center
            dist_from_water = np.sqrt((x - water_center_x)**2 + (y - water_center_y)**2)
            
            # Create depression for water on the road
            if dist_from_water < water_radius:
                # Depression formula to create bowl shape
                water_depth = 0.5  # Shallow puddle
                # Smooth transition to water
                terrain_height = -water_depth * (1 - (dist_from_water/water_radius)**2)
            
            return terrain_height

        # Generate road points with terrain variations
        for i in range(len(centerline_points)):
            x, y = centerline_points[i]
            tangent_x, tangent_y = centerline_tangents[i]
            
            # Create points across the road width
            for offset in np.linspace(-road_width/2, road_width/2, 24):  # Increased density for wider road
                normal_x = -tangent_y
                normal_y = tangent_x
            
                # Road point
                road_x = x + normal_x * offset
                road_y = y + normal_y * offset
                points.append([road_x, road_y, 0.0])
                
                # Road color (darker in center, lighter at edges)
                intensity = 0.3 - 0.1 * abs(offset) / (road_width/2)
                colors.append([intensity, intensity, intensity])
                
                # Add road markings
                # Center double solid line to separate opposing traffic
                if abs(offset) < 0.2:  # Center solid line
                    points.append([road_x, road_y, 0.02])
                    colors.append([1.0, 1.0, 0.0])  # Yellow center line
                
                # Dashed lines between lanes in same direction (inner lanes)
                if abs(abs(offset) - road_width/4) < 0.2:  # Lane dividers at ±road_width/4
                    if i % 4 < 2:  # Dashed line
                        points.append([road_x, road_y, 0.02])
                        colors.append([1.0, 1.0, 1.0])  # White lane divider
                
                # Side lines (edge of road)
                if road_width/2 - 0.3 <= abs(offset) <= road_width/2:  # Edge lines
                    points.append([road_x, road_y, 0.02])
                    colors.append([1.0, 1.0, 1.0])  # White edge line
                    
                # Add crosswalk markings at specific intervals
                if abs(x) < 10.0 and abs(x) % 2 < 1.0 and abs(offset) < road_width/2 - 1.0:
                    points.append([road_x, road_y, 0.03])
                    colors.append([1.0, 1.0, 1.0])  # White crosswalk
                
                # Add stop line
                if abs(x - road_length/3) < 0.5 and abs(offset) < road_width/2 - 0.5:
                    points.append([road_x, road_y, 0.03])
                    colors.append([1.0, 1.0, 1.0])  # White stop line
            
            # Add sidewalks on each side of the road
            for side in [-1, 1]:
                normal_x = -tangent_y * side
                normal_y = tangent_x * side
                
                # Sidewalk width with slight variation
                for sw_offset in np.linspace(0, sidewalk_width, 5):  # Reduced density (was 8)
                    sidewalk_x = x + normal_x * (road_width/2 + sw_offset)
                    sidewalk_y = y + normal_y * (road_width/2 + sw_offset)
                    
                    # Get terrain height at this position
                    terrain_height = get_terrain_height(sidewalk_x, sidewalk_y)
                    
                    points.append([sidewalk_x, sidewalk_y, sidewalk_height + terrain_height])
                    colors.append(left_sidewalk_color if side < 0 else right_sidewalk_color)
                    
                    # Add occasional crack/damage to sidewalk
                    if np.random.random() < 0.01:  # 1% chance
                        crack_size = np.random.uniform(0.1, 0.3)
                        points.append([sidewalk_x, sidewalk_y, sidewalk_height + terrain_height + 0.02])
                        colors.append([0.3, 0.3, 0.3])  # Darker color for cracks
        
        # Add traffic signs
        sign_positions = []
        sign_types = ['stop', 'yield', 'speed_limit', 'no_parking', 'curve_ahead']
        
        # Place traffic signs at specific intervals
        for i in range(0, len(centerline_points), 50):  # Every 50 points
                if i < len(centerline_points):
                    x, y = centerline_points[i]
                if i < len(centerline_tangents):
                    tangent_x, tangent_y = centerline_tangents[i]
                    
                    # Place sign on one side (alternating)
                    side = 1 if i % 100 < 50 else -1
                    normal_x = -tangent_y * side
                    normal_y = tangent_x * side
                    
                    # Position sign near the sidewalk
                    sign_x = x + normal_x * (road_width/2 + sidewalk_width + 0.5)
                    sign_y = y + normal_y * (road_width/2 + sidewalk_width + 0.5)
                    
                    # Get terrain height at this position
                    terrain_height = get_terrain_height(sign_x, sign_y)
                    
                    sign_type = sign_types[i % len(sign_types)]
                    sign_positions.append((sign_x, sign_y, sign_type, terrain_height))
        
        # Generate traffic signs
        for sign_x, sign_y, sign_type, terrain_height in sign_positions:
            # Create sign pole
            pole_height = 2.5
            pole_radius = 0.05
            
            # Generate pole points
            for h in np.arange(0, pole_height, 0.3):
                points.append([sign_x, sign_y, h + terrain_height])
                colors.append(sign_pole_color)
            
            # Create sign face
            sign_width = 0.6
            sign_height = 0.6
            sign_depth = 0.1
            
            # Generate sign face points based on type
            if sign_type == 'stop':
                # Red octagon for stop sign
                sign_color = [0.8, 0.0, 0.0]  # Red
                # Generate octagon points at top of pole
                for angle in np.arange(0, 2*np.pi, np.pi/4):
                    sx = sign_x + 0.3 * np.cos(angle)
                    sy = sign_y + 0.3 * np.sin(angle)
                    points.append([sx, sy, pole_height + terrain_height])
                    colors.append(sign_color)
            elif sign_type == 'yield':
                # Yellow triangle for yield
                sign_color = [1.0, 0.8, 0.0]  # Yellow
                # Triangle points
                for corner in range(3):
                    angle = corner * 2*np.pi/3
                    sx = sign_x + 0.3 * np.cos(angle)
                    sy = sign_y + 0.3 * np.sin(angle)
                    points.append([sx, sy, pole_height + terrain_height])
                    colors.append(sign_color)
            else:
                # Rectangle for other signs
                sign_color = [0.0, 0.3, 0.8]  # Blue for most signs
                if sign_type == 'speed_limit':
                    sign_color = [1.0, 1.0, 1.0]  # White
                
                # Create rectangular sign
                corners = [
                    [-sign_width/2, -sign_height/2],
                    [sign_width/2, -sign_height/2],
                    [sign_width/2, sign_height/2],
                    [-sign_width/2, sign_height/2]
                ]
                
                for dx, dy in corners:
                    points.append([sign_x + dx, sign_y + dy, pole_height + terrain_height])
                    colors.append(sign_color)
        
        # Add environmental elements like trees and houses
        # Place them at lower density (every 30m along the road)
        sampling_step = 30  # Increased from 25
        for i in range(0, len(centerline_points), sampling_step):
                if i < len(centerline_points):
                    x, y = centerline_points[i]
                if i < len(centerline_tangents):
                    tangent_x, tangent_y = centerline_tangents[i]
                    
                    # Place trees on both sides of the road
                    for side in [-1, 1]:
                        normal_x = -tangent_y * side
                        normal_y = tangent_x * side
                        
                        # Place trees at different distances from the road
                        min_dist = road_width/2 + sidewalk_width + 3.0  # Increased min distance
                        max_dist = min_dist + env_range  # Max distance from road center
                        
                        # Get terrain height at approximate position
                        approx_x = x + normal_x * (min_dist + 5)
                        approx_y = y + normal_y * (min_dist + 5)
                        base_terrain_height = get_terrain_height(approx_x, approx_y)
                        
                        # Reduced likelihood of placing trees (50% chance)
                        if np.random.random() < 0.5:
                            # Only 1 tree at each location
                            # Randomize position
                            dist = np.random.uniform(min_dist, max_dist)
                            offset_x = np.random.uniform(-8, 8)
                            offset_y = np.random.uniform(-8, 8)
                            
                            tree_x = x + normal_x * dist + offset_x
                            tree_y = y + normal_y * dist + offset_y
                            
                            # Get terrain height at this position
                            terrain_height = get_terrain_height(tree_x, tree_y)
                            
                            # Create tree trunk
                            trunk_height = np.random.uniform(3, 7)
                            trunk_radius = np.random.uniform(0.3, 0.6)
                            trunk_resolution = 0.5  # Further increased resolution = fewer points
                            trunk_color = [0.3, 0.2, 0.1]  # Brown
                            
                            # Generate trunk points - even more sparse
                            for h in np.arange(0, trunk_height, trunk_resolution):
                                for angle in np.arange(0, 2*np.pi, 1.2):  # Further increased angle step
                                    tx = tree_x + trunk_radius * np.cos(angle)
                                    ty = tree_y + trunk_radius * np.sin(angle)
                                    points.append([tx, ty, h + terrain_height])
                                    colors.append(trunk_color)
                            
                            # Create tree crown (foliage)
                            crown_radius = np.random.uniform(2, 4)
                            crown_height = np.random.uniform(3, 5)
                            crown_resolution = 1.0  # Further increased resolution = fewer points
                            
                            # Select a random color from the tree_colors array
                            crown_color = tree_colors[np.random.randint(len(tree_colors))]
                            
                            # Crown center
                            crown_center_z = trunk_height + crown_height/2 + terrain_height
                            
                            # Generate crown points (sphere-like) - even more sparse
                            for dz in np.arange(-crown_height/2, crown_height/2, crown_resolution):
                                # Radius at this height (ellipsoid shape)
                                radius_at_z = crown_radius * np.sqrt(1 - (dz/(crown_height/2))**2)
                                
                                for angle1 in np.arange(0, np.pi, 1.0):  # Further increased angle step
                                    for angle2 in np.arange(0, 2*np.pi, 1.0):  # Further increased angle step
                                        dx = radius_at_z * np.sin(angle1) * np.cos(angle2)
                                        dy = radius_at_z * np.sin(angle1) * np.sin(angle2)
                                        dz_sphere = radius_at_z * np.cos(angle1)
                                        
                                        # Only add points on the outer shell with more sparsity
                                        if np.sqrt(dx**2 + dy**2 + dz_sphere**2) > 0.9 * radius_at_z:
                                            points.append([tree_x + dx, tree_y + dy, crown_center_z + dz])
                                            colors.append(crown_color)
                        
                        # Add bushes and small plants (higher probability than trees)
                        if np.random.random() < 0.7:
                            # Place 1-3 bushes in cluster
                            num_bushes = np.random.randint(1, 4)
                            cluster_center_x = x + normal_x * (min_dist + np.random.uniform(1, 8))
                            cluster_center_y = y + normal_y * (min_dist + np.random.uniform(1, 8))
                            
                            for _ in range(num_bushes):
                                # Random position within cluster
                                bush_x = cluster_center_x + np.random.uniform(-2, 2)
                                bush_y = cluster_center_y + np.random.uniform(-2, 2)
                                
                                # Get terrain height at this position
                                terrain_height = get_terrain_height(bush_x, bush_y)
                                
                                # Bush size
                                bush_radius = np.random.uniform(0.5, 1.5)
                                bush_height = np.random.uniform(0.5, 1.2)
                                
                                # Create bush (simple hemisphere)
                                for dz in np.arange(0, bush_height, 0.3):
                                    radius_at_z = bush_radius * np.sqrt(1 - (dz/bush_height)**2)
                                    for angle in np.arange(0, 2*np.pi, 0.8):
                                        dx = radius_at_z * np.cos(angle)
                                        dy = radius_at_z * np.sin(angle)
                                        points.append([bush_x + dx, bush_y + dy, dz + terrain_height])
                                        colors.append(bush_color)
                                
                                # Add flowers on some bushes
                                if np.random.random() < 0.5:
                                    flower_color = flower_colors[np.random.randint(len(flower_colors))]
                                    num_flowers = np.random.randint(3, 8)
                                    for _ in range(num_flowers):
                                        angle = np.random.uniform(0, 2*np.pi)
                                        rad = np.random.uniform(0.5, 0.9) * bush_radius
                                        flower_x = bush_x + rad * np.cos(angle)
                                        flower_y = bush_y + rad * np.sin(angle)
                                        flower_z = np.random.uniform(0.3, bush_height) + terrain_height
                                        # Simple flower (single point)
                                        points.append([flower_x, flower_y, flower_z])
                                        colors.append(flower_color)
                        
                        # Add buildings on some spots (less frequent than trees)
                        if np.random.random() < 0.15:  # Further reduced chance of placing a building
                            dist = np.random.uniform(min_dist + 10, max_dist)  # Buildings even further back
                            offset_x = np.random.uniform(-12, 12)
                            offset_y = np.random.uniform(-12, 12)
                            
                            building_x = x + normal_x * dist + offset_x
                            building_y = y + normal_y * dist + offset_y
                            
                            # Get terrain height at this position
                            terrain_height = get_terrain_height(building_x, building_y)
                            
                            # Building dimensions
                            building_width = np.random.uniform(8, 15)
                            building_depth = np.random.uniform(8, 15)
                            building_height = np.random.uniform(5, 15)
                            building_resolution = 1.2  # Further increased resolution = fewer points
                            
                            # Select a random color from the building_colors array
                            building_color = building_colors[np.random.randint(len(building_colors))]
                            
                            # Rotate the building slightly
                            rotation_angle = np.random.uniform(0, np.pi/4)
                            
                            # Generate building points - only key points for structure outline
                            # Corners of the building
                            corners = [
                                [-building_width/2, -building_depth/2],  # Front left
                                [building_width/2, -building_depth/2],   # Front right
                                [building_width/2, building_depth/2],    # Back right
                                [-building_width/2, building_depth/2]    # Back left
                            ]
                            
                            # Generate only key points with larger spacing
                            for corner in corners:
                                dx, dy = corner
                                # Rotate corner
                                rotated_dx = dx * np.cos(rotation_angle) - dy * np.sin(rotation_angle)
                                rotated_dy = dx * np.sin(rotation_angle) + dy * np.cos(rotation_angle)
                                
                                # Vertical edges
                                for dz in np.arange(0, building_height, building_resolution):
                                    points.append([building_x + rotated_dx, building_y + rotated_dy, dz + terrain_height])
                                    colors.append(building_color)
                            
                            # Roof corners
                            for corner in corners:
                                dx, dy = corner
                                # Rotate corner
                                rotated_dx = dx * np.cos(rotation_angle) - dy * np.sin(rotation_angle)
                                rotated_dy = dx * np.sin(rotation_angle) + dy * np.cos(rotation_angle)
                                
                                points.append([building_x + rotated_dx, building_y + rotated_dy, building_height + terrain_height])
                                colors.append(building_color)
                            
                            # Add a few windows - much sparser
                            num_floors = max(1, int(building_height / 3))
                            for floor in range(num_floors):
                                floor_height = 1.5 + floor * 3
                                
                                # Only add 2-3 windows per side
                                for window_idx in range(2):
                                    window_x_offset = -building_width/4 + window_idx * building_width/2
                                    
                                    # Front wall windows
                                    rotated_x = window_x_offset * np.cos(rotation_angle) - (-building_depth/2) * np.sin(rotation_angle)
                                    rotated_y = window_x_offset * np.sin(rotation_angle) + (-building_depth/2) * np.cos(rotation_angle)
                                    points.append([building_x + rotated_x, building_y + rotated_y, floor_height + terrain_height])
                                    colors.append([0.9, 0.9, 0.7])  # Window color
                                    
                                    # Back wall windows
                                    rotated_x = window_x_offset * np.cos(rotation_angle) - (building_depth/2) * np.sin(rotation_angle)
                                    rotated_y = window_x_offset * np.sin(rotation_angle) + (building_depth/2) * np.cos(rotation_angle)
                                    points.append([building_x + rotated_x, building_y + rotated_y, floor_height + terrain_height])
                                    colors.append([0.9, 0.9, 0.7])  # Window color
        
        # Add lampposts along the road - greatly reduced density
        current_distance = 0
        for i in range(1, len(centerline_points)):
            x1, y1 = centerline_points[i-1]
            x2, y2 = centerline_points[i]
            segment_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            current_distance += segment_length
            
            # Place a lamppost every lamp_spacing meters (increased to 40m)
            if current_distance >= lamp_spacing:
                current_distance = 0
                
                # Place lampposts on both sides of the road
                for side in [-1, 1]:
                    tangent_x, tangent_y = centerline_tangents[i]
                    normal_x = -tangent_y * side
                    normal_y = tangent_x * side
                    
                    # Position at the edge of the sidewalk
                    lamp_x = x2 + normal_x * (road_width/2 + sidewalk_width - 0.5)
                    lamp_y = y2 + normal_y * (road_width/2 + sidewalk_width - 0.5)
                    
                    # Get terrain height at this position
                    terrain_height = get_terrain_height(lamp_x, lamp_y)
                    
                    # Generate lamppost points - ultra simplified
                    # Bottom post
                    points.append([lamp_x, lamp_y, terrain_height])
                    colors.append(lamp_color)
                    
                    # Middle post
                    points.append([lamp_x, lamp_y, lamp_height/2 + terrain_height])
                    colors.append(lamp_color)
                    
                    # Top post
                    points.append([lamp_x, lamp_y, lamp_height + terrain_height])
                    colors.append(lamp_color)
                    
                    # Light bulb (single point)
                    points.append([lamp_x + normal_x * 0.5, lamp_y + normal_y * 0.5, lamp_height + terrain_height])
                    colors.append(lamp_light_color)
        
        # Add water feature (lake)
        water_center_x = road_length/4
        water_center_y = 0  # Now in the middle of the road
        water_radius = 5.0  # Smaller radius to fit on the road
        water_resolution = 1.0
        
        # Generate grid of points for water surface
        for dx in np.arange(-water_radius, water_radius, water_resolution):
            for dy in np.arange(-water_radius, water_radius, water_resolution):
                water_x = water_center_x + dx
                water_y = water_center_y + dy
                
                # Only add points within the circular lake
                dist_from_center = np.sqrt(dx**2 + dy**2)
                if dist_from_center < water_radius:
                    # Water depth formula
                    water_depth = 0.5  # Shallower depth for the road puddle
                    water_height = -water_depth * (1 - (dist_from_center/water_radius)**2)
                    
                    # Add small waves to water surface
                    wave_height = 0.05 * np.sin(dx * 2) * np.cos(dy * 2)
                    
                    points.append([water_x, water_y, water_height + wave_height])
                    colors.append(water_color)
        
        # Add some animals (birds, squirrels, etc.)
        num_animals = 10
        for _ in range(num_animals):
            # Choose a random position, generally near trees or water
            if np.random.random() < 0.5:
                # Bird near a tree
                i = np.random.randint(0, len(centerline_points))
                if i < len(centerline_points):
                    x, y = centerline_points[i]
                    side = np.random.choice([-1, 1])
                    if i < len(centerline_tangents):
                        tangent_x, tangent_y = centerline_tangents[i]
                        normal_x = -tangent_y * side
                        normal_y = tangent_x * side
                        
                        dist = np.random.uniform(road_width/2 + sidewalk_width + 5, road_width/2 + sidewalk_width + 20)
                        animal_x = x + normal_x * dist + np.random.uniform(-5, 5)
                        animal_y = y + normal_y * dist + np.random.uniform(-5, 5)
                        
                        # Get terrain height at this position
                        terrain_height = get_terrain_height(animal_x, animal_y)
                        
                        # Bird in the air
                        animal_height = terrain_height + np.random.uniform(5, 10)
                        
                        # Simple bird representation (few points)
                        points.append([animal_x, animal_y, animal_height])
                        colors.append([0.2, 0.2, 0.8])  # Blue for birds
                        
                        # Wing points
                        wing_span = np.random.uniform(0.3, 0.8)
                        points.append([animal_x - wing_span, animal_y, animal_height])
                        colors.append([0.2, 0.2, 0.8])
                        points.append([animal_x + wing_span, animal_y, animal_height])
                        colors.append([0.2, 0.2, 0.8])
            else:
                # Animal near water or on ground
                if np.random.random() < 0.5:
                    # Near water
                    angle = np.random.uniform(0, 2*np.pi)
                    dist = np.random.uniform(water_radius - 3, water_radius + 3)
                    animal_x = water_center_x + dist * np.cos(angle)
                    animal_y = water_center_y + dist * np.sin(angle)
                else:
                    # Random ground position
                    i = np.random.randint(0, len(centerline_points))
                    if i < len(centerline_points):
                        x, y = centerline_points[i]
                        side = np.random.choice([-1, 1])
                        if i < len(centerline_tangents):
                            tangent_x, tangent_y = centerline_tangents[i]
                            normal_x = -tangent_y * side
                            normal_y = tangent_x * side
                            
                            dist = np.random.uniform(road_width/2 + sidewalk_width + 2, road_width/2 + sidewalk_width + 15)
                            animal_x = x + normal_x * dist + np.random.uniform(-3, 3)
                            animal_y = y + normal_y * dist + np.random.uniform(-3, 3)
                
                # Get terrain height at this position
                terrain_height = get_terrain_height(animal_x, animal_y)
                
                # Small animal (squirrel, rabbit, etc.)
                animal_size = np.random.uniform(0.2, 0.4)
                points.append([animal_x, animal_y, terrain_height + animal_size/2])
                if np.random.random() < 0.5:
                    colors.append([0.6, 0.4, 0.2])  # Brown for squirrel
                else:
                    colors.append([0.8, 0.8, 0.8])  # Gray for rabbit
        
        # Convert to numpy arrays
        points_array = np.array(points, dtype=np.float64)
        colors_array = np.array(colors, dtype=np.float64)
        
        # Store the centerline and tangents for use by other parts of the simulation
        self.road_network['centerline'] = centerline_points
        self.road_network['tangents'] = centerline_tangents
        
        # Also store centerline points as a class attribute for access by get_scene_data
        self.centerline_points = centerline_points
        
        return {
            'points': points_array,
            'colors': colors_array
        }
    
    def step(self):
        """Advance the simulation by one time step."""
        self.time += self.time_step
        
        # Apply AI/control logic for autonomous vehicles first
        for obj in self.objects:
            if obj.get('autonomous', False) and obj['type'] == ObjectType.CAR:
                self._update_autonomous_vehicle(obj)
            # Add simple behavior for non-autonomous vehicles
            elif obj['type'] == ObjectType.CAR and not obj.get('autonomous', False):
                self._update_non_autonomous_vehicle(obj)
        
        # Then update positions of all objects
        for obj in self.objects:
            # Ensure all arrays are float64 type
            if not np.issubdtype(obj['position'].dtype, np.floating):
                obj['position'] = obj['position'].astype(np.float64)
            if not np.issubdtype(obj['velocity'].dtype, np.floating):
                obj['velocity'] = obj['velocity'].astype(np.float64)
            
            # Initialize all needed state variables
            if 'prev_rotation' not in obj:
                obj['prev_rotation'] = obj['rotation'].copy()
            if 'prev_position' not in obj:
                obj['prev_position'] = obj['position'].copy()
            if 'prev_velocity' not in obj:
                obj['prev_velocity'] = obj['velocity'].copy()
            if 'steering_history' not in obj:
                obj['steering_history'] = [0.0] * 10  # Increased history buffer
            if 'angular_velocity' not in obj:
                obj['angular_velocity'] = 0.0
            if 'lateral_velocity' not in obj:
                obj['lateral_velocity'] = np.zeros(3, dtype=np.float64)
            if 'current_turn_radius' not in obj:
                obj['current_turn_radius'] = float('inf')  # Straight line by default
            
            # Get current speed 
            speed = np.linalg.norm(obj['velocity'])
            
            # Apply vehicle physics if steering is present
            if 'steering' in obj:
                # Physical constants for realistic car behavior
                wheelbase = obj['dimensions'][0] * 0.55  # Giảm xuống 55% của chiều dài xe 
                mass = 1800.0  # Tăng khối lượng xe để tăng độ ổn định
                tire_grip = 0.99  # Tăng độ bám của lốp gần như tối đa
                
                # Heavily smoothed steering input using an expanded history buffer
                obj['steering_history'].append(obj['steering'])
                if len(obj['steering_history']) > 15:  # Increased from 12 to 15 for smoother input
                    obj['steering_history'].pop(0)
                
                # Sử dụng hàm làm mịn mạnh hơn cho góc lái
                if len(obj['steering_history']) >= 15:
                    # Exponential weights that emphasize recent values but maintain smoothing
                    weights = [0.01, 0.01, 0.02, 0.02, 0.03, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.12, 0.14, 0.14, 0.15]
                    avg_steering = sum(w * s for w, s in zip(weights, obj['steering_history'][-15:]))
                else:
                    avg_steering = sum(obj['steering_history']) / len(obj['steering_history'])
                
                # Apply steering with dynamic limitations based on speed
                if speed > 0.1:
                    # Calculate maximum steering angle based on speed
                    # At high speeds, limit maximum steering angle (realistic car limitation)
                    max_speed_for_full_steering = 2.0  # Reduce max speed for full steering (was 4.0)
                    max_steering_angle = 0.7  # Increase max steering angle (was 0.45)
                    
                    # Tăng giới hạn góc lái ở tốc độ thấp
                    speed_factor = max(0.5, min(1.0, max_speed_for_full_steering / speed))
                    effective_max_steering = max_steering_angle * speed_factor
                    
                    # Limit steering to effective maximum
                    effective_steering = np.clip(avg_steering, -effective_max_steering, effective_max_steering)
                    
                    # Calculate turn radius - used for realistic vehicle dynamics
                    if abs(effective_steering) > 0.001:  # Avoid division by near-zero
                        turn_radius = wheelbase / np.tan(abs(effective_steering))
                        turn_radius = max(2.0, turn_radius)  # Tăng bán kính quay tối thiểu lên 3m
                        obj['current_turn_radius'] = turn_radius * np.sign(effective_steering)
                    else:
                        obj['current_turn_radius'] = float('inf')  # Straight line
                    
                    # Tính toán tốc độ quay dựa trên bán kính quay và tốc độ
                    if abs(obj['current_turn_radius']) < 1000:  # If turning
                        # Increase turn responsiveness - reduced from 1.2 to 0.9 for gentler turns
                        target_yaw_rate = 0.9 * speed / obj['current_turn_radius']
                    else:
                        target_yaw_rate = 0.0
                    
                    # Tăng độ mượt mà cho tốc độ góc
                    angular_acceleration = 2.0 * (target_yaw_rate - obj['angular_velocity'])  # Reduced from 3.0 to 2.0
                    obj['angular_velocity'] += angular_acceleration * self.time_step
                    
                    # Apply smooth angular velocity to rotation
                    new_rotation = obj['rotation'].copy()
                    new_rotation[2] += obj['angular_velocity'] * self.time_step
                    
                    # Even smoother rotation changes (93% new, 7% previous)
                    smooth_factor = 0.93
                    obj['rotation'] = smooth_factor * new_rotation + (1.0 - smooth_factor) * obj['prev_rotation']
                    obj['prev_rotation'] = obj['rotation'].copy()
                    
                    # Calculate vehicle forward direction vector
                    forward_dir = np.array([
                        np.cos(obj['rotation'][2]),
                        np.sin(obj['rotation'][2]), 
                        0
                    ], dtype=np.float64)
                    
                    # Calculate side (lateral) direction vector
                    side_dir = np.array([
                        -np.sin(obj['rotation'][2]),
                        np.cos(obj['rotation'][2]),
                        0
                    ], dtype=np.float64)
                    
                    # Decompose velocity into forward and lateral components
                    forward_speed = np.dot(obj['velocity'], forward_dir)
                    forward_vel = forward_speed * forward_dir
                    
                    lateral_vel = obj['velocity'] - forward_vel
                    obj['lateral_velocity'] = lateral_vel
                    
                    # Apply tire grip based on turning physics
                    # 1. More lateral grip when going straight or slow
                    # 2. Less lateral grip in tight turns at high speeds (realistic drift effect)
                    base_grip = tire_grip
                    
                    # Calculate dynamic grip factor based on turn radius and speed
                    # Tighter turns at higher speeds have less grip (realistic)
                    if abs(obj['current_turn_radius']) < 100:
                        # The centripetal force is proportional to v²/r
                        centripetal_factor = min(0.8, (speed * speed) / (25.0 * abs(obj['current_turn_radius'])))
                        dynamic_grip = base_grip * (1.0 - 0.3 * centripetal_factor)  # Reduced grip loss in turns
                    else:
                        dynamic_grip = base_grip
                    
                    # Apply grip to reduce lateral velocity
                    lateral_reduction = 1.0 - dynamic_grip
                    new_lateral_vel = lateral_vel * lateral_reduction
                    
                    # Calculate lane center for the curved road (matching the formula in autonomous_logic)
                    current_x_phys = obj['position'][0]
                    lane_center_phys = 0.0  # Default for straight section
                    
                    # Curve parameters (must match those in autonomous_logic.py)
                    phys_curve_start = 100.0
                    phys_curve_intensity = 0.2
                    phys_road_half_length = self.autonomous_logic.road_length / 2
                    
                    # Check if in curve section
                    if current_x_phys >= -phys_road_half_length + phys_curve_start:
                        # Calculate curve parameter t
                        phys_curve_t = (current_x_phys - (-phys_road_half_length + phys_curve_start)) / (300.0 - phys_curve_start)
                        phys_curve_t = max(0.0, min(1.0, phys_curve_t))  # Clamp to [0,1]
                        # Calculate lane center using the exact same formula
                        lane_center_phys = phys_curve_intensity * (phys_curve_t * phys_curve_t) * 300.0
                    
                    # Apply additional road centering force when off center
                    # This simulates the natural tendency of cars to follow the camber of the road
                    if abs(obj['position'][1] - lane_center_phys) > 0.2:  # If off center relative to curved road
                        # Reduce the centering force for non-autonomous vehicles to prevent unnatural drift
                        centering_force_factor = 0.01 if not obj.get('autonomous', False) else 0.1  # Reduced from 0.05 for non-autonomous
                        centering_force = -centering_force_factor * (obj['position'][1] - lane_center_phys)  # Proportional to distance from center
                        centering_vector = side_dir * centering_force
                        
                        # Apply less centering for non-autonomous vehicles
                        centering_factor = 0.0 if not obj.get('autonomous', False) else min(1.0, speed / 2.0)  # Changed from min(1.0, speed/4.0) to 0.0
                        centering_vector *= centering_factor
                        
                        # Add centering force to velocity
                        new_lateral_vel += centering_vector
                    
                    # Recombine velocity components
                    obj['velocity'] = forward_vel + new_lateral_vel
                    
                    # Apply very light smoothing to final velocity (less jitter, more responsive)
                    obj['velocity'] = 0.95 * obj['velocity'] + 0.05 * obj['prev_velocity']
                    obj['prev_velocity'] = obj['velocity'].copy()
                
            # Update position with continuous motion integration
            new_position = obj['position'] + obj['velocity'] * self.time_step
            
            # Adaptive position smoothing - almost no smoothing at higher speeds
            # This prevents the vehicle from lagging behind its actual physical position
            speed_factor = min(1.0, speed / 5.0)
            smooth_factor = 0.8 + (0.19 * speed_factor)  # 0.8 to 0.99 based on speed
            obj['position'] = smooth_factor * new_position + (1.0 - smooth_factor) * obj['prev_position']
            obj['prev_position'] = obj['position'].copy()
            
            # Road boundary handling with gradual correction
            # Instead of hardcoded values, use the road parameters from autonomous_logic
            road_width = self.autonomous_logic.road_width  # Get road width from autonomous logic
            
            # Calculate the lane center for current x position using the same formula as in autonomous_logic
            current_x = obj['position'][0]
            current_y = obj['position'][1]
            
            # Calculate lane center for the curved road (same as in autonomous_logic)
            lane_center = 0.0  # Default for straight section
            
            # For lane offset, use values consistent with autonomous_logic
            lanes = 2  # Default assumption
            lane_width = road_width / lanes
            lane_offset = lane_width / 2  # Half a lane width for right side driving
            
            # Use curve parameters from autonomous_logic directly
            curve_start = self.autonomous_logic.curve_start
            curve_intensity = self.autonomous_logic.curve_intensity
            road_half_length = self.autonomous_logic.road_length / 2
            
            # Check if the autonomous logic has stored road endpoints (for curved roads)
            has_stored_road_endpoints = hasattr(self.autonomous_logic, 'road_start') and hasattr(self.autonomous_logic, 'road_end')
            
            # If we have road endpoints, use linear interpolation for lane center (same as autonomous_logic)
            if has_stored_road_endpoints and curve_intensity > 0.01:
                # Get road start and end from autonomous_logic
                start = self.autonomous_logic.road_start
                end = self.autonomous_logic.road_end
                
                # Calculate normalized position along the road (0 to 1)
                # Use same calculation as in autonomous_logic's calculate_lane_center
                road_length = self.autonomous_logic.road_length
                curve_t = (current_x - (-road_length / 2)) / road_length
                curve_t = max(0.0, min(1.0, curve_t))  # Clamp to [0,1]
                
                # Linear interpolation for curved road
                lane_center = curve_t * curve_intensity * road_length
            
            # For non-autonomous vehicles, respect their initial lane position
            # instead of forcing them to the rightmost lane
            if not obj['autonomous']:
                # Get the initial y-position from object's initial setup
                initial_lane = 0
                for orig_obj in self.scenario_data.get('objects', []):
                    if orig_obj.get('type', '').upper() == obj['type'].name and np.array_equal(
                        np.array(orig_obj.get('position', [0, 0, 0]), dtype=np.float64),
                        obj.get('initial_position', obj['position'])):
                        initial_lane = orig_obj.get('position', [0, 0, 0])[1]
                        break
                
                # Keep vehicle in its original lane with minimal lateral drift
                if not hasattr(obj, 'target_lane_position'):
                    obj['target_lane_position'] = initial_lane
                    # Store initial position if not already stored
                    if 'initial_position' not in obj:
                        obj['initial_position'] = np.copy(obj['position'])
                
                # Reduce centering force for non-autonomous vehicles to prevent them from drifting
                centering_factor = 0.01 if speed > 0 else 0.0
                
                # Target is curved lane center plus original offset from center
                lane_center = lane_center + (obj['target_lane_position'] - 0)
            else:
                # Add lane offset to calculate center of the rightmost lane for autonomous vehicle
                lane_center += lane_offset
                centering_factor = 0.05  # Normal centering for autonomous vehicles
            
            # Calculate road edges based on curved lane center
            left_edge = lane_center - road_width/2
            right_edge = lane_center + road_width/2
            
            if current_y < left_edge + 0.5 or current_y > right_edge - 0.5:  # Near edge detection
                # Calculate force to push vehicle back to road
                if current_y < left_edge + 0.5:
                    # Too far left
                    target_y = left_edge + 1.0
                else:
                    # Too far right
                    target_y = right_edge - 1.0
                
                # Create a centering force
                centering_force = 0.8 * (target_y - current_y)
                
                # Apply force immediately to avoid vehicle going off-road
                obj['position'][1] += centering_force * 0.3
                
                # Calculate angle to steer back toward road center
                target_lane_y = lane_center
                target_angle = np.arctan2(target_lane_y - current_y, 1.0)
                angle_diff = (target_angle - obj['rotation'][2] + np.pi) % (2 * np.pi) - np.pi
                
                # Apply stronger angle correction near the edge
                correction_strength = min(0.3, 0.1 + abs(current_y - lane_center) / (road_width/2) * 0.2)
                obj['rotation'][2] += angle_diff * correction_strength
            
            # Mild centering force when not near edge but off center
            elif abs(current_y - lane_center) > 0.5:
                # Apply centering force proportional to distance from center
                # Non-autonomous vehicles use the reduced centering_factor
                centering_force = -centering_factor * min(1.0, speed / 4.0) * (current_y - lane_center)
                obj['position'][1] += centering_force * self.time_step
        
        # Check for collisions between vehicles
        self._check_collisions()
    
    def _update_autonomous_vehicle(self, vehicle):
        """Apply autonomous driving logic to the vehicle."""
        # Get control commands from the autonomous driving logic
        scene_data = self.get_scene_data()
        control = self.autonomous_logic.process(vehicle, scene_data)
        
        # Apply acceleration with giới hạn
        acceleration = control.get('acceleration', 0.0)
        # Giới hạn gia tốc tối đa để tránh tăng/giảm tốc đột ngột
        acceleration = np.clip(acceleration, -self.max_deceleration * 0.7, self.max_acceleration * 0.7)
        
        # Calculate current direction vector
        direction = np.array([
            np.cos(vehicle['rotation'][2]),
            np.sin(vehicle['rotation'][2]),
            0
        ], dtype=np.float64)
        
        # Update velocity based on acceleration with smoothing
        current_speed = np.linalg.norm(vehicle['velocity'])
        target_speed = max(0, current_speed + acceleration * self.time_step)
        
        # Làm mịn thay đổi vận tốc
        if 'prev_speed' not in vehicle:
            vehicle['prev_speed'] = current_speed
            
        # Áp dụng làm mịn vận tốc (80% mới, 20% cũ)
        smoothed_speed = 0.8 * target_speed + 0.2 * vehicle['prev_speed']
        vehicle['prev_speed'] = smoothed_speed
        
        # Cập nhật vận tốc xe dựa trên hướng hiện tại
        vehicle['velocity'] = direction * smoothed_speed
        
        # Apply steering với giới hạn góc lái
        raw_steering = control.get('steering', 0.0)
        
        # Giới hạn góc lái tùy thuộc vào tốc độ
        max_steering_at_speed = 0.5 * (1.0 - min(0.8, current_speed / 10.0))
        limited_steering = np.clip(raw_steering, -max_steering_at_speed, max_steering_at_speed)
        
        vehicle['steering'] = limited_steering
    
    def _update_non_autonomous_vehicle(self, vehicle):
        """Apply simple driving behavior to non-autonomous vehicles."""
        # Get current speed
        current_speed = np.linalg.norm(vehicle['velocity'])
        
        # Simple behavior: gradually slow down if moving
        if current_speed > 0:
            # Calculate deceleration (stronger at higher speeds)
            deceleration = min(0.5, current_speed * 0.05)
            
            # Apply slight speed reduction
            direction = np.array([
                np.cos(vehicle['rotation'][2]),
                np.sin(vehicle['rotation'][2]),
                0
            ], dtype=np.float64)
            
            # Reduce speed by deceleration
            new_speed = max(0, current_speed - deceleration * self.time_step)
            
            # Update velocity
            vehicle['velocity'] = direction * new_speed
            
            # Add simple steering to follow the road curvature
            current_x = vehicle['position'][0]
            road_half_length = self.autonomous_logic.road_length / 2
            curve_start = self.autonomous_logic.curve_start
            curve_intensity = self.autonomous_logic.curve_intensity
            road_length = self.autonomous_logic.road_length
            
            # Check if we have curve parameters and the vehicle is in the curve section
            if curve_intensity > 0.01:
                # Calculate normalized position along the road (0 to 1)
                curve_t = (current_x - (-road_length / 2)) / road_length
                curve_t = max(0.0, min(1.0, curve_t))  # Clamp to [0,1]
                
                # Check if autonomous_logic has stored road endpoints
                if hasattr(self.autonomous_logic, 'road_start') and hasattr(self.autonomous_logic, 'road_end'):
                    # Get road start and end points
                    start = self.autonomous_logic.road_start
                    end = self.autonomous_logic.road_end
                    
                    # Calculate road direction vector
                    road_dir = (end - start) / road_length
                    
                    # Calculate target heading from road direction
                    target_heading = np.arctan2(road_dir[1], road_dir[0])
                else:
                    # Fallback to tangent calculation
                    # For linear interpolation, tangent is constant
                    tangent_x = 1.0  # Always forward
                    tangent_y = curve_intensity  # Derivative of linear curve
                    
                    # Calculate target heading
                    target_heading = np.arctan2(tangent_y, tangent_x)
                
                # Smoothly adjust heading to follow road
                current_heading = vehicle['rotation'][2]
                heading_diff = (target_heading - current_heading + np.pi) % (2 * np.pi) - np.pi
                
                # Apply gentle rotation based on heading difference
                correction_rate = 0.3  # Adjust heading gradually
                new_heading = current_heading + heading_diff * correction_rate * self.time_step
                
                # Update rotation
                vehicle['rotation'][2] = new_heading
        
        # Add steering attribute required for physics calculations
        if 'steering' not in vehicle:
            vehicle['steering'] = 0.0

    def _check_collisions(self):
        """Check for and handle collisions between objects."""
        num_objects = len(self.objects)
        
        for i in range(num_objects):
            for j in range(i + 1, num_objects):
                obj1 = self.objects[i]
                obj2 = self.objects[j]
                
                # Only check collisions between vehicles
                if obj1['type'] in [ObjectType.CAR, ObjectType.TRUCK] and obj2['type'] in [ObjectType.CAR, ObjectType.TRUCK]:
                    # Get positions and dimensions
                    pos1 = obj1['position']
                    pos2 = obj2['position']
                    
                    # Calculate 2D distance between centers
                    distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                    
                    # Combined half-lengths and half-widths
                    combined_half_length = (obj1['dimensions'][0] + obj2['dimensions'][0]) / 2
                    combined_half_width = (obj1['dimensions'][1] + obj2['dimensions'][1]) / 2
                    
                    # Simplified collision check (rectangular bounding boxes)
                    # More accurate would account for rotation, but this is simpler
                    if distance < (combined_half_length + combined_half_width) / 2:
                        # Collision detected!
                        self._handle_collision(obj1, obj2)
                        
                        # Add debug info
                        if 'debug_info' not in obj1:
                            obj1['debug_info'] = {}
                        if 'debug_info' not in obj2:
                            obj2['debug_info'] = {}
                            
                        obj1['debug_info']['collision'] = True
                        obj2['debug_info']['collision'] = True
                        
                        # Print collision information
                        print(f"\n!!! COLLISION DETECTED !!!")
                        print(f"Object {obj1['id']} ({obj1['type']}) and Object {obj2['id']} ({obj2['type']})")
                        print(f"Distance: {distance:.2f}m, Required separation: {(combined_half_length + combined_half_width) / 2:.2f}m")
                        print(f"Positions: {obj1['position']} and {obj2['position']}")
                        print(f"Velocities: {obj1['velocity']} and {obj2['velocity']}")
                        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")

    def _handle_collision(self, obj1, obj2):
        """Handle collision between two objects."""
        # Simple collision response - reduce velocity and apply repulsion force
        
        # Reduce velocities (simulating energy loss)
        obj1['velocity'] *= 0.5
        obj2['velocity'] *= 0.5
        
        # Calculate direction vector from obj1 to obj2
        direction = obj2['position'] - obj1['position']
        dist = np.linalg.norm(direction)
        
        # Avoid division by zero
        if dist > 0.001:
            direction /= dist
        else:
            direction = np.array([1, 0, 0])  # Default direction if objects are at the same position
        
        # Calculate masses based on dimensions (simplified)
        mass1 = np.prod(obj1['dimensions'])
        mass2 = np.prod(obj2['dimensions'])
        
        # Apply a repulsion force to separate objects
        repulsion_strength = 2.0  # Strength of repulsion force
        
        # Apply stronger force to the lighter object
        force1 = -repulsion_strength * mass2 / (mass1 + mass2) * direction
        force2 = repulsion_strength * mass1 / (mass1 + mass2) * direction
        
        # Apply the forces by directly modifying position
        obj1['position'] += force1 * self.time_step
        obj2['position'] += force2 * self.time_step
    
    def add_object(self, obj_type, position, rotation, velocity, dimensions, autonomous=False):
        """Add a new object to the simulation."""
        type_enum = ObjectType[obj_type.upper()]
        self.objects.append({
            'id': len(self.objects),
            'type': type_enum,
            'position': np.array(position, dtype=np.float64),
            'rotation': np.array(rotation, dtype=np.float64),
            'velocity': np.array(velocity, dtype=np.float64),
            'dimensions': np.array(dimensions, dtype=np.float64),
            'color': type_enum.get_color(),
            'autonomous': autonomous
        })
        return self.objects[-1]
    
    def get_scene_data(self):
        """Get the current scene data for visualization."""
        # Ensure the road_network includes centerline data
        if 'centerline' not in self.road_network and hasattr(self, 'centerline_points'):
            self.road_network['centerline'] = self.centerline_points
            
        # If there's no centerline data but we have the road endpoints, generate a smooth centerline
        if 'centerline' not in self.road_network and hasattr(self.autonomous_logic, 'road_start') and hasattr(self.autonomous_logic, 'road_end'):
            # Generate a smooth centerline with many points
            start = self.autonomous_logic.road_start
            end = self.autonomous_logic.road_end
            num_points = 100  # More points for a smoother line
            
            centerline = []
            for i in range(num_points):
                t = i / (num_points - 1)
                x = start[0] + t * (end[0] - start[0])
                y = start[1] + t * (end[1] - start[1])
                centerline.append((x, y))
                
            self.road_network['centerline'] = centerline
        
        return {
            'time': self.time,
            'objects': self.objects,
            'static_point_cloud': self.static_point_cloud,
            'road_network': self.road_network,
            'autonomous_logic': self.autonomous_logic,
            'road_width': self.autonomous_logic.road_width
        }
    
    def set_autonomous(self, object_id, autonomous=True):
        """Set whether an object is autonomously controlled."""
        for obj in self.objects:
            if obj['id'] == object_id:
                obj['autonomous'] = autonomous
                return True
        return False
    
    def cleanup(self):
        """Clean up any resources."""
        pass 