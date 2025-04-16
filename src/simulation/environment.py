import os
import json
import sys
import numpy as np
from scipy.spatial.transform import Rotation

# Add parent directory to path to allow imports from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from .object_types import ObjectType
from .scenario_loader import load_scenario
from .autonomous_logic import get_autonomous_logic
from src.sensors.sensor_manager import SensorManager

class Environment:
    def __init__(self, scenario='basic_intersection', low_quality=False, disable_debug=False):
        """Initialize the simulation environment.
        
        Args:
            scenario (str): Name of the scenario to load
            low_quality (bool): Whether to use lower quality settings for better performance
            disable_debug (bool): Whether to disable debug logging for better performance
        """
        self.time = 0.0
        self.time_step = 0.05  # Giảm từ 0.1 xuống 0.05 để tăng độ mịn của mô phỏng
        
        # Performance settings
        self.low_quality = low_quality
        self.disable_debug = disable_debug
        self.sensor_update_interval = 0.2 if low_quality else 0.1  # Giảm từ 0.3 xuống 0.2
        self.last_sensor_update = 0.0
        
        # Thêm các thông số vật lý chung
        self.max_acceleration = 2.0  # Giảm từ 3.0 xuống 2.0 m/s^2
        self.max_deceleration = 4.0  # Giảm từ 5.0 xuống 4.0 m/s^2
        
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
        
        # Initialize sensor managers for autonomous vehicles
        self.sensor_managers = {}
        self._init_sensors()
        
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
    
    def _init_sensors(self):
        """Initialize sensors for autonomous vehicles."""
        for obj in self.objects:
            if obj.get('autonomous', False):
                # Create a sensor manager for each autonomous vehicle
                sensor_manager = SensorManager(obj, low_quality=self.low_quality)
                sensor_manager.setup_default_sensors()
                self.sensor_managers[obj['id']] = sensor_manager
                
                # Store sensor data in the vehicle object
                obj['sensors'] = sensor_manager
    
    def _load_static_point_cloud(self):
        """Load or generate a static point cloud for the environment representing the road and surroundings."""
        # Generate the centerline points from the road data
        points = []
        colors = []
        
        # Define parameters - use try/except to handle potential missing attributes
        try:
            road_width = self.autonomous_logic.road_width
        except AttributeError:
            road_width = 16.0  # Đường rộng phù hợp cho 4 làn xe
            
        try:
            road_length = self.autonomous_logic.road_length
        except AttributeError:
            road_length = 300.0  # Chiều dài đường mặc định (m)
        
        # Environment parameters
        env_range = 30.0  # Giảm từ 50.0 xuống 30.0
        lamp_spacing = 60.0  # Tăng khoảng cách giữa đèn đường
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
        ])
        
        # Define building colors as numpy array directly
        building_colors = np.array([
            [0.7, 0.7, 0.7],  # Light gray
            [0.8, 0.8, 0.8],  # White
        ])

        # Additional environmental colors
        water_color = [0.0, 0.3, 0.8]  # Blue for water
        grass_color = [0.2, 0.6, 0.1]  # Green for grass
        bush_color = [0.1, 0.4, 0.1]   # Dark green for bushes
        sign_pole_color = [0.4, 0.4, 0.4]  # Gray for sign poles
        sign_face_color = [1.0, 1.0, 1.0]  # White for sign face
        
        # Generate the centerline points
        # For lane-keeping scenario, generate a curved road
        # Natural curve parameters - use parameters from autonomous_logic
        curve_start = self.autonomous_logic.curve_start
        curve_intensity = self.autonomous_logic.curve_intensity
        resolution = 4.0  # Giảm độ phân giải để tạo ít điểm hơn
        
        # Generate the center line path with a natural curve
        num_points = int(road_length / resolution)
        centerline_points = []
        centerline_tangents = []
        
        # Check if autonomous_logic has stored road endpoints for a curved road
        has_stored_road_endpoints = hasattr(self.autonomous_logic, 'road_start') and hasattr(self.autonomous_logic, 'road_end')
        
        # Tạo đường cong
        if has_stored_road_endpoints and curve_intensity > 0.001:
            # Use the same linear interpolation as in autonomous_logic's calculate_lane_center
            start = self.autonomous_logic.road_start
            end = self.autonomous_logic.road_end
            
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
        else:
            # Original curve generation (fallback)
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
            # Simplified terrain height - just a small constant value
            terrain_height = 0.0
            
            # Simple undulations only far from the road
            if abs(y) > road_width/2 + sidewalk_width + 5.0:
                terrain_height = 0.2
            
            return terrain_height

        # Generate road points with terrain variations - reduced density
        for i in range(0, len(centerline_points), 2):  # Sample every other point
            if i < len(centerline_points):
                x, y = centerline_points[i]
            if i < len(centerline_tangents):
                tangent_x, tangent_y = centerline_tangents[i]
            
                    # Create points across the road width - reduced density
            for offset in np.linspace(-road_width/2, road_width/2, 16):  # Reduced from 24
                normal_x = -tangent_y
                normal_y = tangent_x
            
                # Road point
                road_x = x + normal_x * offset
                road_y = y + normal_y * offset
                points.append([road_x, road_y, 0.0])
                
                # Road color (darker in center, lighter at edges)
                intensity = 0.3 - 0.1 * abs(offset) / (road_width/2)
                colors.append([intensity, intensity, intensity])
                
                        # Add road markings - only critical ones
                        # Center line
                if abs(offset) < 0.2:  # Center solid line
                    points.append([road_x, road_y, 0.02])
                    colors.append([1.0, 1.0, 0.0])  # Yellow center line
                
                # Side lines (edge of road)
                if road_width/2 - 0.3 <= abs(offset) <= road_width/2:  # Edge lines
                    points.append([road_x, road_y, 0.02])
                    colors.append([1.0, 1.0, 1.0])  # White edge line
                    
                    # Add sidewalks on each side of the road - simplified
            for side in [-1, 1]:
                normal_x = -tangent_y * side
                normal_y = tangent_x * side
                
                # Sidewalk width with slight variation
                for sw_offset in np.linspace(0, sidewalk_width, 3):  # Reduced from 5
                    sidewalk_x = x + normal_x * (road_width/2 + sw_offset)
                    sidewalk_y = y + normal_y * (road_width/2 + sw_offset)
                    
                    # Get terrain height at this position
                    terrain_height = get_terrain_height(sidewalk_x, sidewalk_y)
                    
                    points.append([sidewalk_x, sidewalk_y, sidewalk_height + terrain_height])
                    colors.append(left_sidewalk_color if side < 0 else right_sidewalk_color)
                    
        # Add environmental elements very sparsely
        sampling_step = 80  # Increased from 30 for much fewer objects
        for i in range(0, len(centerline_points), sampling_step):
                if i < len(centerline_points):
                    x, y = centerline_points[i]
                if i < len(centerline_tangents):
                    tangent_x, tangent_y = centerline_tangents[i]
                    
                    # Place trees on both sides of the road - very sparse 
                    for side in [-1, 1]:
                        normal_x = -tangent_y * side
                        normal_y = tangent_x * side
                        
                        # Place trees at different distances from the road
                        min_dist = road_width/2 + sidewalk_width + 3.0
                        max_dist = min_dist + env_range
                        
                        # Get terrain height
                        approx_x = x + normal_x * (min_dist + 5)
                        approx_y = y + normal_y * (min_dist + 5)
                        terrain_height = get_terrain_height(approx_x, approx_y)
                        
                        # Very low chance of placing a tree
                        if np.random.random() < 0.3:
                            dist = np.random.uniform(min_dist, max_dist)
                            offset_x = np.random.uniform(-8, 8)
                            offset_y = np.random.uniform(-8, 8)
                            
                            tree_x = x + normal_x * dist + offset_x
                            tree_y = y + normal_y * dist + offset_y
                            
                            # Get terrain height
                            terrain_height = get_terrain_height(tree_x, tree_y)
                            
                            # Create ultra-simplified tree (just a trunk and top)
                            trunk_height = np.random.uniform(3, 5)
                            trunk_color = [0.3, 0.2, 0.1]  # Brown
                            
                            # Only 2 points for trunk - bottom and top
                            points.append([tree_x, tree_y, terrain_height])
                            colors.append(trunk_color)
                            points.append([tree_x, tree_y, trunk_height + terrain_height])
                            colors.append(trunk_color)
                            
                            # One point for crown
                            crown_color = tree_colors[np.random.randint(len(tree_colors))]
                            points.append([tree_x, tree_y, trunk_height + 2 + terrain_height])
                            colors.append(crown_color)
                        
                        # Buildings - extremely rare
                        if np.random.random() < 0.1:
                            dist = np.random.uniform(min_dist + 10, max_dist)
                            building_x = x + normal_x * dist + np.random.uniform(-10, 10)
                            building_y = y + normal_y * dist + np.random.uniform(-10, 10)
                            
                            # Get terrain height at this position
                            terrain_height = get_terrain_height(building_x, building_y)
                            
                            # Building dimensions
                            building_width = np.random.uniform(8, 12)
                            building_depth = np.random.uniform(8, 12)
                            building_height = np.random.uniform(5, 10)
                            
                            # Select building color
                            building_color = building_colors[np.random.randint(len(building_colors))]
                            
                            # Ultra-simplified building - just 5 points for corners
                            points.append([building_x - building_width/2, building_y - building_depth/2, terrain_height])
                            colors.append(building_color)
                            points.append([building_x + building_width/2, building_y - building_depth/2, terrain_height])
                            colors.append(building_color)
                            points.append([building_x + building_width/2, building_y + building_depth/2, terrain_height])
                            colors.append(building_color)
                            points.append([building_x - building_width/2, building_y + building_depth/2, terrain_height])
                            colors.append(building_color)
                            # Roof center
                            points.append([building_x, building_y, building_height + terrain_height])
                            colors.append(building_color)
                            
        # Add lampposts - extremely sparse
        current_distance = 0
        for i in range(1, len(centerline_points), int(len(centerline_points)/4)):  # Only 4 lampposts total
            if i < len(centerline_points):
                x, y = centerline_points[i]
                if i < len(centerline_tangents):
                    tangent_x, tangent_y = centerline_tangents[i]
                    
                    # Place lampposts on one side of the road only
                    side = 1
                    normal_x = -tangent_y * side
                    normal_y = tangent_x * side
                    
                    # Position at the edge of the sidewalk
                    lamp_x = x + normal_x * (road_width/2 + sidewalk_width - 0.5)
                    lamp_y = y + normal_y * (road_width/2 + sidewalk_width - 0.5)
                    
                    # Get terrain height at this position
                    terrain_height = get_terrain_height(lamp_x, lamp_y)
                    
                    # Ultra-simplified lamppost (2 points)
                    # Bottom post
                    points.append([lamp_x, lamp_y, terrain_height])
                    colors.append(lamp_color)
                    
                    # Light bulb
                    points.append([lamp_x, lamp_y, lamp_height + terrain_height])
                    colors.append(lamp_light_color)
        
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
        """Update the simulation by one time step."""
        self.time += self.time_step
        
        # Update sensor data for all autonomous vehicles (throttled in low quality mode)
        enough_time_passed = self.time - self.last_sensor_update >= self.sensor_update_interval
        
        if enough_time_passed:
            scene_data = self.get_scene_data()
            for obj_id, sensor_manager in self.sensor_managers.items():
                sensor_data = sensor_manager.update(scene_data, self.time)
                # Store sensor data in the object
                for obj in self.objects:
                    if obj['id'] == obj_id:
                        obj['sensor_data'] = sensor_data
                        break
            
            self.last_sensor_update = self.time
        
        # Apply AI/control logic for autonomous vehicles first
        for obj in self.objects:
            if obj.get('autonomous', False):
                self._update_autonomous_vehicle(obj)
            else:
                self._update_non_autonomous_vehicle(obj)
                
        # Apply physics update - update all object positions based on velocity
        self._update_physics()
                
        # Check for collisions
        self._check_collisions()
        
        # Print debug info only if not disabled and less frequently
        if not self.disable_debug and int(self.time * 10) % 200 == 0:  # Reduced from 50 to 200 (less frequent)
            # Log simulation stats every ~20 seconds instead of ~5 seconds
            active_sensors = sum(len(sm.sensors) for sm in self.sensor_managers.values())
            print(f"Simulation time: {self.time:.1f}s, FPS: {1.0/self.time_step:.1f}, Active sensors: {active_sensors}")
    
    def _update_physics(self):
        """Update the physics state of all objects based on their velocity."""
        for obj in self.objects:
            # Get current position and velocity
            position = np.array(obj['position'])
            velocity = np.array(obj.get('velocity', [0, 0, 0]))
            rotation = np.array(obj['rotation'])
            
            # Apply velocity to position (p = p + v*t)
            position += velocity * self.time_step
            
            # Update position in the object
            obj['position'] = position.tolist()
            
            # Update rotation based on steering for vehicles
            if obj.get('type').name in ['CAR', 'TRUCK']:
                # Get steering angle if available
                steering_angle = obj.get('steering', 0)
                
                # Only update rotation if vehicle is moving and steering
                speed = np.linalg.norm(velocity)
                if abs(steering_angle) > 0.01 and speed > 0.1:
                    # Calculate rotation change based on steering and speed
                    # Simple model: rotation_rate = steering_angle * speed / wheelbase
                    wheelbase = obj.get('dimensions', [4, 2, 1.5])[0] * 0.8  # Use length as approximation
                    rotation_rate = steering_angle * speed / wheelbase
                    
                    # Update yaw (rotation around Z-axis)
                    rotation[2] += rotation_rate * self.time_step
                    
                    # Normalize angle to [-180, 180]
                    rotation[2] = (rotation[2] + 180) % 360 - 180
                    
                    # Update rotation in the object
                    obj['rotation'] = rotation.tolist()
    
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
        # Convert positions to numpy arrays before subtraction
        pos1 = np.array(obj1['position'])
        pos2 = np.array(obj2['position'])
        direction = pos2 - pos1
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
        new_obj = {
            'id': len(self.objects),
            'type': type_enum,
            'position': np.array(position, dtype=np.float64),
            'rotation': np.array(rotation, dtype=np.float64),
            'velocity': np.array(velocity, dtype=np.float64),
            'dimensions': np.array(dimensions, dtype=np.float64),
            'color': type_enum.get_color(),
            'autonomous': autonomous
        }
        
        self.objects.append(new_obj)
        
        # Initialize sensors if autonomous
        if autonomous:
            sensor_manager = SensorManager(new_obj, low_quality=self.low_quality)
            sensor_manager.setup_default_sensors()
            self.sensor_managers[new_obj['id']] = sensor_manager
            new_obj['sensors'] = sensor_manager
            
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
            'road_width': self.autonomous_logic.road_width,
            'sensor_managers': self.sensor_managers
        }
    
    def set_autonomous(self, object_id, autonomous=True):
        """Set whether an object is autonomously controlled."""
        for obj in self.objects:
            if obj['id'] == object_id:
                previous_state = obj.get('autonomous', False)
                obj['autonomous'] = autonomous
                
                # Add sensors if newly autonomous
                if autonomous and not previous_state:
                    sensor_manager = SensorManager(obj, low_quality=self.low_quality)
                    sensor_manager.setup_default_sensors()
                    self.sensor_managers[obj['id']] = sensor_manager
                    obj['sensors'] = sensor_manager
                
                # Remove sensors if no longer autonomous
                elif not autonomous and previous_state and obj['id'] in self.sensor_managers:
                    del self.sensor_managers[obj['id']]
                    if 'sensors' in obj:
                        del obj['sensors']
                
                return True
        return False
    
    def get_sensor_debug_data(self, object_id):
        """Get sensor data for debugging visualization.
        
        Args:
            object_id: ID of the vehicle to get sensor data for
            
        Returns:
            dict: Sensor debug data or None if no sensors for this vehicle
        """
        if object_id in self.sensor_managers:
            sensor_manager = self.sensor_managers[object_id]
            return {
                'status': sensor_manager.get_sensor_status(),
                'data': sensor_manager.get_sensor_data()
            }
        return None
    
    def set_sensor_visualization(self, object_id, camera=None, lidar=None, radar=None, fusion=None):
        """Set visualization flags for vehicle sensors.
        
        Args:
            object_id: ID of the vehicle to set visualization for
            camera: Enable/disable camera visualization
            lidar: Enable/disable LiDAR visualization
            radar: Enable/disable radar visualization
            fusion: Enable/disable sensor fusion visualization
            
        Returns:
            bool: Success or failure
        """
        if object_id in self.sensor_managers:
            self.sensor_managers[object_id].set_visualization(camera, lidar, radar, fusion)
            return True
        return False
    
    def cleanup(self):
        """Clean up any resources."""
        # Close any open visualization windows
        for _, manager in self.sensor_managers.items():
            for name, sensor in manager.sensors.items():
                if hasattr(sensor, 'cleanup') and callable(sensor.cleanup):
                    sensor.cleanup()
        pass 