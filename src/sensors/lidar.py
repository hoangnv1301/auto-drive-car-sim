"""
LiDAR sensor for autonomous vehicle simulation.
"""

import numpy as np
import time
import open3d as o3d
from .base_sensor import BaseSensor


class LidarSensor(BaseSensor):
    """LiDAR sensor for autonomous vehicle simulation."""
    
    def __init__(self, vehicle, num_layers=16, points_per_layer=1800, range_m=100.0, 
                 update_freq_hz=10.0, fov_h=360.0, fov_v=30.0, noise_std=0.01, low_quality=False):
        """Initialize LiDAR sensor.
        
        Args:
            vehicle: Vehicle this sensor is attached to
            num_layers: Number of vertical layers (lines)
            points_per_layer: Number of points per layer
            range_m: Maximum range in meters
            update_freq_hz: Update frequency in Hz
            fov_h: Horizontal field of view in degrees
            fov_v: Vertical field of view in degrees
            noise_std: Standard deviation of noise
            low_quality: Whether to use lower quality settings for better performance
        """
        super().__init__(vehicle, range_m, update_freq_hz)
        
        self.low_quality = low_quality
        
        # Apply quality reductions if in low quality mode
        if low_quality:
            # Reduce number of points for better performance
            num_layers = max(4, num_layers // 2)
            points_per_layer = max(450, points_per_layer // 4)
            # Decrease raycast range
            range_m = range_m * 0.7
            # Reduce noise computation
            noise_std = 0
        
        self.num_layers = num_layers
        self.points_per_layer = points_per_layer
        self.fov_h = fov_h
        self.fov_v = fov_v
        self.noise_std = noise_std
        
        # Initialize sensor data structures
        self.points = []  # List of (x, y, z, intensity) points
        self.points_np = np.zeros((0, 4))  # For more efficient numpy operations
        
        # Calculate the angular resolution
        self.angular_resolution_h = fov_h / points_per_layer if points_per_layer > 0 else 0
        self.angular_resolution_v = fov_v / num_layers if num_layers > 0 else 0
        
        # Set default position and orientation
        self.position_offset = np.array([0.0, 0.0, 1.8])  # Top of vehicle
        self.orientation_offset = np.array([0.0, 0.0, 0.0])  # Forward
        
        # Make sure update_interval is set correctly
        self.update_frequency = update_freq_hz
        self.update_interval = 1.0 / update_freq_hz if update_freq_hz > 0 else 0.1
        
        # Time of last update
        self.last_update_time = 0
        
        # Visualization window (initialized when needed)
        self.vis_window = None
        
        # Cache for ground plane points to avoid recalculating every update
        self._ground_points_cache = None
        self._last_vehicle_position = None
        
        # Optimization: Pre-compute ray directions for better performance
        self.ray_directions = self._precompute_ray_directions()
        
        # Set ground recalculation distance based on quality setting
        self.ground_recalc_distance = 10.0 if low_quality else 5.0
        
        # Skip ratio for update to further improve performance in low quality mode
        self.skip_counter = 0
        self.skip_ratio = 2 if low_quality else 1  # Skip every other update in low quality
        
    def _precompute_ray_directions(self):
        """Pre-compute ray directions for better performance."""
        directions = []
        
        # Calculate ray directions based on sensor parameters
        for layer in range(self.num_layers):
            # Calculate vertical angle
            vert_angle = -self.fov_v / 2.0 + layer * self.angular_resolution_v
            
            # Calculate horizontal angles and directions for this layer
            for point in range(self.points_per_layer):
                # Calculate horizontal angle
                horz_angle = -self.fov_h / 2.0 + point * self.angular_resolution_h
                
                # Convert to radians
                vert_rad = np.radians(vert_angle)
                horz_rad = np.radians(horz_angle)
                
                # Calculate direction vector (normalized)
                x = np.cos(vert_rad) * np.cos(horz_rad)
                y = np.cos(vert_rad) * np.sin(horz_rad)
                z = np.sin(vert_rad)
                
                # Add to list of directions
                directions.append([x, y, z])
        
        return np.array(directions)
    
    def update(self, scene_data, current_time):
        """Update LiDAR sensor with current scene data.
        
        Args:
            scene_data: Current scene data
            current_time: Current simulation time
            
        Returns:
            bool: Whether the sensor was updated
        """
        # Check if update is needed based on time interval
        if not self.enabled or current_time - self.last_update_time < self.update_interval:
            return False
        
        # Skip updates based on skip ratio in low quality mode
        if self.low_quality:
            self.skip_counter += 1
            if self.skip_counter % self.skip_ratio != 0:
                return False
            
        # Get vehicle state and environment objects
        vehicle_pos = np.array(self.vehicle['position'])
        vehicle_rot = np.array(self.vehicle['rotation'])
        
        # Calculate sensor position and orientation in world frame
        sensor_pos, sensor_rot = self._get_sensor_world_transform()
        
        # Check if the vehicle position has changed significantly
        needs_ground_recalculation = (
            self._last_vehicle_position is None or 
            np.linalg.norm(vehicle_pos - self._last_vehicle_position) > self.ground_recalc_distance
        )
        
        # Update ground points cache if needed
        if needs_ground_recalculation:
            self._ground_points_cache = self._generate_ground_points(sensor_pos)
            self._last_vehicle_position = vehicle_pos.copy()
            
        # Get objects from the scene
        road_objects = scene_data.get('objects', [])
        
        # Skip ray casting for objects beyond a certain distance for performance
        max_obj_distance = self.range + (10.0 if not self.low_quality else 5.0)
        nearby_objects = [
            obj for obj in road_objects 
            if np.linalg.norm(np.array(obj['position']) - sensor_pos) < max_obj_distance
        ]
        
        # Create point cloud using parallel batch processing for better performance
        all_points = self._cast_rays_batch(sensor_pos, sensor_rot, nearby_objects)
        
        # Add ground points to point cloud
        if self._ground_points_cache is not None:
            all_points = np.vstack([all_points, self._ground_points_cache])
            
        # Add noise to point cloud
        if self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, size=(len(all_points), 3))
            all_points[:, :3] += noise
            
        # Update point cloud
        self.points_np = all_points
        
        # Update time
        self.last_update_time = current_time
        
        return True
        
    def _cast_rays_batch(self, sensor_pos, sensor_rot, objects):
        """Cast all rays in a batch for better performance.
        
        Args:
            sensor_pos: Sensor position in world frame
            sensor_rot: Sensor rotation in world frame
            objects: List of objects in the scene
            
        Returns:
            np.ndarray: Array of points (x, y, z, intensity)
        """
        # Optimization: Process rays in smaller batches instead of all at once
        batch_size = min(1000, len(self.ray_directions))
        all_points = []
        
        # Set sampling rate - in low quality mode, we'll sample fewer rays
        sampling_rate = 2 if self.low_quality else 1
        
        # Process rays in batches
        for batch_start in range(0, len(self.ray_directions), batch_size * sampling_rate):
            batch_end = min(batch_start + batch_size * sampling_rate, len(self.ray_directions))
            # If in low quality mode, take every Nth ray direction
            if self.low_quality:
                indices = range(batch_start, batch_end, sampling_rate)
                batch_directions = self.ray_directions[indices]
            else:
                batch_directions = self.ray_directions[batch_start:batch_end]
            
            # Transform ray directions to world frame
            # First, create rotation matrix from sensor rotation
            rotation_matrix = self._rotation_matrix_from_euler(sensor_rot)
            
            # Apply rotation to all directions in the batch
            world_directions = np.dot(batch_directions, rotation_matrix.T)
            
            # Initialize points for this batch
            batch_points = []
            
            # Process each ray in the batch
            for i, direction in enumerate(world_directions):
                # Normalize direction
                direction = direction / np.linalg.norm(direction)
                
                # Ray origin and direction
                ray_origin = sensor_pos
                ray_direction = direction
                
                # Initialize variables to track closest hit
                closest_dist = self.range + (10.0 if not self.low_quality else 5.0)
                closest_point = None
                closest_intensity = 0
                
                # Check intersections with objects
                for obj in objects:
                    # Skip objects that are too far away (quick rejection)
                    obj_pos = np.array(obj['position'])
                    obj_dist = np.linalg.norm(obj_pos - ray_origin)
                    if obj_dist > self.range + obj['dimensions'][0]:
                        continue
                    
                    # Check intersection with object
                    hit, dist, hit_point, normal = self._check_object_intersection(
                        ray_origin, ray_direction, obj
                    )
                    
                    # Update closest hit if intersection found
                    if hit and dist < closest_dist:
                        closest_dist = dist
                        closest_point = hit_point
                        
                        # Calculate intensity based on object type and angle
                        if normal is not None:
                            # Calculate angle between ray and normal
                            cos_angle = np.abs(np.dot(ray_direction, normal))
                            
                            # Look up reflectivity based on object type
                            reflectivity = self._get_reflectivity(obj['type'])
                            
                            # Calculate intensity based on reflectivity and angle
                            intensity = reflectivity * cos_angle * (1.0 - dist / self.range)
                            closest_intensity = max(0.1, min(1.0, intensity))
                        else:
                            closest_intensity = 0.5  # Default
                
                # Add point to batch if intersection found
                if closest_point is not None:
                    point_with_intensity = np.append(closest_point, closest_intensity)
                    batch_points.append(point_with_intensity)
            
            # Add batch points to all points
            if batch_points:
                all_points.extend(batch_points)
        
        # Convert to numpy array
        return np.array(all_points) if all_points else np.zeros((0, 4))
    
    def _check_object_intersection(self, ray_origin, ray_direction, obj):
        """Check intersection between ray and object.
        
        Args:
            ray_origin: Ray origin in world frame
            ray_direction: Ray direction in world frame
            obj: Object to check intersection with
            
        Returns:
            tuple: (hit, distance, hit_point, normal)
        """
        # In low quality mode, use simpler intersection test for certain object types
        if self.low_quality and hasattr(obj['type'], 'name') and obj['type'].name in ['PEDESTRIAN', 'TRAFFIC_SIGN']:
            # Use simpler sphere intersection test for small objects
            obj_pos = np.array(obj['position'])
            obj_dim = np.array(obj['dimensions'])
            
            # Use maximum dimension as sphere radius
            radius = max(obj_dim) / 2.0
            
            # Vector from ray origin to sphere center
            oc = ray_origin - obj_pos
            
            # Quadratic coefficients
            a = np.dot(ray_direction, ray_direction)
            b = 2.0 * np.dot(oc, ray_direction)
            c = np.dot(oc, oc) - radius * radius
            
            # Discriminant
            disc = b * b - 4 * a * c
            
            if disc < 0:
                return False, np.inf, None, None
            
            # Find closest intersection
            sqrt_disc = np.sqrt(disc)
            t1 = (-b - sqrt_disc) / (2.0 * a)
            t2 = (-b + sqrt_disc) / (2.0 * a)
            
            # Get smallest positive t
            t = t1 if t1 > 0 else t2
            
            if t < 0:
                return False, np.inf, None, None
                
            # Calculate hit point
            hit_point = ray_origin + ray_direction * t
            
            # Calculate normal (pointing from center to hit point)
            normal = (hit_point - obj_pos) / radius
            
            return True, t, hit_point, normal
        
        # Simplified bounding box intersection test for performance
        # This approximates objects as oriented bounding boxes
        
        # Get object properties
        obj_pos = np.array(obj['position'])
        obj_rot = np.array(obj['rotation'])
        obj_dim = np.array(obj['dimensions'])
        
        # Create inverse transformation to object space
        inv_rot_mat = self._rotation_matrix_from_euler(obj_rot).T
        
        # Transform ray to object space
        obj_ray_origin = ray_origin - obj_pos
        obj_ray_origin = np.dot(inv_rot_mat, obj_ray_origin)
        obj_ray_dir = np.dot(inv_rot_mat, ray_direction)
        
        # Half dimensions
        half_dim = obj_dim / 2.0
        
        # Check intersection with AABB in object space
        t_min = -np.inf
        t_max = np.inf
        
        normal = None
        
        # For each axis
        for i in range(3):
            if np.abs(obj_ray_dir[i]) < 1e-6:
                # Ray is parallel to this axis
                if obj_ray_origin[i] < -half_dim[i] or obj_ray_origin[i] > half_dim[i]:
                    # Ray origin is outside box on this axis, no intersection
                    return False, np.inf, None, None
            else:
                # Compute intersection with the two slabs for this axis
                t1 = (-half_dim[i] - obj_ray_origin[i]) / obj_ray_dir[i]
                t2 = (half_dim[i] - obj_ray_origin[i]) / obj_ray_dir[i]
                
                # Ensure t1 <= t2
                if t1 > t2:
                    t1, t2 = t2, t1
                
                # Update t_min and t_max
                if t1 > t_min:
                    t_min = t1
                    # Normal points out from the box surface
                    normal_idx = i
                    normal_sign = -1 if obj_ray_dir[i] > 0 else 1
                
                if t2 < t_max:
                    t_max = t2
                
                # No intersection if t_max < 0 or t_min > t_max
                if t_max < 0 or t_min > t_max:
                    return False, np.inf, None, None
        
        # If we get here, there's an intersection
        # t_min is the distance to the first intersection
        if t_min < 0:
            # Ray origin is inside the box
            t = t_max
        else:
            t = t_min
        
        # Calculate hit point in world space
        hit_point = ray_origin + ray_direction * t
        
        # Calculate normal in world space
        if normal is None:
            # Default normal
            normal = np.array([0, 0, 1])
        else:
            # Create normal vector
            normal = np.zeros(3)
            normal[normal_idx] = normal_sign
            
            # Transform normal to world space
            rotation_matrix = self._rotation_matrix_from_euler(obj_rot)
            normal = np.dot(rotation_matrix, normal)
        
        return True, t, hit_point, normal
    
    def _generate_ground_points(self, sensor_pos):
        """Generate points representing the ground plane.
        
        Args:
            sensor_pos: Sensor position in world frame
            
        Returns:
            np.ndarray: Array of ground points (x, y, z, intensity)
        """
        # Determine ground height (using simple flat ground model)
        ground_height = 0.0
        
        # Calculate distance from sensor to ground
        height_above_ground = sensor_pos[2] - ground_height
        
        # Maximum radius to generate ground points (limited by sensor range)
        max_radius = min(self.range, 50.0)  # Limit for performance
        
        # Generate grid of ground points with increasing spacing for efficiency
        ground_points = []
        
        # Adjust grid density based on quality setting
        if self.low_quality:
            num_rings = 10  # Half the rings in low quality
            points_per_ring = 36  # Every 10 degrees in low quality
        else:
            num_rings = 20
            points_per_ring = 72  # Every 5 degrees
        
        for ring in range(1, num_rings + 1):
            # Radius increases exponentially for better distribution
            radius = max_radius * (ring / num_rings) ** 1.5
            
            for point in range(points_per_ring):
                angle = 2 * np.pi * point / points_per_ring
                
                # Calculate x, y coordinates
                x = sensor_pos[0] + radius * np.cos(angle)
                y = sensor_pos[1] + radius * np.sin(angle)
                z = ground_height
                
                # Calculate intensity based on distance
                dist = np.sqrt((x - sensor_pos[0])**2 + (y - sensor_pos[1])**2 + (z - sensor_pos[2])**2)
                intensity = max(0.1, 0.5 * (1.0 - dist / self.range))
                
                # Add ground point
                ground_points.append([x, y, z, intensity])
        
        return np.array(ground_points)
    
    def _get_sensor_world_transform(self):
        """Get sensor position and orientation in world frame.
        
        Returns:
            tuple: (position, rotation) in world frame
        """
        # Get vehicle transform
        vehicle_pos = np.array(self.vehicle['position'])
        vehicle_rot = np.array(self.vehicle['rotation'])
        
        # Create rotation matrix from vehicle rotation
        vehicle_rotation_matrix = self._rotation_matrix_from_euler(vehicle_rot)
        
        # Transform sensor position offset to world frame
        sensor_pos_offset_world = np.dot(vehicle_rotation_matrix, self.position_offset)
        
        # Calculate sensor position in world frame
        sensor_pos = vehicle_pos + sensor_pos_offset_world
        
        # Calculate sensor rotation in world frame (simple addition for Euler angles)
        # Note: This is a simplification, proper rotation composition would use quaternions
        sensor_rot = vehicle_rot + self.orientation_offset
        
        return sensor_pos, sensor_rot
    
    def _rotation_matrix_from_euler(self, euler_angles):
        """Create rotation matrix from Euler angles (in degrees).
        
        Args:
            euler_angles: Euler angles [roll, pitch, yaw] in degrees
            
        Returns:
            np.ndarray: 3x3 rotation matrix
        """
        # Convert to radians
        roll = np.radians(euler_angles[0])
        pitch = np.radians(euler_angles[1])
        yaw = np.radians(euler_angles[2])
        
        # Create rotation matrix around z-axis (yaw)
        R_z = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        
        # Create rotation matrix around y-axis (pitch)
        R_y = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        
        # Create rotation matrix around x-axis (roll)
        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        
        # Combine rotations: R = R_z * R_y * R_x
        return np.dot(R_z, np.dot(R_y, R_x))
    
    def _get_reflectivity(self, object_type):
        """Get reflectivity value based on object type.
        
        Args:
            object_type: Type of object
            
        Returns:
            float: Reflectivity value (0-1)
        """
        # Typical reflectivity values for different materials
        reflectivity_map = {
            'vehicle': 0.8,
            'pedestrian': 0.5,
            'road': 0.1,
            'traffic_sign': 0.9,
            'barrier': 0.7,
            'building': 0.6,
            'vegetation': 0.4
        }
        
        # Default reflectivity if type not found
        return reflectivity_map.get(object_type, 0.5)
    
    def get_point_cloud(self):
        """Get the current point cloud.
        
        Returns:
            np.ndarray: Array of points (x, y, z, intensity)
        """
        return self.points_np
    
    def visualize_point_cloud(self, color_by='intensity'):
        """Visualize the point cloud.
        
        Args:
            color_by: How to color the points ('height', 'intensity', or 'distance')
        """
        import open3d as o3d
        
        # Check if we have points
        if len(self.points_np) == 0:
            print("No LiDAR points to visualize")
            return
        
        # Extract point coordinates (x, y, z)
        points = self.points_np[:, :3]
        
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Color points based on specified property
        if color_by == 'height':
            # Color by height (z value)
            height = points[:, 2]
            min_height = np.min(height)
            max_height = np.max(height)
            height_range = max_height - min_height if max_height > min_height else 1.0
            
            # Create colormap (blue to red)
            colors = np.zeros((len(points), 3))
            norm_height = (height - min_height) / height_range
            colors[:, 0] = norm_height  # Red
            colors[:, 2] = 1.0 - norm_height  # Blue
            
        elif color_by == 'intensity':
            # Color by intensity
            intensity = self.points_np[:, 3]
            
            # Create colormap (gray)
            colors = np.zeros((len(points), 3))
            for i in range(3):
                colors[:, i] = intensity
                
        elif color_by == 'distance':
            # Color by distance from sensor
            dist = np.linalg.norm(points, axis=1)
            max_dist = np.max(dist)
            
            # Create colormap (blue to red)
            colors = np.zeros((len(points), 3))
            norm_dist = dist / max_dist if max_dist > 0 else dist
            colors[:, 0] = norm_dist  # Red
            colors[:, 1] = 0.5 * (1.0 - norm_dist)  # Green
            colors[:, 2] = 1.0 - norm_dist  # Blue
            
        else:
            # Default color (white)
            colors = np.ones((len(points), 3))
        
        # Set colors
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Create visualizer if not already created
        if self.vis_window is None:
            self.vis_window = o3d.visualization.Visualizer()
            self.vis_window.create_window(window_name="LiDAR Point Cloud", width=800, height=600)
            
            # Add point cloud
            self.vis_window.add_geometry(pcd)
            
            # Set initial viewpoint
            view = self.vis_window.get_view_control()
            view.set_zoom(0.8)
            view.set_up([0, 0, 1])  # Set up direction
            view.set_front([1, 0, 0])  # Set front direction
            
            # Create coordinate frame
            coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=5.0, origin=[0, 0, 0]
            )
            self.vis_window.add_geometry(coordinate_frame)
            
        else:
            # Update point cloud
            self.vis_window.update_geometry(pcd)
        
        # Update visualization
        self.vis_window.poll_events()
        self.vis_window.update_renderer()
        
    def close_visualization(self):
        """Close the visualization window."""
        if self.vis_window is not None:
            self.vis_window.destroy_window()
            self.vis_window = None

    def get_data(self):
        """Get the LiDAR point cloud data.
        
        Returns:
            np.ndarray: Array of points (x, y, z, intensity)
        """
        return self.points_np
    
    def visualize(self):
        """Visualize LiDAR point cloud.
        
        Returns:
            tuple: (fig, ax) the matplotlib figure and axis
        """
        try:
            # Skip visualization if not enough data has changed since last visualization
            if hasattr(self, '_last_visualization_time') and hasattr(self, 'last_update_time'):
                # Only update visualization if at least 0.5 seconds have passed since the last one
                if self.last_update_time - self._last_visualization_time < 0.5:
                    if hasattr(self, 'fig') and hasattr(self, 'ax') and self.fig is not None and self.ax is not None:
                        return self.fig, self.ax
                    else:
                        return None, None
                        
            # Call the point cloud visualization with error handling
            result = self.visualize_point_cloud(color_by='intensity')
            
            # Store visualization time
            self._last_visualization_time = self.last_update_time
            
            return result
        except Exception as e:
            print(f"LiDAR visualization error: {e}")
            return None, None 