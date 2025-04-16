"""
Sensor fusion module for autonomous vehicle simulation.
Combines data from various sensors (radar, lidar, camera) for improved perception.
"""

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import time


class SensorFusion:
    """Combines data from different sensors for improved perception and visualization."""
    
    def __init__(self, sensor_manager):
        """Initialize the sensor fusion.
        
        Args:
            sensor_manager: The sensor manager containing all sensors
        """
        self.sensor_manager = sensor_manager
        self.last_update_time = 0
        self.update_interval = 0.1  # Update at 10Hz by default
        
        # Visualization objects
        self.vis_window = None
        self.detected_objects = []
        
        # Store current fused data
        self.current_lidar_points = np.zeros((0, 4))
        self.current_radar_points = []
        self.current_radar_velocities = []
        self.current_camera_image = None
        
        # Store detected objects for tracking
        self.tracked_objects = []
        
    def update(self, scene_data, current_time):
        """Update the sensor fusion with data from all sensors.
        
        Args:
            scene_data: Current scene data
            current_time: Current simulation time
            
        Returns:
            bool: Whether the sensor fusion was updated
        """
        # Check if update is needed based on time interval
        if current_time - self.last_update_time < self.update_interval:
            return False
            
        # Get sensor data from all sensors
        sensor_data = self.sensor_manager.get_sensor_data()
        
        # Extract LiDAR data
        if 'lidar' in sensor_data:
            self.current_lidar_points = sensor_data['lidar']
        
        # Extract Radar data
        if 'front_radar' in sensor_data:
            self.current_radar_points, self.current_radar_velocities = sensor_data['front_radar']
        
        # Extract Camera data
        if 'front_camera' in sensor_data:
            self.current_camera_image = sensor_data['front_camera']
        
        # Perform fusion to detect and track objects
        self._fuse_sensor_data(scene_data)
        
        # Update last update time
        self.last_update_time = current_time
        
        return True
    
    def _fuse_sensor_data(self, scene_data):
        """Perform actual sensor fusion by combining data from different sensors.
        
        Args:
            scene_data: Current scene data
        """
        # Get actual objects from scene data for ground truth
        # In a real system, these would be detected from sensor data
        self.detected_objects = scene_data.get('objects', [])
        
        # In a real system, we would run various algorithms here:
        # 1. Object detection from LiDAR (clustering, segmentation)
        # 2. Object detection from camera (deep learning)
        # 3. Object detection from radar (clustering)
        # 4. Association between detections from different sensors
        # 5. Tracking of objects over time (Kalman filter, etc.)
        # ...
    
    def visualize(self):
        """Visualize the fused sensor data using Open3D."""
        import open3d as o3d
        
        # Check if we have LiDAR points
        if len(self.current_lidar_points) == 0:
            return
        
        # Extract point coordinates (x, y, z)
        points = self.current_lidar_points[:, :3]
        
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Assign default colors (by height)
        colors = self._color_by_height(points)
        
        # Enhance with radar velocity information if available
        if len(self.current_radar_points) > 0:
            colors = self._incorporate_radar_velocity(points, colors)
        
        # Create bounding boxes for detected objects
        bboxes = self._create_bounding_boxes()
        
        # Set point cloud colors
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Create visualizer if not already created
        if self.vis_window is None:
            self.vis_window = o3d.visualization.Visualizer()
            self.vis_window.create_window(window_name="Sensor Fusion", width=1024, height=768)
            
            # Add point cloud
            self.vis_window.add_geometry(pcd)
            
            # Add coordinate frame
            coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=5.0, origin=[0, 0, 0]
            )
            self.vis_window.add_geometry(coordinate_frame)
            
            # Add bounding boxes
            for bbox in bboxes:
                self.vis_window.add_geometry(bbox)
            
            # Set initial viewpoint
            view = self.vis_window.get_view_control()
            view.set_zoom(0.8)
            view.set_up([0, 0, 1])  # Z is up
            view.set_front([1, 0, 0])  # Looking forward along X axis
            
        else:
            # Update point cloud
            self.vis_window.update_geometry(pcd)
            
            # Remove old bounding boxes
            for idx in range(len(bboxes)):
                self.vis_window.remove_geometry(bboxes[idx], False)
                
            # Add new bounding boxes
            for bbox in bboxes:
                self.vis_window.add_geometry(bbox, False)
        
        # Update visualization
        self.vis_window.poll_events()
        self.vis_window.update_renderer()
    
    def _color_by_height(self, points):
        """Color points by height (z value).
        
        Args:
            points: Array of (x, y, z) coordinates
            
        Returns:
            np.ndarray: Array of (r, g, b) colors
        """
        # Get height (z value)
        height = points[:, 2]
        min_height = np.min(height)
        max_height = np.max(height)
        height_range = max_height - min_height if max_height > min_height else 1.0
        
        # Create color map (ground = blue, high objects = red)
        colors = np.zeros((len(points), 3))
        norm_height = (height - min_height) / height_range
        colors[:, 0] = norm_height  # Red
        colors[:, 2] = 1.0 - norm_height  # Blue
        
        return colors
    
    def _incorporate_radar_velocity(self, lidar_points, base_colors):
        """Add velocity information from radar to LiDAR points.
        
        Args:
            lidar_points: Array of (x, y, z) coordinates from LiDAR
            base_colors: Initial colors assigned to points
            
        Returns:
            np.ndarray: Updated colors with radar velocity information
        """
        # If no radar points or no LiDAR points, return base colors
        if len(self.current_radar_points) == 0 or len(lidar_points) == 0:
            return base_colors
            
        # Create updated colors array
        enhanced_colors = base_colors.copy()
        
        # Convert radar points from spherical to Cartesian coordinates
        radar_cart_points = []
        for point in self.current_radar_points:
            # Extract distance, azimuth, elevation
            distance, azimuth_deg, elevation_deg = point
            
            # Convert to radians
            azimuth = np.radians(azimuth_deg)
            elevation = np.radians(elevation_deg)
            
            # Convert to Cartesian
            x = distance * np.cos(elevation) * np.cos(azimuth)
            y = distance * np.cos(elevation) * np.sin(azimuth)
            z = distance * np.sin(elevation)
            
            radar_cart_points.append([x, y, z])
        
        radar_cart_points = np.array(radar_cart_points)
        
        # Associate radar points with LiDAR points (nearest neighbor)
        if len(radar_cart_points) > 0:
            # For each radar point, find nearby LiDAR points
            for i, radar_point in enumerate(radar_cart_points):
                # Calculate squared distances to all LiDAR points
                # (Faster than calculating actual distances)
                squared_dists = np.sum((lidar_points - radar_point) ** 2, axis=1)
                
                # Find points within a threshold
                nearby_mask = squared_dists < 9.0  # Within 3 meters
                
                # Apply velocity-based coloring to nearby points
                if np.any(nearby_mask):
                    velocity = self.current_radar_velocities[i]
                    
                    # Normalize velocity to color scale (-10 to +10 m/s)
                    norm_vel = max(-1.0, min(1.0, velocity / 10.0))
                    
                    # Adjust colors: moving toward = green, away = red
                    if velocity < 0:  # Moving toward radar (negative radial velocity)
                        enhanced_colors[nearby_mask, 0] = 0.0  # Red
                        enhanced_colors[nearby_mask, 1] = -norm_vel  # Green
                        enhanced_colors[nearby_mask, 2] = 0.0  # Blue
                    else:  # Moving away
                        enhanced_colors[nearby_mask, 0] = norm_vel  # Red
                        enhanced_colors[nearby_mask, 1] = 0.0  # Green
                        enhanced_colors[nearby_mask, 2] = 0.0  # Blue
        
        return enhanced_colors
    
    def _create_bounding_boxes(self):
        """Create 3D bounding boxes for detected objects.
        
        Returns:
            list: List of Open3D line sets for bounding boxes
        """
        bboxes = []
        
        for obj in self.detected_objects:
            # Skip if it's the ego vehicle
            if obj.get('autonomous', False):
                continue
                
            # Get object position, rotation, and dimensions
            pos = np.array(obj.get('position', [0, 0, 0]))
            rot = np.array(obj.get('rotation', [0, 0, 0]))
            dims = np.array(obj.get('dimensions', [1, 1, 1]))
            
            # Get object velocity for color
            vel = np.array(obj.get('velocity', [0, 0, 0]))
            velocity_magnitude = np.linalg.norm(vel)
            
            # Create a box
            box = o3d.geometry.OrientedBoundingBox()
            box.center = pos + np.array([0, 0, dims[2]/2])  # Center at object position
            box.R = self._rotation_matrix_from_euler(rot)  # Rotation
            box.extent = dims  # Dimensions
            
            # Set color based on object type and velocity
            obj_type = obj.get('type', 'UNKNOWN')
            
            if obj_type == 'CAR':
                color = [0, 1, 0]  # Green
            elif obj_type == 'TRUCK':
                color = [0, 0.8, 0.8]  # Cyan
            elif obj_type == 'PEDESTRIAN':
                color = [1, 0, 1]  # Magenta
            else:
                color = [1, 1, 0]  # Yellow
                
            # Adjust brightness based on velocity
            brightness = min(1.0, 0.5 + velocity_magnitude / 10.0)
            color = [c * brightness for c in color]
            
            # Set color
            box.color = color
            
            # Add to list
            bboxes.append(box)
            
            # Add velocity vector if moving
            if velocity_magnitude > 0.5:
                line_points = [pos.tolist(), (pos + vel).tolist()]
                # Create line set for velocity vector
                line = o3d.geometry.LineSet()
                line.points = o3d.utility.Vector3dVector(line_points)
                line.lines = o3d.utility.Vector2iVector([[0, 1]])
                line.colors = o3d.utility.Vector3dVector([[1, 0, 0]])  # Red for velocity
                bboxes.append(line)
        
        return bboxes
    
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
    
    def close(self):
        """Close the visualization window."""
        if self.vis_window is not None:
            self.vis_window.destroy_window()
            self.vis_window = None 