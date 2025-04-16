"""
Camera sensor for autonomous vehicle simulation.
"""

import numpy as np
import cv2
import threading
import queue
import time
from .base_sensor import BaseSensor

# Global visualization queue and thread objects
vis_queue = queue.Queue()
vis_thread = None
vis_running = False

# Visualization thread function
def visualization_thread_func():
    global vis_running
    vis_running = True
    
    window_created = False
    
    while vis_running:
        try:
            # Get image from queue with timeout
            try:
                img_data = vis_queue.get(timeout=0.5)
                if img_data is None:
                    break
                
                img, window_name = img_data
                
                # Check if image exists
                if img is None or img.size == 0:
                    continue
                
                # Show image using OpenCV
                cv2.imshow(window_name, img)
                cv2.waitKey(1)  # Update window and wait for 1ms
                window_created = True
            except queue.Empty:
                # No image available, just update windows
                if window_created:
                    cv2.waitKey(1)
            
            # Short sleep to avoid high CPU usage
            time.sleep(0.05)
            
        except Exception as e:
            print(f"Visualization thread error: {e}")
    
    # Close all windows when thread exits
    cv2.destroyAllWindows()
    vis_running = False

class CameraSensor(BaseSensor):
    """Camera sensor for autonomous vehicle simulation."""
    
    def __init__(self, vehicle, width=1280, height=720, fov=70.0, 
                 range_m=100.0, update_freq_hz=10.0, low_quality=False):
        """Initialize camera sensor.
        
        Args:
            vehicle: Vehicle this sensor is attached to
            width: Image width in pixels
            height: Image height in pixels
            fov: Horizontal field of view in degrees
            range_m: Maximum range in meters
            update_freq_hz: Update frequency in Hz
            low_quality: Whether to use lower quality settings for better performance
        """
        super().__init__(vehicle, range_m, update_freq_hz)
        
        self.low_quality = low_quality
        
        # Apply quality reductions if in low quality mode
        if low_quality:
            # Reduce resolution
            width = max(640, width // 2)
            height = max(360, height // 2)
            # Reduce range for faster processing
            range_m = range_m * 0.7
        
        self.width = width
        self.height = height
        self.fov = fov
        
        # Create empty image
        self.image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Set default position and orientation
        self.position_offset = np.array([0.0, 0.0, 1.5])  # Front of vehicle, slightly elevated
        self.orientation_offset = np.array([0.0, 0.0, 0.0])  # Forward
        
        # Calculate camera intrinsic parameters
        self.fx = width / (2 * np.tan(np.radians(fov / 2)))
        self.fy = self.fx
        self.cx = width / 2
        self.cy = height / 2
        
        # Build camera matrix
        self.camera_matrix = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ])
        
        # Ensure update_interval is set based on update frequency
        self.update_frequency = update_freq_hz  # Store the frequency
        self.update_interval = 1.0 / update_freq_hz if update_freq_hz > 0 else 0.1  # Calculate interval in seconds
        
        # Time of last update
        self.last_update_time = 0
        
        # Skip ratio for update to further improve performance in low quality mode
        self.skip_counter = 0
        self.skip_ratio = 2 if low_quality else 1  # Skip every other update in low quality
        
    def update(self, scene_data, current_time):
        """Update camera image with current scene data.
        
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
        
        # Create new image
        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Draw ground plane and sky
        self._draw_ground_plane(image, sensor_pos, sensor_rot)
        
        # Get objects from the scene
        objects = scene_data.get('objects', [])
        
        # Skip rendering objects beyond a certain distance for performance
        max_obj_distance = self.range + (10.0 if not self.low_quality else 5.0)
        nearby_objects = [
            obj for obj in objects 
            if np.linalg.norm(np.array(obj['position']) - sensor_pos) < max_obj_distance
        ]
        
        # Sort objects by distance (render far to near)
        nearby_objects.sort(key=lambda obj: -np.linalg.norm(np.array(obj['position']) - sensor_pos))
        
        # Render objects
        for obj in nearby_objects:
            # Skip object if the quality is low and object is far away
            obj_pos = np.array(obj['position'])
            obj_dist = np.linalg.norm(obj_pos - sensor_pos)
            
            # Skip small distant objects in low quality mode
            if self.low_quality and obj_dist > self.range * 0.7:
                # Check if object is small
                if 'dimensions' in obj:
                    obj_dim = np.array(obj['dimensions'])
                    if max(obj_dim) < 2.0:  # Skip small objects
                        continue
            
            self._render_object(image, obj, sensor_pos, sensor_rot)
        
        # Apply optional post-processing
        if not self.low_quality:
            # Add some image processing effects (like anti-aliasing)
            image = cv2.GaussianBlur(image, (3, 3), 0)
        
        # Update camera image
        self.image = image
        
        # Update time
        self.last_update_time = current_time
        
        return True
    
    def _draw_ground_plane(self, image, sensor_pos, sensor_rot):
        """Draw ground plane and sky in the image.
        
        Args:
            image: Image to draw on
            sensor_pos: Sensor position in world frame
            sensor_rot: Sensor rotation in world frame
        """
        # Create horizon line
        # Calculate camera direction vector in world frame
        rotation_matrix = self._rotation_matrix_from_euler(sensor_rot)
        camera_forward = rotation_matrix[:, 0]  # First column is forward direction
        
        # Calculate horizon line
        # The horizon is the line where the camera forward direction (projected to horizontal plane) intersects the image
        horizon_y = int(self.cy - self.fy * camera_forward[2] / np.sqrt(camera_forward[0]**2 + camera_forward[1]**2))
        
        # Fill sky (light blue)
        if horizon_y >= 0:
            sky_color = (204, 153, 102) if horizon_y < image.shape[0] else (0, 0, 0)
            image[:horizon_y, :] = sky_color
        
        # Fill ground (green/gray)
        if horizon_y < image.shape[0]:
            ground_height = 0.0  # Assume flat ground at z=0
            distance_to_ground = sensor_pos[2] - ground_height
            
            # Create ground grid for perspective effect
            # In low quality mode, use a simpler ground representation
            if self.low_quality:
                # Simple gradient ground
                ground_color = (90, 120, 90)  # Dark green
                horizon_y = max(0, horizon_y)
                image[horizon_y:, :] = ground_color
            else:
                # More detailed ground with grid
                # Draw ground with perspective grid
                for y in range(max(0, horizon_y), image.shape[0]):
                    # Calculate distance based on y-coordinate (perspective)
                    dist_factor = (y - horizon_y) / (image.shape[0] - horizon_y) if y > horizon_y else 0
                    dist_factor = min(1.0, dist_factor * 2.0)  # Scale for better visual effect
                    
                    # Calculate ground color based on distance (darker with distance)
                    ground_color = np.array([90, 120, 90], dtype=np.uint8)  # Base color (dark green)
                    brightness = 1.0 - 0.5 * dist_factor  # Darken with distance
                    color = (ground_color * brightness).astype(np.uint8)
                    
                    # Fill row with ground color
                    image[y, :] = color
                
                # Add grid lines for perspective effect
                grid_spacing_m = 5.0  # Grid spacing in meters
                grid_color = (50, 50, 50)  # Dark gray
                
                # Draw grid lines parallel to camera view
                num_lines = 10
                for i in range(1, num_lines + 1):
                    # Calculate distance in meters
                    dist_m = i * grid_spacing_m
                    
                    # Skip if beyond range
                    if dist_m > self.range:
                        break
                    
                    # Project distance to image y-coordinate (simplified perspective projection)
                    y = int(horizon_y + (image.shape[0] - horizon_y) * (1.0 - np.exp(-dist_m / 20.0)))
                    
                    if 0 <= y < image.shape[0]:
                        cv2.line(image, (0, y), (image.shape[1], y), grid_color, 1)
                
                # Draw grid lines perpendicular to camera view
                num_lines = 20
                for i in range(-num_lines // 2, num_lines // 2 + 1):
                    # Calculate angle in degrees
                    angle_deg = i * (self.fov / num_lines)
                    
                    # Calculate x-coordinate at bottom of image
                    bottom_x = int(self.cx + self.width * np.tan(np.radians(angle_deg)) / (2 * np.tan(np.radians(self.fov / 2))))
                    
                    if 0 <= bottom_x < image.shape[1]:
                        cv2.line(image, (int(self.cx), horizon_y), (bottom_x, image.shape[0] - 1), grid_color, 1)
    
    def _render_object(self, image, obj, sensor_pos, sensor_rot):
        """Render object in the camera image.
        
        Args:
            image: Image to render on
            obj: Object to render
            sensor_pos: Sensor position in world frame
            sensor_rot: Sensor rotation in world frame
        """
        # Get object properties
        obj_pos = np.array(obj['position'])
        obj_rot = np.array(obj.get('rotation', [0, 0, 0]))
        obj_dim = np.array(obj.get('dimensions', [1, 1, 1]))
        obj_type = obj.get('type', 'unknown')
        
        # Calculate object corners in 3D space
        corners_3d = self._get_object_corners(obj_pos, obj_rot, obj_dim)
        
        # Project corners to image
        corners_2d = self._project_points_to_image(corners_3d, sensor_pos, sensor_rot)
        
        # Check if object is behind camera or all corners are outside the image
        if corners_2d is None or len(corners_2d) == 0:
            return
        
        # Determine object color based on type
        color = self._get_object_color(obj_type)
        
        # Render based on object distance and quality setting
        obj_dist = np.linalg.norm(obj_pos - sensor_pos)
        
        # In low quality mode, use simpler rendering for distant objects
        if self.low_quality and obj_dist > self.range * 0.4:
            # Simple bounding box for distant objects
            # Find bounding rectangle
            x_coords = [int(p[0]) for p in corners_2d if 0 <= p[0] < self.width]
            y_coords = [int(p[1]) for p in corners_2d if 0 <= p[1] < self.height]
            
            if len(x_coords) > 1 and len(y_coords) > 1:
                x_min, x_max = max(0, min(x_coords)), min(self.width - 1, max(x_coords))
                y_min, y_max = max(0, min(y_coords)), min(self.height - 1, max(y_coords))
                
                # Draw simple box
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 1)
                
                # Draw filled center to indicate object type
                center_x = (x_min + x_max) // 2
                center_y = (y_min + y_max) // 2
                size = max(2, min((x_max - x_min) // 4, (y_max - y_min) // 4))
                cv2.circle(image, (center_x, center_y), size, color, -1)
            
        else:
            # Advanced rendering for close objects
            # Draw edges of the object
            # Define edges as pairs of corner indices
            edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
                (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
                (0, 4), (1, 5), (2, 6), (3, 7)   # Connecting edges
            ]
            
            # Draw each edge
            for i, j in edges:
                if i < len(corners_2d) and j < len(corners_2d):
                    p1 = corners_2d[i].astype(int)
                    p2 = corners_2d[j].astype(int)
                    
                    # Check if points are within image bounds (with some margin)
                    if (-100 <= p1[0] < self.width + 100 and -100 <= p1[1] < self.height + 100 and
                        -100 <= p2[0] < self.width + 100 and -100 <= p2[1] < self.height + 100):
                        cv2.line(image, (p1[0], p1[1]), (p2[0], p2[1]), color, 1)
            
            # Draw visible faces with transparency
            faces = [
                [0, 1, 2, 3],  # Bottom face
                [4, 5, 6, 7],  # Top face
                [0, 1, 5, 4],  # Front face
                [1, 2, 6, 5],  # Right face
                [2, 3, 7, 6],  # Back face
                [3, 0, 4, 7]   # Left face
            ]
            
            # Create mask for filled polygons
            if not self.low_quality:
                mask = np.zeros_like(image)
                
                # Draw each face
                for face in faces:
                    # Get face corners
                    face_corners = [corners_2d[i] for i in face if i < len(corners_2d)]
                    
                    # Skip if not enough corners
                    if len(face_corners) < 3:
                        continue
                    
                    # Check if face normal is pointing towards camera (back-face culling)
                    # This is a simplified version, calculating normal correctly requires more work
                    
                    # Convert to integer points and correct format for fillPoly
                    face_points = np.array([[[int(p[0]), int(p[1])]] for p in face_corners], dtype=np.int32)
                    
                    # Draw filled polygon with alpha blending
                    alpha = 0.3  # Transparency factor
                    
                    # Fill face with color in mask
                    cv2.fillPoly(mask, [face_points], color)
                
                # Blend mask with image
                cv2.addWeighted(image, 1.0, mask, 0.3, 0, image)
            
            # Add object type label for close objects
            if obj_dist < self.range * 0.3:
                # Get centroid of object
                centroid_3d = obj_pos
                centroid_2d = self._project_points_to_image(centroid_3d.reshape(1, 3), sensor_pos, sensor_rot)
                
                if centroid_2d is not None and len(centroid_2d) > 0:
                    x, y = int(centroid_2d[0][0]), int(centroid_2d[0][1])
                    
                    # Check if within image bounds
                    if 0 <= x < self.width and 0 <= y < self.height:
                        label = str(obj_type)
                        cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    def _get_object_corners(self, position, rotation, dimensions):
        """Get corners of object bounding box in 3D world space.
        
        Args:
            position: Object position [x, y, z]
            rotation: Object rotation [roll, pitch, yaw] in degrees
            dimensions: Object dimensions [length, width, height]
            
        Returns:
            np.ndarray: 8x3 array of corner coordinates
        """
        # Half dimensions
        half_l, half_w, half_h = dimensions / 2
        
        # Define corner offsets from center (in object space)
        corners = np.array([
            [-half_l, -half_w, -half_h],  # Bottom face
            [half_l, -half_w, -half_h],
            [half_l, half_w, -half_h],
            [-half_l, half_w, -half_h],
            [-half_l, -half_w, half_h],   # Top face
            [half_l, -half_w, half_h],
            [half_l, half_w, half_h],
            [-half_l, half_w, half_h]
        ])
        
        # Create rotation matrix from object rotation
        rotation_matrix = self._rotation_matrix_from_euler(rotation)
        
        # Transform corners to world space
        corners_world = np.dot(corners, rotation_matrix.T) + position
        
        return corners_world
    
    def _project_points_to_image(self, points_3d, camera_pos, camera_rot):
        """Project 3D points to 2D image coordinates.
        
        Args:
            points_3d: Nx3 array of 3D points in world frame
            camera_pos: Camera position in world frame
            camera_rot: Camera rotation in world frame
            
        Returns:
            np.ndarray: Nx2 array of 2D image coordinates, or None if all points are behind camera
        """
        # Ensure points_3d is at least 2D
        points_3d = np.atleast_2d(points_3d)
        
        # Create rotation matrix from camera rotation
        rotation_matrix = self._rotation_matrix_from_euler(camera_rot)
        
        # Transform points to camera frame
        points_camera = []
        for point in points_3d:
            # Vector from camera to point
            point_rel = point - camera_pos
            
            # Transform to camera frame
            point_camera = np.dot(rotation_matrix, point_rel)
            points_camera.append(point_camera)
        
        points_camera = np.array(points_camera)
        
        # Check if any points are in front of camera
        if np.all(points_camera[:, 0] <= 0):
            return None  # All points are behind camera
        
        # Keep only points in front of camera for projection
        front_indices = points_camera[:, 0] > 0
        points_front = points_camera[front_indices]
        
        if len(points_front) == 0:
            return None
        
        # Project to image plane
        points_2d = []
        for point in points_front:
            # Perspective projection
            x, y, z = point
            u = self.cx + self.fx * y / x
            v = self.cy - self.fy * z / x
            
            points_2d.append([u, v])
        
        return np.array(points_2d)
    
    def _get_object_color(self, object_type):
        """Get color based on object type.
        
        Args:
            object_type: Type of object
            
        Returns:
            tuple: BGR color
        """
        # Color map for different object types
        color_map = {
            'vehicle': (0, 0, 255),        # Red
            'pedestrian': (0, 255, 255),   # Yellow
            'traffic_sign': (255, 0, 0),   # Blue
            'traffic_light': (0, 255, 0),  # Green
            'barrier': (255, 0, 255),      # Magenta
            'building': (128, 128, 128),   # Gray
            'vegetation': (0, 128, 0)      # Dark green
        }
        
        # Default color if type not found
        return color_map.get(object_type, (200, 200, 200))
    
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
    
    def get_image(self):
        """Get the current camera image.
        
        Returns:
            np.ndarray: Camera image
        """
        return self.image
    
    def get_data(self):
        """Get the sensor data.
        
        Returns:
            np.ndarray: Camera image
        """
        return self.image
    
    def draw_debug_overlay(self, display_image=False):
        """Draw debug overlay on the camera image and optionally display it.
        
        Args:
            display_image: Whether to display the image with the overlay
        """
        if self.image is None or self.image.size == 0:
            return
            
        # Create a copy of the image to draw on
        debug_image = self.image.copy()
        
        # Draw camera info
        text_color = (255, 255, 255)  # White
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Add camera parameters at the top
        cv2.putText(debug_image, f"FOV: {self.fov:.1f}°, Range: {self.range:.1f}m", 
                   (10, 20), font, 0.5, text_color, 1, cv2.LINE_AA)
                   
        # Add frame info
        cv2.putText(debug_image, f"Resolution: {self.width}x{self.height}", 
                   (10, 40), font, 0.5, text_color, 1, cv2.LINE_AA)
                   
        # Add quality mode info
        quality_text = "LOW QUALITY" if self.low_quality else "HIGH QUALITY"
        cv2.putText(debug_image, quality_text, 
                   (self.width - 150, 20), font, 0.5, text_color, 1, cv2.LINE_AA)
        
        # Draw crosshair in the center
        center_x, center_y = self.width // 2, self.height // 2
        cv2.line(debug_image, (center_x - 20, center_y), (center_x + 20, center_y), (0, 255, 0), 1)
        cv2.line(debug_image, (center_x, center_y - 20), (center_x, center_y + 20), (0, 255, 0), 1)
        
        # Draw field of view lines
        fov_rad = np.radians(self.fov / 2)
        line_length = 50
        x1 = int(center_x - line_length * np.sin(fov_rad))
        y1 = int(center_y + line_length * np.cos(fov_rad))
        x2 = int(center_x + line_length * np.sin(fov_rad))
        y2 = y1
        
        cv2.line(debug_image, (center_x, center_y), (x1, y1), (0, 255, 255), 1)
        cv2.line(debug_image, (center_x, center_y), (x2, y2), (0, 255, 255), 1)
        
        # Display the image if requested
        if display_image:
            cv2.imshow("Camera View", debug_image)
            cv2.waitKey(1)  # Update window and wait for 1ms
            
        # Store the debug image
        self.debug_image = debug_image
    
    def visualize_image(self, window_name='Camera'):
        """Visualize the camera image.
        
        Args:
            window_name: Name of the window
        """
        global vis_thread, vis_running
        
        # Check if image exists
        if self.image is None or self.image.size == 0:
            return
        
        # Start visualization thread if not already running
        if vis_thread is None or not vis_running:
            vis_thread = threading.Thread(target=visualization_thread_func)
            vis_thread.daemon = True  # Thread will exit when main program exits
            vis_thread.start()
        
        # Put image in queue for visualization thread
        try:
            # Don't block if queue is full
            vis_queue.put((self.image.copy(), window_name), block=False)
        except queue.Full:
            pass  # Skip frame if queue is full
        
    def visualize(self):
        """Visualize the camera image.
        
        Returns:
            tuple: (None, None) to match the return type of other sensor visualize methods
        """
        self.visualize_image("Camera")
        return None, None 