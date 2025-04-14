import numpy as np
import open3d as o3d
import time
import copy
import math
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

class Renderer:
    def __init__(self, width=1280, height=720, fullscreen=False):
        """Initialize the renderer.
        
        Args:
            width (int): Window width
            height (int): Window height
            fullscreen (bool): Whether to run in fullscreen mode
        """
        self.width = width
        self.height = height
        self.fullscreen = fullscreen
        
        # Initialize Open3D visualization
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(width=width, height=height, visible=True)
        
        # Set rendering options
        render_option = self.vis.get_render_option()
        render_option.background_color = np.array([0.05, 0.05, 0.05])  # Dark background
        render_option.point_size = 3.0  # Larger point size for better visibility
        render_option.show_coordinate_frame = True
        render_option.line_width = 5.0  # Thicker lines
        
        # Camera settings
        self.view_control = self.vis.get_view_control()
        
        # Object geometry cache
        self.geometries = {}
        self.point_cloud = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.point_cloud)
        
        # Create color legend using simple boxes
        self._create_legend()
        
        # Store maximum steering angle
        self.max_steering_angle = 0.5  # radians
        
        # Disable complex steering wheel visualizations that might cause hanging
        self.use_simple_steering_display = True
        
        # Create a visual steering wheel
        self.steering_wheel = self._create_steering_wheel()
        self.vis.add_geometry(self.steering_wheel)
        
        # Initialize camera after all objects are added
        self._setup_camera()
    
    def _setup_camera(self):
        """Setup the camera for the visualization."""
        # Get the view control
        self.view_control = self.vis.get_view_control()
        
        # Set initial camera position for a good starting view
        self.view_control.set_zoom(0.7)
        self.view_control.set_front([0, -1, -1])  # Looking from behind and above
        self.view_control.set_lookat([5.0, 0.0, 0.0])  # Looking ahead on the road
        self.view_control.set_up([0, 0, 1])  # Z-axis is up
        
        # Additional rendering options
        render_option = self.vis.get_render_option()
        render_option.point_size = 5
        render_option.background_color = np.array([0.1, 0.1, 0.2])
        
        # Keep track of last camera position to avoid jumpy behavior
        self.last_camera_pos = [0, -10, 10]
        self.camera_initialized = False
    
    def _create_legend(self):
        """Create a legend in the Open3D window using simple geometries."""
        # Define colors and labels - matches the colors in object_types.py
        legend_items = [
            ("Car", [0, 1, 0]),         # Green
            ("Truck", [0, 0.8, 0.8]),   # Teal
            ("Tricar", [1, 0.65, 0]),   # Orange
            ("Cyclist", [0, 0, 1]),     # Blue
            ("Pedestrian", [1, 0, 1])   # Magenta
        ]
        
        # Position for the legend items (top-left corner)
        x, y, z = -95, -40, 10
        box_size = 3
        spacing = 6
        
        # Add each legend item as a small cube
        for i, (_, color) in enumerate(legend_items):
            # Create a small cube
            cube = o3d.geometry.TriangleMesh.create_box(width=box_size, height=box_size, depth=box_size)
            cube.translate([x, y + i * spacing, z])
            cube.paint_uniform_color(color)
            self.vis.add_geometry(cube)
    
    def render(self, scene_data):
        """Render the current scene data.
        
        Args:
            scene_data (dict): Current scene data from the environment
        """
        try:
            # Update static point cloud if needed
            if 'static_point_cloud' in scene_data:
                self.update_static_point_cloud(scene_data['static_point_cloud'])
            
            # Update or add all objects
            if 'objects' in scene_data:
                self.update_objects(scene_data['objects'])
                
                # Find the autonomous vehicle to follow
                car_to_follow = None
                for obj in scene_data['objects']:
                    if obj.get('autonomous', False):
                        car_to_follow = obj
                        break
                
                # If we found a car to follow, update camera position
                if car_to_follow:
                    self._update_third_person_camera(car_to_follow)
                
            # Update the visualization
            self.vis.poll_events()
            self.vis.update_renderer()
            
            self._render_debug_info(scene_data)
            
        except Exception as e:
            print(f"Render error: {e}")
    
    def update_static_point_cloud(self, point_cloud_data):
        """Update the static point cloud.
        
        Args:
            point_cloud_data (dict): Point cloud data with 'points' and 'colors' keys
        """
        try:
            if not isinstance(point_cloud_data, dict) or 'points' not in point_cloud_data:
                return
                
            points = point_cloud_data['points']
            colors = point_cloud_data.get('colors', None)
            
            if len(points) == 0:
                return
                
            # Remove existing point cloud
            self.vis.remove_geometry(self.point_cloud, False)
            
            # Remove any existing road visualization
            if hasattr(self, 'road_edges'):
                self.vis.remove_geometry(self.road_edges, False)
            if hasattr(self, 'centerline'):
                self.vis.remove_geometry(self.centerline, False)
            
            # Update point cloud
            self.point_cloud = o3d.geometry.PointCloud()
            self.point_cloud.points = o3d.utility.Vector3dVector(points)
            
            if colors is not None and len(colors) == len(points):
                self.point_cloud.colors = o3d.utility.Vector3dVector(colors)
            else:
                # Use a light gray color for the road
                default_colors = np.ones_like(points) * np.array([0.7, 0.7, 0.7])
                self.point_cloud.colors = o3d.utility.Vector3dVector(default_colors)
            
            self.vis.add_geometry(self.point_cloud, False)
            
            # Extract road boundary and centerline from point cloud
            # Analyze the point cloud to find the road shape
            road_points = np.asarray(points)
            
            # Check if we have road_network data with centerline
            has_centerline_data = 'road_network' in point_cloud_data and 'centerline' in point_cloud_data['road_network']
            
            # Use the actual smooth centerline from road_network when available
            if has_centerline_data:
                # Get the centerline points directly from the environmental model
                # This ensures the centerline is smooth and matches the actual road curve
                centerline_data = point_cloud_data['road_network']['centerline']
                
                # Convert to numpy array if it's not already
                if not isinstance(centerline_data, np.ndarray):
                    centerline_data = np.array(centerline_data)
                
                # Create 3D points from 2D centerline data
                centerline_points = []
                for point in centerline_data:
                    # Add a small height to make centerline visible above road
                    if len(point) == 2:
                        centerline_points.append([point[0], point[1], 0.02])
                    else:
                        # If already 3D point, just ensure z is slightly above road
                        point_3d = point.copy()
                        point_3d[2] = 0.02
                        centerline_points.append(point_3d)
            else:
                # Fallback to extracting centerline from point cloud if data not available
                # Extract road centerline by finding points near y=0 for each x value
                # Group points by their x coordinate (rounded to nearest integer)
                x_values = road_points[:, 0].astype(int)
                unique_x = np.unique(x_values)
                
                centerline_points = []
                left_edge_points = []
                right_edge_points = []
                
                # For each x value, find corresponding road points to determine centerline and edges
                for x in unique_x:
                    # Find all points with this x coordinate
                    mask = (x_values == x)
                    points_at_x = road_points[mask]
                    
                    if len(points_at_x) > 0:
                        # Sort points by y-coordinate
                        sorted_points = points_at_x[points_at_x[:, 1].argsort()]
                        
                        # The centerline is approximately the middle point
                        if len(sorted_points) > 2:
                            middle_idx = len(sorted_points) // 2
                            center_point = sorted_points[middle_idx].copy()
                            center_point[2] = 0.02  # Slightly above road
                            centerline_points.append(center_point)
                            
                            # Left edge is the minimum y
                            left_edge = sorted_points[0].copy()
                            left_edge[2] = 0.01  # Slightly above road
                            left_edge_points.append(left_edge)
                            
                            # Right edge is the maximum y
                            right_edge = sorted_points[-1].copy()
                            right_edge[2] = 0.01  # Slightly above road
                            right_edge_points.append(right_edge)
            
            # Create road edges from extracted points (keep the original edge extraction)
            # Group points by their x coordinate for edge extraction
            x_values = road_points[:, 0].astype(int)
            unique_x = np.unique(x_values)
            
            left_edge_points = []
            right_edge_points = []
            
            # Extract road edges
            for x in unique_x:
                # Find all points with this x coordinate
                mask = (x_values == x)
                points_at_x = road_points[mask]
                
                if len(points_at_x) > 0:
                    # Sort points by y-coordinate
                    sorted_points = points_at_x[points_at_x[:, 1].argsort()]
                    
                    if len(sorted_points) > 2:
                        # Left edge is the minimum y
                        left_edge = sorted_points[0].copy()
                        left_edge[2] = 0.01  # Slightly above road
                        left_edge_points.append(left_edge)
                        
                        # Right edge is the maximum y
                        right_edge = sorted_points[-1].copy()
                        right_edge[2] = 0.01  # Slightly above road
                        right_edge_points.append(right_edge)
            
            # Sort points by x-coordinate to ensure proper line connections
            if centerline_points:
                # For the centerline from road_network, it should already be sorted
                if not has_centerline_data:
                    centerline_points.sort(key=lambda p: p[0])
                centerline_points = np.array(centerline_points)
                
                # Create centerline geometry using the extracted points
                centerline = o3d.geometry.LineSet()
                centerline.points = o3d.utility.Vector3dVector(centerline_points)
                
                # Connect points to form a line
                centerline_lines = [[i, i + 1] for i in range(len(centerline_points) - 1)]
                centerline.lines = o3d.utility.Vector2iVector(centerline_lines)
                
                # Set centerline color to yellow
                centerline_colors = [[1, 1, 0] for _ in range(len(centerline_lines))]
                centerline.colors = o3d.utility.Vector3dVector(centerline_colors)
                
                self.centerline = centerline
                self.vis.add_geometry(centerline, False)
            
            # Create road edges from extracted points
            if left_edge_points and right_edge_points:
                left_edge_points.sort(key=lambda p: p[0])
                right_edge_points.sort(key=lambda p: p[0])
                
                # Combine edges into a single line set
                road_edges = o3d.geometry.LineSet()
                
                # Combine all edge points
                edge_points = np.vstack((left_edge_points, right_edge_points))
                road_edges.points = o3d.utility.Vector3dVector(edge_points)
                
                # Create lines for left and right edges
                left_lines = [[i, i + 1] for i in range(len(left_edge_points) - 1)]
                right_lines = [[i + len(left_edge_points), i + 1 + len(left_edge_points)] 
                               for i in range(len(right_edge_points) - 1)]
                
                edge_lines = left_lines + right_lines
                road_edges.lines = o3d.utility.Vector2iVector(edge_lines)
                
                # Set line color to white
                edge_colors = [[1, 1, 1] for _ in range(len(edge_lines))]
                road_edges.colors = o3d.utility.Vector3dVector(edge_colors)
                
                self.road_edges = road_edges
                self.vis.add_geometry(road_edges, False)
            
        except Exception as e:
            print(f"Error updating point cloud: {e}")
    
    def update_objects(self, objects):
        """Update all objects in the scene.
        
        Args:
            objects (list): List of object dictionaries
        """
        try:
            # Track current objects to remove ones that are no longer present
            current_object_ids = set()
            
            for obj in objects:
                obj_id = obj['id']
                current_object_ids.add(obj_id)
                
                # Create or update the object
                if obj_id not in self.geometries:
                    # Create new geometry
                    geometry = self._create_object_geometry(obj)
                    if geometry is not None:
                        self.geometries[obj_id] = geometry
                        self.vis.add_geometry(geometry, False)
                else:
                    # Update existing geometry
                    success = self._update_object_geometry(self.geometries[obj_id], obj)
                    if success:
                        self.vis.update_geometry(self.geometries[obj_id])
            
            # Remove objects that are no longer in the scene
            removed_ids = set(self.geometries.keys()) - current_object_ids
            for obj_id in removed_ids:
                self.vis.remove_geometry(self.geometries[obj_id], False)
                del self.geometries[obj_id]
        except Exception as e:
            print(f"Error updating objects: {e}")
    
    def _create_object_geometry(self, obj):
        """Create a 3D geometry for an object.
        
        Args:
            obj (dict): Object data
            
        Returns:
            open3d.geometry.Geometry or None: The created geometry or None if failed
        """
        try:
            # Get object properties
            dimensions = obj['dimensions']
            color = np.array(obj['color'], dtype=np.float64) / 255.0  # Convert to [0,1] range
            
            # Create a box for the object
            box = o3d.geometry.TriangleMesh.create_box(
                width=float(dimensions[0]),
                height=float(dimensions[1]),
                depth=float(dimensions[2])
            )
            
            # Center the box
            box.translate([-dimensions[0]/2, -dimensions[1]/2, -dimensions[2]/2])
            
            # Apply rotation
            rotation = obj['rotation']
            R = self._euler_to_rotation_matrix(rotation)
            box.rotate(R)
            
            # Apply translation
            position = obj['position']
            box.translate(position)
            
            # Set color
            box.paint_uniform_color(color)
            
            return box
        except Exception as e:
            print(f"Error creating object geometry: {e}")
            return None
    
    def _update_object_geometry(self, geometry, obj):
        """Update an existing geometry with new object data.
        
        Args:
            geometry (open3d.geometry.Geometry): The geometry to update
            obj (dict): New object data
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create a new box
            dimensions = obj['dimensions']
            box = o3d.geometry.TriangleMesh.create_box(
                width=float(dimensions[0]),
                height=float(dimensions[1]),
                depth=float(dimensions[2])
            )
            
            # Center the box
            box.translate([-dimensions[0]/2, -dimensions[1]/2, -dimensions[2]/2])
            
            # Apply rotation
            rotation = obj['rotation']
            R = self._euler_to_rotation_matrix(rotation)
            box.rotate(R)
            
            # Apply translation
            position = obj['position']
            box.translate(position)
            
            # Update vertices and triangles
            geometry.vertices = box.vertices
            geometry.triangles = box.triangles
            
            # Set color
            color = np.array(obj['color'], dtype=np.float64) / 255.0
            geometry.paint_uniform_color(color)
            
            return True
        except Exception as e:
            print(f"Error updating object geometry: {e}")
            return False
    
    def _euler_to_rotation_matrix(self, euler):
        """Convert Euler angles to rotation matrix."""
        # ZYX rotation (common in vehicle dynamics)
        x, y, z = euler
        
        # Z rotation (yaw)
        Rz = np.array([
            [np.cos(z), -np.sin(z), 0],
            [np.sin(z), np.cos(z), 0],
            [0, 0, 1]
        ])
        
        # Y rotation (pitch)
        Ry = np.array([
            [np.cos(y), 0, np.sin(y)],
            [0, 1, 0],
            [-np.sin(y), 0, np.cos(y)]
        ])
        
        # X rotation (roll)
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(x), -np.sin(x)],
            [0, np.sin(x), np.cos(x)]
        ])
        
        # Combined rotation matrix
        R = Rz @ Ry @ Rx
        return R
    
    def should_quit(self):
        """Check if the renderer should quit.
        
        Returns:
            bool: True if should quit, False otherwise
        """
        return not self.vis.poll_events()
    
    def cleanup(self):
        """Clean up resources."""
        # Close any open windows/resources safely
        try:
            # Ensure the main visualization window is closed properly
            self.vis.destroy_window()
        except:
            pass

    def _update_third_person_camera(self, car):
        """Update camera to follow the car while maintaining a stable view.
        
        Args:
            car (dict): Car object to follow
        """
        # Get car position and orientation
        car_pos = np.array(car['position'])
        
        # Initialize camera state if needed
        if not hasattr(self, 'camera_state'):
            self.camera_state = {
                'target_pos': car_pos.copy(),  # Current target position
                'last_target_pos': car_pos.copy(),  # Previous target position
                'smooth_factor': 0.05,  # Smoothing factor (lower = smoother)
                'initialized': False
            }
        
        # Update the target position (car's position)
        self.camera_state['last_target_pos'] = self.camera_state['target_pos'].copy()
        self.camera_state['target_pos'] = car_pos.copy()
        
        # If first time, just look at the car directly
        if not self.camera_state['initialized']:
            # Just set the look-at point to the car
            self.view_control.set_lookat(car_pos)
            self.camera_state['initialized'] = True
            return
            
        # Compute a smoothed target position - prevent jumpy camera
        smoothed_target = np.array(self.camera_state['last_target_pos']) + \
                         self.camera_state['smooth_factor'] * \
                         (np.array(self.camera_state['target_pos']) - np.array(self.camera_state['last_target_pos']))
        
        # Just update the look-at point to the smoothed car position
        # This keeps the viewing angle and distance as user has set them
        self.view_control.set_lookat(smoothed_target)

    def _create_steering_wheel(self):
        """Create a visual steering wheel display in the 3D view.
        
        Returns:
            open3d.geometry.TriangleMesh: The steering wheel mesh
        """
        # Create a cylinder to represent the steering wheel
        steering_wheel = o3d.geometry.TriangleMesh.create_cylinder(
            radius=4.0,  # Larger radius for better visibility
            height=0.5,  # Thicker for better visibility
            resolution=36  # Higher resolution for smoother appearance
        )
        
        # Create wheel spokes
        spoke_length = 3.6  # Longer spokes
        spoke_width = 0.4   # Wider spokes
        spoke_height = 0.6  # Thicker spokes
        
        for angle in [0, 45, 90, 135, 180, 225, 270, 315]:  # More spokes for better appearance
            spoke = o3d.geometry.TriangleMesh.create_box(
                width=spoke_length,
                height=spoke_width,
                depth=spoke_height
            )
            
            # Position the spoke
            spoke.translate([-spoke_length/2, -spoke_width/2, -spoke_height/2])
            
            # Rotate the spoke to the right angle
            R = self._euler_to_rotation_matrix([0, 0, np.radians(angle)])
            spoke.rotate(R)
            
            # Add spoke to steering wheel
            steering_wheel += spoke
        
        # Add a center hub
        center_hub = o3d.geometry.TriangleMesh.create_cylinder(
            radius=1.0,
            height=0.7,
            resolution=20
        )
        steering_wheel += center_hub
        
        # Paint the steering wheel with a more visible color
        steering_wheel.paint_uniform_color([0.9, 0.7, 0.1])  # Gold/yellow color
        
        # Position the steering wheel in a more visible position
        # We'll place it in front of the car but at a fixed position in the view
        steering_wheel.translate([0, -50, 30])  # More centered horizontally, higher and closer to camera
        
        # Store the initial position for future reference
        self.steering_wheel_position = [0, -50, 30]
        
        return steering_wheel
    
    def _update_steering_wheel(self, steering_angle):
        """Update the steering wheel rotation based on the steering angle.
        
        Args:
            steering_angle (float): Current steering angle in radians
        """
        if not hasattr(self, 'steering_wheel') or self.steering_wheel is None:
            return
            
        try:
            # Remove existing steering wheel
            self.vis.remove_geometry(self.steering_wheel, False)
            
            # Create a new steering wheel
            self.steering_wheel = self._create_steering_wheel()
            
            # Apply rotation around Z axis based on steering angle
            # Note: We use a negative angle because of how the coordinate system is set up
            rotation_matrix = self._euler_to_rotation_matrix([0, 0, -steering_angle * 6])  # Amplify for more visible effect
            self.steering_wheel.rotate(rotation_matrix)
            
            # Add back to the visualization
            self.vis.add_geometry(self.steering_wheel, False)
            
        except Exception as e:
            print(f"Error updating steering wheel: {e}")
    
    def _render_debug_info(self, scene_data):
        """Render debug information on screen.
        
        Args:
            scene_data (dict): Current scene data
        """
        if not scene_data or 'objects' not in scene_data:
            return
            
        # Find the main vehicle
        main_vehicle = None
        for obj in scene_data['objects']:
            if obj.get('autonomous', False):
                main_vehicle = obj
                break
        
        if main_vehicle is None:
            return
            
        # Get steering and position data
        steering_angle = main_vehicle.get('steering', 0)
        pos_y = main_vehicle['position'][1]
        heading = main_vehicle['rotation'][2]
        
        # Get actual target lane center and position from the autonomous logic
        lane_center = 0.0  # Default
        target_position = 0.0  # Default
        
        if 'debug_info' in main_vehicle:
            debug_info = main_vehicle.get('debug_info', {})
            lane_center = debug_info.get('lane_center', 0.0)
            target_position = debug_info.get('target_position', 0.0)
        
        # If we couldn't get lane center from debug_info, try to calculate it
        if lane_center == 0.0 and 'autonomous_logic' in scene_data:
            autonomous_logic = scene_data['autonomous_logic']
            if hasattr(autonomous_logic, 'calculate_lane_center'):
                lane_center = autonomous_logic.calculate_lane_center(main_vehicle['position'][0])
                
            # And if we have a target position calculator, use it
            if hasattr(autonomous_logic, 'calculate_target_position'):
                target_position = autonomous_logic.calculate_target_position(main_vehicle['position'][0])
        
        # Get driving side and lane offset if available
        driving_side = "center"
        lane_offset = 0.0
        if 'autonomous_logic' in scene_data:
            autonomous_logic = scene_data['autonomous_logic']
            if hasattr(autonomous_logic, 'driving_side'):
                driving_side = autonomous_logic.driving_side
            if hasattr(autonomous_logic, 'lane_offset'):
                lane_offset = autonomous_logic.lane_offset
        
        # Calculate distance to road edges based on the lane center
        road_width = 10.0  # Default from environment.py
        if hasattr(scene_data.get('autonomous_logic', None), 'road_width'):
            road_width = scene_data['autonomous_logic'].road_width
        
        left_edge = lane_center - road_width/2
        right_edge = lane_center + road_width/2
        distance_to_left = abs(pos_y - left_edge)
        distance_to_right = abs(pos_y - right_edge)
        
        # Update the 3D steering wheel
        self._update_steering_wheel(steering_angle)
        
        # Create simple terminal-based steering wheel visualization instead of GUI
        # This avoids the hanging issues with matplotlib or Open3D windows
        if self.use_simple_steering_display:
            # Use only for special frames to avoid overwhelming the terminal
            if not hasattr(self, 'frame_count'):
                self.frame_count = 0
            
            # Display steering wheel on every frame for more responsive UI
            self._display_text_steering_wheel(steering_angle)
                
        # Format steering indicator for console output
        steering_percentage = abs(steering_angle / 0.5) * 100  # Assuming max_steering_angle is 0.5
        steering_dir = "RIGHT" if steering_angle > 0 else "LEFT" if steering_angle < 0 else "CENTER"
        steering_bar = "=" * int(steering_percentage / 5)
        
        # Create text for console log
        text = [
            f"Steering: {steering_angle:.2f} rad ({steering_dir} {steering_percentage:.0f}%)",
            f"Position Y: {pos_y:.2f} (Lane Center: {lane_center:.2f}, Target: {target_position:.2f})",
            f"Heading: {heading:.2f} rad",
            f"Lane Position: {driving_side.upper()} with {lane_offset:.1f}m offset",
            f"Left edge: {distance_to_left:.2f}m | Right edge: {distance_to_right:.2f}m"
        ]
        
        # Add to debug log
        if hasattr(self, 'frame_count'):
            self.frame_count += 1
        else:
            self.frame_count = 0
            
        if self.frame_count % 30 == 0:  # Print full debug info less frequently
            print("\n=== STEERING DEBUG INFO ===")
            for line in text:
                print(line)
            print("==============================")
            
    def _display_text_steering_wheel(self, steering_angle):
        """Display a simple text-based steering wheel visualization in the terminal.
        
        Args:
            steering_angle (float): Current steering angle in radians
        """
        # Convert steering angle to an integer position for the display
        # Our steering wheel is 25 characters wide (-12 to +12)
        wheel_width = 25
        center = wheel_width // 2
        
        # Scale the steering angle to fit our display
        # Get max steering from the autonomous logic if available
        max_angle = self.max_steering_angle
        
        # Calculate steering percentage (0-100%)
        steering_percent = min(100, max(0, abs(steering_angle / max_angle * 100)))
        
        # Create more granular position for smoother visualization
        # Scale by 12 instead of 10 for more sensitivity to small changes
        position = int(round(steering_angle / max_angle * 12))
        
        # Clamp to our display range
        position = max(-12, min(12, position))
        
        # Create the wheel display with enhanced visuals
        wheel = [" "] * wheel_width
        wheel[center] = "|"  # Center mark
        
        # Create a more noticeable indicator with arrow showing direction
        indicator_pos = center + position
        if 0 <= indicator_pos < wheel_width:
            if position < 0:
                wheel[indicator_pos] = "◀"  # Left arrow for left steering
            elif position > 0:
                wheel[indicator_pos] = "▶"  # Right arrow for right steering
            else:
                wheel[indicator_pos] = "O"  # Center dot for straight
        
        # Create the display string
        wheel_str = "".join(wheel)
        
        # Create tick marks for reference with clear marks at intervals
        ticks = [" "] * wheel_width
        ticks[center - 12] = "L"   # Left max
        ticks[center - 8] = "|"    # Left 2/3
        ticks[center - 4] = "|"    # Left 1/3
        ticks[center] = "|"        # Center
        ticks[center + 4] = "|"    # Right 1/3
        ticks[center + 8] = "|"    # Right 2/3
        ticks[center + 12] = "R"   # Right max
        ticks_str = "".join(ticks)
        
        # Print the steering wheel display with enhanced border
        direction = "RIGHT" if steering_angle > 0 else "LEFT" if steering_angle < 0 else "CENTER"
        angle_deg = abs(np.rad2deg(steering_angle))
        
        # Create a progress bar style display for the steering amount
        progress_width = 20
        filled_width = int(steering_percent / 100 * progress_width)
        progress_bar = "█" * filled_width + "░" * (progress_width - filled_width)
        
        # Enhanced visualization with thicker borders and more information
        print("\n====== STEERING WHEEL DISPLAY ======")
        print(f"Direction: {direction} ({angle_deg:.1f}°) - {steering_percent:.0f}% of max")
        print(f"L    -    |    -    R")
        print(wheel_str)
        print("═════════════════════════════")
        
        # Print advanced steering visualization showing intensity
        if filled_width > 0:
            if position < 0:  # Left steering
                print(f"{'←' * filled_width}")
            else:  # Right steering
                print(f"{' ' * (progress_width - filled_width)}{'→' * filled_width}") 