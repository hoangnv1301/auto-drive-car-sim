import numpy as np
import math
import time

class AutonomousLogic:
    """Base class for autonomous driving logic."""
    
    def __init__(self):
        """Initialize the autonomous driving logic."""
        self.target_speed = 5.0  # m/s
        self.follow_distance = 10.0  # m
        self.lane_width = 4.0  # m
        self.max_acceleration = 3.0  # m/s^2
        self.max_deceleration = 5.0  # m/s^2
        self.max_steering_angle = 0.5  # radians
        self._prev_position = None
        self._prev_steering = None  # Store previous steering angle for rate limiting
        self._last_debug = {
            "lane": {},
            "steering": {},
            "speed": {}
        }
    
    def process(self, vehicle, scene_data):
        """Process the current scene and return control commands.
        
        Args:
            vehicle (dict): The vehicle to control
            scene_data (dict): Current scene data
            
        Returns:
            dict: Control commands (acceleration, steering)
        """
        # By default, just continue at current speed
        return {
            'acceleration': 0.0,
            'steering': 0.0
        }

class LaneKeepingLogic(AutonomousLogic):
    """Lane keeping autonomous driving logic."""
    
    def __init__(self, road_network):
        """Initialize the lane keeping logic."""
        super().__init__()
        
        self.road_network = road_network
        
        # Steering parameters
        self.steering_gain = 0.4  # Reduced from 0.5 for smoother steering
        self.lane_keeping_factor = 0.5  # Reduced from 0.6 to be less aggressive
        self.curve_following_factor = 0.5  # Reduced from 0.6 for gentler curve handling
        self.max_steering_angle = 0.7  # Maximum steering angle in radians
        self.steering_smoothing = 0.95  # Increased from 0.9 for even smoother transitions
        self.max_change_rate = 0.03  # Reduced for more gradual changes
        self.max_steering_history = 20  # Increased history for better smoothing
        self.steering_history = []
        
        # Lane related parameters
        self.lane_center_tolerance = 0.2  # How close to lane center is acceptable
        self.lane_change_smoothness = 0.4  # Increased for smoother lane transitions
        
        # Speed related parameters
        self.target_speed = 3.0  # Reduced from 5.0 for better stability
        self.speed_smoothing = 0.9
        
        # Lookahead parameters
        self.lookahead_base = 15.0  # Increased from 12.0 to look further ahead
        self.lookahead_min = 12.0  # Increased from 10.0
        self.lookahead_speed_factor = 0.8
        
        # Curve handling
        self.curve_adjustment = 3.0  # Reduced from 4.0
        self.curve_speed_reduction = 0.7  # How much to slow down in curves
        self.curve_transition_smoothing = 0.95  # Increased from 0.9
        self.curve_response = 0.8  # How strongly to respond to curves (higher = more response)
        
        # PID controller for lane tracking
        self.pid_kp = 0.2  # Reduced from 0.25 - lower proportional gain for less aggressive response
        self.pid_ki = 0.0001  # Reduced from 0.0003 - much smaller integral gain to prevent overshoot
        self.pid_kd = 2.0  # Increased from 1.5 - stronger derivative gain to dampen oscillations
        self.error_history = []
        self.error_sum = 0.0
        self.pid_max_error_sum = 1.5  # Reduced from 2.0 - lower integral windup limit
        self.last_error = 0.0
        self.last_derivative = 0.0
        self.derivative_smoothing = 0.7  # Increased from 0.8 for stronger smoothing of derivative
        self.prev_pid_steering = 0.0  # Add variable for exponential moving average filter
        
        # Oscillation detection and prevention
        self.oscillation_detection_window = 10
        self.steering_direction_changes = 0
        self.previous_steering_direction = None
        self.direction_change_damping = 1.5  # Increased from 3.0 for stronger damping on direction changes
        self.oscillation_detected_counter = 0
        self.oscillation_recovery_limit = 10
        
        # Adaptive steering
        self.adaptive_gain_enabled = True
        self.min_gain = 0.2
        self.max_gain = 0.5
        
        # General parameters
        self.min_speed = 0.0
        self.max_acceleration = 3.0
        self.max_deceleration = 5.0
        self.lookahead_base = 12.0  # Increased from 10.0 for better anticipation
        self.max_steering_angle = 0.7
        self.max_steering = 0.5  # Maximum steering wheel rotation in radians
        self.previous_steering = 0.0
        
        # Add a steering history array for stronger filtering
        self.steering_history = []
        self.max_steering_history = 15  # Increased from 10 for better averaging
        
        # Lane keeping parameters
        self.target_lane_offset = 2.5  # Positive = target right lane, negative = target left lane
        self.lane_keeping_factor = 0.08  # How strongly to keep in lane
        self.curve_following_factor = 0.6  # Reduced from 0.8 for smoother curve handling
        self.obstacle_avoidance_factor = 1.5  # Kept same
        self.emergency_steering_factor = 1.8  # Kept same
        self.current_target_lane = 0.0
        self.lane_change_smoothness = 0.3  # Increased from 0.2 for smoother lane changes
        
        # Core parameters
        self.lane_offset = 1.5  # meters, positive is right lane, negative is left lane
        self.lookahead_min = 10.0  # Increased from 8.0 for better anticipation
        self.lookahead_factor = 0.75  # Increased from 0.65 for smoother steering
        self.steering_gain = 0.15  # How quickly to correct steering
        self.max_acceleration = 2.5  # Increased from 2.0 for better acceleration
        self.max_deceleration = 4.5  # Increased from 4.0 for stronger braking
        self.straight_road_speed_boost = 1.05
        
        # Road parameters - note, will be replaced with data from scene
        # For now, keeping variables for backward compatibility
        self.road_width = road_network.width
        self.road_length = np.linalg.norm(road_network.end - road_network.start)
        self.curve_start = 50.0
        self.curve_intensity = 0.005
        self.driving_side = "right"  # or "left"
        
        # Physical driving parameters to make motion more naturalistic
        self.heading_correction_gain = 0.15  # Reduced from 0.2 for smoother corrections
        self.max_heading_correction = 0.015  # Reduced from 0.02 for less aggressive corrections
        self.steering_smoothing = 0.8  # Reduced from 0.85 for more responsive steering
        self.natural_understeer = 0.1
        
        # Collision avoidance parameters
        self.avoidance_lane_shift = 3.5  # Tăng từ 3.0 để né rộng hơn
        self.avoidance_shift_speed = 0.15  # Tăng từ 0.1 để né nhanh hơn
        self.obstacle_recovery_rate = 0.02
        self.current_lane_shift = 0.0
        self.returning_to_lane = False
        self.obstacle_detection_distance = 60.0  # Tăng từ 45.0 để phát hiện sớm hơn
        self.lateral_safety_margin = 1.5  # Tăng từ 1.2 để giữ khoảng cách lớn hơn
        
        # Emergency management
        self.emergency_mode = False
        self.emergency_counter = 0
        self.max_emergency_frames = 10
        
        # Debugging
        self.debug = True
        self.debug_interval = 30
        self.frame_counter = 0
        self.prev_steering = 0.0
        self.speed_history = []
        
        # Lane keeping parameters
        self.target_lane_offset = 1.5  # meters from center to the right
        self.lookahead_base = 14.0  # Increased from 12.0 for better anticipation
        self.previous_steering = 0.0
        self.max_steering_angle = 0.45  # Increased from 0.4 for more steering range
        self.warning_distance = 3.0  # meters
        self.collision_threshold = 12.0  # Distance to start collision avoidance
        self.emergency_braking_factor = 0.7  # Reduced from 0.8 for less aggressive braking
        self.min_emergency_speed = 1.0  # Increased from 0.2 for better maneuverability
        self.mode = "NORMAL"
        self.emergency_timer = 0
        self.emergency_cooldown = 30  # frames
        self.lateral_shift_factor = 1.2  # Increased from 1.0 for more lateral movement
        self.braking_intensity = 0.9  # Reduced from 1.0 for smoother braking
        self.in_curve = False
        self.curve_detection_threshold = 0.08  # Reduced from 0.1 for smoother curve detection
        self.max_steering = 0.5  # Maximum steering wheel rotation in radians
        
        # PID controller parameters for lane keeping
        self.pid_kp = 0.3  # Reduced from 0.5 - less aggressive proportional response
        self.pid_ki = 0.0005  # Reduced from 0.001 - less integral buildup
        self.pid_kd = 1.2  # Increased from 0.8 - stronger derivative action to dampen oscillations
        self.pid_error_sum = 0.0  # Accumulated error for integral term
        self.pid_max_error_sum = 3.0  # Reduced from 5.0 to limit integral windup
        self.pid_previous_error = 0.0  # Previous error for derivative term
        
        # Apply more aggressive damping when error changes direction
        self.pid_damping_factor = 2.0  # Increased from 1.2 for stronger oscillation damping
        self.pid_last_error_sign = 0  # Track error sign changes
        
        # Steering oscillation detection and correction
        self.oscillation_detection_window = 10
        self.steering_direction_changes = 0
        self.previous_steering_direction = 0
        
        # Additional curve smoothing parameters
        self.curve_smoothing_factor = 0.85  # Higher factor for smoother curve transitions
        self.adaptive_lookahead = True  # Enable adaptive lookahead based on curvature
        self.curve_lookahead_reduction = 0.6  # Reduce lookahead in curves for better tracking
    
    def calculate_lane_center(self, x_position):
        """Calculate the ideal lane center y-coordinate at a given x position to follow the curve."""
        # Before the curve starts, lane center is straight
        if x_position < -self.road_length / 2 + self.curve_start:
            return 0.0
        
        # After curve start, calculate the centerline of the curved road
        # For a road from (start_x, start_y) to (end_x, end_y), calculate the y position
        # Normalized position along the road (0 to 1)
        curve_t = (x_position - (-self.road_length / 2)) / self.road_length
        curve_t = max(0.0, min(1.0, curve_t))  # Clamp to [0,1] for stability
        
        # Linear interpolation for curved road
        y = curve_t * self.curve_intensity * self.road_length
        
        # Reduce debug logging - only print every 1000 frames instead of 100
        if hasattr(self, 'frame_counter') and self.frame_counter % 1000 == 0 and self.debug:
            # Road curve debug info disabled for performance optimization
            # print(f"\n=== ROAD CURVE DEBUG INFO ===")
            # print(f"Road params: length={self.road_length}, curve_start={self.curve_start}, intensity={self.curve_intensity}")
            # print(f"Position x={x_position}, curve_t={curve_t}, calculated y={y}")
            # print(f"==============================\n")
            pass
        
        if hasattr(self, 'frame_counter'):
            self.frame_counter += 1
        else:
            self.frame_counter = 0
        
        return y
    
    def calculate_ideal_heading(self, x_position):
        """Calculate the ideal heading (tangent to the road) at a given x position."""
        # Before the curve, heading straight ahead
        if x_position < -self.road_length / 2 + self.curve_start:
            return 0.0
        
        # After curve start, calculate the tangent to the curve
        # Derivative of y = curve_intensity * t^2 * road_length
        curve_t = (x_position - (-self.road_length / 2 + self.curve_start)) / (self.road_length - self.curve_start)
        curve_t = max(0.0, min(1.0, curve_t))  # Clamp to [0,1] for stability
        
        # dy/dt = 2 * curve_intensity * t * road_length
        dy_dt = 2 * self.curve_intensity * curve_t * self.road_length
        
        # Convert to dx/dt
        dt_dx = 1.0 / (self.road_length - self.curve_start)
        dy_dx = dy_dt * dt_dx
        
        # Calculate heading angle (arctan of slope)
        heading = np.arctan(dy_dx)
        
        # Add a small natural adjustment based on curve intensity to simulate real driving
        # This makes the car proactively adjust its heading slightly before the curve
        if curve_t > 0 and curve_t < 0.3:
            # In the early part of the curve, slightly anticipate
            anticipation = 0.05 * curve_t * dy_dx  # Small anticipation proportional to slope
            heading += anticipation
        
        return heading
    
    def calculate_lookahead_point(self, x_position, lookahead_distance):
        """Calculate a point ahead on the road to aim for (better curve anticipation)."""
        # Look ahead on the path
        lookahead_x = x_position + lookahead_distance
        lookahead_y = self.calculate_lane_center(lookahead_x)
        return lookahead_x, lookahead_y
    
    def calculate_target_position(self, x_position):
        """Calculate the target y-coordinate including lane offset."""
        # Get the lane center 
        lane_center = self.calculate_lane_center(x_position)
        
        # Apply the lane offset
        if self.driving_side == "right":
            # Drive on the right side of the road (positive offset from center)
            target_y = lane_center + self.lane_offset
        else:
            # Drive on the left side of the road (negative offset from center)
            target_y = lane_center - self.lane_offset
            
        return target_y
    
    def _get_road_info(self, position, scene_data):
        """Get information about the road and vehicle position.
        
        Args:
            position: The position to get road info for (can be a numpy array or a vehicle dictionary)
            scene_data: The scene data
            
        Returns:
            dict: Road information
        """
        # Handle either a position array or a vehicle dictionary
        if isinstance(position, dict) and 'position' in position:
            current_x = position['position'][0]
            current_y = position['position'][1]
        else:
            # Assume position is a numpy array or list
            current_x = position[0]
            current_y = position[1]
        
        # Always update road parameters from scene_data if available
        if 'road_width' in scene_data:
            self.road_width = scene_data['road_width']
        if 'road_length' in scene_data:
            self.road_length = scene_data['road_length']
        
        # Debug print control - only print every 300 frames
        should_debug_print = False
        if not hasattr(self, 'road_debug_counter'):
            self.road_debug_counter = 0
        self.road_debug_counter += 1
        if self.road_debug_counter % 300 == 0:
            should_debug_print = True
        
        # FOR DEBUGGING - force print road parameters
        if should_debug_print:
            print(f"\n===== ROAD PARAMS DEBUG =====")
            print(f"Before processing: road_width={self.road_width}, road_length={self.road_length}")
            print(f"curve_start={self.curve_start}, curve_intensity={self.curve_intensity}")
        
        # Extract curve parameters and road data ONCE
        road_network = scene_data.get('road_network', None)
        current_road = None
        lane_center_y = None
        road_angle = 0.0
        
        if road_network and 'roads' in road_network and len(road_network['roads']) > 0:
            # Store the road parameters directly from the scene data
            first_road = road_network['roads'][0]
            if should_debug_print:
                print(f"Road data: {first_road}")
            
            # Update road width if available
            if 'width' in first_road:
                self.road_width = first_road['width']
                if should_debug_print:
                    print(f"Setting road_width to {self.road_width}")
                
            # Extract start and end points
            if 'start' in first_road and 'end' in first_road:
                start = np.array(first_road['start'])
                end = np.array(first_road['end'])
                if should_debug_print:
                    print(f"Road start: {start}, end: {end}")
                
                # Calculate road length
                road_vec = end - start
                self.road_length = np.linalg.norm(road_vec)
                if should_debug_print:
                    print(f"Setting road_length to {self.road_length}")
                
                # Calculate road angle/heading
                road_dir = road_vec / self.road_length
                road_angle = np.arctan2(road_dir[1], road_dir[0]) if 'road_dir' in locals() else np.arctan2(end[1] - start[1], end[0] - start[0])
                if should_debug_print:
                    print(f"Road angle: {road_angle}")
                
                # CRITICAL SECTION: Determine if this is a curved road
                if abs(end[1] - start[1]) > 0.1:
                    if should_debug_print:
                        print(f"CURVED ROAD DETECTED: y-diff = {abs(end[1] - start[1])}")
                    # This is a curved road
                    self.curve_start = 0  # Start curve immediately
                    # Calculate curve intensity
                    self.curve_intensity = abs(end[1] - start[1]) / self.road_length
                    if should_debug_print:
                        print(f"Setting curve_start=0, curve_intensity={self.curve_intensity}")
                    
                    # Store road endpoints for lane calculation
                    self.road_start = start
                    self.road_end = end
                else:
                    if should_debug_print:
                        print("STRAIGHT ROAD DETECTED")
                    self.curve_start = self.road_length  # No curve
                    self.curve_intensity = 0
                    if should_debug_print:
                        print(f"Setting curve_start={self.road_length}, curve_intensity=0")
                
            # Find the current road segment the vehicle is on
            current_road = first_road  # Default to first road
            for road in road_network['roads']:
                start = np.array(road['start'])
                end = np.array(road['end'])
                road_vec = end - start
                road_length = np.linalg.norm(road_vec)
                road_dir = road_vec / road_length
                
                # Check if vehicle is near this road segment
                vehicle_pos = np.array([current_x, current_y, 0])
                proj = np.dot(vehicle_pos - start, road_dir)
                
                # If projection is within road length and lateral distance is reasonable
                if 0 <= proj <= road_length:
                    # Calculate lateral distance
                    projected_point = start + proj * road_dir
                    distance = np.linalg.norm(vehicle_pos - projected_point)
                    
                    if distance <= road['width'] * 1.5:  # Allow some margin
                        current_road = road
                        if should_debug_print:
                            print(f"Vehicle on road segment: {road}")
                        break
            
            # Calculate lane center based on the current road
            if current_road:
                road_width = current_road['width']
                lanes = current_road['lanes']
                lane_width = road_width / lanes
                
                # Calculate lane center
                start = np.array(current_road['start'])
                end = np.array(current_road['end'])
                
                # Linear interpolation for curved road
                # Calculate the proportion of the vehicle's position along the road
                total_road_length = np.linalg.norm(end - start)
                vehicle_x_normalized = (current_x - start[0]) / (end[0] - start[0]) if end[0] != start[0] else 0
                vehicle_x_normalized = max(0.0, min(1.0, vehicle_x_normalized))  # Clamp to [0,1]
                
                # Interpolate Y position based on road endpoints
                interpolated_y = start[1] + vehicle_x_normalized * (end[1] - start[1])
                
                # Calculate lane offset based on driving side
                if self.driving_side == "right":
                    # For right-side driving, target the right lane
                    lane_offset = lane_width / 2
                else:
                    # For left-side driving, target the left lane
                    lane_offset = -lane_width / 2
                
                # Calculate road direction at this point
                road_dir = (end - start) / total_road_length
                # Create perpendicular vector for lateral offset (positive = right)
                lateral_dir = np.array([-road_dir[1], road_dir[0], 0])
                
                # Apply lane offset in the direction perpendicular to the road
                lane_center_point = np.array([current_x, interpolated_y, 0]) + lateral_dir * lane_offset
                
                lane_center_y = lane_center_point[1]
                if should_debug_print:
                    print(f"Calculated lane_center_y = {lane_center_y}")
        
        # DEBUGGING - After processing road network
        if should_debug_print:
            print(f"After processing: road_width={self.road_width}, road_length={self.road_length}")
            print(f"curve_start={self.curve_start}, curve_intensity={self.curve_intensity}")
            print(f"===== END ROAD PARAMS DEBUG =====\n")
        
        # If road network processing didn't yield a lane center, fallback to original calculation
        if lane_center_y is None:
            lane_center_y = self.calculate_lane_center(current_x)
            if should_debug_print:
                print(f"Using fallback lane_center_y = {lane_center_y}")
        
        # Check if we're approaching a curve
        curves_ahead = False
        curve_distance = float('inf')
        curve_angle = 0.0
        
        # Only calculate this if we have curve logic enabled
        if self.curve_intensity > 0:
            curves_ahead = True
            curve_distance = 0
            curve_angle = road_angle
        
        # Force increment the frame counter for debug output
        if hasattr(self, 'frame_counter'):
            self.frame_counter += 1
        else:
            self.frame_counter = 1
            
        return {
            'lane_center_y': lane_center_y,
            'lane_width': self.road_width / 2,  # Assuming 2 lanes
            'road_width': self.road_width,
            'curves_ahead': curves_ahead, 
            'curve_distance': curve_distance,
            'curve_angle': curve_angle,
            'road_heading': road_angle
        }
    
    def process(self, vehicle, scene_data):
        """Process the current scene data to generate control commands.
        
        Args:
            vehicle (dict): The vehicle to control
            scene_data (dict): The current scene data
            
        Returns:
            dict: Control commands
        """
        # Create a combined scene_data if it doesn't already have the vehicle
        if 'ego_vehicle' not in scene_data:
            scene_data = scene_data.copy()
            scene_data['ego_vehicle'] = vehicle
            
        # Package together vehicle and scene data
        dt = scene_data.get('dt', 0.033)  # Default to 33ms if not provided
        
        # Calculate current position
        current_pos = vehicle['position']
        
        # Extract some useful properties
        vehicle_heading = vehicle['rotation'][2]  # Yaw in radians
        vehicle_velocity = vehicle.get('velocity', [0, 0, 0])
        current_speed = math.sqrt(vehicle_velocity[0]**2 + vehicle_velocity[1]**2)  # In m/s
        
        # Get road information if available
        road = scene_data.get('road', None)
        road_width = 8.0  # Default road width if not available
        if road is not None and 'width' in road:
            road_width = road['width']
        
        # Check for obstacles using our _check_for_obstacles method
        all_objects = scene_data.get('objects', [])
        has_obstacles, closest_obstacle, distance_to_obstacle, obstacle_info = self._check_for_obstacles(
            current_pos, vehicle_heading, vehicle['dimensions'], all_objects, vehicle['id'], None
        )
        
        # Calculate safe distance for avoiding obstacles
        # Larger margins for higher speeds
        safe_distance = self.collision_threshold * (1.0 + 0.5 * min(1.0, current_speed / 10.0))
        
        # Determine left-right position in lane
        lane_pos = 0.0  # 0 = center, -1 = left edge, 1 = right edge
        
        # Get lane position from localization if available
        if 'localization' in scene_data and 'lane_position' in scene_data['localization']:
            lane_pos = scene_data['localization']['lane_position']
        
        # Initialize enhanced avoidance data with more detailed information
        avoidance_data = {
            "has_obstacles": has_obstacles,
            "closest_obstacle": closest_obstacle,
            "distance_to_obstacle": distance_to_obstacle,
            "obstacle_info": obstacle_info,
            "safe_distance": safe_distance,
            "lane_position": lane_pos,
            "avoidance_direction": 0  # Default: no avoidance
        }
        
        # If we have obstacles, determine avoidance direction with improved logic
        if has_obstacles and closest_obstacle is not None:
            # Get the relative position to determine which way to steer
            rel_pos = self._calculate_relative_position(vehicle, closest_obstacle)
            
            # Add relative position to obstacle info for enhanced processing
            if obstacle_info is None:
                obstacle_info = {}
            obstacle_info['rel_pos'] = rel_pos
            avoidance_data["obstacle_info"] = obstacle_info
            
            # If obstacle is directly ahead, use lane position to decide direction
            if abs(rel_pos[1]) < vehicle['dimensions'][1] * 0.7:  # Increased from 0.5 for earlier response
                # If we're left of center, go right and vice versa
                avoidance_data["avoidance_direction"] = -1 if lane_pos < 0 else 1
            else:
                # Otherwise avoid based on obstacle position (opposite direction)
                avoidance_data["avoidance_direction"] = -1 if rel_pos[1] > 0 else 1
            
            # Check road edges to ensure we don't go off road
            # If already near edge, override to avoid going off road
            if lane_pos > 0.7 and avoidance_data["avoidance_direction"] > 0:
                # Too close to right edge, force left
                avoidance_data["avoidance_direction"] = -1
            elif lane_pos < -0.7 and avoidance_data["avoidance_direction"] < 0:
                # Too close to left edge, force right
                avoidance_data["avoidance_direction"] = 1
        
        # Package data with the enhanced avoidance data
        data = {
            "vehicle": vehicle,
            "scene": scene_data,
            "avoidance_data": avoidance_data  # Package enhanced avoidance data
        }
        
        # Get key vehicle properties
        current_pos = np.array(vehicle['position'])
        current_x = current_pos[0]
        current_y = current_pos[1]
        vehicle_heading = vehicle['rotation'][2]
        current_speed = self._calculate_speed(vehicle)
        
        # Create road information
        road_info = self._get_road_info(current_pos, scene_data)
        lane_center_y = road_info["lane_center_y"]  # Use the improved lane center
        
        # Create a simplified segment representation
        road_width = scene_data.get('road_width', self.road_width)
        lane_width = road_width / 2  # Assuming 2 lanes
        
        # Simplified segment data with curve awareness
        segment = {
            'width': road_width,
            'lanes': 2,
            'center': np.array([current_x, lane_center_y, 0]),
            'start': np.array([current_x - 10, self.calculate_lane_center(current_x - 10), 0]),
            'end': np.array([current_x + 10, self.calculate_lane_center(current_x + 10), 0])
        }
        
        # Calculate target position (offset from center to stay in right lane)
        lane_offset = self.target_lane_offset
        if segment['lanes'] == 1:
            lane_offset = 0  # Stay centered in a single lane road
            
        # Calculate segment direction (tangent)
        segment_direction = segment['end'] - segment['start']
        segment_length = np.linalg.norm(segment_direction)
        if segment_length > 0.001:
            tangent = segment_direction / segment_length
        else:
            tangent = np.array([1, 0, 0])  # Default to forward
        
        perpendicular = np.array([-tangent[1], tangent[0], 0])
        
        # Target position with lane offset
        target_position = segment['center'] + perpendicular * lane_offset
        
        # Calculate relative lane position for debugging
        lane_position = "CENTER"
        lane_position_value = 0.0
        
        # Project vehicle position onto perpendicular axis
        lane_center_position = np.array([current_x, lane_center_y, 0])
        projection = np.dot(current_pos - lane_center_position, perpendicular)
        
        if projection > 0:
            lane_position = "RIGHT"
            lane_position_value = projection
        elif projection < 0:
            lane_position = "LEFT"
            lane_position_value = -projection
            
        # Calculate distance to left and right edge
        left_edge = (segment['width'] / 2) + projection
        right_edge = (segment['width'] / 2) - projection
        
        # IMPROVEMENT: Check if vehicle is near the edge of the road - if so, prioritize return to lane
        near_edge = left_edge < 1.0 or right_edge < 1.0
        
        # Update lane position in avoidance data for better decision making
        avoidance_data["lane_position"] = projection / (lane_width * 0.5)  # Normalized -1 to 1
        
        # Check for curves in the road
        is_curved_road = road_info.get('curves_ahead', False)
        
        # Calculate steering angle for normal driving (lane keeping logic)
        # Adjust lookahead distance based on speed and curve conditions
        if is_curved_road:
            # Shorter lookahead on curves for better curve tracking
            speed_lookahead_factor = max(0.7, current_speed / 8.0)
            lookahead_distance = self.lookahead_base * speed_lookahead_factor
            
            # Add curve intensity factor - tighter curves need shorter lookahead
            curve_factor = max(0.6, 1.0 - self.curve_intensity * 5.0)
            lookahead_distance *= curve_factor
        else:
            # Longer lookahead on straight sections
            speed_lookahead_factor = max(1.0, current_speed / 5.0)
            lookahead_distance = self.lookahead_base * speed_lookahead_factor * 1.2
        
        # Add emergency adjustment for obstacles
        if has_obstacles and distance_to_obstacle < safe_distance * 2.0:
            lookahead_distance = max(8.0, lookahead_distance * 0.7)
        
        # Generate lookahead points for steering calculation
        main_lookahead_position = current_pos + tangent * lookahead_distance
        near_lookahead = lookahead_distance * 0.3
        mid_lookahead = lookahead_distance * 0.6
        
        near_lookahead_position = current_pos + tangent * near_lookahead
        mid_lookahead_position = current_pos + tangent * mid_lookahead
        
        # Calculate vectors to lookahead points
        to_near = near_lookahead_position - current_pos
        to_mid = mid_lookahead_position - current_pos
        to_far = main_lookahead_position - current_pos
        
        # Calculate the vector from vehicle to target lane position
        to_target = target_position - current_pos
        
        # Determine lane deviation
        lane_deviation = abs(projection - lane_offset)
        
        # Adjust lane keeping factor based on whether we're on a curved road
        if is_curved_road:
            # Increase lane keeping weight on curves for better tracking
            # Use a quadratic weight calculation for smoother transitions
            lane_deviation_factor = min(1.0, (lane_deviation / (lane_width * 0.5))**2)
            lane_keeping_weight = 0.5 + 0.3 * lane_deviation_factor * self.curve_intensity * 8.0
            lane_keeping_weight = min(0.8, lane_keeping_weight)  # Cap at 0.8 (reduced from 0.85)
        else:
            # Normal lane keeping on straight roads
            lane_deviation_factor = min(1.0, (lane_deviation / (lane_width * 0.5))**2)
            lane_keeping_weight = 0.4 + 0.4 * lane_deviation_factor  # Reduced from 0.5 factor
        
        # IMPROVEMENT: Increase lane keeping weight when near road edge for stronger correction
        if near_edge:
            # Apply smoother weight increase for roads
            edge_closeness = max(0, 1.0 - min(left_edge, right_edge) / 2.0)
            edge_weight = 0.7 + 0.2 * edge_closeness  # Max 0.9 at edge
            # Smoothly blend between normal and edge weights
            lane_keeping_weight = lane_keeping_weight * (1 - edge_closeness) + edge_weight * edge_closeness
        
        # Apply temporal smoothing to lane keeping weight to prevent sudden changes
        if hasattr(self, 'previous_lane_keeping_weight'):
            # 80% previous, 20% current - strong smoothing
            lane_keeping_weight = 0.8 * self.previous_lane_keeping_weight + 0.2 * lane_keeping_weight
        self.previous_lane_keeping_weight = lane_keeping_weight
        
        # Calculate the weights, priorities and steering
        pure_pursuit_weight = 1.0 - lane_keeping_weight
        
        # Get the target position and calculate the steering angle using pure pursuit
        target_position = lane_center_position + perpendicular * lane_offset
        
        # Calculate pure pursuit steering component
        to_target = target_position - current_pos
        target_angle = np.arctan2(to_target[1], to_target[0])
        heading_error = (target_angle - vehicle_heading + np.pi) % (2 * np.pi) - np.pi
        pure_pursuit_steering = np.clip(heading_error, -self.max_steering, self.max_steering)
        
        # For curved roads, also consider the road's heading/direction
        if is_curved_road:
            # Get the road's direction/heading at this point
            road_heading = road_info['road_heading']
            
            # Calculate error between vehicle heading and road heading
            road_heading_error = (road_heading - vehicle_heading + np.pi) % (2 * np.pi) - np.pi
            
            # Blend with the regular pure pursuit steering (70% road direction, 30% target direction)
            pure_pursuit_steering = 0.7 * road_heading_error + 0.3 * pure_pursuit_steering
            pure_pursuit_steering = np.clip(pure_pursuit_steering, -self.max_steering, self.max_steering)
        
        # Calculate lane keeping steering component (cross-track error)
        # Increase lane keeping strength for curved roads
        lane_factor = 1.0 if not is_curved_road else 1.2  # Reduced from 1.5
        
        # Calculate traditional lane keeping steering
        lane_error = (lane_offset - projection) * self.lane_keeping_factor * lane_factor
        
        # Calculate PID-based lane keeping
        pid_steering = self._calculate_pid_steering(lane_offset - projection)
        
        # Blend traditional with PID-based steering (80% PID, 20% traditional)
        lane_keeping_steering = 0.8 * pid_steering + 0.2 * lane_error
        
        # Apply steering limits
        lane_keeping_steering = np.clip(lane_keeping_steering, -self.max_steering, self.max_steering)
        
        # Blend the steering components
        steering_angle = (lane_keeping_weight * lane_keeping_steering + 
                           pure_pursuit_weight * pure_pursuit_steering)
        
        # Add curve adjustment - stronger for higher curve intensity
        curve_adjustment = 0.0
        if is_curved_road:
            # Calculate curve adjustment based on curvature
            curve_adjustment = road_info['curve_angle'] * self.curve_response
            # Scale down for higher speeds
            curve_adjustment *= max(0.5, 1.0 - current_speed / 15.0)
            
        # Apply curve adjustment
        steering_angle += curve_adjustment
        
        # Store base steering angle before obstacle avoidance for debugging
        base_steering = steering_angle
        
        # Package avoidance data with the current steering
        avoidance_data["current_steering"] = steering_angle
        
        # Apply obstacle avoidance if needed using improved handler
        avoidance_result = self._handle_obstacle_avoidance(data, steering_angle)
        
        # Check if obstacle avoidance has modified the steering
        if avoidance_result is not None:
            steering_angle = avoidance_result["steering_angle"]
            is_emergency = avoidance_result["is_emergency"]
            emergency_blend = avoidance_result.get("emergency_blend", 0.0)
            avoidance_bias = avoidance_result.get("avoidance_bias", 0.0)
        else:
            is_emergency = False
            emergency_blend = 0.0
            avoidance_bias = 0.0
        
        # Apply speed control logic
        # Calculate target speed - base target speed + adjustments
        target_speed = self.target_speed
        
        # Decrease speed for curves
        if is_curved_road:
            # Stronger speed reduction for sharper curves (using curve intensity)
            curve_speed_factor = max(0.6, 1.0 - self.curve_intensity * 8.0)  # 0.6 to 1.0
            
            # More gradual speed reduction for gentle curves
            target_speed *= curve_speed_factor
            
            # Debug the curve speed adjustment
            if hasattr(self, 'debug_enabled') and self.debug_enabled:
                print(f"Speed reduced for curve: {curve_speed_factor:.2f} * {self.target_speed:.1f} = {target_speed:.1f}")
        
        # Apply speed control logic when approaching obstacles
        if has_obstacles and distance_to_obstacle < safe_distance * 3.0:
            # Pack data for speed control
            data["emergency"] = is_emergency
            data["obstacle_distance"] = distance_to_obstacle
            
            # Use the improved speed control logic
            speed_control_result = self._handle_speed_control(data, is_emergency)
            target_speed = speed_control_result.get("target_speed", target_speed)
        
        # Apply final speed control
        speed_error = target_speed - current_speed
        
        # Calculate acceleration based on speed error
        if speed_error > 0:
            # Accelerate - smooth acceleration curve with reduced intensity
            acceleration = min(self.max_acceleration, speed_error * 0.8)
        else:
            # Brake - more responsive braking, especially in emergency situations
            base_deceleration = abs(speed_error) * 1.2  # More responsive braking 
            
            # Apply stronger braking in emergency with enhanced variable braking force
            if is_emergency:
                # Scale braking force by emergency blend
                emergency_braking = self.max_deceleration * emergency_blend
                # Combine with base deceleration for a smooth blend
                acceleration = -max(base_deceleration, emergency_braking)
            else:
                acceleration = -min(self.max_deceleration, base_deceleration)
        
        # Debug info package
        debug_info = {
            "lane_position": lane_position,
            "lane_position_value": lane_position_value,
            "lane_center_y": lane_center_y,
            "heading": vehicle_heading,
            "lane_deviation": lane_deviation,
            "lane_keeping_weight": lane_keeping_weight,
            "pure_pursuit_weight": pure_pursuit_weight,
            "lane_keeping_steering": lane_keeping_steering,
            "pure_pursuit_steering": pure_pursuit_steering,
            "has_obstacles": has_obstacles,
            "distance_to_obstacle": distance_to_obstacle,
            "is_emergency": is_emergency,
            "emergency_blend": emergency_blend,
            "avoidance_bias": avoidance_bias,
            "base_steering": base_steering,
            "final_steering": steering_angle,
            "current_speed": current_speed,
            "target_speed": target_speed,
            "acceleration": acceleration
        }
        
        # Return final control commands
        return {
            "acceleration": acceleration,
            "steering": steering_angle,
            "debug": debug_info
        }

    def _handle_obstacle_avoidance(self, data, steering_angle=None, emergency=False):
        """Handle obstacle avoidance behavior with improved object type awareness and certainty-based decision making.
        
        Args:
            data: Dictionary containing vehicle, scene, and obstacle data
            steering_angle: Current steering angle (if None, will be derived from data)
            emergency: Flag indicating if we're already in emergency mode
            
        Returns:
            dict: Updated steering and emergency status
        """
        # Get vehicle data
        vehicle = data.get("vehicle", None)
        if vehicle is None and "avoidance_data" in data:
            # If using the new avoidance_data format
            avoidance_data = data["avoidance_data"]
            # Extract key data from avoidance_data
            has_obstacles = avoidance_data.get("has_obstacles", False)
            closest_obstacle = avoidance_data.get("closest_obstacle", None)
            distance_to_obstacle = avoidance_data.get("distance_to_obstacle", float('inf'))
            obstacle_info = avoidance_data.get("obstacle_info", None)
            avoidance_direction = avoidance_data.get("avoidance_direction", 0)
        else:
            # Backwards compatibility with old data format
            has_obstacles = data.get("has_obstacles", False)
            closest_obstacle = data.get("closest_obstacle", None)
            distance_to_obstacle = data.get("distance_to_obstacle", float('inf'))
            obstacle_info = data.get("obstacle_info", None)
            avoidance_direction = data.get("avoidance_direction", 0)
            vehicle = data.get("vehicle", None)
        
        # Use provided steering angle or extract from data
        if steering_angle is None:
            steering_angle = data.get("current_steering", 0.0)
        
        # Initialize emergency state tracking if not present
        if not hasattr(self, 'emergency_blend'):
            self.emergency_blend = 0.0  # 0 = normal, 1 = full emergency
            
        # Calculate smooth emergency blend factor based on obstacle distance and type
        if has_obstacles and closest_obstacle is not None:
            # Get object type with enhanced fallback logic
            obj_type = "UNKNOWN"
            if obstacle_info is not None:
                obj_type = obstacle_info.get('type', "UNKNOWN")
            elif hasattr(closest_obstacle, 'type') and closest_obstacle.get('type'):
                obj_type = closest_obstacle['type'] if isinstance(closest_obstacle['type'], str) else closest_obstacle['type'].name
            
            # Adjust thresholds based on object type
            distance_multiplier = 1.0
            
            # Kiểm tra xem đây có phải là xe đỗ hay người đi bộ cắt ngang không
            is_parked_vehicle = obstacle_info is not None and obstacle_info.get('is_parked_vehicle', False)
            is_crossing_pedestrian = obstacle_info is not None and obstacle_info.get('is_crossing_pedestrian', False)
            
            # Vulnerable road users require extra caution
            if obj_type == 'PEDESTRIAN':
                distance_multiplier = 2.0  # Tăng từ 1.5 lên 2.0 cho người đi bộ
                # Thêm điều chỉnh cho người đi bộ cắt ngang
                if is_crossing_pedestrian:
                    distance_multiplier = 2.5  # Tăng khoảng cách an toàn cho người đi bộ cắt ngang
            elif obj_type == 'CYCLIST':
                distance_multiplier = 1.8  # Tăng từ 1.3 lên 1.8 cho xe đạp
            elif is_parked_vehicle:
                distance_multiplier = 1.2  # Tăng khoảng cách an toàn cho xe đỗ
            
            # Define distance thresholds with type-based adjustment
            critical_distance = self.collision_threshold * 1.0 * distance_multiplier  # Tăng từ 0.8 lên 1.0
            safe_distance = self.collision_threshold * 2.0 * distance_multiplier  # Tăng từ 1.5 lên 2.0
            
            # Get certainty information if available
            detection_certainty = 1.0  # Default to full certainty
            if obstacle_info and 'detection_certainty' in obstacle_info:
                detection_certainty = obstacle_info['detection_certainty']
            
            # Adjust thresholds based on certainty - lower certainty requires earlier reaction
            if detection_certainty < 0.9:
                # Increase distances for uncertain detections
                certainty_factor = max(1.0, 1.5 - 0.5 * detection_certainty)  # 1.0-1.5x based on certainty
                critical_distance *= certainty_factor
                safe_distance *= certainty_factor
            
            # Adjust for collision risk if available (moving object on collision course)
            if obstacle_info and obstacle_info.get('is_moving_toward_vehicle', False):
                collision_risk = obstacle_info.get('collision_risk', 0)
                # Increase emergency level for potential collisions based on certainty
                risk_factor = collision_risk * max(0.7, detection_certainty)  # Scale risk by certainty
                critical_distance *= (1.0 + risk_factor * 0.5)
                safe_distance *= (1.0 + risk_factor * 0.5)
            
            # Calculate target emergency level based on distance
            if distance_to_obstacle <= critical_distance:
                # When very close, quickly increase emergency level
                target_emergency = 1.0
            elif distance_to_obstacle >= safe_distance:
                # When far enough, decrease emergency level
                target_emergency = 0.0
            else:
                # Gradual transition based on distance
                proximity = 1.0 - ((distance_to_obstacle - critical_distance) / 
                                  (safe_distance - critical_distance))
                target_emergency = max(0.0, min(1.0, proximity))
            
            # Điều chỉnh mức độ khẩn cấp cho xe đỗ
            if is_parked_vehicle:
                # Xe đỗ tạo mức độ khẩn cấp vừa phải, không cần dừng lại hoàn toàn
                # Nhưng cần điều chỉnh lớn hơn nếu khoảng cách gần
                if distance_to_obstacle < critical_distance * 1.5:
                    target_emergency = min(1.0, target_emergency * 1.2)
            
            # Adjust target emergency based on object type
            if obj_type == 'PEDESTRIAN':
                # More cautious with pedestrians
                target_emergency = min(1.0, target_emergency * 1.3)
            elif obj_type == 'CYCLIST':
                # Also more cautious with cyclists
                target_emergency = min(1.0, target_emergency * 1.2)
            
            # Scale emergency level by certainty
            target_emergency *= max(0.6, detection_certainty)
            
            # Gradually approach target emergency level (faster increase, slower decrease)
            if target_emergency > self.emergency_blend:
                # Quick response to danger - increase speed depends on certainty
                increase_rate = max(0.1, (target_emergency - self.emergency_blend) * 0.4 * detection_certainty)
                
                # Tăng cấp phản ứng cho người đi bộ cắt ngang
                if is_crossing_pedestrian:
                    increase_rate *= 1.5
                
                self.emergency_blend = min(1.0, self.emergency_blend + increase_rate)
            else:
                # Slower return to normal - more gradual with lower certainty
                decrease_rate = 0.05 * min(1.0, 0.5 + 0.5 * detection_certainty)
                
                # Giảm tốc độ giảm mức khẩn cấp cho người đi bộ cắt ngang
                if is_crossing_pedestrian:
                    decrease_rate *= 0.5
                
                self.emergency_blend = max(0.0, self.emergency_blend - decrease_rate)
        else:
            # No obstacles, gradually reduce emergency level
            self.emergency_blend = max(0.0, self.emergency_blend - 0.05)
            
        # Replace binary emergency mode with blended approach
        emergency_factor = self.emergency_blend
        normal_factor = 1.0 - emergency_factor
            
        # Original steering angle (from lane keeping, curve following, etc.)
        original_steering = steering_angle
        
        # Default outputs
        result_steering = original_steering
        is_emergency = emergency_factor > 0.25  # Consider it emergency if blend is significant
        
        avoidance_bias = 0.0
        
        # Only process avoidance if obstacles exist
        if has_obstacles and closest_obstacle is not None:
            # Get relative position between vehicle and obstacle
            rel_pos = None
            
            if vehicle and closest_obstacle:
                rel_pos = self._calculate_relative_position(vehicle, closest_obstacle)
            elif obstacle_info and 'rel_pos' in obstacle_info:
                rel_pos = obstacle_info['rel_pos']
            
            if rel_pos is not None:
                # Calculate relative angle to obstacle
                rel_angle = np.arctan2(rel_pos[1], rel_pos[0])
                
                # Calculate distance to obstacle (reuse provided distance if available)
                distance = distance_to_obstacle if distance_to_obstacle < float('inf') else self._calculate_distance(rel_pos)
                
                # Xử lý đặc biệt cho xe đỗ - né với biên độ lớn hơn
                if is_parked_vehicle:
                    # Tính khoảng cách ngang đến xe đỗ
                    lateral_distance = abs(rel_pos[1])
                    
                    # Quyết định hướng né tránh
                    # Lấy hướng từ avoidance_direction hoặc tính toán
                    if avoidance_direction == 0:
                        # Nếu không có hướng sẵn, tính hướng né dựa trên vị trí ngang
                        avoidance_direction = -1 if rel_pos[1] > 0 else 1
                    
                    # Tính cường độ né tránh dựa trên khoảng cách
                    # Áp dụng cường độ cao hơn cho xe đỗ
                    intensity = max(0.0, min(1.0, self.collision_threshold * 1.2 / max(0.1, distance)))
                    intensity *= 1.3  # Tăng cường độ né tránh cho xe đỗ
                
                # Xử lý đặc biệt cho người đi bộ cắt ngang - dừng lại và không đổi hướng
                elif is_crossing_pedestrian:
                    # Chỉ áp dụng né tránh nhỏ, ưu tiên dừng lại
                    intensity = max(0.0, min(1.0, self.collision_threshold / max(0.1, distance)))
                    intensity *= 0.7  # Giảm cường độ né tránh vì ưu tiên dừng hơn là né
                else:
                    # Calculate intensity based on distance (closer = stronger response)
                    # Smoother intensity curve with improved response near collision threshold
                    intensity = max(0.0, min(1.0, self.collision_threshold / max(0.1, distance)))
                
                # Apply certainty scaling if available
                if obstacle_info and 'detection_certainty' in obstacle_info:
                    certainty = obstacle_info['detection_certainty']
                    # Apply certainty-based scaling that's stronger for closer objects
                    # For far objects, low certainty reduces intensity more
                    # For close objects, even low certainty maintains high intensity for safety
                    certainty_scale = min(1.0, 0.6 + 0.4 * (intensity ** 0.5) + certainty * 0.4)
                    intensity *= certainty_scale
                
                # Get avoidance direction from data
                # Check if using new or old data format
                if "avoidance_data" in data and "avoidance_direction" in data["avoidance_data"]:
                    avoidance_direction = data["avoidance_data"]["avoidance_direction"]
                
                # Adjust avoidance direction based on predicted path if available
                if obstacle_info and obstacle_info.get('predicted_path') and obstacle_info.get('is_moving_toward_vehicle'):
                    # Check if object is predicted to cross our path from left or right
                    predicted_path = obstacle_info['predicted_path']
                    prediction_certainty = obstacle_info.get('prediction_certainty', 0.7)  # Default 70% certainty
                    
                    if predicted_path:
                        # Check the lateral position (y) trend in the prediction
                        current_y = closest_obstacle['position'][1] if 'position' in closest_obstacle else rel_pos[1]
                        future_y = predicted_path[-1][1]
                        
                        # If object is predicted to move across our path, adjust avoidance direction
                        if abs(future_y - current_y) > 1.0:  # Significant lateral movement
                            # Determine which way the object is moving laterally
                            moving_right = future_y > current_y
                            
                            # If moving right, prefer going left and vice versa
                            preferred_direction = -1 if moving_right else 1
                            
                            # Blend with current avoidance direction based on certainty
                            # Lower certainty = less influence from prediction
                            time_to_collision = obstacle_info.get('collision_risk', 0) * 5.0  # 0-5 seconds
                            if time_to_collision < 3.0:  # Only override for imminent collisions
                                # Stronger influence for close collisions, scaled by prediction certainty
                                prediction_weight = max(0, 1.0 - time_to_collision / 3.0) * prediction_certainty
                                avoidance_direction = (1 - prediction_weight) * avoidance_direction + prediction_weight * preferred_direction
                
                # Calculate avoidance steering (smooth and proportional)
                max_avoidance_angle = 0.2  # Maximum steering angle for avoidance
                avoidance_steering = avoidance_direction * intensity * max_avoidance_angle
                
                # Điều chỉnh góc lái cho xe đỗ - tăng góc né tránh
                if is_parked_vehicle:
                    avoidance_steering *= 1.5  # Tăng góc né tránh cho xe đỗ
                
                # Giảm né tránh cho người đi bộ cắt ngang - ưu tiên dừng lại
                if is_crossing_pedestrian:
                    avoidance_steering *= 0.7  # Giảm né tránh cho người đi bộ
                
                # Record avoidance_bias for debugging
                avoidance_bias = avoidance_direction * intensity
                
                # Apply smoother blending between normal and emergency steering
                result_steering = (normal_factor * original_steering) + (emergency_factor * avoidance_steering)
                
                # Add result to steering history for filtering
                if not hasattr(self, 'steering_history'):
                    self.steering_history = []
                    self.max_steering_history = 15
                
                self.steering_history.append(result_steering)
                if len(self.steering_history) > self.max_steering_history:
                    self.steering_history.pop(0)
                    
                # Apply weighted moving average filter to smooth steering
                if len(self.steering_history) >= 5:  # Increased from 3 for more data points
                    # Exponential weights that emphasize recent values but with stronger smoothing
                    if emergency_factor > 0.5:
                        weights = [0.05, 0.1, 0.15, 0.2, 0.5]  # More responsive in emergency
                    else:
                        weights = [0.05, 0.1, 0.15, 0.3, 0.4]  # More smoothing in normal driving
                    
                    # Normalize weights to ensure they sum to 1.0
                    weight_sum = sum(weights)
                    weights = [w/weight_sum for w in weights]
                    
                    result_steering = sum(w * s for w, s in zip(weights, self.steering_history[-5:]))
                
                # Avoid sudden steering changes by limiting change rate
                if hasattr(self, 'previous_steering'):
                    # Limit steering rate of change (more strict in normal driving)
                    max_change_rate = 0.05 * (1.0 - emergency_factor) + 0.15 * emergency_factor
                    steering_delta = result_steering - self.previous_steering
                    if abs(steering_delta) > max_change_rate:
                        # Limit the change rate
                        result_steering = self.previous_steering + np.sign(steering_delta) * max_change_rate
                        
                # Save current steering for next iteration
                self.previous_steering = result_steering
        
        # Return updated data with avoidance information
        result = {
            "steering_angle": result_steering,
            "is_emergency": is_emergency,
            "emergency_blend": emergency_factor,
            "avoidance_bias": avoidance_bias
        }
        
        return result

    def _handle_speed_control(self, data, is_emergency=False):
        """Calculate speed and acceleration based on the current situation.
        
        Args:
            data: Data dictionary containing scene and vehicle information
            is_emergency: Whether we're in an emergency situation
        
        Returns:
            Dict with target_speed and acceleration values
        """
        # Extract key info
        vehicle = data.get("vehicle", {})
        current_speed = self._calculate_speed(vehicle)
        
        # Get obstacle information
        has_obstacles = data.get("has_obstacles", False)
        distance_to_obstacle = data.get("obstacle_distance", float('inf'))
        obstacle_info = data.get("obstacle_info", {})
        
        # Improved obstacle certainty handling
        certainty = 1.0  # Default to full certainty if not provided
        if obstacle_info and "certainty" in obstacle_info:
            certainty = obstacle_info["certainty"]
        
        # Get emergency status
        emergency = is_emergency or data.get("emergency", False)
        emergency_blend = data.get("emergency_blend", 0) if "emergency_blend" in data else 0.0
        
        # Default values
        target_speed = self.target_speed
        
        # If we're in an emergency situation, reduce speed dramatically
        if emergency:
            # Scale emergency braking by certainty and emergency_blend
            emergency_factor = certainty * emergency_blend
            
            # Tăng cường phanh khi có người đi bộ đang sang đường
            is_crossing_pedestrian = obstacle_info and obstacle_info.get('is_crossing_pedestrian', False)
            is_parked_vehicle = obstacle_info and obstacle_info.get('is_parked_vehicle', False)
            
            # Progressive speed reduction based on obstacle distance and emergency level
            if distance_to_obstacle < 8.0:  # Tăng từ 5.0 lên 8.0
                # Very close obstacle - almost stop
                target_speed = 0.0  # Dừng hoàn toàn
            elif distance_to_obstacle < 15.0:  # Tăng từ 10.0 lên 15.0
                # Close obstacle - slow down significantly
                base_speed = 1.0 + (1.0 - emergency_factor) * 2.0
                # Dừng lại hoàn toàn khi người đi bộ cắt ngang ở gần
                if is_crossing_pedestrian and distance_to_obstacle < 12.0:
                    target_speed = 0.0
                else:
                    target_speed = base_speed
            else:
                # Distant obstacle but still emergency - moderate slowdown
                # Allow higher speeds for less certain detections
                certainty_factor = min(1.0, certainty + 0.3)  # Bias toward caution
                if is_crossing_pedestrian:
                    # Giảm tốc mạnh hơn nếu người đi bộ đang cắt ngang
                    target_speed = max(2.0, self.target_speed * (1.0 - certainty_factor * 0.8))
                else:
                    target_speed = max(3.0, self.target_speed * (1.0 - certainty_factor * 0.7))
                
            # Cap emergency target speed by current speed to prevent acceleration in emergency
            target_speed = min(target_speed, current_speed)
            
        # For normal obstacle avoidance (not emergency)
        elif has_obstacles and distance_to_obstacle < 40.0:
            # Adjust target speed based on distance to obstacle
            # Using sigmoid function for smooth transition
            distance_factor = 1.0 / (1.0 + math.exp(-(distance_to_obstacle - 20.0) / 5.0))
            
            # Apply certainty to the distance factor
            certainty_adjusted_factor = distance_factor + (1.0 - certainty) * (1.0 - distance_factor)
            
            # Adjust target speed gradually based on obstacle distance
            min_cruise_speed = 3.0
            target_speed = min_cruise_speed + (self.target_speed - min_cruise_speed) * certainty_adjusted_factor
            
            # Add speed margin based on object type if available
            if obstacle_info and "type" in obstacle_info:
                obj_type = obstacle_info["type"]
                
                # Cautious around pedestrians, more relaxed around vehicles
                if obj_type == "PEDESTRIAN":
                    if is_crossing_pedestrian:
                        target_speed *= 0.5  # Giảm tốc độ mạnh hơn cho người đi bộ cắt ngang
                    else:
                        target_speed *= 0.7  # Slower around pedestrians
                elif obj_type in ["BICYCLE", "CYCLIST"]:
                    target_speed *= 0.8  # Slower around cycles
                elif is_parked_vehicle:
                    target_speed *= 0.8  # Giảm tốc khi đi qua xe đỗ
        
        # Calculate acceleration based on speed error
        speed_error = target_speed - current_speed
        
        # More sophisticated acceleration control
        if speed_error > 0:
            # Accelerate - smoother acceleration for comfort
            acceleration = min(self.max_acceleration, speed_error * 0.5)
        else:
            # Brake - progressive braking force
            # Base deceleration proportional to speed error
            base_deceleration = min(self.max_deceleration, abs(speed_error) * 0.8)
            
            # Enhanced braking in emergency situations
            if emergency:
                # Scale by emergency_blend for smoother transitions
                emergency_decel = self.max_deceleration * emergency_blend
                # Combine approaches - take the stronger of the two
                deceleration = max(base_deceleration, emergency_decel)
            else:
                deceleration = base_deceleration
                
            # Apply final deceleration
            acceleration = -deceleration
        
        # Return calculated values
        return {
            "target_speed": target_speed,
            "acceleration": acceleration
        }

    def _print_debug_info(self, scene_data):
        # Reduce debug information frequency - changed from 1500 to 5000 frames
        if self.debug and hasattr(self, 'debug_counter') and self.debug_counter % 5000 == 0:
            # Find the ego vehicle (autonomous vehicle) in the objects list
            ego_vehicle = None
            for obj in scene_data['objects']:
                if obj.get('autonomous', False):
                    ego_vehicle = obj
                    break
                    
            if ego_vehicle:
                position = ego_vehicle['position']
                heading = ego_vehicle['rotation'][2]  # Z-axis rotation is the heading
                velocity_vector = ego_vehicle['velocity']
                speed = np.linalg.norm(velocity_vector)  # Calculate speed from velocity vector
                
                # Get steering angle from vehicle if available
                steering_angle = ego_vehicle.get('steering', 0.0)
                
                # Get target speed if available, otherwise use current speed
                target_speed = getattr(self, 'target_speed', speed)
                
                # Reduce console output by combining multiple lines into fewer lines
                print(f"\n--- VEHICLE CONTROL DEBUG ---")
                print(f"Position: ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}) | " 
                      f"Heading: {heading:.2f}° | Speed: {speed:.2f} m/s")
                print(f"Steering angle: {steering_angle:.2f}° | " 
                      f"Target speed: {target_speed:.2f} m/s")
                print(f"-----------------------------\n")

        # Update counter
        if hasattr(self, 'debug_counter'):
            self.debug_counter += 1
        else:
            self.debug_counter = 0

    def _calculate_relative_position(self, vehicle, obj):
        """Calculate the position of obj relative to vehicle in vehicle's coordinate frame."""
        # Get positions
        vehicle_pos = np.array(vehicle['position'])
        obj_pos = np.array(obj['position'])
        
        # Calculate relative position in world coordinates
        rel_pos_world = obj_pos - vehicle_pos
        
        # Get vehicle heading
        vehicle_heading = vehicle['rotation'][2]
        
        # Rotate to vehicle's coordinate frame
        cos_heading = np.cos(vehicle_heading)
        sin_heading = np.sin(vehicle_heading)
        
        # Forward is x, left is y in vehicle coordinates
        rel_x = cos_heading * rel_pos_world[0] + sin_heading * rel_pos_world[1]
        rel_y = -sin_heading * rel_pos_world[0] + cos_heading * rel_pos_world[1]
        
        return [rel_x, rel_y, rel_pos_world[2]]
    
    def _calculate_distance(self, rel_pos):
        """Calculate distance from relative position."""
        # Just the planar distance (ignoring height)
        return np.sqrt(rel_pos[0]**2 + rel_pos[1]**2)
    
    def _calculate_speed(self, vehicle):
        """Calculate the current speed of the vehicle in m/s."""
        if 'velocity' in vehicle:
            return np.linalg.norm(vehicle['velocity'])
        return 0.0
    
    def _check_for_obstacles(self, vehicle_position, vehicle_heading, vehicle_dimensions, objects, vehicle_id, road=None):
        """Check for obstacles in the scene with improved detection logic.
        
        Args:
            vehicle_position: The position of the vehicle
            vehicle_heading: The heading of the vehicle in radians
            vehicle_dimensions: The dimensions of the vehicle
            objects: All objects in the scene
            vehicle_id: The ID of the vehicle to exclude
            road: Road data (can be None)
            
        Returns:
            tuple: (has_obstacles, closest_obstacle, distance_to_obstacle, obstacle_info)
        """
        has_obstacles = False
        closest_obstacle = None
        min_distance = float('inf')
        
        # Create vehicle object for relative position calculation
        vehicle = {
            'position': vehicle_position,
            'rotation': [0, 0, vehicle_heading]
        }
        
        # Vehicle forward direction vector (unit vector)
        forward_vec = np.array([np.cos(vehicle_heading), np.sin(vehicle_heading), 0])
        
        # Enhanced detection parameters
        # 1. Immediate detection zone (narrower but longer)
        immediate_zone_width = vehicle_dimensions[1] * 2.5  # Tăng từ 2.0 lên 2.5
        immediate_zone_length = vehicle_dimensions[0] * 12.0  # Tăng từ 10.0 lên 12.0
        
        # 2. Forward detection zone (wider but shorter)
        forward_zone_width = vehicle_dimensions[1] * 4.0  # Tăng từ 3.5 lên 4.0
        forward_zone_length = vehicle_dimensions[0] * 20.0  # Tăng từ 18.0 lên 20.0
        
        # 3. Extended awareness zone (widest but with diminishing importance)
        awareness_zone_width = vehicle_dimensions[1] * 5.5  # Tăng từ 4.5 lên 5.5
        awareness_zone_length = vehicle_dimensions[0] * 35.0  # Tăng từ 30.0 lên 35.0
        
        # Keep track of priority obstacles
        priority_obstacles = []
        
        # Object type weights for priority calculation
        object_type_weights = {
            'PEDESTRIAN': 3.0,    # Tăng từ 2.5 lên 3.0
            'CYCLIST': 2.5,       # Tăng từ 2.0 lên 2.5
            'TRICAR': 1.5,        # Giữ nguyên
            'CAR': 1.2,           # Giữ nguyên
            'TRUCK': 1.0          # Giữ nguyên
        }
        
        # Initialize/update velocity history if needed
        if not hasattr(self, 'velocity_history'):
            self.velocity_history = {}
        
        # Check each object
        for obj in objects:
            # Skip our own vehicle
            if obj['id'] == vehicle_id:
                continue
            
            # Calculate relative position and distance
            rel_pos = self._calculate_relative_position(vehicle, obj)
            distance = self._calculate_distance(rel_pos)
            
            # Project the obstacle onto the forward axis to get true forward distance
            forward_distance = rel_pos[0]
            lateral_distance = abs(rel_pos[1])  # Distance to sides
            
            # Only consider obstacles ahead of the vehicle (with small buffer behind for large vehicles)
            if forward_distance > -vehicle_dimensions[0]:
                # Assign priority based on which zone the obstacle is in
                zone_priority = 0
                
                # Check immediate zone (highest priority)
                if forward_distance < immediate_zone_length and lateral_distance < immediate_zone_width:
                    zone_priority = 3  # Highest priority - immediate danger
                    
                # Check forward zone (medium priority)
                elif forward_distance < forward_zone_length and lateral_distance < forward_zone_width:
                    zone_priority = 2  # Medium priority - approaching obstacle
                    
                # Check awareness zone (lowest priority)
                elif forward_distance < awareness_zone_length and lateral_distance < awareness_zone_width:
                    # Lower priority for more distant and lateral obstacles
                    zone_priority = 1  # Low priority - distant obstacle
                
                # If obstacle is in any detection zone
                if zone_priority > 0:
                    has_obstacles = True
                    
                    # Get object type
                    obj_type = obj.get('type', None)
                    obj_type_name = obj_type.name if obj_type else 'CAR'
                    
                    # Calculate base type priority
                    type_priority = object_type_weights.get(obj_type_name, 1.0)
                    
                    # Track velocity for moving objects
                    obj_velocity = np.array(obj.get('velocity', [0, 0, 0]))
                    obj_speed = np.linalg.norm(obj_velocity)
                    obj_id = obj['id']
                    
                    # Store velocity history for this object
                    if obj_id not in self.velocity_history:
                        self.velocity_history[obj_id] = {
                            'positions': [np.array(obj['position'])],
                            'velocities': [obj_velocity],
                            'timestamps': [time.time() if 'time' not in locals() else time],
                            'predicted_path': []
                        }
                    else:
                        # Update history (keep last 5 positions)
                        history = self.velocity_history[obj_id]
                        history['positions'].append(np.array(obj['position']))
                        history['velocities'].append(obj_velocity)
                        history['timestamps'].append(time.time() if 'time' not in locals() else time)
                        
                        # Limit history size
                        if len(history['positions']) > 5:
                            history['positions'].pop(0)
                            history['velocities'].pop(0)
                            history['timestamps'].pop(0)
                        
                        # Predict future path if we have at least 2 positions
                        if len(history['positions']) >= 2:
                            # Simple linear prediction for 3 seconds ahead
                            current_pos = history['positions'][-1]
                            avg_velocity = np.mean(history['velocities'], axis=0)
                            
                            # Generate prediction points (1, 2, and 3 seconds ahead)
                            predicted_path = []
                            for t in range(1, 4):
                                future_pos = current_pos + avg_velocity * t
                                predicted_path.append(future_pos)
                            
                            history['predicted_path'] = predicted_path
                    
                    # Check if object is moving toward vehicle's path
                    is_moving_toward_vehicle = False
                    collision_risk = 0
                    
                    # Xác định xem đây có phải là xe đỗ không
                    is_parked_vehicle = obj_speed < 0.05
                    
                    # Xác định xem đây có phải là người đi bộ đang cắt ngang đường không
                    is_crossing_pedestrian = False
                    if obj_type_name == 'PEDESTRIAN' and obj_speed > 0.2:
                        # Tính góc giữa hướng di chuyển của người đi bộ và đường
                        if abs(rel_pos[1]) < forward_zone_width and abs(rel_pos[0]) < forward_zone_length:
                            # Người đi bộ ở gần đường
                            if hasattr(self, 'road_start') and hasattr(self, 'road_end'):
                                road_dir = self.road_end - self.road_start
                                road_dir = road_dir / np.linalg.norm(road_dir)
                                ped_dir = obj_velocity / obj_speed
                                
                                # Tính góc giữa hướng đi của người đi bộ và đường
                                crossing_angle = np.arccos(np.clip(np.dot(road_dir[:2], ped_dir[:2]), -1.0, 1.0))
                                
                                # Nếu người đi bộ di chuyển vuông góc với đường
                                if abs(crossing_angle - np.pi/2) < np.pi/4:  # Trong khoảng 45 độ so với vuông góc
                                    is_crossing_pedestrian = True
                                    collision_risk = 0.8  # Đặt mức độ nguy hiểm cao cho người đi bộ cắt ngang
                    
                    if obj_speed > 0.2:  # Only consider moving objects
                        # Calculate object's heading
                        obj_heading = np.arctan2(obj_velocity[1], obj_velocity[0])
                        
                        # Calculate angle between object's heading and vehicle-to-object vector
                        to_vehicle_angle = np.arctan2(-rel_pos[1], -rel_pos[0])
                        angle_diff = abs((obj_heading - to_vehicle_angle + np.pi) % (2 * np.pi) - np.pi)
                        
                        # If object is moving in the general direction of the vehicle
                        if angle_diff < np.pi / 3:  # Within 60 degrees
                            is_moving_toward_vehicle = True
                            
                            # Calculate potential collision point
                            # Simple time-to-collision estimate
                            if distance > 0.1:  # Avoid division by zero
                                time_to_collision = distance / max(0.1, obj_speed)
                                if time_to_collision < 5.0:  # Within 5 seconds
                                    collision_risk = 1.0 - (time_to_collision / 5.0)
                    
                    # Adjust priority based on motion analysis
                    motion_priority = 1.0
                    if is_moving_toward_vehicle:
                        # Increase priority if object is moving toward vehicle
                        motion_priority += collision_risk * 0.5
                    
                    # Adjust priority if object is a pedestrian or cyclist and moving unpredictably
                    if obj_type_name in ['PEDESTRIAN', 'CYCLIST']:
                        # Check for sudden direction changes in history
                        if obj_id in self.velocity_history and len(self.velocity_history[obj_id]['velocities']) > 2:
                            velocities = self.velocity_history[obj_id]['velocities']
                            # Calculate angular change between consecutive velocity vectors
                            if np.linalg.norm(velocities[-1]) > 0.1 and np.linalg.norm(velocities[-2]) > 0.1:
                                v1 = velocities[-2] / np.linalg.norm(velocities[-2])
                                v2 = velocities[-1] / np.linalg.norm(velocities[-1])
                                
                                # Check for sudden direction change
                                direction_stability = np.dot(v1, v2)  # 1 = same direction, -1 = opposite
                                
                                # Lower stability means more unpredictable
                                if direction_stability < 0.7:  # More than ~45 degree change
                                    # Add unpredictability factor to type priority
                                    motion_priority += 0.5 * (1.0 - direction_stability)
                    
                    # Adjust centrality factor based on object type (pedestrians more important even on edges)
                    base_centrality = 1.0 - 0.6 * (lateral_distance / immediate_zone_width)
                    if obj_type_name == 'PEDESTRIAN':
                        # Pedestrians are important even if off to the side
                        centrality_factor = max(0.6, base_centrality)
                    elif obj_type_name == 'CYCLIST':
                        # Cyclists are also important on the side
                        centrality_factor = max(0.5, base_centrality)
                    else:
                        centrality_factor = max(0.4, base_centrality)
                    
                    # Calculate final priority score combining all factors
                    priority_score = zone_priority * type_priority * motion_priority
                    
                    # Calculate effective distance (for priority sorting)
                    effective_distance = distance / (centrality_factor * priority_score)
                    
                    # Create obstacle info dictionary with all relevant data
                    obstacle_info = {
                        'id': obj['id'],
                        'type': obj_type_name,
                        'distance': distance,
                        'rel_pos': rel_pos,
                        'priority': priority_score,
                        'is_moving_toward_vehicle': is_moving_toward_vehicle,
                        'collision_risk': collision_risk,
                        'predicted_path': self.velocity_history.get(obj_id, {}).get('predicted_path', []),
                        'is_parked_vehicle': is_parked_vehicle,
                        'is_crossing_pedestrian': is_crossing_pedestrian,
                        'speed': obj_speed
                    }
                    
                    # Add to priority list
                    priority_obstacles.append((obj, distance, effective_distance, priority_score, obstacle_info))
        
        # Find the highest priority obstacle, with distance as tiebreaker
        obstacle_info = None
        if priority_obstacles:
            # Sort by priority (descending) then by distance (ascending)
            priority_obstacles.sort(key=lambda x: (-x[3], x[1]))
            closest_obstacle = priority_obstacles[0][0]
            min_distance = priority_obstacles[0][1]
            obstacle_info = priority_obstacles[0][4]
            
            # Also keep track of top 3 obstacles for multi-obstacle scenarios
            if len(priority_obstacles) > 1:
                top_obstacles = priority_obstacles[:min(3, len(priority_obstacles))]
                obstacle_info['nearby_obstacles'] = [
                    {'id': ob[0]['id'], 'distance': ob[1], 'priority': ob[3]} 
                    for ob in top_obstacles[1:]  # Skip the first one (it's the closest)
                ]
        
        return has_obstacles, closest_obstacle, min_distance, obstacle_info

    def _calculate_pid_steering(self, error, dt=0.033):
        """Calculate steering angle using PID control for smoother lane keeping"""
        # Proportional term
        p_term = self.pid_kp * error
        
        # Check if sign of error has changed (crossing the center line)
        error_sign_changed = (error * self.last_error) < 0
        
        # Integral term with anti-windup
        # Only add to integral if we're not oscillating heavily and not changing direction
        if not error_sign_changed or abs(error) < 0.2:
            self.error_sum += error * dt
        else:
            # Reset integral term when crossing centerline to prevent overshoot
            self.error_sum = 0
            
        if abs(self.error_sum) > self.pid_max_error_sum:
            self.error_sum = math.copysign(self.pid_max_error_sum, self.error_sum)
        i_term = self.pid_ki * self.error_sum
        
        # Calculate derivative with smoothing
        if dt > 0:
            derivative = (error - self.last_error) / dt
            # Apply stronger low-pass filter to derivative term
            self.last_derivative = (self.derivative_smoothing * self.last_derivative + 
                                    (1 - self.derivative_smoothing) * derivative)
            d_term = self.pid_kd * self.last_derivative
        else:
            d_term = 0
        
        # If we detect a sign change in the error (crossing centerline), apply extra damping
        damping = 1.0
        if error_sign_changed and abs(error) > 0.1:
            damping = self.direction_change_damping
            # Apply stronger damping when crossing centerline
            d_term *= 1.5
        
        # Apply adaptive gain if enabled
        if self.adaptive_gain_enabled:
            # Reduce gain when error is large to prevent overshooting
            error_factor = min(1.0, 1.0 / (1.0 + abs(error) * 2))
            adaptive_p = max(self.min_gain, self.max_gain * error_factor)
            p_term = adaptive_p * error
        
        # Store history for oscillation detection
        self.error_history.append(error)
        if len(self.error_history) > self.oscillation_detection_window:
            self.error_history.pop(0)
        
        # Detect oscillation by checking for frequent direction changes
        if len(self.error_history) >= 3:
            changes = 0
            prev_sign = math.copysign(1, self.error_history[0])
            for e in self.error_history[1:]:
                current_sign = math.copysign(1, e)
                if current_sign != prev_sign:
                    changes += 1
                    prev_sign = current_sign
            
            # If too many changes detected, we're oscillating
            if changes >= 3:
                self.oscillation_detected_counter = min(self.oscillation_recovery_limit, 
                                                       self.oscillation_detected_counter + 1)
                # Apply strong damping to break oscillation
                damping *= (1 + self.oscillation_detected_counter * 0.5)
                # Reduce integral component during oscillation
                i_term *= 0.2
                # Add extra derivative damping during oscillation
                d_term *= 1.5
            else:
                self.oscillation_detected_counter = max(0, self.oscillation_detected_counter - 1)
        
        # Save last error for next iteration
        self.last_error = error
        
        # Calculate raw steering value
        steering = (p_term + i_term + d_term) / damping
        
        # Apply exponential moving average filter to smooth steering output
        if hasattr(self, 'prev_pid_steering'):
            # Stronger smoothing during oscillation
            smoothing = 0.85 if self.oscillation_detected_counter > 0 else 0.7
            steering = smoothing * self.prev_pid_steering + (1 - smoothing) * steering
            
        # Save for next iteration
        self.prev_pid_steering = steering
        
        # Return the clamped steering value
        return max(-self.max_steering_angle, min(self.max_steering_angle, steering))

    def _calculate_lane_keeping_steering(self, car_pos, heading, lane_pos, lane_center):
        # Calculate the cross-track error
        cross_track_error = lane_pos - lane_center
        
        # Calculate the heading error (normalize to [-pi, pi])
        desired_heading = math.atan2(self.road_network.end[1] - self.road_network.start[1], 
                                   self.road_network.end[0] - self.road_network.start[0])
        heading_error = self._normalize_angle(heading - desired_heading)
        
        # Blend PID-based steering (70%) with traditional steering (30%)
        pid_steering = self._calculate_pid_steering(cross_track_error)
        traditional_steering = -(self.steering_gain * cross_track_error + 
                                self.lane_keeping_factor * heading_error)
        
        # Weighted blend
        steering = 0.7 * pid_steering + 0.3 * traditional_steering
        
        # Apply smoothing with current history
        if len(self.steering_history) > 0:
            # Calculate weighted moving average with exponential weights
            total_weight = 0
            weighted_sum = 0
            for i, past_steering in enumerate(reversed(self.steering_history)):
                weight = math.exp(-i * 0.3)  # Exponential weighting
                weighted_sum += past_steering * weight
                total_weight += weight
            
            if total_weight > 0:
                avg_steering = weighted_sum / total_weight
                steering = (1 - self.steering_smoothing) * steering + self.steering_smoothing * avg_steering
        
        # Limit the rate of change of steering
        if len(self.steering_history) > 0:
            last_steering = self.steering_history[-1]
            max_change = self.max_change_rate
            
            # Even stronger limitation when changing direction
            if last_steering * steering < 0:  # Direction change
                max_change *= 0.5  # Further limit during direction changes
            
            steering = max(last_steering - max_change, min(last_steering + max_change, steering))
        
        # Update steering history
        self.steering_history.append(steering)
        if len(self.steering_history) > self.max_steering_history:
            self.steering_history.pop(0)
        
        # Apply emergency limit based on previous steering
        if len(self.steering_history) > 2:
            # If steering has been changing directions frequently, apply additional smoothing
            direction_changes = 0
            for i in range(1, len(self.steering_history)):
                if self.steering_history[i] * self.steering_history[i-1] < 0:
                    direction_changes += 1
            
            if direction_changes >= 2:  # Multiple direction changes
                # Apply stronger stabilizing: pull toward average of recent history
                recent_avg = sum(self.steering_history[-5:]) / min(5, len(self.steering_history))
                steering = 0.7 * recent_avg + 0.3 * steering
        
        return steering

    def _handle_steering(self, data):
        """
        Process steering logic based on lane following and obstacle avoidance
        
        Args:
            data: Dictionary containing relevant scene data
            
        Returns:
            Float representing steering angle in radians
        """
        # Get lane following data
        lane_data = self._handle_lane_following(data)
        
        # Get obstacle avoidance data
        obstacle_data = self._handle_obstacle_avoidance(data)
        
        # Base steering from lane following
        base_steering = lane_data.get("steering_angle", 0.0)
        
        # Extract vehicle data for contextual decision making
        vehicle = data.get("vehicle", {})
        veh_vel = np.array(vehicle.get("velocity", [0, 0, 0]))
        speed = np.linalg.norm(veh_vel)
        
        # Initialize final steering value
        final_steering = base_steering
        
        # If obstacles are detected, consider avoidance steering
        if obstacle_data.get("has_obstacles", False):
            # Get avoidance direction (negative for left, positive for right)
            avoidance_direction = obstacle_data.get("avoidance_direction", 0.0)
            
            # If we're in emergency situation, blend avoidance steering more strongly
            if obstacle_data.get("emergency", False):
                # Get emergency blend factor
                emergency_blend = obstacle_data.get("emergency_blend", 0.0)
                
                # Calculate max avoidance steering based on current speed
                # At low speeds, allow sharper turns for avoidance
                max_avoidance_angle = np.radians(30.0)  # 30 degrees max
                if speed < 5.0:
                    max_avoidance_angle = np.radians(45.0)  # 45 degrees at low speed
                elif speed > 20.0:
                    max_avoidance_angle = np.radians(15.0)  # 15 degrees at high speed
                
                # Calculate avoidance steering (scale by direction and emergency blend)
                avoidance_steering = avoidance_direction * max_avoidance_angle * emergency_blend
                
                # Blend base steering with avoidance steering based on emergency level
                # Higher emergency gives more weight to avoidance
                final_steering = base_steering * (1.0 - emergency_blend) + avoidance_steering * emergency_blend
            else:
                # For non-emergency situations, apply gentler avoidance
                # Calculate avoidance influence based on obstacle distance
                obstacle_distance = obstacle_data.get("obstacle_distance", float('inf'))
                avoidance_influence = 0.0
                
                # Only apply gentle avoidance for obstacles within a reasonable distance
                if obstacle_distance < 30.0:
                    avoidance_influence = 0.3 * (1.0 - min(1.0, obstacle_distance / 30.0))
                    
                    # Get obstacle information for more intelligent decision making
                    obstacle_info = obstacle_data.get("obstacle_info", {})
                    obstacle_type = obstacle_info.get("type", "UNKNOWN")
                    will_cross_path = obstacle_info.get("will_cross_path", False)
                    
                    # Adjust influence based on obstacle type and movement pattern
                    if obstacle_type == "PEDESTRIAN":
                        # More cautious around pedestrians
                        avoidance_influence *= 1.5
                    elif will_cross_path:
                        # More cautious with objects predicted to cross our path
                        avoidance_influence *= 1.3
                
                # Calculate gentle avoidance steering
                gentle_avoidance = avoidance_direction * np.radians(10.0) * avoidance_influence
                
                # Blend with base steering
                final_steering = base_steering + gentle_avoidance
        
        # Apply steering limits and smoothing
        max_steering_angle = np.radians(25.0)  # 25 degrees max
        
        # Limit steering angle
        final_steering = np.clip(final_steering, -max_steering_angle, max_steering_angle)
        
        # Apply steering rate limiting for smoother control
        # Get previous steering angle
        prev_steering = self._prev_steering
        
        # Calculate maximum change in steering per step
        max_steering_delta = np.radians(3.0)  # 3 degrees max change per step
        
        # Apply rate limiting
        if prev_steering is not None:
            steering_delta = final_steering - prev_steering
            if abs(steering_delta) > max_steering_delta:
                # Limit the rate of change
                steering_delta = np.sign(steering_delta) * max_steering_delta
                final_steering = prev_steering + steering_delta
        
        # Store current steering for next iteration
        self._prev_steering = final_steering
        
        # Create debug data for visualization and logging
        debug_data = {
            "base_steering": np.degrees(base_steering),
            "final_steering": np.degrees(final_steering),
            "has_obstacles": obstacle_data.get("has_obstacles", False),
            "obstacle_distance": obstacle_data.get("obstacle_distance", float('inf')),
            "emergency": obstacle_data.get("emergency", False),
            "avoidance_direction": obstacle_data.get("avoidance_direction", 0.0)
        }
        
        # Store debug data in the autonomous state
        self._last_debug["steering"] = debug_data
        
        return final_steering

# Factory method to get different autonomous logic implementations
def get_autonomous_logic(logic_type='lane_keeping'):
    """Get an autonomous driving logic implementation.
    
    Args:
        logic_type (str): Type of autonomous logic to use
        
    Returns:
        AutonomousLogic: An implementation of autonomous driving logic
    """
    # Create a simple road network object to initialize the logic
    class SimpleRoadNetwork:
        def __init__(self):
            self.start = np.array([-100, 0, 0])
            self.end = np.array([100, 0, 0])
            self.width = 10.0
            self.lanes = 2
    
    road_network = SimpleRoadNetwork()
    
    if logic_type == 'basic':
        return BasicAutonomousLogic()
    elif logic_type == 'lane_keeping':
        return LaneKeepingLogic(road_network)
    else:
        # Default to lane keeping logic
        return LaneKeepingLogic(road_network) 