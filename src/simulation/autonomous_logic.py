import numpy as np
import math

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
        self.target_speed = 5.0  # Reduced from 6.0 for better stability
        self.speed_smoothing = 0.9
        
        # Lookahead parameters
        self.lookahead_base = 15.0  # Increased from 12.0 to look further ahead
        self.lookahead_min = 12.0  # Increased from 10.0
        self.lookahead_speed_factor = 0.8
        
        # Curve handling
        self.curve_adjustment = 3.0  # Reduced from 4.0
        self.curve_speed_reduction = 0.7  # How much to slow down in curves
        self.curve_transition_smoothing = 0.95  # Increased from 0.9
        
        # PID controller for lane tracking
        self.pid_kp = 0.2  # Reduced from 0.25 - lower proportional gain for less aggressive response
        self.pid_ki = 0.0001  # Reduced from 0.0003 - much smaller integral gain to prevent overshoot
        self.pid_kd = 2.0  # Increased from 1.5 - stronger derivative gain to dampen oscillations
        self.error_history = []
        self.error_sum = 0.0
        self.pid_max_error_sum = 1.5  # Reduced from 2.0 - lower integral windup limit
        self.last_error = 0.0
        self.last_derivative = 0.0
        self.derivative_smoothing = 0.9  # Increased from 0.8 for stronger smoothing of derivative
        self.prev_pid_steering = 0.0  # Add variable for exponential moving average filter
        
        # Oscillation detection and prevention
        self.oscillation_detection_window = 10
        self.steering_direction_changes = 0
        self.previous_steering_direction = None
        self.direction_change_damping = 3.5  # Increased from 3.0 for stronger damping on direction changes
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
        self.max_steering = 0.7
        self.steering_smoothing = 0.9  # Increased from 0.8 for much smoother steering
        self.previous_steering = 0.0
        
        # Add a steering history array for stronger filtering
        self.steering_history = []
        self.max_steering_history = 15  # Increased from 10 for better averaging
        
        # Lane keeping parameters
        self.target_lane_offset = 2.5  # Positive = target right lane, negative = target left lane
        self.lane_keeping_factor = 0.6  # Reduced from 0.8 to make lane corrections less aggressive
        self.curve_following_factor = 0.6  # Reduced from 0.8 for smoother curve handling
        self.obstacle_avoidance_factor = 1.5  # Kept same
        self.emergency_steering_factor = 1.8  # Kept same
        self.current_target_lane = 0.0
        self.lane_change_smoothness = 0.3  # Increased from 0.2 for smoother lane changes
        
        # Core parameters
        self.lane_offset = 1.5  # meters, positive is right lane, negative is left lane
        self.lookahead_min = 10.0  # Increased from 8.0 for better anticipation
        self.lookahead_factor = 0.75  # Increased from 0.65 for smoother steering
        self.steering_gain = 0.5  # Reduced from 0.7 for less responsive but more stable steering
        self.max_acceleration = 2.5  # Increased from 2.0 for better acceleration
        self.max_deceleration = 4.5  # Increased from 4.0 for stronger braking
        self.straight_road_speed_boost = 1.05
        
        # Road parameters - note, will be replaced with data from scene
        # For now, keeping variables for backward compatibility
        self.road_width = 10.0
        self.road_length = 200.0
        self.curve_start = 50.0
        self.curve_intensity = 0.005
        self.driving_side = "right"  # or "left"
        
        # Physical driving parameters to make motion more naturalistic
        self.heading_correction_gain = 0.15  # Reduced from 0.2 for smoother corrections
        self.max_heading_correction = 0.015  # Reduced from 0.02 for less aggressive corrections
        self.steering_smoothing = 0.8  # Reduced from 0.85 for more responsive steering
        self.natural_understeer = 0.1
        
        # Collision avoidance parameters
        self.avoidance_lane_shift = 3.0  # Increased from 2.5 for wider avoidance
        self.avoidance_shift_speed = 0.1  # Increased from 0.08 for faster lateral movements
        self.obstacle_recovery_rate = 0.02
        self.current_lane_shift = 0.0
        self.returning_to_lane = False
        self.obstacle_detection_distance = 45.0  # Increased from 40.0 for earlier detection
        self.lateral_safety_margin = 1.2  # Increased from 1.0 for wider safety margin
        
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
        self.collision_threshold = 0.1  # meters
        self.emergency_braking_factor = 0.7  # Reduced from 0.8 for less aggressive braking
        self.min_emergency_speed = 1.0  # Increased from 0.2 for better maneuverability
        self.mode = "NORMAL"
        self.emergency_timer = 0
        self.emergency_cooldown = 30  # frames
        self.lateral_shift_factor = 1.2  # Increased from 1.0 for more lateral movement
        self.braking_intensity = 0.9  # Reduced from 1.0 for smoother braking
        self.in_curve = False
        self.curve_detection_threshold = 0.08  # Reduced from 0.1 for smoother curve detection
        self.max_steering = 0.7  # Add the missing attribute
        
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
        
        # Print debug info periodically
        if hasattr(self, 'frame_counter') and self.frame_counter % 100 == 0:
            print(f"\n=== ROAD CURVE DEBUG INFO ===")
            print(f"Road params: length={self.road_length}, curve_start={self.curve_start}, intensity={self.curve_intensity}")
            print(f"Position x={x_position}, curve_t={curve_t}, calculated y={y}")
            print(f"==============================\n")
            self.frame_counter += 1
        elif not hasattr(self, 'frame_counter'):
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
    
    def _get_road_info(self, vehicle, scene_data):
        """Get information about the road and vehicle position."""
        current_x = vehicle['position'][0]
        current_y = vehicle['position'][1]
        
        # Always update road parameters from scene_data if available
        if 'road_width' in scene_data:
            self.road_width = scene_data['road_width']
        if 'road_length' in scene_data:
            self.road_length = scene_data['road_length']
        
        # FOR DEBUGGING - force print road parameters
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
            print(f"Road data: {first_road}")
            
            # Update road width if available
            if 'width' in first_road:
                self.road_width = first_road['width']
                print(f"Setting road_width to {self.road_width}")
                
            # Extract start and end points
            if 'start' in first_road and 'end' in first_road:
                start = np.array(first_road['start'])
                end = np.array(first_road['end'])
                print(f"Road start: {start}, end: {end}")
                
                # Calculate road length
                road_vec = end - start
                self.road_length = np.linalg.norm(road_vec)
                print(f"Setting road_length to {self.road_length}")
                
                # Calculate road angle/heading
                road_dir = road_vec / self.road_length
                road_angle = np.arctan2(road_dir[1], road_dir[0]) if 'road_dir' in locals() else np.arctan2(end[1] - start[1], end[0] - start[0])
                print(f"Road angle: {road_angle}")
                
                # CRITICAL SECTION: Determine if this is a curved road
                if abs(end[1] - start[1]) > 0.1:
                    print(f"CURVED ROAD DETECTED: y-diff = {abs(end[1] - start[1])}")
                    # This is a curved road
                    self.curve_start = 0  # Start curve immediately
                    # Calculate curve intensity
                    self.curve_intensity = abs(end[1] - start[1]) / self.road_length
                    print(f"Setting curve_start=0, curve_intensity={self.curve_intensity}")
                    
                    # Store road endpoints for lane calculation
                    self.road_start = start
                    self.road_end = end
                else:
                    print("STRAIGHT ROAD DETECTED")
                    self.curve_start = self.road_length  # No curve
                    self.curve_intensity = 0
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
                print(f"Calculated lane_center_y = {lane_center_y}")
        
        # DEBUGGING - After processing road network
        print(f"After processing: road_width={self.road_width}, road_length={self.road_length}")
        print(f"curve_start={self.curve_start}, curve_intensity={self.curve_intensity}")
        print(f"===== END ROAD PARAMS DEBUG =====\n")
        
        # If road network processing didn't yield a lane center, fallback to original calculation
        if lane_center_y is None:
            lane_center_y = self.calculate_lane_center(current_x)
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
        """Process the current scene and return control commands.
        
        Args:
            vehicle (dict): The vehicle to control
            scene_data (dict): Current scene data
            
        Returns:
            dict: Control commands (acceleration, steering)
        """
        # Package data
        data = {
            "vehicle": vehicle,
            "scene": scene_data
        }
        
        # Get key vehicle properties
        current_pos = np.array(vehicle['position'])
        current_x = current_pos[0]
        current_y = current_pos[1]
        vehicle_heading = vehicle['rotation'][2]
        current_speed = self._calculate_speed(vehicle)
        
        # Create road information
        road_info = self._get_road_info(vehicle, scene_data)
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
        
        # Check for obstacles using our _check_for_obstacles method
        all_objects = scene_data.get('objects', [])
        has_obstacles, closest_obstacle, distance_to_obstacle = self._check_for_obstacles(
            current_pos, vehicle_heading, vehicle['dimensions'], all_objects, vehicle['id'], None
        )
        
        # Calculate safe distance for obstacle avoidance
        safe_distance = 2.0
        if has_obstacles and closest_obstacle is not None:
            # Increase safe distance for better obstacle avoidance
            safe_distance = 2.5 * (vehicle['dimensions'][0] + closest_obstacle.get('dimensions', [4.0, 2.0, 1.5])[0])
        
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
        
        # IMPROVEMENT: Calculate avoidance direction based on obstacle and road position
        avoidance_direction = 0
        if has_obstacles and closest_obstacle is not None:
            # Calculate relative position of obstacle
            rel_pos = self._calculate_relative_position(vehicle, closest_obstacle)
            
            # Determine direction to steer (positive = right, negative = left)
            # Add hysteresis to avoid rapid direction changes
            if not hasattr(self, 'last_avoidance_direction'):
                self.last_avoidance_direction = 0
                
            # Make avoidance direction proportional to obstacle's relative position with temporal smoothing
            avoidance_factor = min(1.0, abs(rel_pos[1]) / 5.0)  # Normalize by 5m distance
            raw_direction = -avoidance_factor if rel_pos[1] >= 0 else avoidance_factor
            
            # Apply stronger temporal smoothing (60% new direction, 40% previous direction) for better stability
            # This prevents rapid direction changes that cause zigzagging
            avoidance_direction = 0.6 * raw_direction + 0.4 * self.last_avoidance_direction
            
            # Apply direction change threshold to prevent small oscillations
            # Only change direction if the new direction differs significantly
            if abs(avoidance_direction - self.last_avoidance_direction) < 0.15:
                avoidance_direction = self.last_avoidance_direction
                
            self.last_avoidance_direction = avoidance_direction
            
            # But if near edge, prioritize avoiding the edge over avoiding the obstacle
            if near_edge:
                if left_edge < 1.0:  # Near left edge, prefer right turns
                    avoidance_direction = 1
                elif right_edge < 1.0:  # Near right edge, prefer left turns
                    avoidance_direction = -1
        
        # Check if we're on a curved road
        is_curved_road = self.curve_intensity > 0.01
        
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
        curve_adjustment = 0
        if is_curved_road:
            # Direction of the curve (positive = right curve, negative = left curve)
            curve_direction = np.sign(self.road_end[1] - self.road_start[1]) if hasattr(self, 'road_end') else 0
            
            # Adjust based on curve intensity and current deviation
            # Reduced factor from 4.0 to 3.0 for smoother curve following
            curve_adjustment = curve_direction * self.curve_intensity * self.curve_following_factor * 3.0
            
            # Apply stronger smoothing
            if not hasattr(self, 'prev_curve_adjustment'):
                self.prev_curve_adjustment = 0
            
            # Increased smoothing factor to 0.9 (from 0.8) for even smoother curve transitions
            curve_adjustment = 0.9 * self.prev_curve_adjustment + 0.1 * curve_adjustment
            self.prev_curve_adjustment = curve_adjustment
            
            # Apply exponential curve adjustment to prevent rapid changes
            # This creates a more gradual application of curve steering
            curve_adjustment_sign = 1 if curve_adjustment > 0 else -1
            curve_adjustment_mag = abs(curve_adjustment)
            smoothed_mag = curve_adjustment_mag ** 0.8  # Apply non-linear scaling to smooth changes
            curve_adjustment = curve_adjustment_sign * smoothed_mag
        
        # Add the curve adjustment
        steering_angle += curve_adjustment
        
        # Print steering debug info periodically
        if hasattr(self, 'steering_debug_counter') and self.steering_debug_counter % 50 == 0:
            print(f"\n=== STEERING DEBUG INFO ===")
            print(f"Steering: {round(steering_angle, 2)} rad ({'LEFT' if steering_angle < 0 else 'RIGHT'} {abs(round(steering_angle/self.max_steering*100))}%)")
            print(f"Position Y: {round(current_y, 2)} (Lane Center: {round(lane_center_y, 2)}, Target: {round(lane_center_y + lane_offset, 2)})")
            print(f"Heading: {round(vehicle_heading, 2)} rad")
            print(f"Lane Position: {lane_position} with {1.5}m offset")
            print(f"Left edge: {round(left_edge, 2)}m | Right edge: {round(right_edge, 2)}m")
            print(f"==============================")
            self.steering_debug_counter += 1
        elif not hasattr(self, 'steering_debug_counter'):
            self.steering_debug_counter = 0
        else:
            self.steering_debug_counter += 1
        
        # Package data for obstacle avoidance
        data.update({
            "avoidance_direction": avoidance_direction,
            "has_obstacles": has_obstacles,
            "closest_obstacle": closest_obstacle,
            "distance_to_obstacle": distance_to_obstacle if 'distance_to_obstacle' in locals() else float('inf'),
            "obstacle_id": 'obstacle_id' in locals() and obstacle_id,
            "current_steering": steering_angle,
            "lane_position": lane_position,
            "lane_position_value": lane_position_value,
            "left_edge": left_edge,
            "right_edge": right_edge,
            "current_speed": current_speed,
            "is_curved_road": is_curved_road
        })
        
        # Handle obstacle avoidance
        obstacle_result = self._handle_obstacle_avoidance(data, steering_angle, emergency=False)
        steering_angle = obstacle_result["steering_angle"]
        is_emergency = obstacle_result["is_emergency"]
        
        # Apply final steering limits
        steering_angle = np.clip(steering_angle, -self.max_steering, self.max_steering)
        
        # Handle speed control
        speed_result = self._handle_speed_control(data, emergency=is_emergency)
        target_speed = speed_result["target_speed"]
        acceleration = speed_result["acceleration"]
        
        # Print debug information
        self._print_debug_info(data, steering_angle, target_speed, is_emergency)
        
        # Return control commands
        return {
            'steering': steering_angle,
            'acceleration': acceleration
        }

    def _handle_obstacle_avoidance(self, data, steering_angle, emergency=False):
        """Handle obstacle avoidance behavior."""
        # Get vehicle data
        vehicle = data["vehicle"]
        scene_data = data["scene"]
        
        # Extract obstacle information
        has_obstacles = data.get("has_obstacles", False)
        closest_obstacle = data.get("closest_obstacle", None)
        distance_to_obstacle = data.get("distance_to_obstacle", float('inf'))
        obstacle_id = data.get("obstacle_id", None)
        
        # Initialize emergency state tracking if not present
        if not hasattr(self, 'emergency_blend'):
            self.emergency_blend = 0.0  # 0 = normal, 1 = full emergency
            
        # Calculate smooth emergency blend factor based on obstacle distance
        if has_obstacles and closest_obstacle is not None:
            # Define distance thresholds
            critical_distance = self.collision_threshold * 0.8
            safe_distance = self.collision_threshold * 1.5
            
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
            
            # Gradually approach target emergency level (faster increase, slower decrease)
            if target_emergency > self.emergency_blend:
                # Quick response to danger
                self.emergency_blend = min(1.0, self.emergency_blend + max(0.1, (target_emergency - self.emergency_blend) * 0.4))
            else:
                # Slower return to normal
                self.emergency_blend = max(0.0, self.emergency_blend - 0.05)
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
            rel_pos = self._calculate_relative_position(vehicle, closest_obstacle)
            
            # Calculate relative angle to obstacle
            rel_angle = np.arctan2(rel_pos[1], rel_pos[0])
            
            # Calculate distance to obstacle
            distance = self._calculate_distance(rel_pos)
            
            # Calculate intensity based on distance (closer = stronger response)
            # Smoother intensity curve with improved response near collision threshold
            intensity = max(0.0, min(1.0, self.collision_threshold / max(0.1, distance)))
            
            # Apply non-linear intensity scale - more responsive at medium distances
            if distance < self.collision_threshold * 0.5:
                # Very close - maximum response
                intensity = 1.0
            else:
                # Adjust response curve for medium distances
                proximity = 1.0 - (distance - self.collision_threshold * 0.5) / (self.collision_threshold * 0.5)
                intensity = max(0.0, min(1.0, proximity ** 1.5))  # Non-linear curve
            
            # Apply smoother avoidance behavior
            # Get avoidance direction from vehicle data (-1 = left, 1 = right)
            avoidance_direction = data.get("avoidance_direction", 0)
            
            # Calculate avoidance steering (smooth and proportional)
            max_avoidance_angle = 0.6  # Maximum steering angle for avoidance
            avoidance_steering = avoidance_direction * intensity * max_avoidance_angle
            
            # Record avoidance_bias for debugging
            avoidance_bias = avoidance_direction * intensity
            
            # Apply smoother blending between normal and emergency steering
            result_steering = (normal_factor * original_steering) + (emergency_factor * avoidance_steering)
            
            # Add result to steering history for filtering
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
        data["emergency"] = is_emergency
        data["obstacle_distance"] = distance_to_obstacle
        data["obstacle_id"] = obstacle_id
        data["avoidance_bias"] = avoidance_bias
        data["emergency_blend"] = emergency_factor
        
        return {
            "steering_angle": result_steering,
            "is_emergency": is_emergency
        }

    def _handle_speed_control(self, data, emergency=False):
        """Handle speed control based on vehicle state and obstacles."""
        # Get vehicle data
        vehicle = data["vehicle"]
        
        # Get current speed and emergency information
        current_speed = data.get("current_speed", 0)
        has_obstacles = data.get("has_obstacles", False)
        distance_to_obstacle = data.get("distance_to_obstacle", float('inf'))
        emergency_blend = data.get("emergency_blend", 0.0)
        avoidance_direction = data.get("avoidance_direction", 0)
        is_curved_road = data.get("is_curved_road", False)
        
        # Initialize target speed to normal cruise speed
        target_speed = self.target_speed
        
        # For curved roads, reduce target speed
        if is_curved_road:
            # Reduce speed proportional to curve intensity
            curve_factor = 1.0 - min(0.4, self.curve_intensity * 4.0)  # Max 40% reduction
            target_speed *= curve_factor
            
            # Ensure minimum speed on curves
            target_speed = max(5.0, target_speed)
        
        # Apply speed control based on obstacle distance with smooth blending
        if has_obstacles and distance_to_obstacle < float('inf'):
            # Calculate appropriate speed based on distance to obstacle
            # More gradual speed reduction curve
            distance_factor = min(1.0, max(0.0, distance_to_obstacle / (self.collision_threshold * 3.0)))
            
            # Apply non-linear scaling for smoother transitions
            distance_factor = distance_factor ** 1.5  # More responsive at medium distances
            
            # Calculate obstacle-based target speed
            obstacle_target_speed = self.target_speed * distance_factor
            
            # Set minimum speed based on emergency level for better control
            min_speed = 2.0 + (1.0 - emergency_blend) * 3.0  # 2.0 to 5.0 based on emergency level
            obstacle_target_speed = max(min_speed, obstacle_target_speed)
            
            # Blend between normal and obstacle-based speed using emergency blend factor
            target_speed = (1.0 - emergency_blend) * target_speed + emergency_blend * obstacle_target_speed
            
            # If actively steering to avoid obstacle, maintain adequate speed for effective steering
            if abs(avoidance_direction) > 0.3 and emergency_blend > 0.5:
                steering_min_speed = 3.5  # Minimum speed needed for effective steering
                target_speed = max(steering_min_speed, target_speed)
        
        # Calculate required acceleration/deceleration
        speed_diff = target_speed - current_speed
        
        # Apply acceleration with smoother profile
        if speed_diff > 0:
            # Accelerate - more responsive at lower speeds
            accel_factor = 1.0
            if current_speed < 3.0:
                # Boost acceleration at very low speeds for better responsiveness
                accel_factor = 1.3
                
            acceleration = min(speed_diff, self.max_acceleration * accel_factor)
        else:
            # Decelerate - with emergency-based scaling
            decel_factor = 1.0 + emergency_blend * 0.5  # 1.0 to 1.5 based on emergency level
            
            # Reduce deceleration when actively steering to prevent understeer
            if abs(avoidance_direction) > 0.5 and emergency_blend > 0.7:
                decel_factor *= 0.7  # Reduce braking during active steering
                
            acceleration = max(speed_diff, -self.max_deceleration * decel_factor)
        
        # Return speed control results
        return {
            "target_speed": target_speed,
            "acceleration": acceleration
        }

    def _print_debug_info(self, data, steering_angle, target_speed, emergency):
        """Print debug information about vehicle state and control decisions."""
        # Get basic vehicle data
        vehicle = data["vehicle"]
        current_speed = data.get("current_speed", 0)
        lane_position = data.get("lane_position", "UNKNOWN")
        lane_position_value = data.get("lane_position_value", 0)
        left_edge = data.get("left_edge", 0)
        right_edge = data.get("right_edge", 0)
        emergency_blend = data.get("emergency_blend", 0)
        
        # Determine vehicle mode based on emergency blend
        if emergency_blend > 0.6:
            mode = "EMERGENCY MODE"
        elif emergency_blend > 0.25:
            mode = "CAUTION MODE"
        else:
            mode = "NORMAL"
            
        # Create debug info dictionary
        debug_info = {
            'mode': mode,
            'heading': round(vehicle['rotation'][2], 2),
            'lane_position': lane_position + f" with {abs(round(lane_position_value, 1))}m offset",
            'left_edge': round(left_edge, 2),
            'right_edge': round(right_edge, 2),
            'current_speed': round(current_speed, 2),
            'target_speed': round(target_speed, 2),
            'steering_angle': round(steering_angle, 2),
            'direction': "LEFT" if steering_angle < 0 else "RIGHT",
        }
        
        # Add obstacle information if present
        if data.get("has_obstacles", False):
            debug_info['obstacle_distance'] = round(data.get("distance_to_obstacle", 0), 2)
            if "obstacle_id" in data and data["obstacle_id"] is not None:
                debug_info['obstacle_id'] = data["obstacle_id"]
            if "avoidance_bias" in data:
                debug_info['avoidance_bias'] = round(data.get("avoidance_bias", 0), 2)
        
        # Print debug information
        debug_string = "\n--- VEHICLE CONTROL DEBUG ---\n"
        for key, value in debug_info.items():
            debug_string += f"{key}: {value}\n"
        debug_string += "-----------------------------\n"
        print(debug_string)

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
            tuple: (has_obstacles, closest_obstacle, distance_to_obstacle)
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
        immediate_zone_width = vehicle_dimensions[1] * 2.0  # Increased from 1.8
        immediate_zone_length = vehicle_dimensions[0] * 10.0  # Increased from 8.0
        
        # 2. Forward detection zone (wider but shorter)
        forward_zone_width = vehicle_dimensions[1] * 3.5  # Increased from 3.0
        forward_zone_length = vehicle_dimensions[0] * 18.0  # Increased from 15.0
        
        # 3. Extended awareness zone (widest but with diminishing importance)
        awareness_zone_width = vehicle_dimensions[1] * 4.5  # Increased from 4.0
        awareness_zone_length = vehicle_dimensions[0] * 30.0  # Increased from 25.0
        
        # Keep track of priority obstacles
        priority_obstacles = []
        
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
                priority = 0
                
                # Check immediate zone (highest priority)
                if forward_distance < immediate_zone_length and lateral_distance < immediate_zone_width:
                    priority = 3  # Highest priority - immediate danger
                    
                # Check forward zone (medium priority)
                elif forward_distance < forward_zone_length and lateral_distance < forward_zone_width:
                    priority = 2  # Medium priority - approaching obstacle
                    
                # Check awareness zone (lowest priority)
                elif forward_distance < awareness_zone_length and lateral_distance < awareness_zone_width:
                    # Lower priority for more distant and lateral obstacles
                    priority = 1  # Low priority - distant obstacle
                
                # If obstacle is in any detection zone
                if priority > 0:
                    has_obstacles = True
                    
                    # Adjust effective distance based on priority and lateral position
                    # Obstacles directly ahead feel closer than ones to the side
                    centrality_factor = 1.0 - 0.6 * (lateral_distance / immediate_zone_width)  # Reduced from 0.7
                    centrality_factor = max(0.4, centrality_factor)  # Increased from 0.3 - don't discount too much
                    
                    # Calculate effective distance (used for prioritization)
                    effective_distance = distance / (centrality_factor * priority)
                    
                    # Add to priority list
                    priority_obstacles.append((obj, distance, effective_distance, priority))
        
        # Find the highest priority obstacle, with distance as tiebreaker
        if priority_obstacles:
            # Sort by priority (descending) then by distance (ascending)
            priority_obstacles.sort(key=lambda x: (-x[3], x[1]))
            closest_obstacle = priority_obstacles[0][0]
            min_distance = priority_obstacles[0][1]
        
        return has_obstacles, closest_obstacle, min_distance

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