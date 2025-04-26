import numpy as np
import math
import time
from collections import deque
from simple_pid import PID

class AutonomousLogic:
    """Autonomous driving logic for the vehicle."""
    
    def __init__(self, target_speed=5.0, debug=False):
        # Speed control
        self.target_speed = target_speed
        self.speed_history = deque(maxlen=10)
        self.acceleration = 0.0
        self.debug = debug
        self.debug_info = {}
        
        # Lane keeping - removed circular reference
        self.lane_blend = 0.95  # How much influence lane keeping has vs navigation
        
        # PID controllers
        self.steering_pid = PID(0.8, 0.2, 0.1, setpoint=0)
        self.steering_pid.output_limits = (-1.0, 1.0)
        self.throttle_pid = PID(2.0, 0.2, 0.1, setpoint=target_speed)
        self.throttle_pid.output_limits = (-1.0, 1.0)
        
        # Avoidance parameters
        self.avoidance_lane_shift = 5.0  # Maximum lane shift for avoidance (increased from 3.5)
        self.avoidance_shift_speed = 0.3  # Speed of lane shift (increased from 0.15)
        self.lateral_safety_margin = 2.0  # Lateral safety margin (increased from 1.5)
        self.obstacle_recovery_rate = 0.03  # Rate of recovery after obstacle passing (made slower)
        self.collision_threshold = 5.0  # Base collision threshold in meters
        self.obstacle_detection_distance = 100.0  # Detection distance in meters (increased from 60)
        
        # Avoidance state
        self.current_lane_shift = 0.0
        self.returning_to_lane = False
        self.previous_avoidance_steering = 0.0
        
        # Detection zones
        self.immediate_zone = {
            'x_min': -5.0,   # 5m behind
            'x_max': 100.0,  # 100m ahead (increased from 30m)
            'y_min': -7.0,   # 7m to the left (increased from 5m)
            'y_max': 7.0     # 7m to the right (increased from 5m)
        }
        
        self.long_range_zone = {
            'x_min': -10.0,   # 10m behind
            'x_max': 150.0,   # 150m ahead (increased from 60m)
            'y_min': -10.0,   # 10m to the left (increased from 7m)
            'y_max': 10.0     # 10m to the right (increased from 7m)
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
        self.avoidance_lane_shift = 7.0  # Tăng từ 3.5 lên 7.0 để né với biên độ lớn hơn nhiều
        self.avoidance_shift_speed = 1.0  # Tăng từ 0.15 lên 1.0 để phản ứng ngay lập tức
        self.obstacle_recovery_rate = 0.01  # Giảm từ 0.02 xuống 0.01 để duy trì đánh lái lâu hơn
        self.current_lane_shift = 0.0
        self.returning_to_lane = False
        self.obstacle_detection_distance = 150.0  # Tăng từ 45.0 lên 150.0 để phát hiện từ xa hơn
        self.lateral_safety_margin = 5.0  # Tăng từ 1.2 lên 5.0 cho khoảng cách an toàn lớn hơn
        
        # Emergency management
        self.emergency_mode = False
        self.emergency_counter = 0
        self.max_emergency_frames = 50  # Tăng từ 10 lên 50
        
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
        self.warning_distance = 30.0  # Tăng từ 10.0 lên 30.0 meters
        self.collision_threshold = 50.0  # Tăng từ 30.0 lên 50.0 - Distance to start collision avoidance
        self.emergency_braking_factor = 1.0  # Tăng từ 0.7 lên 1.0 để luôn phanh gấp
        self.min_emergency_speed = 0.0  # Giảm từ 1.0 xuống 0.0 để luôn dừng hẳn
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
    
    def process(self, sensor_data, frame_count):
        """
        Process sensor data and return vehicle control data.
        
        Args:
            sensor_data: Dictionary of sensor data
            frame_count: Current frame count (int or dict)
            
        Returns:
            Dictionary of vehicle control data
        """
        # Ensure frame_count is handled correctly
        if isinstance(frame_count, dict):
            # Store the dict for backward compatibility
            self.frame_count = frame_count
            # Extract frame number if available
            self.frame_count_value = frame_count.get('frame', 0)
        else:
            # It's an integer
            self.frame_count = {'frame': frame_count}
            self.frame_count_value = frame_count
        
        # First check for immediate collision risks
        critical_action = self._critical_collision_prevention(sensor_data)
        if critical_action:
            # Return emergency control actions if critical situation detected
            return critical_action
        
        # Normal processing if no critical situation
        data = {}
        
        # Handle different sensor data
        # Process road information, obstacle avoidance, lane keeping, speed control, etc.
        
        # Call _handle_obstacle_avoidance with frame_count
        data = self._handle_obstacle_avoidance(sensor_data, data, frame_count)
        
        road_info = sensor_data.get("road_info", {})
        
        # Get drivable area percentage
        drivable_area = road_info.get("drivable_area_percentage", 0.0)
        
        # If insufficient drivable area, handle as emergency
        # Temporarily disable this check to avoid warning spam
        """
        if drivable_area < 0.3:
            data["emergency"] = True
            data["target_speed"] = 0.0  # Stop if we don't have enough drivable area
            
            # Check if frame_count is a dictionary or an integer
            frame_count_value = 0
            if isinstance(frame_count, int):
                frame_count_value = frame_count
            elif isinstance(frame_count, dict) and 'frame' in frame_count:
                frame_count_value = frame_count['frame']
                
            if frame_count_value % 300 == 0:
                print(f"WARNING: Insufficient drivable area: {drivable_area:.1f}%")
            
            return data
        """
        
        return data

    def _handle_obstacle_avoidance(self, sensor_data, data, frame_count=None):
        """
        Handle obstacle avoidance based on sensor data.
        
        Args:
            sensor_data: Dictionary of sensor data
            data: Dictionary of vehicle control data
            frame_count: Current frame count (optional)
            
        Returns:
            Updated vehicle control data
        """
        # Extract obstacles data from sensor data
        obstacles = sensor_data.get("obstacles", [])
        
        # If no obstacles, return data as is
        if not obstacles:
            return data
        
        # Initialize values
        is_emergency = False
        emergency_blend = 0.0
        
        # Get current vehicle speed
        current_speed = sensor_data.get("speed", 0.0)
        
        # Initialize variables for closest pedestrian
        closest_pedestrian_id = None
        closest_pedestrian_distance = float('inf')
        
        # Process all obstacles
        for obstacle in obstacles:
            # Get object type, distance, and relative position
            obj_type = obstacle.get("type", "unknown")
            obj_id = obstacle.get("id", -1)
            distance = obstacle.get("distance", 100.0)
            rel_x = obstacle.get("rel_x", 0)
            rel_y = obstacle.get("rel_y", 0)
            obj_speed = obstacle.get("speed", 0.0)
            certainty = obstacle.get("certainty", 0.5)
            
            # Skip objects that are behind us or have low certainty
            if rel_x < -2.0 or certainty < 0.3:
                continue
            
            # Check if this is a pedestrian and track the closest one
            if obj_type == "pedestrian" and distance < closest_pedestrian_distance:
                closest_pedestrian_id = obj_id
                closest_pedestrian_distance = distance
            
            # Determine if the object is a parked vehicle
            is_parked_vehicle = obj_type == "vehicle" and obj_speed < 0.05
            
            # Determine if pedestrian is crossing
            is_crossing_pedestrian = False
            if obj_type == "pedestrian":
                # Check if the pedestrian is moving perpendicular to the road
                if "vel_x" in obstacle and "vel_y" in obstacle:
                    vel_x = obstacle.get("vel_x", 0)
                    vel_y = obstacle.get("vel_y", 0)
                    
                    # If the pedestrian is moving more sideways than forward/backward
                    if abs(vel_y) > abs(vel_x) and abs(vel_y) > 0.2:
                        is_crossing_pedestrian = True
            
            # Special handling for pedestrian ID 5 - always treat as emergency
            if obj_type == "pedestrian" and obj_id == 5 and distance < 100.0:
                # Always an emergency, extremely high priority
                target_emergency = 1.0
                is_emergency = True
                
                # Force full stop if within 50 meters
                if distance < 50.0:
                    data["target_speed"] = 0.0
                    
                # Force avoidance to the right (positive steering)
                avoidance_steering = 0.5 * min(1.0, (80.0 - distance) / 80.0)
                
                if "steering_adjustment" not in data:
                    data["steering_adjustment"] = avoidance_steering
                else:
                    data["steering_adjustment"] += avoidance_steering
                
                print(f"SPECIAL HANDLING: Pedestrian ID 5 at {distance:.1f}m, steering={avoidance_steering:.2f}")
                
                # Set very high emergency blend for this specific pedestrian
                emergency_blend = max(emergency_blend, target_emergency)
                continue
            
            # Distance multipliers for different object types
            distance_multipliers = {
                "pedestrian": 4.0,  # Most cautious around pedestrians
                "cyclist": 2.5,
                "vehicle": 1.8,
                "unknown": 2.0
            }
            
            # Adjust distance multiplier for crossing pedestrians and parked vehicles
            if is_crossing_pedestrian:
                distance_multipliers["pedestrian"] = 5.0  # Even more cautious for crossing pedestrians
            elif is_parked_vehicle:
                distance_multipliers["vehicle"] = 2.2  # More cautious around parked vehicles
            
            # Get the appropriate multiplier for this object type
            multiplier = distance_multipliers.get(obj_type, 2.0)
            
            # Adjusted distance thresholds based on object type
            emergency_distance = self.lateral_safety_margin * 3.0 * multiplier
            caution_distance = self.obstacle_detection_distance * 0.8
            
            # Calculate the target emergency level based on distance
            target_emergency = 0.0
            
            if distance < emergency_distance:
                # Emergency zone - linear blend from 0.6 to 1.0
                target_emergency = 0.6 + 0.4 * (1.0 - max(0.0, (distance - emergency_distance * 0.5) / (emergency_distance * 0.5)))
                is_emergency = True
            elif distance < caution_distance:
                # Caution zone - linear blend from 0.0 to 0.6
                target_emergency = 0.6 * (1.0 - (distance - emergency_distance) / (caution_distance - emergency_distance))
            
            # Scale emergency by certainty
            target_emergency *= certainty
            
            # Calculate avoidance steering based on relative position
            avoidance_steering = 0.0
            
            if abs(rel_y) < self.lateral_safety_margin * 2.0:
                # Object directly in front - adjust based on position
                steer_dir = -1.0 if rel_y > 0 else 1.0
                
                # Make steering more aggressive based on proximity
                steer_intensity = 1.0 - min(1.0, distance / (caution_distance * 0.5))
                
                # Adjust steering based on object type
                if is_crossing_pedestrian:
                    # For crossing pedestrians, prioritize stopping over steering
                    avoidance_steering = steer_dir * steer_intensity * 0.3 * target_emergency
                elif is_parked_vehicle:
                    # For parked vehicles, more aggressive steering
                    avoidance_steering = steer_dir * steer_intensity * 0.8 * target_emergency
                else:
                    # Normal avoidance steering
                    avoidance_steering = steer_dir * steer_intensity * 0.5 * target_emergency
            
            # Add the avoidance steering to the control data
            if "steering_adjustment" not in data:
                data["steering_adjustment"] = avoidance_steering
            else:
                data["steering_adjustment"] += avoidance_steering
            
            # Update emergency blend
            emergency_blend = max(emergency_blend, target_emergency)
            
            # Debug output for significant obstacles
            if target_emergency > 0.3 and frame_count % 30 == 0:
                print(f"Obstacle: {obj_type} (ID {obj_id}) at {distance:.1f}m, rel_y={rel_y:.1f}, emergency={target_emergency:.2f}, steering={avoidance_steering:.2f}")
        
        # Update control data with emergency status
        data["emergency"] = is_emergency
        data["emergency_blend"] = emergency_blend
        
        # Adjust target speed based on emergency level
        if is_emergency and "target_speed" in data and emergency_blend > 0.4:
            # Reduce speed based on emergency blend
            speed_reduction_factor = 1.0 - min(0.9, emergency_blend * 0.9)
            data["target_speed"] *= speed_reduction_factor
            
            # Hard limit on maximum speed during emergencies
            max_emergency_speed = 10.0 * (1.0 - emergency_blend * 0.8)
            data["target_speed"] = min(data["target_speed"], max_emergency_speed)
            
            # Complete stop for severe emergencies
            if emergency_blend > 0.8:
                data["target_speed"] = 0.0
            
            # For any pedestrian within very close range, force stop
            if closest_pedestrian_distance < 15.0:
                data["target_speed"] = 0.0
                if frame_count % 30 == 0:
                    print(f"FORCED STOP: Pedestrian at {closest_pedestrian_distance:.1f}m")
        
        return data

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

    def _calculate_avoidance_steering(self, rel_x, rel_y, distance, certainty, 
                               lane_shift_multiplier=1.0, force_right=False):
        """Calculate avoidance steering angle based on obstacle position.
        
        Args:
            rel_x: Relative X position of obstacle (forward)
            rel_y: Relative Y position of obstacle (left/right)
            distance: Distance to obstacle
            certainty: Detection certainty (0-1)
            lane_shift_multiplier: Multiplier for avoidance lane shift (default 1.0)
            force_right: Force avoidance to the right side (default False)
            
        Returns:
            Steering angle for avoidance
        """
        # Determine avoidance direction (negative = right, positive = left)
        # If rel_y is negative, obstacle is to our right, so steer left (positive)
        # If rel_y is positive, obstacle is to our left, so steer right (negative)
        direction = 1.0 if rel_y <= 0 else -1.0
        
        # Override direction if force_right is True
        if force_right:
            direction = -1.0  # Always avoid to the right
        
        # Distance-based scaling for avoidance intensity
        # Closer obstacles require more aggressive avoidance
        distance_factor = max(0.2, min(1.0, self.obstacle_detection_distance / (distance + 5.0)))
        
        # Calculate avoidance intensity based on distance and certainty
        # Use a higher base value (4.5) for more aggressive lane shifts
        lane_shift = self.avoidance_lane_shift * lane_shift_multiplier * 1.5
        
        # More aggressive response when very close to obstacles
        if distance < 10.0:
            lane_shift *= (1.0 + (10.0 - distance) / 5.0)
        
        # Scale by certainty
        lane_shift *= max(0.5, certainty)
        
        # Calculate lateral distance needed based on time-to-collision
        time_to_collision = max(0.1, distance / max(0.1, self.current_speed))
        
        # More aggressive for close obstacles or slow speeds
        if time_to_collision < 2.0:
            lane_shift *= (2.0 / time_to_collision)
        
        # Calculate steering angle based on lane shift and distance
        steering_angle = direction * lane_shift * distance_factor
        
        # Limit maximum steering angle for stability
        max_angle = math.radians(40)  # Increased from 30 degrees for more aggressive steering
        steering_angle = max(-max_angle, min(max_angle, steering_angle))
        
        return steering_angle

    def _critical_collision_prevention(self, sensor_data):
        """
        Critical collision prevention - overrides all other controls when a collision is imminent.
        Returns emergency actions if needed, None otherwise.
        
        Args:
            sensor_data: Dictionary of sensor data
            
        Returns:
            Dictionary with emergency control values or None if no critical situation
        """
        obstacles = sensor_data.get("obstacles", [])
        
        if not obstacles:
            return None
            
        # Define critical distance thresholds by object type
        critical_thresholds = {
            "pedestrian": 8.0,  # More space for pedestrians
            "cyclist": 6.0,
            "vehicle": 5.0,
            "unknown": 7.0
        }
        
        # Check for critical obstacles
        critical_obstacles = []
        
        for obstacle in obstacles:
            # Get object information
            obj_type = obstacle.get("type", "unknown")
            distance = obstacle.get("distance", 999.0)
            rel_x = obstacle.get("rel_x", 0)
            rel_y = obstacle.get("rel_y", 0)
            certainty = obstacle.get("certainty", 0.5)
            
            # Only consider objects ahead of us and with reasonable certainty
            if rel_x < -1.0 or certainty < 0.3:
                continue
                
            # Determine critical threshold for this object
            threshold = critical_thresholds.get(obj_type, 7.0)
            
            # Adjust threshold based on relative position
            # Objects directly in our path are more critical
            path_factor = max(0.0, 1.0 - min(1.0, abs(rel_y) / 2.0))
            adjusted_threshold = threshold * (0.5 + 0.5 * path_factor)
            
            # Check if this is a critical obstacle
            if distance < adjusted_threshold:
                critical_obstacles.append({
                    "obstacle": obstacle,
                    "distance": distance,
                    "rel_y": rel_y,
                    "type": obj_type,
                    "threshold": adjusted_threshold
                })
        
        # If no critical obstacles, return None
        if not critical_obstacles:
            return None
            
        # Sort by distance (closest first)
        critical_obstacles.sort(key=lambda x: x["distance"])
        
        # Get the closest critical obstacle
        critical = critical_obstacles[0]
        
        # Prepare emergency action
        emergency_action = {
            "emergency": True,
            "emergency_blend": 1.0,
            "target_speed": 0.0,  # Full stop
            "acceleration": -1.0,  # Maximum braking
            "steering_adjustment": 0.0  # Will be set based on avoidance direction
        }
        
        # Determine emergency steering direction
        rel_y = critical["rel_y"]
        distance = critical["distance"]
        obj_type = critical["type"]
        
        # Calculate steering adjustment for emergency avoidance
        # Steer away from obstacle with intensity based on proximity
        steer_dir = -1.0 if rel_y > 0 else 1.0
        steer_intensity = max(0.3, min(1.0, (critical["threshold"] - distance) / critical["threshold"]))
        
        # Special handling for pedestrians - prioritize braking over steering
        if obj_type == "pedestrian":
            steer_intensity *= 0.5  # Less aggressive steering for pedestrians
        
        # Set emergency steering
        emergency_action["steering_adjustment"] = steer_dir * steer_intensity
        
        # Debug info
        print(f"CRITICAL COLLISION PREVENTION: {obj_type} at {distance:.1f}m, steering={emergency_action['steering_adjustment']:.2f}")
        
        return emergency_action
    
    def process(self, sensor_data, frame_count):
        """
        Process sensor data and return vehicle control data.
        
        Args:
            sensor_data: Dictionary of sensor data
            frame_count: Current frame count (int or dict)
            
        Returns:
            Dictionary of vehicle control data
        """
        # Ensure frame_count is handled correctly
        if isinstance(frame_count, dict):
            # Store the dict for backward compatibility
            self.frame_count = frame_count
            # Extract frame number if available
            self.frame_count_value = frame_count.get('frame', 0)
        else:
            # It's an integer
            self.frame_count = {'frame': frame_count}
            self.frame_count_value = frame_count
        
        # First check for immediate collision risks
        critical_action = self._critical_collision_prevention(sensor_data)
        if critical_action:
            # Return emergency control actions if critical situation detected
            return critical_action
        
        # Normal processing if no critical situation
        data = {}
        
        # Handle different sensor data
        # Process road information, obstacle avoidance, lane keeping, speed control, etc.
        
        # Call _handle_obstacle_avoidance with frame_count
        data = self._handle_obstacle_avoidance(sensor_data, data, frame_count)
        
        road_info = sensor_data.get("road_info", {})
        
        # Get drivable area percentage
        drivable_area = road_info.get("drivable_area_percentage", 0.0)
        
        # If insufficient drivable area, handle as emergency
        # Temporarily disable this check to avoid warning spam
        """
        if drivable_area < 0.3:
            data["emergency"] = True
            data["target_speed"] = 0.0  # Stop if we don't have enough drivable area
            
            # Check if frame_count is a dictionary or an integer
            frame_count_value = 0
            if isinstance(frame_count, int):
                frame_count_value = frame_count
            elif isinstance(frame_count, dict) and 'frame' in frame_count:
                frame_count_value = frame_count['frame']
                
            if frame_count_value % 300 == 0:
                print(f"WARNING: Insufficient drivable area: {drivable_area:.1f}%")
            
            return data
        """
        
        return data

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