"""
Radar sensor for autonomous vehicle simulation.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from .base_sensor import BaseSensor


class RadarSensor(BaseSensor):
    """Radar sensor for autonomous vehicle perception."""
    
    def __init__(self, vehicle, range_m=150.0, fov_deg=120.0, 
                 update_freq_hz=20.0, num_points=100, low_quality=False):
        """Initialize the radar sensor.
        
        Args:
            vehicle: The vehicle this sensor is attached to
            range_m: Maximum detection range in meters
            fov_deg: Field of view in degrees
            update_freq_hz: Sensor update frequency in Hz
            num_points: Number of radar points to generate
            low_quality: Whether to use low-quality settings for better performance
        """
        super().__init__(vehicle, range_m, update_freq_hz)
        
        self.low_quality = low_quality
        
        # Apply quality reductions if in low quality mode
        if low_quality:
            # Reduce range and number of points
            range_m = range_m * 0.7
            num_points = num_points // 2
        
        self.fov_deg = fov_deg
        self.num_points = num_points
        
        # Initialize radar data
        self.points = []
        self.velocities = []
        
        # Radar position offset from vehicle center (front of vehicle)
        self.position_offset = np.array([2.0, 0.0, 0.5])  # Front center, slightly elevated
        self.orientation_offset = np.array([0.0, 0.0, 0.0])  # Forward
        
        # Make sure update_interval is set correctly
        self.update_frequency = update_freq_hz
        self.update_interval = 1.0 / update_freq_hz if update_freq_hz > 0 else 0.1
        
        # Time of last update
        self.last_update_time = 0
        
        # Skip ratio for update to further improve performance in low quality mode
        self.skip_counter = 0
        self.skip_ratio = 1 if low_quality else 1  # Don't skip updates for better detection
        
        # Noise parameters
        self.distance_noise_std = 0.5 if not low_quality else 0.1  # Reduce noise computation in low quality
        self.velocity_noise_std = 0.2 if not low_quality else 0.05
        
        # Store the current figure and axis for reuse
        self.fig = None
        self.ax = None
        
    def update(self, scene_data, current_time):
        """Update radar data with current scene data.
        
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
        
        # Get vehicle state
        vehicle_pos = np.array(self.vehicle['position'])
        vehicle_rot = np.array(self.vehicle['rotation'])
        vehicle_vel = np.array(self.vehicle.get('velocity', [0, 0, 0]))
        
        # Calculate sensor position and orientation in world frame
        sensor_pos, sensor_rot = self._get_sensor_world_transform()
        
        # Get objects from the scene
        objects = scene_data.get('objects', [])
        
        # New radar points will be stored here
        new_points = []
        new_velocities = []
        
        # Maximum object distance for radar
        max_obj_distance = self.range + (5.0 if not self.low_quality else 0.0)
        
        # Process objects to generate radar points
        for obj in objects:
            # Skip processing if it's the vehicle itself (avoid self-detection)
            if 'id' in obj and 'id' in self.vehicle and obj['id'] == self.vehicle['id']:
                continue
                
            # Get object position and dimensions
            obj_pos = np.array(obj['position'])
            
            # Compute distance to object center
            distance = np.linalg.norm(obj_pos - sensor_pos)
            
            # Skip if object is too far
            if distance > max_obj_distance:
                continue
            
            # Get relative velocity if available
            obj_vel = np.array(obj.get('velocity', [0, 0, 0]))
            rel_vel = obj_vel - vehicle_vel
            
            # Project relative velocity onto the radar-to-object direction
            dir_vector = (obj_pos - sensor_pos) / max(0.1, distance)
            radial_vel = np.dot(rel_vel, dir_vector)
            
            # Get object dimensions or use default
            obj_dims = np.array(obj.get('dimensions', [1, 1, 1]))
            
            # In low quality mode, simplify point generation for distant objects
            if self.low_quality and distance > self.range * 0.6:
                # Generate more points for better detection
                num_points = np.random.randint(1, 4)
            else:
                # Calculate number of radar points based on object size and distance
                obj_size = np.linalg.norm(obj_dims)
                angular_size = np.arctan2(obj_size, distance) * 180 / np.pi
                # More points for larger/closer objects
                num_points = max(1, int(angular_size * 0.8))
                
                # Limit the number of points in low quality mode
                if self.low_quality:
                    num_points = min(num_points, 6)
                else:
                    num_points = min(num_points, 18)
            
            # Generate points around the object
            for _ in range(num_points):
                # Random offset within object dimensions
                offset = np.random.uniform(-0.5, 0.5, 3) * obj_dims
                
                # Calculate point position
                point_pos = obj_pos + offset
                
                # Calculate distance from radar to this point
                point_distance = np.linalg.norm(point_pos - sensor_pos)
                
                # Skip if outside radar range
                if point_distance > self.range:
                    continue
                
                # Calculate angle to the point
                # Convert to radar local coordinates
                relative_point = point_pos - sensor_pos
                
                # Create rotation matrix from sensor rotation
                sensor_rotation_matrix = self._rotation_matrix_from_euler(sensor_rot)
                
                # Transform to radar local frame
                local_point = np.dot(sensor_rotation_matrix.T, relative_point)
                
                # Calculate azimuth and elevation
                azimuth = np.arctan2(local_point[1], local_point[0]) * 180 / np.pi
                elevation = np.arctan2(local_point[2], np.sqrt(local_point[0]**2 + local_point[1]**2)) * 180 / np.pi
                
                # Check if point is within radar FOV
                if abs(azimuth) <= self.fov_deg / 2 and abs(elevation) <= self.fov_deg / 4:
                    # Add noise to distance and velocity
                    if not self.low_quality:
                        noisy_distance = point_distance + np.random.normal(0, self.distance_noise_std)
                        noisy_radial_vel = radial_vel + np.random.normal(0, self.velocity_noise_std)
                    else:
                        # Less noise computation in low quality mode
                        noisy_distance = point_distance
                        noisy_radial_vel = radial_vel
                    
                    # Store radar point (distance, azimuth, elevation)
                    new_points.append([noisy_distance, azimuth, elevation])
                    new_velocities.append(noisy_radial_vel)
        
        # Update radar data
        self.points = new_points
        self.velocities = new_velocities
        
        # Update last update time
        self.last_update_time = current_time
        
        return True
    
    def get_data(self):
        """Get radar detection data.
        
        Returns:
            tuple: (points, velocities) where points is a list of [distance, azimuth, elevation]
                  and velocities is a list of radial velocities
        """
        return self.points, self.velocities
    
    def visualize(self, ax=None):
        """Visualize radar points.
        
        Args:
            ax: Matplotlib axis to plot on, or None to create a new one
            
        Returns:
            tuple: (fig, ax) the matplotlib figure and axis
        """
        try:
            # Skip visualization if not enough data has changed since last visualization
            if hasattr(self, '_last_visualization_time') and hasattr(self, 'last_update_time'):
                # Only update visualization if at least 0.5 seconds have passed since the last one
                if self.last_update_time - self._last_visualization_time < 0.5:
                    if self.fig is not None and self.ax is not None:
                        return self.fig, self.ax
                    else:
                        return None, None
            
            # Reuse existing figure if available
            if self.fig is None or self.ax is None:
                try:
                    if ax is None:
                        self.fig, self.ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
                        # Make it non-blocking
                        self.fig.canvas.draw_idle()
                        plt.ion()
                    else:
                        self.fig = ax.figure
                        self.ax = ax
                except Exception as e:
                    print(f"Error creating radar figure: {e}")
                    return None, None
            
            # Check if figure is still valid (not closed)
            if not plt.fignum_exists(self.fig.number if self.fig is not None else -1):
                try:
                    # Figure was closed, create a new one
                    self.fig, self.ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
                    self.fig.canvas.draw_idle()
                    plt.ion()
                except Exception as e:
                    print(f"Error recreating radar figure: {e}")
                    return None, None
            
            # Clear previous points
            if self.ax is not None:
                self.ax.clear()
            else:
                return None, None
            
            # Convert radar points to polar coordinates for visualization
            if self.points:
                # Extract distances and azimuths (convert to radians)
                distances = [p[0] for p in self.points]
                azimuths = [np.radians(p[1]) for p in self.points]
                
                # Get velocities for color
                velocities = self.velocities
                
                # Create colormap based on velocities
                if velocities:
                    # Normalize velocities for colormap
                    norm_vel = np.array(velocities)
                    vmin = min(-10, norm_vel.min())
                    vmax = max(10, norm_vel.max())
                    
                    # Plot scatter points - reduced point size in low quality mode
                    scatter = self.ax.scatter(azimuths, distances, 
                                             c=norm_vel, cmap='coolwarm', 
                                             s=20 if self.low_quality else 30, 
                                             alpha=0.8, vmin=vmin, vmax=vmax)
                    
                    # Add colorbar - in low quality mode, update less frequently
                    if not self.low_quality or not hasattr(self, 'cbar') or self.cbar is None:
                        if hasattr(self, 'cbar') and self.cbar is not None:
                            try:
                                self.cbar.remove()
                            except:
                                pass  # Ignore errors if colorbar can't be removed
                        self.cbar = self.fig.colorbar(scatter, ax=self.ax)
                        self.cbar.set_label('Radial Velocity (m/s)')
                else:
                    # Plot without velocity information
                    self.ax.scatter(azimuths, distances, c='blue', s=20 if self.low_quality else 30, alpha=0.8)
            
            # Set plot properties
            self.ax.set_theta_zero_location('N')  # 0 degrees at the top
            self.ax.set_theta_direction(-1)  # clockwise
            self.ax.set_rlabel_position(135)  # rotation of radial labels
            
            # Set radar range
            self.ax.set_ylim(0, self.range)
            
            # Add radar FOV lines
            left_fov = np.radians(-self.fov_deg / 2)
            right_fov = np.radians(self.fov_deg / 2)
            self.ax.plot([left_fov, left_fov], [0, self.range], 'r--', lw=2, alpha=0.7)
            self.ax.plot([right_fov, right_fov], [0, self.range], 'r--', lw=2, alpha=0.7)
            
            # Add grid and title
            self.ax.grid(True)
            self.ax.set_title('Radar View')
            
            # Redraw and update with a very brief pause
            try:
                self.fig.canvas.draw()
                plt.pause(0.001)
                
                # Update last visualization time
                self._last_visualization_time = self.last_update_time
            except Exception as e:
                print(f"Error updating radar visualization: {e}")
            
            return self.fig, self.ax
            
        except Exception as e:
            print(f"Radar visualization error: {e}")
            # Reset the figure and axis - will be recreated next time
            self.fig = None
            self.ax = None
            return None, None
    
    def cleanup(self):
        """Close any open figures to free resources."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
    
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