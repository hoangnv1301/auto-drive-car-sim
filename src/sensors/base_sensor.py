"""
Base sensor class for autonomous vehicle sensor implementations.
"""

import numpy as np


class BaseSensor:
    """Base class for all vehicle sensors."""
    
    def __init__(self, vehicle, range_m=100.0, update_freq_hz=10):
        """Initialize the sensor.
        
        Args:
            vehicle: The vehicle this sensor is attached to
            range_m: Maximum detection range in meters
            update_freq_hz: Sensor update frequency in Hz
        """
        self.vehicle = vehicle
        self.range = range_m
        self.update_frequency = update_freq_hz
        self.last_update_time = 0
        self.data = None
        self.enabled = True
        self.position_offset = np.array([0.0, 0.0, 0.0])  # Local offset from vehicle center
        self.orientation_offset = np.array([0.0, 0.0, 0.0])  # Rotation offset in radians (roll, pitch, yaw)
        
    def update(self, scene_data, current_time):
        """Update sensor data if enough time has passed since last update.
        
        Args:
            scene_data: Current scene data
            current_time: Current simulation time
            
        Returns:
            bool: True if sensor was updated, False otherwise
        """
        if not self.enabled:
            return False
            
        # Check if it's time to update
        update_interval = 1.0 / self.update_frequency
        if current_time - self.last_update_time < update_interval:
            return False
            
        # Update sensor
        self.last_update_time = current_time
        self._update_impl(scene_data)
        return True
        
    def _update_impl(self, scene_data):
        """Implement actual sensor update logic in derived classes.
        
        Args:
            scene_data: Current scene data
        """
        raise NotImplementedError("Derived sensors must implement _update_impl")
        
    def get_sensor_position(self):
        """Get the global position of the sensor.
        
        Returns:
            numpy.ndarray: Global position of the sensor
        """
        # Get vehicle position and orientation
        vehicle_pos = np.array(self.vehicle['position'])
        vehicle_yaw = self.vehicle['rotation'][2]
        
        # Calculate the offset position in global coordinates
        cos_yaw = np.cos(vehicle_yaw)
        sin_yaw = np.sin(vehicle_yaw)
        
        # Rotate local offset to global coordinates
        global_offset_x = cos_yaw * self.position_offset[0] - sin_yaw * self.position_offset[1]
        global_offset_y = sin_yaw * self.position_offset[0] + cos_yaw * self.position_offset[1]
        
        # Add offset to vehicle position
        return vehicle_pos + np.array([global_offset_x, global_offset_y, self.position_offset[2]])
        
    def get_sensor_orientation(self):
        """Get the global orientation of the sensor.
        
        Returns:
            numpy.ndarray: Global orientation of the sensor (roll, pitch, yaw)
        """
        # Add vehicle orientation and sensor offset
        return np.array(self.vehicle['rotation']) + self.orientation_offset
    
    def set_position_offset(self, x, y, z):
        """Set the position offset relative to vehicle center.
        
        Args:
            x: Forward offset (positive is forward)
            y: Lateral offset (positive is right)
            z: Vertical offset (positive is up)
        """
        self.position_offset = np.array([x, y, z])
        
    def set_orientation_offset(self, roll, pitch, yaw):
        """Set the orientation offset relative to vehicle orientation.
        
        Args:
            roll: Roll offset in radians
            pitch: Pitch offset in radians
            yaw: Yaw offset in radians
        """
        self.orientation_offset = np.array([roll, pitch, yaw])
        
    def enable(self):
        """Enable the sensor."""
        self.enabled = True
        
    def disable(self):
        """Disable the sensor."""
        self.enabled = False 