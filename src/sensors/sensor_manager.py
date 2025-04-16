"""
Sensor manager for handling all sensors in the autonomous vehicle.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from .camera import CameraSensor
from .lidar import LidarSensor
from .radar import RadarSensor
from .sensor_fusion import SensorFusion


class SensorManager:
    """Manager for all sensors in the autonomous vehicle."""
    
    def __init__(self, vehicle, camera_enabled=False, lidar_enabled=False, 
                 radar_enabled=False, sensor_fusion_enabled=False, low_quality=False):
        """Initialize the sensor manager.
        
        Args:
            vehicle: The vehicle this sensor manager is attached to
            camera_enabled: Whether camera sensor is enabled
            lidar_enabled: Whether lidar sensor is enabled
            radar_enabled: Whether radar sensor is enabled
            sensor_fusion_enabled: Whether sensor fusion is enabled
            low_quality: Whether to use low-quality settings for better performance
        """
        self.vehicle = vehicle
        self.low_quality = low_quality
        
        # Store visualization flags
        self.camera_vis_enabled = camera_enabled
        self.lidar_vis_enabled = lidar_enabled
        self.radar_vis_enabled = radar_enabled
        self.sensor_fusion_vis_enabled = sensor_fusion_enabled
        
        # Initialize sensors
        self.sensors = {}
        self.setup_default_sensors()
        
        # Initialize sensor fusion if enabled
        if sensor_fusion_enabled:
            self.setup_sensor_fusion()
        else:
            self.sensor_fusion = None
            
        # Initialize visualization figures
        self.figures = {}
    
    def setup_default_sensors(self):
        """Set up default sensors for the vehicle."""
        # Camera parameters
        camera_width = 640 if not self.low_quality else 320
        camera_height = 480 if not self.low_quality else 240
        
        # Initialize camera sensor (always create, even if visualization is disabled)
        self.sensors['camera'] = CameraSensor(
            vehicle=self.vehicle,
            width=camera_width,
            height=camera_height,
            range_m=150.0,
            update_freq_hz=20.0,
            low_quality=self.low_quality
        )
        self.sensors['camera'].enabled = self.camera_vis_enabled
        
        # Initialize lidar sensor (always create, even if visualization is disabled)
        self.sensors['lidar'] = LidarSensor(
            vehicle=self.vehicle,
            range_m=150.0,
            fov_h=360.0,
            update_freq_hz=20.0,
            num_layers=16 if not self.low_quality else 8,
            points_per_layer=2048 if not self.low_quality else 1024,
            low_quality=self.low_quality
        )
        self.sensors['lidar'].enabled = self.lidar_vis_enabled
        
        # Initialize radar sensor (always create, even if visualization is disabled)
        self.sensors['radar'] = RadarSensor(
            vehicle=self.vehicle,
            range_m=150.0,
            fov_deg=120.0,
            update_freq_hz=25.0,
            num_points=150 if not self.low_quality else 80,
            low_quality=self.low_quality
        )
        self.sensors['radar'].enabled = self.radar_vis_enabled
    
    def setup_sensor_fusion(self):
        """Set up sensor fusion."""
        self.sensor_fusion = SensorFusion(self)
        self.sensor_fusion.enabled = self.sensor_fusion_vis_enabled
    
    def add_sensor(self, name, sensor):
        """Add a sensor to the manager.
        
        Args:
            name: Name of the sensor
            sensor: The sensor object
        """
        self.sensors[name] = sensor
    
    def get_sensor(self, name):
        """Get a sensor by name.
        
        Args:
            name: Name of the sensor
            
        Returns:
            The sensor object, or None if not found
        """
        return self.sensors.get(name)
    
    def get_camera(self):
        """Get the camera sensor.
        
        Returns:
            The camera sensor, or None if not found
        """
        return self.sensors.get('camera')
    
    def get_lidar(self):
        """Get the lidar sensor.
        
        Returns:
            The lidar sensor, or None if not found
        """
        return self.sensors.get('lidar')
    
    def get_radar(self):
        """Get the radar sensor.
        
        Returns:
            The radar sensor, or None if not found
        """
        return self.sensors.get('radar')
    
    def get_sensor_data(self):
        """Get data from all sensors.
        
        Returns:
            dict: Dictionary of sensor data
        """
        sensor_data = {}
        
        for name, sensor in self.sensors.items():
            if sensor.enabled:
                sensor_data[name] = sensor.get_data()
        
        return sensor_data
    
    def update(self, scene_data, current_time):
        """Update all sensors with current scene data.
        
        Args:
            scene_data: Current scene data
            current_time: Current simulation time
            
        Returns:
            dict: Dictionary of updated sensor data
        """
        sensor_data = {}
        
        # Update all sensors
        for name, sensor in self.sensors.items():
            updated = sensor.update(scene_data, current_time)
            if updated:
                sensor_data[name] = sensor.get_data()
        
        # Update sensor fusion if enabled
        if self.sensor_fusion and self.sensor_fusion.enabled:
            self.sensor_fusion.update(scene_data, current_time)
            
        return sensor_data
    
    def visualize(self):
        """Visualize all enabled sensors.
        
        Returns:
            list: List of (fig, ax) tuples for all visualizations
        """
        visualizations = []
        
        # Visualize camera if enabled
        if self.camera_vis_enabled and 'camera' in self.sensors:
            camera = self.sensors['camera']
            if hasattr(camera, 'visualize'):
                fig_camera, ax_camera = camera.visualize()
                if fig_camera and ax_camera:
                    visualizations.append((fig_camera, ax_camera))
                    self.figures['camera'] = fig_camera
        
        # Visualize lidar if enabled
        if self.lidar_vis_enabled and 'lidar' in self.sensors:
            lidar = self.sensors['lidar']
            if hasattr(lidar, 'visualize'):
                fig_lidar, ax_lidar = lidar.visualize()
                if fig_lidar and ax_lidar:
                    visualizations.append((fig_lidar, ax_lidar))
                    self.figures['lidar'] = fig_lidar
        
        # Visualize radar if enabled
        if self.radar_vis_enabled and 'radar' in self.sensors:
            radar = self.sensors['radar']
            if hasattr(radar, 'visualize'):
                fig_radar, ax_radar = radar.visualize()
                if fig_radar and ax_radar:
                    visualizations.append((fig_radar, ax_radar))
                    self.figures['radar'] = fig_radar
                    
        # Visualize sensor fusion if enabled
        if self.sensor_fusion_vis_enabled and self.sensor_fusion:
            if hasattr(self.sensor_fusion, 'visualize'):
                fig_fusion, ax_fusion = self.sensor_fusion.visualize()
                if fig_fusion and ax_fusion:
                    visualizations.append((fig_fusion, ax_fusion))
                    self.figures['sensor_fusion'] = fig_fusion
        
        return visualizations
    
    def close_visualizations(self):
        """Close all visualization figures."""
        for name, fig in self.figures.items():
            try:
                plt.close(fig)
            except:
                pass  # Ignore errors
        
        self.figures = {}

    def set_sensor_enabled(self, sensor_name, enabled):
        """Enable or disable a sensor.
        
        Args:
            sensor_name: Name of the sensor
            enabled: Whether the sensor should be enabled
        """
        if sensor_name in self.sensors:
            self.sensors[sensor_name].enabled = enabled
            
            # Update visualization flags
            if sensor_name == 'camera':
                self.camera_vis_enabled = enabled
            elif sensor_name == 'lidar':
                self.lidar_vis_enabled = enabled
            elif sensor_name == 'radar':
                self.radar_vis_enabled = enabled
        
        # Update sensor fusion if the sensor name is sensor_fusion
        if sensor_name == 'sensor_fusion' and self.sensor_fusion:
            self.sensor_fusion.enabled = enabled
            self.sensor_fusion_vis_enabled = enabled
            
    def set_visualization(self, camera=None, lidar=None, radar=None, fusion=None):
        """Set visualization flags for vehicle sensors.
        
        Args:
            camera: Enable/disable camera visualization
            lidar: Enable/disable LiDAR visualization
            radar: Enable/disable radar visualization
            fusion: Enable/disable sensor fusion visualization
        """
        if camera is not None:
            self.set_sensor_enabled('camera', camera)
            self.camera_vis_enabled = camera
            
        if lidar is not None:
            self.set_sensor_enabled('lidar', lidar)
            self.lidar_vis_enabled = lidar
            
        if radar is not None:
            self.set_sensor_enabled('radar', radar)
            self.radar_vis_enabled = radar
            
        if fusion is not None and self.sensor_fusion:
            self.sensor_fusion.enabled = fusion
            self.sensor_fusion_vis_enabled = fusion 