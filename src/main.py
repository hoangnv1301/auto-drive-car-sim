#!/usr/bin/env python3
import os
import sys
import argparse
import time

from simulation.environment import Environment
from visualization.renderer import Renderer

# Simple non-blocking windows for sensor visualization
import cv2
import numpy as np
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='Autonomous Driving Simulation')
    parser.add_argument('--scenario', type=str, default='basic_intersection',
                        help='Scenario to load (default: basic_intersection)')
    parser.add_argument('--resolution', type=str, default='1280x720',
                        help='Display resolution (default: 1280x720)')
    parser.add_argument('--fullscreen', action='store_true',
                        help='Run in fullscreen mode')
    parser.add_argument('--show-camera', action='store_true',
                        help='Enable camera sensor')
    parser.add_argument('--show-lidar', action='store_true',
                        help='Enable LiDAR sensor')
    parser.add_argument('--show-radar', action='store_true',
                        help='Enable radar sensor')
    parser.add_argument('--show-fusion', action='store_true',
                        help='Enable sensor fusion')
    parser.add_argument('--render-sensors', action='store_true',
                        help='Enable sensor visualization rendering in separate windows')
    parser.add_argument('--separate-windows', action='store_true',
                        help='Use separate non-blocking windows for visualization (more stable)')
    parser.add_argument('--low-quality', action='store_true',
                        help='Run with lower quality settings for better performance')
    parser.add_argument('--disable-debug', action='store_true',
                        help='Disable debug logging for better performance')
    parser.add_argument('--adaptive-quality', action='store_true',
                        help='Dynamically adjust quality based on performance')
    return parser.parse_args()

def show_camera_window(camera_sensor):
    """Show camera image in a non-blocking window."""
    if camera_sensor and camera_sensor.enabled:
        try:
            image = camera_sensor.get_image()
            if image is not None and image.size > 0:
                cv2.imshow('Camera View', image)
                cv2.waitKey(1)
                return True
        except Exception as e:
            print(f"Camera window error: {e}")
    return False

def show_lidar_window(lidar_sensor):
    """Show a simple LiDAR view."""
    if lidar_sensor and lidar_sensor.enabled:
        try:
            points = lidar_sensor.get_data()
            if points is not None and len(points) > 0:
                # Create a simple 2D top-down view of the LiDAR points
                img_size = 800
                img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
                
                # Scale and center points
                scale = 5.0  # 1 meter = 5 pixels
                center_x, center_y = img_size // 2, img_size // 2
                
                for point in points:
                    x, y, z, intensity = point
                    
                    # Skip points that are too far or too high/low
                    if abs(x) > 80 or abs(y) > 80 or abs(z) > 5:
                        continue
                    
                    # Calculate pixel coordinates
                    px = int(center_x + y * scale)  # Y is left/right
                    py = int(center_y - x * scale)  # X is forward/backward
                    
                    # Check if within image bounds
                    if 0 <= px < img_size and 0 <= py < img_size:
                        # Color by height (blue to red)
                        normalized_z = min(1.0, max(0.0, (z + 2) / 4.0))
                        r = int(255 * normalized_z)
                        b = int(255 * (1 - normalized_z))
                        g = 0
                        
                        # Draw point
                        cv2.circle(img, (px, py), 1, (b, g, r), -1)
                
                # Draw origin
                cv2.circle(img, (center_x, center_y), 5, (0, 255, 0), -1)
                
                # Draw range indicators (10m, 20m, etc.)
                for r in range(10, 81, 10):
                    radius = int(r * scale)
                    cv2.circle(img, (center_x, center_y), radius, (50, 50, 50), 1)
                    # Add label
                    cv2.putText(img, f"{r}m", (center_x + radius, center_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
                
                # Show image
                cv2.imshow('LiDAR View (Top-Down)', img)
                cv2.waitKey(1)
                return True
        except Exception as e:
            print(f"LiDAR window error: {e}")
    return False

def show_radar_window(radar_sensor):
    """Show a simple radar view."""
    if radar_sensor and radar_sensor.enabled:
        try:
            radar_data = radar_sensor.get_data()
            if radar_data is not None:
                points, velocities = radar_data
                
                if len(points) > 0:
                    # Create a simple top-down radar view
                    img_size = 600
                    img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
                    
                    # Draw radar field of view
                    center_x, center_y = img_size // 2, img_size // 2
                    fov_deg = radar_sensor.fov_deg
                    max_range = radar_sensor.range
                    scale = img_size / (2 * max_range)
                    
                    # Draw range circles
                    for r in range(0, int(max_range) + 1, 20):
                        radius = int(r * scale)
                        cv2.circle(img, (center_x, center_y), radius, (50, 50, 50), 1)
                        # Add label
                        cv2.putText(img, f"{r}m", (center_x + radius, center_y), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
                    
                    # Draw FOV lines
                    left_angle = -fov_deg / 2
                    right_angle = fov_deg / 2
                    
                    # Calculate endpoints of FOV lines
                    left_x = center_x + int(max_range * scale * np.sin(np.radians(left_angle)))
                    left_y = center_y - int(max_range * scale * np.cos(np.radians(left_angle)))
                    right_x = center_x + int(max_range * scale * np.sin(np.radians(right_angle)))
                    right_y = center_y - int(max_range * scale * np.cos(np.radians(right_angle)))
                    
                    # Draw FOV lines
                    cv2.line(img, (center_x, center_y), (left_x, left_y), (0, 0, 255), 1)
                    cv2.line(img, (center_x, center_y), (right_x, right_y), (0, 0, 255), 1)
                    
                    # Draw radar points
                    for i, point in enumerate(points):
                        distance, azimuth, elevation = point
                        
                        # Calculate point position in image
                        angle_rad = np.radians(azimuth)
                        x = center_x + int(distance * scale * np.sin(angle_rad))
                        y = center_y - int(distance * scale * np.cos(angle_rad))
                        
                        if 0 <= x < img_size and 0 <= y < img_size:
                            # Color by velocity - Ensure i is within range of velocities
                            if i < len(velocities):
                                vel = velocities[i]
                                # Red for approaching, green for moving away
                                if vel < 0:  # Approaching
                                    color = (0, 0, 255)  # Red
                                else:  # Moving away
                                    color = (0, 255, 0)  # Green
                                
                                # Size by velocity magnitude
                                size = max(2, min(8, int(abs(vel) + 2)))
                            else:
                                color = (255, 255, 0)  # Yellow
                                size = 3
                            
                            # Draw the radar point
                            cv2.circle(img, (x, y), size, color, -1)
                    
                    # Show image
                    cv2.imshow('Radar View', img)
                    cv2.waitKey(1)
                    return True
        except Exception as e:
            print(f"Radar window error: {e}")
    return False

def main():
    args = parse_args()
    width, height = map(int, args.resolution.split('x'))
    
    print("Initializing simulation environment...")
    
    # Initialize simulation environment with performance settings
    env = Environment(
        scenario=args.scenario, 
        low_quality=args.low_quality,
        disable_debug=args.disable_debug
    )
    
    print("Initializing visualization...")
    
    # Initialize visualization
    renderer = Renderer(width, height, fullscreen=args.fullscreen)
    
    # Enable sensor visualization for the first autonomous vehicle
    auto_vehicle_id = None
    for obj in env.objects:
        if obj.get('autonomous', False):
            auto_vehicle_id = obj['id']
            env.set_sensor_visualization(
                auto_vehicle_id,
                camera=args.show_camera,
                lidar=args.show_lidar,
                radar=args.show_radar,
                fusion=args.show_fusion
            )
            break
            
    if auto_vehicle_id is not None:
        print(f"Sensor visualization enabled for vehicle {auto_vehicle_id}")
        print(f"Camera: {'ON' if args.show_camera else 'OFF'}, "
              f"LiDAR: {'ON' if args.show_lidar else 'OFF'}, "
              f"Radar: {'ON' if args.show_radar else 'OFF'}, "
              f"Fusion: {'ON' if args.show_fusion else 'OFF'}")
        print(f"Sensor rendering: {'ON' if args.render_sensors else 'OFF'}")
        print(f"Separate windows: {'ON' if args.separate_windows else 'OFF'}")
        if args.low_quality:
            print("Running in LOW QUALITY mode for better performance")
        if args.disable_debug:
            print("Debug logging disabled for better performance")
        if args.adaptive_quality:
            print("Adaptive quality enabled - will adjust based on performance")
    else:
        print("No autonomous vehicle found to enable sensor visualization")
            
    print("Starting simulation loop...")
    
    # Main simulation loop
    try:
        running = True
        last_time = time.time()
        frame_count = 0
        last_sensor_render_time = 0
        sensor_render_interval = 0.5  # Render sensor views 2 times per second (more conservative)
        
        # Get sensor managers once
        sensor_manager = None
        if auto_vehicle_id is not None and auto_vehicle_id in env.sensor_managers:
            sensor_manager = env.sensor_managers[auto_vehicle_id]
        
        # Performance monitoring variables for adaptive quality
        fps_history = []
        fps_window_size = 5
        current_quality_level = 1  # 0=ultra low, 1=low, 2=medium
        low_fps_threshold = 8.0
        high_fps_threshold = 15.0
        quality_change_cooldown = 0
        
        while running:
            # Timing for FPS calculation
            current_time = time.time()
            delta_time = current_time - last_time
            
            if delta_time >= 1.0:
                fps = frame_count / delta_time
                print(f"FPS: {fps:.1f}, Objects: {len(env.objects)}")
                
                # Adaptive quality adjustment
                if args.adaptive_quality and sensor_manager:
                    # Add to fps history and keep window size
                    fps_history.append(fps)
                    if len(fps_history) > fps_window_size:
                        fps_history.pop(0)
                    
                    # Only adjust if we have enough samples and not on cooldown
                    if len(fps_history) >= fps_window_size and quality_change_cooldown <= 0:
                        avg_fps = sum(fps_history) / len(fps_history)
                        
                        # Decrease quality if FPS is too low
                        if avg_fps < low_fps_threshold and current_quality_level > 0:
                            current_quality_level -= 1
                            quality_change_cooldown = 5  # Wait 5 seconds before next change
                            
                            # Apply quality changes
                            if current_quality_level == 0:
                                print("Switching to ULTRA LOW quality mode due to performance")
                                # Reduce update frequency drastically
                                for sensor_name in sensor_manager.sensors:
                                    if sensor_name in sensor_manager.sensors:  # Kiểm tra lại để đảm bảo sensor vẫn tồn tại
                                        sensor = sensor_manager.sensors[sensor_name]
                                        sensor.update_frequency = max(5.0, sensor.update_frequency / 2)
                                sensor_render_interval = 1.0  # Reduce to once per second
                            
                        # Increase quality if FPS is high enough
                        elif avg_fps > high_fps_threshold and current_quality_level < 1:
                            current_quality_level += 1
                            quality_change_cooldown = 5  # Wait 5 seconds before next change
                            
                            # Apply quality changes
                            print("Returning to LOW quality mode")
                            for sensor_name in sensor_manager.sensors:
                                if sensor_name in sensor_manager.sensors:  # Kiểm tra lại để đảm bảo sensor vẫn tồn tại
                                    sensor = sensor_manager.sensors[sensor_name]
                                    # Restore default update frequencies
                                    if sensor_name == 'camera':
                                        sensor.update_frequency = 20.0
                                    elif sensor_name == 'lidar':
                                        sensor.update_frequency = 10.0
                                    elif sensor_name == 'radar':
                                        sensor.update_frequency = 25.0
                            sensor_render_interval = 0.5  # Back to normal
                    
                    # Decrease cooldown timer
                    if quality_change_cooldown > 0:
                        quality_change_cooldown -= 1
                
                last_time = current_time
                frame_count = 0
            
            # Update simulation physics
            env.step()
            
            # Render the scene
            renderer.render(env.get_scene_data())
            
            # Render sensor visualizations on a lower frequency if enabled
            if args.separate_windows and sensor_manager and current_time - last_sensor_render_time >= sensor_render_interval:
                try:
                    # Show simple sensor visualizations in separate windows
                    if args.show_camera:
                        show_camera_window(sensor_manager.get_camera())
                    
                    if args.show_lidar:
                        show_lidar_window(sensor_manager.get_lidar())
                    
                    if args.show_radar:
                        show_radar_window(sensor_manager.get_radar())
                    
                    last_sensor_render_time = current_time
                except Exception as e:
                    print(f"Sensor visualization error: {e}")
            # Use regular visualization approach if separate windows not enabled
            elif args.render_sensors and not args.separate_windows and sensor_manager and current_time - last_sensor_render_time >= sensor_render_interval:
                try:
                    # Use original visualization methods
                    sensor_manager.visualize()
                    last_sensor_render_time = current_time
                except Exception as e:
                    print(f"Sensor visualization error: {e}")
            
            # Check for exit condition
            if renderer.should_quit():
                running = False
                
            frame_count += 1
            
            # Cap update rate
            time.sleep(0.016)  # ~60 FPS target
            
    except KeyboardInterrupt:
        print("\nExiting simulation...")
    except Exception as e:
        print(f"Error in simulation: {e}")
    finally:
        # Clean up resources
        print("Cleaning up...")
        renderer.cleanup()
        env.cleanup()
        # Close all OpenCV windows
        cv2.destroyAllWindows()
        
    print("Simulation ended.")

if __name__ == "__main__":
    main() 