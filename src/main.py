#!/usr/bin/env python3
import os
import sys
import argparse
import time

from simulation.environment import Environment
from visualization.renderer import Renderer

def parse_args():
    parser = argparse.ArgumentParser(description='Autonomous Driving Simulation')
    parser.add_argument('--scenario', type=str, default='basic_intersection',
                        help='Scenario to load (default: basic_intersection)')
    parser.add_argument('--resolution', type=str, default='1280x720',
                        help='Display resolution (default: 1280x720)')
    parser.add_argument('--fullscreen', action='store_true',
                        help='Run in fullscreen mode')
    return parser.parse_args()

def main():
    args = parse_args()
    width, height = map(int, args.resolution.split('x'))
    
    print("Initializing simulation environment...")
    
    # Initialize simulation environment
    env = Environment(scenario=args.scenario)
    
    print("Initializing visualization...")
    
    # Initialize visualization
    renderer = Renderer(width, height, fullscreen=args.fullscreen)
    
    print("Starting simulation loop...")
    
    # Main simulation loop
    try:
        running = True
        last_time = time.time()
        frame_count = 0
        
        while running:
            # Timing for FPS calculation
            current_time = time.time()
            delta_time = current_time - last_time
            
            if delta_time >= 1.0:
                fps = frame_count / delta_time
                print(f"FPS: {fps:.1f}, Objects: {len(env.objects)}")
                last_time = current_time
                frame_count = 0
            
            # Update simulation physics
            env.step()
            
            # Render the scene
            renderer.render(env.get_scene_data())
            
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
        
    print("Simulation ended.")

if __name__ == "__main__":
    main() 