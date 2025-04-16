def _render_debug_info(self, steering_angle, fps, frame_counter):
    # Convert from radians to degrees for display
    steering_angle_deg = np.degrees(steering_angle)
    
    # Reduce debug information frequency - changed from 30 to 300 frames
    if frame_counter % 300 == 0:
        # Print steering debug info
        print("\nSTEERING DEBUG INFO:")
        print(f"Steering angle: {steering_angle_deg:.1f}°")
        
        if self.autonomous_logic:
            print(f"Position from lane center: {self.autonomous_logic.position_from_lane_center:.2f} m")
            print(f"Heading: {self.autonomous_logic.current_heading:.2f}°")
            print(f"Lane position: {self.autonomous_logic.lane_position} | "
                  f"Lane offset: {self.autonomous_logic.lane_offset:.2f}")
            print(f"Road direction at current pos: {self.autonomous_logic.road_direction:.2f}°")
        print("")
    
    # Update steering wheel display every frame (keep this frequent for responsive display)
    self._update_steering_wheel_display(steering_angle_deg)
    
    # Display FPS - keep this but simplify it
    fps_text = f"{fps:.1f} FPS"
    fps_surface = self.debug_font.render(fps_text, True, (255, 255, 255))
    fps_rect = fps_surface.get_rect(topleft=(10, 10))
    self.screen.blit(fps_surface, fps_rect)
 