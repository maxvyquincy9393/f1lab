# -*- coding: utf-8 -*-
"""
arcade_replay_window.py
~~~~~~~~~~~~~~~~~~~~~~~
Standalone Arcade window for smooth F1 race replay.

Launch from command line:
    python arcade_replay_window.py --year 2025 --race "Australian Grand Prix" --session R

:copyright: (c) 2025 F1 Analytics
:license: MIT
"""

import os
import sys
import argparse
import arcade
import numpy as np
from pathlib import Path
import logging

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from race_replay_data import (
    get_race_replay_frames,
    enable_cache,
    get_circuit_rotation,
    format_race_time
)
import fastf1

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Window settings
SCREEN_WIDTH = 1400
SCREEN_HEIGHT = 900
SCREEN_TITLE = "F1 Race Replay"

# Colors
BACKGROUND_COLOR = (14, 17, 23)
TRACK_COLOR = (60, 60, 60)
TRACK_EDGE_COLOR = (80, 80, 80)
TEXT_COLOR = arcade.color.WHITE

# Tyre compound colors
TYRE_COLORS = {
    1: (255, 51, 51),    # SOFT - Red
    2: (255, 221, 0),    # MEDIUM - Yellow
    3: (238, 238, 238),  # HARD - White
    4: (57, 181, 74),    # INTER - Green
    5: (0, 170, 255),    # WET - Blue
}


class F1ReplayWindow(arcade.Window):
    """Main Arcade window for F1 race replay visualization."""
    
    def __init__(self, replay_data: dict, title: str = "F1 Race Replay"):
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, title, resizable=True)
        arcade.set_background_color(BACKGROUND_COLOR)
        
        self.replay_data = replay_data
        self.frames = replay_data['frames']
        self.track = replay_data['track']
        self.driver_colors = replay_data['driver_colors']
        self.total_laps = replay_data['total_laps']
        self.event_name = replay_data.get('event_name', 'Race')
        
        # Playback state
        self.current_frame_idx = 0
        self.is_playing = True
        self.playback_speed = 1.0
        self.frame_accumulator = 0.0
        
        # Display settings
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.selected_driver = None
        
        # Calculate track bounds for auto-scaling
        self._calculate_track_bounds()
        
        # Pre-compute track shape list
        self.track_shape_list = None
        self._create_track_shapes()
        
    def _calculate_track_bounds(self):
        """Calculate track coordinate bounds for proper scaling."""
        if self.track is None:
            return
            
        x = self.track['x']
        y = self.track['y']
        
        self.track_min_x = np.min(x)
        self.track_max_x = np.max(x)
        self.track_min_y = np.min(y)
        self.track_max_y = np.max(y)
        
        track_width = self.track_max_x - self.track_min_x
        track_height = self.track_max_y - self.track_min_y
        
        # Calculate scale to fit track in window with padding
        padding = 100
        scale_x = (SCREEN_WIDTH - padding * 2) / track_width
        scale_y = (SCREEN_HEIGHT - padding * 2 - 100) / track_height  # Extra for HUD
        
        self.scale = min(scale_x, scale_y) * 0.85
        
        # Center track
        self.offset_x = SCREEN_WIDTH / 2 - (self.track_min_x + track_width / 2) * self.scale
        self.offset_y = SCREEN_HEIGHT / 2 - (self.track_min_y + track_height / 2) * self.scale + 50
    
    def _create_track_shapes(self):
        """Pre-create track shapes for efficient rendering."""
        if self.track is None:
            return
            
        self.track_shape_list = arcade.ShapeElementList()
        
        x = self.track['x']
        y = self.track['y']
        
        # Transform coordinates
        transformed_points = []
        for i in range(len(x)):
            tx = x[i] * self.scale + self.offset_x
            ty = y[i] * self.scale + self.offset_y
            transformed_points.append((tx, ty))
        
        # Create track line
        if len(transformed_points) > 1:
            track_line = arcade.create_line_strip(
                transformed_points,
                TRACK_COLOR,
                line_width=20
            )
            self.track_shape_list.append(track_line)
            
            # Track edge
            edge_line = arcade.create_line_strip(
                transformed_points,
                TRACK_EDGE_COLOR,
                line_width=25
            )
            # Insert at beginning so it renders behind
            self.track_shape_list = arcade.ShapeElementList()
            self.track_shape_list.append(edge_line)
            self.track_shape_list.append(track_line)
    
    def _transform_coord(self, x, y):
        """Transform track coordinates to screen coordinates."""
        tx = x * self.scale + self.offset_x
        ty = y * self.scale + self.offset_y
        return tx, ty
    
    def _get_driver_color(self, driver_code: str) -> tuple:
        """Get RGB color for driver."""
        hex_color = self.driver_colors.get(driver_code, '#FFFFFF')
        if isinstance(hex_color, str):
            hex_color = hex_color.lstrip('#')
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return hex_color
    
    def on_draw(self):
        """Render the screen."""
        self.clear()
        
        # Draw track
        if self.track_shape_list:
            self.track_shape_list.draw()
        
        # Get current frame
        if not self.frames or self.current_frame_idx >= len(self.frames):
            return
            
        frame = self.frames[self.current_frame_idx]
        drivers_data = frame.get('drivers', {})
        
        # Draw drivers
        for code, data in drivers_data.items():
            x, y = self._transform_coord(data['x'], data['y'])
            color = self._get_driver_color(code)
            
            # Driver marker
            arcade.draw_circle_filled(x, y, 12, color)
            arcade.draw_circle_outline(x, y, 12, arcade.color.WHITE, 2)
            
            # Driver code label
            arcade.draw_text(
                code,
                x, y + 18,
                TEXT_COLOR,
                font_size=10,
                anchor_x="center",
                anchor_y="bottom",
                bold=True
            )
        
        # Draw HUD
        self._draw_hud(frame)
        
        # Draw leaderboard
        self._draw_leaderboard(drivers_data)
        
        # Draw selected driver telemetry
        if self.selected_driver and self.selected_driver in drivers_data:
            self._draw_telemetry(drivers_data[self.selected_driver])
    
    def _draw_hud(self, frame):
        """Draw heads-up display with race info."""
        leader_lap = frame.get('leader_lap', 0)
        race_time = frame.get('time', 0)
        
        # Title bar
        arcade.draw_rectangle_filled(
            SCREEN_WIDTH / 2, SCREEN_HEIGHT - 30,
            SCREEN_WIDTH, 60,
            (20, 20, 30)
        )
        
        # Event name
        arcade.draw_text(
            self.event_name,
            20, SCREEN_HEIGHT - 40,
            TEXT_COLOR,
            font_size=18,
            bold=True
        )
        
        # Lap counter
        arcade.draw_text(
            f"LAP {leader_lap}/{self.total_laps}",
            SCREEN_WIDTH / 2, SCREEN_HEIGHT - 40,
            TEXT_COLOR,
            font_size=16,
            anchor_x="center",
            bold=True
        )
        
        # Race time
        time_str = format_race_time(race_time)
        arcade.draw_text(
            time_str,
            SCREEN_WIDTH - 20, SCREEN_HEIGHT - 40,
            TEXT_COLOR,
            font_size=16,
            anchor_x="right"
        )
        
        # Playback info
        status = "▶ PLAYING" if self.is_playing else "⏸ PAUSED"
        arcade.draw_text(
            f"{status} | Speed: {self.playback_speed}x",
            20, 20,
            arcade.color.GRAY,
            font_size=12
        )
        
        # Controls hint
        arcade.draw_text(
            "SPACE: Play/Pause | ←→: Seek | ↑↓: Speed | 1-4: Quick Speed | Click driver to select",
            SCREEN_WIDTH / 2, 20,
            arcade.color.GRAY,
            font_size=10,
            anchor_x="center"
        )
    
    def _draw_leaderboard(self, drivers_data: dict):
        """Draw live leaderboard on right side."""
        # Sort by position
        sorted_drivers = sorted(
            drivers_data.items(),
            key=lambda x: x[1]['position']
        )[:10]  # Top 10
        
        # Background
        arcade.draw_rectangle_filled(
            SCREEN_WIDTH - 100, SCREEN_HEIGHT / 2 + 100,
            180, 350,
            (20, 20, 30, 200)
        )
        
        arcade.draw_text(
            "LEADERBOARD",
            SCREEN_WIDTH - 100, SCREEN_HEIGHT - 100,
            TEXT_COLOR,
            font_size=12,
            anchor_x="center",
            bold=True
        )
        
        y_pos = SCREEN_HEIGHT - 130
        for code, data in sorted_drivers:
            color = self._get_driver_color(code)
            pos = data['position']
            tyre = int(data['tyre'])
            tyre_color = TYRE_COLORS.get(tyre, (150, 150, 150))
            
            # Position
            arcade.draw_text(
                f"{pos}",
                SCREEN_WIDTH - 180, y_pos,
                TEXT_COLOR,
                font_size=11,
                bold=True
            )
            
            # Team color bar
            arcade.draw_rectangle_filled(
                SCREEN_WIDTH - 155, y_pos + 6,
                4, 14,
                color
            )
            
            # Driver code
            arcade.draw_text(
                code,
                SCREEN_WIDTH - 145, y_pos,
                TEXT_COLOR,
                font_size=11
            )
            
            # Tyre indicator
            arcade.draw_circle_filled(
                SCREEN_WIDTH - 40, y_pos + 6,
                6,
                tyre_color
            )
            
            y_pos -= 28
    
    def _draw_telemetry(self, driver_data: dict):
        """Draw telemetry for selected driver."""
        # Background
        arcade.draw_rectangle_filled(
            150, 150,
            280, 180,
            (20, 20, 30, 220)
        )
        
        code = self.selected_driver
        speed = driver_data.get('speed', 0)
        gear = driver_data.get('gear', 0)
        pos = driver_data.get('position', 0)
        lap = driver_data.get('lap', 0)
        
        color = self._get_driver_color(code)
        
        arcade.draw_text(
            f"TELEMETRY: {code}",
            150, 230,
            color,
            font_size=14,
            anchor_x="center",
            bold=True
        )
        
        # Speed
        arcade.draw_text(f"SPEED", 50, 190, arcade.color.GRAY, font_size=10)
        arcade.draw_text(f"{speed:.0f} km/h", 50, 170, TEXT_COLOR, font_size=16, bold=True)
        
        # Gear
        arcade.draw_text(f"GEAR", 150, 190, arcade.color.GRAY, font_size=10, anchor_x="center")
        arcade.draw_text(f"{gear}", 150, 160, color, font_size=32, anchor_x="center", bold=True)
        
        # Position
        arcade.draw_text(f"POS", 250, 190, arcade.color.GRAY, font_size=10, anchor_x="right")
        arcade.draw_text(f"P{pos}", 250, 170, TEXT_COLOR, font_size=16, anchor_x="right", bold=True)
        
        # Lap
        arcade.draw_text(f"LAP {lap}", 150, 90, arcade.color.GRAY, font_size=11, anchor_x="center")
    
    def on_update(self, delta_time: float):
        """Update animation state."""
        if not self.is_playing or not self.frames:
            return
        
        # Accumulate time for frame stepping
        self.frame_accumulator += delta_time * self.playback_speed * 25  # 25 FPS base
        
        # Step frames
        frames_to_advance = int(self.frame_accumulator)
        if frames_to_advance > 0:
            self.current_frame_idx = min(
                self.current_frame_idx + frames_to_advance,
                len(self.frames) - 1
            )
            self.frame_accumulator -= frames_to_advance
    
    def on_key_press(self, key, modifiers):
        """Handle keyboard input."""
        if key == arcade.key.SPACE:
            self.is_playing = not self.is_playing
        
        elif key == arcade.key.LEFT:
            # Rewind 5 seconds (125 frames at 25fps)
            self.current_frame_idx = max(0, self.current_frame_idx - 125)
        
        elif key == arcade.key.RIGHT:
            # Fast forward 5 seconds
            self.current_frame_idx = min(
                len(self.frames) - 1,
                self.current_frame_idx + 125
            )
        
        elif key == arcade.key.UP:
            # Increase speed
            speeds = [0.5, 1.0, 2.0, 4.0, 8.0]
            current_idx = speeds.index(self.playback_speed) if self.playback_speed in speeds else 1
            self.playback_speed = speeds[min(current_idx + 1, len(speeds) - 1)]
        
        elif key == arcade.key.DOWN:
            # Decrease speed
            speeds = [0.5, 1.0, 2.0, 4.0, 8.0]
            current_idx = speeds.index(self.playback_speed) if self.playback_speed in speeds else 1
            self.playback_speed = speeds[max(current_idx - 1, 0)]
        
        elif key == arcade.key.KEY_1:
            self.playback_speed = 0.5
        elif key == arcade.key.KEY_2:
            self.playback_speed = 1.0
        elif key == arcade.key.KEY_3:
            self.playback_speed = 2.0
        elif key == arcade.key.KEY_4:
            self.playback_speed = 4.0
        
        elif key == arcade.key.R:
            # Reset to beginning
            self.current_frame_idx = 0
        
        elif key == arcade.key.ESCAPE:
            arcade.close_window()
    
    def on_mouse_press(self, x, y, button, modifiers):
        """Handle mouse clicks to select driver."""
        if button == arcade.MOUSE_BUTTON_LEFT and self.frames:
            frame = self.frames[self.current_frame_idx]
            drivers_data = frame.get('drivers', {})
            
            # Check if clicked on a driver
            for code, data in drivers_data.items():
                dx, dy = self._transform_coord(data['x'], data['y'])
                distance = ((x - dx) ** 2 + (y - dy) ** 2) ** 0.5
                
                if distance < 20:
                    self.selected_driver = code
                    return
            
            # Clicked elsewhere - deselect
            self.selected_driver = None


def main():
    """Main entry point for standalone replay."""
    parser = argparse.ArgumentParser(description='F1 Race Replay - Desktop Viewer')
    parser.add_argument('--year', type=int, default=2025, help='Race year')
    parser.add_argument('--race', type=str, required=True, help='Race name')
    parser.add_argument('--session', type=str, default='R', help='Session type (R=Race, S=Sprint)')
    args = parser.parse_args()
    
    # Setup cache
    enable_cache()
    
    print(f"Loading {args.year} {args.race} ({args.session})...")
    print("This may take 1-2 minutes for first load...")
    
    try:
        # Load session
        session = fastf1.get_session(args.year, args.race, args.session)
        session.load(telemetry=True, weather=False)
        
        # Get replay frames
        replay_data = get_race_replay_frames(session, session_type=args.session)
        
        print(f"Loaded {len(replay_data['frames'])} frames")
        print("Launching replay window...")
        
        # Create and run window
        title = f"F1 {args.year} - {args.race}"
        window = F1ReplayWindow(replay_data, title)
        arcade.run()
        
    except Exception as e:
        print(f"Error: {e}")
        logger.exception("Failed to load replay")
        sys.exit(1)


if __name__ == "__main__":
    main()
