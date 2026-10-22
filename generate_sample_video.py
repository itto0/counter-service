"""Generate a sample conveyor video with moving chicken silhouettes for testing."""

import argparse
from pathlib import Path
import cv2
import numpy as np


def generate_sample_video(
    output_path: str = "sample_conveyor.mp4",
    duration_sec: float = 10.0,
    fps: int = 30,
    width: int = 640,
    height: int = 480,
    num_chickens: int = 10,
) -> str:
    """Generate an MP4 video of a moving poultry conveyor with distinct chicken silhouettes."""
    total_frames = int(duration_sec * fps)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        # Fallback to AVI if mp4v is not supported on this platform
        alt_path = str(Path(output_path).with_suffix(".avi"))
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(alt_path, fourcc, fps, (width, height))
        output_path = alt_path

    # Compute spaced entry points for chickens
    chicken_spacing = total_frames / num_chickens
    speed = 4.0  # pixels per frame downward

    print(f"Generating '{output_path}' ({width}x{height}, {fps} fps, {duration_sec}s, {num_chickens} chickens)...")

    for frame_idx in range(total_frames):
        # Conveyor background: dark grey
        frame = np.full((height, width, 3), (35, 35, 35), dtype=np.uint8)

        # Side guard rails
        rail_width = 40
        frame[:, :rail_width] = (60, 60, 60)
        frame[:, width - rail_width :] = (60, 60, 60)
        cv2.line(frame, (rail_width, 0), (rail_width, height), (90, 90, 90), 2)
        cv2.line(frame, (width - rail_width, 0), (width - rail_width, height), (90, 90, 90), 2)

        # Moving belt rib textures
        rib_spacing = 40
        rib_offset = int((frame_idx * speed) % rib_spacing)
        for y in range(-rib_spacing + rib_offset, height + rib_spacing, rib_spacing):
            if 0 <= y < height:
                cv2.line(frame, (rail_width, y), (width - rail_width, y), (48, 48, 48), 1)

        # Draw each chicken
        for i in range(num_chickens):
            start_frame = i * chicken_spacing
            if frame_idx >= start_frame:
                dist_traveled = (frame_idx - start_frame) * speed
                cy = int(dist_traveled - 40)
                # Alternate slightly across conveyor belt center
                lane_offset = ((i % 3) - 1) * 35
                cx = (width // 2) + lane_offset

                if -50 <= cy <= height + 50:
                    # Chicken body (white/bright oval)
                    cv2.ellipse(frame, (cx, cy), (28, 38), 0, 0, 360, (230, 230, 230), -1)
                    # Chicken head
                    cv2.circle(frame, (cx, cy + 28), 14, (240, 240, 240), -1)
                    # Beak (yellow/orange)
                    beak_pts = np.array([[cx - 4, cy + 42], [cx + 4, cy + 42], [cx, cy + 49]], np.int32)
                    cv2.fillPoly(frame, [beak_pts], (0, 180, 255))
                    # Eye (small black dot)
                    cv2.circle(frame, (cx - 5, cy + 30), 2, (10, 10, 10), -1)
                    cv2.circle(frame, (cx + 5, cy + 30), 2, (10, 10, 10), -1)

        out.write(frame)

    out.release()
    file_size_kb = Path(output_path).stat().st_size / 1024
    print(f"Generated sample video successfully: {output_path} ({file_size_kb:.1f} KB)")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate sample conveyor video with chickens for testing.")
    parser.add_argument("--output", "-o", default="sample_conveyor.mp4", help="Output file path (default: sample_conveyor.mp4)")
    parser.add_argument("--duration", "-d", type=float, default=10.0, help="Duration in seconds (default: 10.0)")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second (default: 30)")
    parser.add_argument("--width", type=int, default=640, help="Video width (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Video height (default: 480)")
    parser.add_argument("--chickens", "-c", type=int, default=10, help="Number of chickens passing through (default: 10)")

    args = parser.parse_args()
    generate_sample_video(
        output_path=args.output,
        duration_sec=args.duration,
        fps=args.fps,
        width=args.width,
        height=args.height,
        num_chickens=args.chickens,
    )


if __name__ == "__main__":
    main()
