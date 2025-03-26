"""Recording module for CARLA simulator to capture the spectator's view."""

import platform
from pathlib import Path
from queue import Empty, Queue
import subprocess

import carla
import cv2
import numpy as np
from sympy import capture

from carla_icts2.config import logger

# Define codec preferences per platform
# If no codec works, you may need to install additional codecs for your platform
# For Fedora, visit:
# https://www.reddit.com/r/Fedora/comments/cr9wpu/multimedia_codecs_on_fedora/
# https://discussion.fedoraproject.org/t/cleanest-way-to-install-all-video-codecs-on-fedora-kde-40/134005
# https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html
CODEC_PREFERENCES = {
    "Linux": ["MJPG", "H264", "AVC1", "XVID", "X264", "DIVX", "WMV1", "WMV2"],
    "Windows": ["DIVX", "XVID"],  # More to be tested
    "Darwin": ["X264", "MJPG", "DIVX"],  # macOS
}


def get_best_codec() -> str | None:
    """Return the best available video codec based on the current platform."""
    system = platform.system()
    codecs_to_test = CODEC_PREFERENCES.get(system, [])

    if not codecs_to_test:
        logger.error(f"Unsupported platform ({system}): No suitable codec found.")
        return None

    # Test each codec in order of preference
    for codec in codecs_to_test:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        try:
            # Try to create a dummy VideoWriter to test the codec
            test_writer = cv2.VideoWriter("test.avi", fourcc, 1, (100, 100))
            if test_writer.isOpened():
                test_writer.release()
                logger.info(f"Working video codec: {codec}")
                return codec  # Return the first working codec
        except Exception as e:
            logger.warning(f"Failed to initialize codec {codec}: {e}")

    # Remove the test file if it was created
    if Path("test.avi").exists():
        Path("test.avi").unlink()

    # If no codec is found, log an error
    logger.error(
        "No suitable codec found for video writing. "
        f"Current CODEC_PREFERENCES: {CODEC_PREFERENCES}",
    )
    return None


def check_ffmpeg_installed() -> bool:
    """Check if ffmpeg is installed on the system."""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)  # noqa: S603, S607
    except subprocess.CalledProcessError:
        logger.error("ffmpeg is not installed or not found in the system path.")
        return False
    else:
        return True


def execute_ffmpeg_frames_to_video(
    frames_path: str | Path,
    output_path: str | Path,
    video_fps: int = 25,
) -> subprocess.CompletedProcess | None:
    """Convert frames to video using `ffmpeg`.

    Args:
        frames_path (str | Path): Path to the directory containing frames.
        output_path (str | Path): Path to save the output video.
        video_fps (int): Frames per second for the output video.

    Returns:
        (subprocess.CompletedProcess | None): Result of the ffmpeg command execution or None if
            ffmpeg is not installed.
    """
    if not check_ffmpeg_installed():
        logger.error("ffmpeg is not installed. Cannot create video.")
        return None

    # Construct ffmpeg command
    cmd_parts = [
        "ffmpeg",
        "-y",  # Overwrite output files without asking
        "-f image2 -pattern_type glob -i",
        "'" + str(frames_path) + "/*.png'",  # Input frames
        "-r",
        str(video_fps),  # Set FPS
        "-c:v",
        "libx264",  # Use H.264 codec
        "-preset slow",
        "-crf 10",  #  Lower CRF = better quality (0 = lossless, 10 = nearly lossless)
        "-pix_fmt yuv420p",  # Ensures compatibility
        "'" + str(output_path) + "'",  # Output file
        "> /dev/null 2>&1",  # Suppress output
    ]

    cmd = " ".join(cmd_parts)  # Join command list into a single string
    logger.info(f"Executing: {cmd}")

    # Execute command
    # Not checking safety of subprocess.run (S603) as this is a controlled environment
    result = subprocess.run([cmd], capture_output=True, check=True, shell=True)  # noqa: S602

    if result.returncode != 0:
        logger.error(f"Error creating video with ffmpeg: {result.stderr.decode()}")

    return result


def resize_video(
    input_path: str | Path,
    output_path: str | Path,
    target_width: int,
    target_height: int,
    fps: int | None = None,
) -> str | Path:
    """Resize a video to the specified dimensions.

    Args:
        input_path (str | Path): Path to the input video.
        output_path (str | Path): Path to save the resized video.
        target_width (int): Target width for the resized video.
        target_height (int): Target height for the resized video.
        fps (int | None): Frames per second for the resized video. If None, uses original FPS.

    Returns:
        (str | Path): Path to the resized video.
    """
    cap = cv2.VideoCapture(str(input_path))

    # Get original video properties
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps = int(cap.get(cv2.CAP_PROP_FPS))
    # frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Use original fps if not specified
    if fps is None:
        fps = original_fps

    logger.info(
        f"Resizing video: {original_width}x{original_height} -> {target_width}x{target_height}",
    )

    # Create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*get_best_codec())
    out = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))

    # Process frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame
        resized_frame = cv2.resize(frame, (target_width, target_height))
        out.write(resized_frame)

    # Release resources
    cap.release()
    out.release()

    logger.info(f"Resized video saved to {output_path}")
    return output_path


def stitch_videos_side_by_side(
    video_paths: list[Path],
    output_path: Path,
    fps: int = 25,
    width: int = 1920,
    height: int = 1080,
) -> None:
    """Stitch multiple videos side by side into one video.

    Args:
        video_paths (list[Path]): List of video file paths to stitch together.
        output_path (Path): Path to save the final stitched video.
        fps (int): Frames per second for the final video.
        width (int): Width that each video has to be resized to, before stitching.
        height (int): Height that each video has to be resized to, before stitching.
    """
    # Standard dimensions - use the width and height from the EEG videos
    standard_width = width
    standard_height = height

    # Process the videos to ensure they all have the same dimensions
    processed_videos = []
    for i, video_path in enumerate(video_paths):
        cap = cv2.VideoCapture(str(video_path))
        v_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        v_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # If dimensions don't match, resize the video
        if v_width != standard_width or v_height != standard_height:
            temp_path = f"temp_resized_video_{i}.mp4"
            processed_videos.append(
                resize_video(video_path, temp_path, standard_width, standard_height, fps),
            )
        else:
            processed_videos.append(video_path)

    # Open all processed videos
    videos = [cv2.VideoCapture(video) for video in processed_videos]

    # Use the shortest video length
    frame_counts = [int(video.get(cv2.CAP_PROP_FRAME_COUNT)) for video in videos]
    shortest_video_frames = min(frame_counts)

    logger.info(f"Using {shortest_video_frames} frames for the stitched video")

    # Calculate total width and use standard height
    total_width = standard_width * len(videos)

    # Create output video
    logger.info(f"Creating output video with dimensions: {total_width}x{standard_height}")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_out = cv2.VideoWriter(output_path, fourcc, fps, (total_width, standard_height))

    # Read and stitch frames
    for frame_idx in range(shortest_video_frames):
        frames = []
        for video in videos:
            ret, frame = video.read()
            if not ret:
                logger.error(f"Error reading frame {frame_idx} from a video")
                break
            frames.append(frame)

        if len(frames) == len(videos):
            # Concatenate frames horizontally
            stitched_frame = np.hstack(frames)

            # Write the stitched frame
            video_out.write(stitched_frame)

    video_out.release()

    # Release the video captures
    for video in videos:
        video.release()

    # Clean up temporary resized videos
    for path in processed_videos:
        if Path(path).stem.startswith("temp_resized_"):
            Path(path).unlink()
            logger.info(f"Removed temporary file: {path}")


def videos_in_folder(
    path: str | Path,
    formats: tuple[str, ...] = (".mp4", ".avi"),
    startswith: tuple[str, ...] = (),
) -> list[Path]:
    """List all video files in a given folder.

    Args:
        path (str | Path): Path to the folder containing video files.
        formats (tuple[str, ...]): Tuple of file extensions to consider as video files.
            Default is (".mp4", ".avi").
        startswith (tuple[str, ...]): Tuple of prefixes to filter video filenames.

    Returns:
        list[Path]: List of video file paths.
    """
    path = Path(path)
    if not path.is_dir():
        logger.error(f"{path} is not a valid directory.")
        return []

    video_files: list[Path] = []
    for ext in formats:
        if startswith:
            for prefix in startswith:
                video_files.extend(path.glob(f"{prefix}*{ext}"))
        else:
            video_files.extend(path.glob(f"*{ext}"))

    if not video_files:
        logger.warning(f"No video files found in {path}")

    return video_files


class SpectatorRecorder:
    """Records the spectator's view in CARLA and saves it as a video."""

    def __init__(  # noqa: PLR0913
        self,
        world: carla.World,
        config: dict,
        width: int = 1920,
        height: int = 1080,
        scenario: str | None = "unnamed",
        camera_type: str | None = "untyped",
    ) -> None:
        """Initialize the SpectatorRecorder.

        Args:
            world (carla.World): The CARLA world instance.
            config (dict): Configuration dictionary containing output path and FPS.
            width (int): Width of the video frames.
            height (int): Height of the video frames.
            scenario (str | None): Name of the scenario.
            camera_type (str | None): Type of the camera.
        """
        self.world = world
        self.config = config
        self.frames: list[np.ndarray] = []
        self.queue: Queue[tuple[int, np.ndarray]] = Queue()
        self.video_fps = config["fps"]
        self.output_path = self._build_output_path(scenario, camera_type)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Get dimensions from Config
        self.width = width
        self.height = height

        # Create a camera sensor blueprint
        self.camera_bp = self.world.get_blueprint_library().find("sensor.camera.rgb")
        self.camera_bp.set_attribute("image_size_x", str(self.width))
        self.camera_bp.set_attribute("image_size_y", str(self.height))
        self.camera_bp.set_attribute("fov", "105")  # Field of view in degrees

        # Get the spectator object
        self.spectator = self.world.get_spectator()

        # Get initial transform of spectator
        self.transform = self.spectator.get_transform()

        # Spawn the camera at the spectator's location
        self.camera = self.world.spawn_actor(self.camera_bp, self.transform)

        # Attach listener to capture frames (synchronized using a queue)
        self.camera.listen(lambda image: self._process_frame(image))

        # Save the frames to disk
        if self.config["save_frames"]:
            self.frames_path = self.output_path.parent / "frames"
            self.frames_path.mkdir(parents=True, exist_ok=True)

            # Remove the previous contents inside the folder
            self.__clean_frames(verbose=1)

        logger.info(f"Spectator recording started: {self.output_path}")

    def _build_output_path(self, scenario: str | None, camera_type: str | None) -> Path:
        """Build the output path for the video file.

        * If `output_path` is set to `/path/to/output`, `scenario_as_folder` is `True`,
        and `camera_in_filename` is `True`, the output path will be:
            `/path/to/output/scenario_name/camera_type_filename.mp4`
        * If `output_path` is set to `/path/to/output`, `scenario_as_folder` is `False`,
        and `camera_in_filename` is `False`, the output path will be:
            `/path/to/output/filename.mp4`

        Args:
            scenario (str): The name of the scenario.
            camera_type (str): The type of camera.

        Returns:
            Path: The constructed output path.
        """
        # Build output path
        output_path = Path(self.config["output_path"])
        if self.config["scenario_as_folder"] and scenario:
            output_path = output_path / scenario

        # Build filename
        filename = self.config["filename"] + "." + self.config["format"]
        if self.config["camera_in_filename"] and camera_type:
            filename = camera_type + "_" + filename

        return output_path / filename

    def _process_frame(self, image: carla.Image) -> None:
        """Process the camera frame and adds it to the queue for synchronization."""
        array = np.array(image.raw_data, dtype=np.uint8)
        array = array.reshape((self.height, self.width, 4))[:, :, :3]  # Convert to RGB

        # Save the image to disk if save_frames is enabled or use_ffmpeg is set
        if self.config["save_frames"] or self.config["use_ffmpeg"]:
            logger.debug(f"Writing frame {image.frame} to disk at {self.frames_path}")
            cv2.imwrite(
                str(self.frames_path / f"frame_{image.frame}.png"),
                array,
                [cv2.IMWRITE_PNG_COMPRESSION, 0],
            )

        self.queue.put((image.frame, array))  # Store frame in queue

    def __clean_frames(self, verbose: int = 0) -> None:
        """Remove all frames from the frames directory.

        If `verbose` is set to `> 0`, a warning will be logged if frames are found.
        """
        png_files = list(self.frames_path.glob("*.png"))
        for i, f in enumerate(png_files):
            if verbose > 0 and i == 0:
                logger.warning(
                    f"Cleaning up previously contained {len(png_files)} "
                    f"frames in {self.frames_path}",
                )
            f.unlink()

    def tick(self, world_frame: int, discard=True) -> None:
        """Synchronize recorded frames with simulation.

        Args:
            world_frame (int): The current simulation frame.
            discard (bool): If True, discard frames that don't match the simulation tick.
        """
        try:
            # Ensure the camera follows the spectator exactly
            # Keep rotation & position in sync
            self.camera.set_transform(self.spectator.get_transform())

            while not self.queue.empty():
                frame_id, frame_data = self.queue.get(True, timeout=1.0)
                self.frames.append(frame_data)
                if frame_id == world_frame and discard:  # Ensure frames match simulation tick
                    break  # Only process one frame per tick

        except Empty:
            logger.warning("Missing frame for spectator recording.")

    def save_video(self) -> None:
        """Save the recorded frames as an MP4 video."""
        if not self.frames:
            logger.warning("No frames recorded. Skipping video save.")
            return

        if self.config["use_ffmpeg"]:
            # Save frames to video using ffmpeg
            _ = execute_ffmpeg_frames_to_video(
                self.frames_path,
                self.output_path,
                self.video_fps,
            )

            # Delete the frames after conversion
            if not self.config["save_frames"]:
                self.__clean_frames()
        else:
            fourcc = cv2.VideoWriter_fourcc(*get_best_codec())
            video_writer = cv2.VideoWriter(
                self.output_path,
                fourcc,
                self.video_fps,
                (self.width, self.height),
            )

            # Increase quality by setting a higher bitrate
            quality_params = [
                cv2.IMWRITE_JPEG_QUALITY,
                100,  # Save raw images at max quality
                cv2.IMWRITE_PNG_COMPRESSION,
                0,  # PNG has lossless compression
            ]

            for frame in self.frames:
                # Convert to JPEG (optional, but helps with smoother encoding)
                _, buffer = cv2.imencode(".jpg", frame, quality_params)
                save_frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
                video_writer.write(save_frame)

            video_writer.release()

        logger.success(f"Spectator view video saved at {self.output_path}")

    def destroy(self) -> None:
        """Stop recording and releases resources."""
        # Make sure that the queue is empty before destroying the camera
        import time

        # TODO: Find a way to synchronize this without using sleep
        # This is a workaround to ensure all frames are processed
        time.sleep(20)
        self.tick(0, discard=False)
        logger.info(f"Final frame count: {len(self.frames)}.")
        self.camera.destroy()
        self.save_video()
