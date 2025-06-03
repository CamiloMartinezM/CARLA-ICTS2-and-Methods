"""Recording module for CARLA simulator to capture the spectator's view."""

import json
import math
import platform
import subprocess
import time  # Added for sleep
from pathlib import Path
from queue import Empty, Queue

import carla
import cv2
import numpy as np

from carla_icts2.benchmark.environment.world import World
from carla_icts2.config import logger

# Define codec preferences per platform
# If no codec works, you may need to install additional codecs for your platform
# For Fedora, visit:
# https://www.reddit.com/r/Fedora/comments/cr9wpu/multimedia_codecs_on_fedora/
# https://discussion.fedoraproject.org/t/cleanest-way-to-install-all-video-codecs-on-fedora-kde-40/134005
# https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html
CODEC_PREFERENCES = {
    "Linux": ["H264", "MJPG", "AVC1", "XVID", "X264", "DIVX", "WMV1", "WMV2"],
    "Windows": ["DIVX", "XVID"],  # More to be tested
    "Darwin": ["X264", "MJPG", "DIVX"],  # macOS
}


def get_best_codec(video_format: str = "mp4") -> str | None:
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
            test_writer = cv2.VideoWriter(f"test.{video_format}", fourcc, 1, (100, 100))
            if test_writer.isOpened():
                test_writer.release()
                logger.info(f"Working video codec: {codec}")
                return codec  # Return the first working codec
        except Exception as e:
            logger.warning(f"Failed to initialize codec {codec}: {e}")

    # Remove the test file if it was created
    if Path(f"test.{video_format}").exists():
        Path(f"test.{video_format}").unlink()

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
    skip_first_frames: int = 0,
) -> subprocess.CompletedProcess | None:
    """Convert frames to video using `ffmpeg`.

    Args:
        frames_path (str | Path): Path to the directory containing frames.
        output_path (str | Path): Path to save the output video.
        video_fps (int): Frames per second for the output video.
        skip_first_frames (int): Number of frames to skip at the beginning.
            Default is 0 (no frames skipped).

    Returns:
        (subprocess.CompletedProcess | None): Result of the ffmpeg command execution or None if
            ffmpeg is not installed.
    """
    if not check_ffmpeg_installed():
        logger.error("ffmpeg is not installed. Cannot create video.")
        return None

    if skip_first_frames > 0:
        # Sort the frames in the directory, then remove the first N frames
        frames = sorted(Path(frames_path).glob("*.png"), key=lambda x: x.name)
        for frame in frames[:skip_first_frames]:
            try:
                frame.unlink()  # Remove the file
            except Exception as e:  # noqa: PERF203
                logger.error(f"Error removing frame {frame}: {e}")
                continue

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
    *,
    use_ffmpeg: bool = False,
) -> None:
    """Stitch multiple videos side by side into one video.

    Args:
        video_paths (list[Path]): List of video file paths to stitch together.
        output_path (Path): Path to save the final stitched video.
        fps (int): Frames per second for the final video.
        width (int): Width that each video has to be resized to, before stitching.
        height (int): Height that each video has to be resized to, before stitching.
        use_ffmpeg (bool): Whether to use ffmpeg for stitching instead of OpenCV.
    """
    # Standard dimensions - use the width and height from the EEG videos
    standard_width = width
    standard_height = height

    # Process the videos to ensure they all have the same dimensions
    processed_videos = []
    for i, video_path in enumerate(video_paths):
        # Get video dimensions
        if use_ffmpeg:
            # Get dimensions using ffprobe
            cmd = [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "json",
                str(video_path),
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)
            v_width = int(data["streams"][0]["width"])
            v_height = int(data["streams"][0]["height"])
        else:
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

    # Calculate total width and use standard height
    total_width = standard_width * len(processed_videos)

    if use_ffmpeg:
        # Create a filter complex string to stitch videos side by side
        inputs = []
        filter_parts = []

        # Create input arguments for each video
        for i, video in enumerate(processed_videos):
            inputs.extend(["-i", str(video)])
            filter_parts.append(f"[{i}:v]")

        # Add the hstack filter
        filter_complex = f"{' '.join(filter_parts)}hstack=inputs={len(processed_videos)}[outv]"

        # Create the ffmpeg command
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output files without asking
            "-loglevel",
            "error",  # Minimize verbosity (or use "quiet" for complete silence)
        ]
        cmd.extend(inputs)
        cmd.extend(
            [
                "-filter_complex",
                filter_complex,
                "-map",
                "[outv]",
                "-r",
                str(fps),
                "-c:v",
                "libx264",  # Use H.264 codec
                "-preset",
                "medium",  # Medium quality/speed tradeoff
                "-crf",
                "23",  # Constant Rate Factor (quality)
                str(output_path),
            ]
        )

        # Execute the command
        logger.info("Running FFmpeg command to stitch videos")
        subprocess.run(cmd, check=True)  # noqa: S603

    else:
        # Use OpenCV method (original implementation)
        # Open all processed videos
        videos = [cv2.VideoCapture(str(video)) for video in processed_videos]

        # Use the shortest video length
        frame_counts = [int(video.get(cv2.CAP_PROP_FRAME_COUNT)) for video in videos]
        shortest_video_frames = min(frame_counts)

        logger.info(f"Using {shortest_video_frames} frames for the stitched video")

        # Create output video
        logger.info(f"Creating output video with dimensions: {total_width}x{standard_height}")
        fourcc = cv2.VideoWriter_fourcc(
            *get_best_codec(video_format="mp4")
        )  # TODO: Do not hardcode mp4
        video_out = cv2.VideoWriter(str(output_path), fourcc, fps, (total_width, standard_height))

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
        if isinstance(path, (str, Path)) and Path(path).stem.startswith("temp_resized_"):
            Path(path).unlink()
            logger.info(f"Removed temporary file: {path}")


def videos_in_folder(
    path: str | Path,
    formats: tuple[str, ...] = (".mp4", ".avi"),
    startswith: list[str] | tuple[str, ...] = (),
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

    def get_prefix_priority(file_path: Path) -> int:
        """Get the priority index based on startswith order."""
        filename = file_path.name
        for i, prefix in enumerate(startswith):
            if filename.startswith(prefix):
                return i
        return len(startswith)  # Files that don't match any prefix go to the end

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

    # Make sure that video_files has the list of files in the same order as startswith to allow
    # for "priority" lookup
    if startswith:
        video_files.sort(key=get_prefix_priority)

    return video_files


# Define standard camera configurations
CAMERA_CONFIGS = {
    "vehicle_pov": {
        # x (+) brings the camera to the right side of the vehicle
        # y (-) brings the camera forward
        "relative_transform": carla.Transform(carla.Location(x=-0.35, y=-0.7, z=1.25)),
        "attach_to": "vehicle",  # Special keyword for vehicle actor
        "attachment_type": carla.AttachmentType.Rigid,
        "fov": "75",
    },
    "pedestrian_pov": {
        "attach_to": "pedestrian",  # Special keyword for pedestrian actor
        "attachment_type": carla.AttachmentType.Rigid,
        "fov": "75",
    },
    "pedestrian_frontal": {
        # Positioned IN FRONT of the pedestrian, looking AT them.
        # x=2.0 means 2 meters in front of the pedestrian's local +X axis.
        # z=1.6 means 1.6 meters up from the pedestrian's origin.
        # Rotation(yaw=180.0) means the camera, initially facing its own +X,
        # is turned 180 degrees to face back towards the pedestrian's origin.
        "relative_transform": carla.Transform(
            carla.Location(x=2.5, y=0.0, z=1.6),
            carla.Rotation(yaw=180, pitch=-10),
        ),
        "attach_to": "pedestrian",
        "attachment_type": carla.AttachmentType.SpringArm,
        "fov": "95",  # Slightly narrower FOV might be good, ~70
        "distance_from_ped": 2.5,
        "height_offset": 1.6,
    },
    # TODO: Check if this works
    "bev_static": {
        # Note: Static BEV needs a world location. We'll calculate it based on actors later.
        "relative_transform": carla.Transform(carla.Location(z=50), carla.Rotation(pitch=-90)),
        "attachment_type": carla.AttachmentType.Rigid,  # Doesn't matter if not attached
    },
    "bev_follow_vehicle": {
        # High above, following vehicle, looking down
        "relative_transform": carla.Transform(
            carla.Location(z=8),
            carla.Rotation(yaw=90.0, roll=90, pitch=-110),
        ),
        "attach_to": "vehicle",
        "attachment_type": carla.AttachmentType.SpringArm,  # Smoother following
        "fov": "95",
    },
    "bev_follow_pedestrian": {
        # High above, following pedestrian, looking down
        "relative_transform": carla.Transform(
            carla.Location(z=8),
            carla.Rotation(yaw=180.0, pitch=-90),
        ),
        "attach_to": "pedestrian",
        "attachment_type": carla.AttachmentType.SpringArm,  # Smoother following
    },
    "spectator": {  # Replicates SpectatorRecorder
        "relative_transform": carla.Transform(),  # Will follow spectator exactly
        "attach_to": "spectator",
        "attachment_type": carla.AttachmentType.Rigid,
    },
}


class MultiCameraRecorder:
    """Records multiple camera views in CARLA and saves them as videos."""

    def __init__(
        self,
        world: World,
        config: dict,
        camera_views: list[str],
        width: int = 1920,
        height: int = 1080,
        scenario: str | None = "unnamed",
        *,
        debug: bool = False,
    ) -> None:
        self.world = world
        self.carla_world = self.world.world
        self.config = config
        self.camera_views = camera_views
        # Store frames in memory ONLY if not using ffmpeg and not just saving frames
        self._store_in_memory = not config.get("use_ffmpeg", False) and not config.get(
            "save_frames",
            False,
        )
        self.frames: dict[str, list[np.ndarray]] = (
            {view: [] for view in camera_views} if self._store_in_memory else {}
        )
        self.cameras: dict[str, carla.Actor] = {}
        # Using separate queues might simplify debugging/logic if one sensor lags
        # self.queues: dict[str, Queue[tuple[int, np.ndarray]]] = {view: Queue()
        #                                                          for view in camera_views}
        # Sticking to one queue for now for simplicity, but add view_name
        self.queue: Queue[tuple[int, str, np.ndarray]] = Queue()
        self.video_fps = config["fps"]
        self.output_path_base = self._build_output_base_path(scenario)
        self.output_path_base.mkdir(parents=True, exist_ok=True)
        self.width = width
        self.height = height
        self.debug = debug

        # --- Frame Saving Setup ---
        self.save_frames_enabled = self.config.get("save_frames", False)
        self.use_ffmpeg = self.config.get("use_ffmpeg", False)
        # We need to save frames temporarily if using ffmpeg
        self._intermediate_frame_saving = self.save_frames_enabled or self.use_ffmpeg
        self.frames_base_path = self.output_path_base / "frames"
        if self._intermediate_frame_saving:
            self.frames_base_path.mkdir(parents=True, exist_ok=True)
            self.__clean_frames(verbose=1)

        # --- Spawn Cameras ---
        self.spectator = self.carla_world.get_spectator()
        self.latest_frame_ids: dict[str, int] = {}  # Track last processed frame ID per camera

        for view_name in self.camera_views:
            if view_name not in CAMERA_CONFIGS:
                logger.warning(f"Camera view '{view_name}' not found in CAMERA_CONFIGS. Skipping.")
                continue

            view_config = CAMERA_CONFIGS[view_name]
            camera = self._spawn_camera(view_name, view_config)
            if camera:
                self.cameras[view_name] = camera
                # Use functools.partial to pass the view_name to the callback
                # Assign the callback using lambda with default argument capture
                camera.listen(lambda image, view=view_name: self._process_frame(image, view))

                if self._intermediate_frame_saving:
                    (self.frames_base_path / view_name).mkdir(exist_ok=True)
                logger.info(f"Camera '{view_name}' spawned and listening.")
                self.latest_frame_ids[view_name] = -1  # Initialize frame tracking
            else:
                logger.error(f"Failed to spawn camera for view '{view_name}'.")

        logger.info(f"Multi-camera recording initialized for views: {list(self.cameras.keys())}")
        # Add a small delay after spawning sensors
        time.sleep(5.0)

    def _build_output_base_path(self, scenario: str | None) -> Path:
        """Build the base output directory."""
        output_path = Path(self.config["output_path"])
        if self.config.get("scenario_as_folder", False) and scenario:
            output_path = output_path / scenario
        return output_path

    def _get_camera_transform(self, view_name: str, view_config: dict) -> carla.Transform:
        """Calculate the world transform for a camera."""
        if view_name == "spectator":
            return self.spectator.get_transform()

        # Attached to an actor (vehicle or pedestrian)
        if view_config["attach_to"] == "vehicle":
            ref_transform = self.world.player.get_transform()
        elif view_config["attach_to"] == "pedestrian":
            ref_transform = self.get_walker_pov_camera()

            # For other pedestrian-attached cameras like "pedestrian_frontal"
            if view_name == "pedestrian_frontal":
                pedestrian_transform = self.world.walker.get_transform()

                # TODO: It's better practice to pass these from run_config.yaml eventually
                distance_in_front = view_config["distance_from_ped"]
                height_offset = view_config["height_offset"]
                camera_yaw_offset = view_config["relative_transform"].rotation.yaw
                camera_pitch = view_config["relative_transform"].rotation.pitch

                # Calculate offset in pedestrian's local frame
                # Pedestrian's +X is forward. Camera needs to be at +X relative to ped.
                local_offset = carla.Location(x=distance_in_front, y=0, z=height_offset)

                # Transform this local offset to world coordinates based on pedestrian's transform
                camera_location_world = pedestrian_transform.transform(local_offset)

                # Camera rotation: pedestrian's yaw + 180 degrees to face them, plus pitch
                camera_rotation_world = carla.Rotation(
                    pitch=pedestrian_transform.rotation.pitch
                    + camera_pitch,  # Add pitch relative to ped's pitch
                    yaw=pedestrian_transform.rotation.yaw + camera_yaw_offset,
                    roll=pedestrian_transform.rotation.roll,
                )

                ref_transform = carla.Transform(
                    camera_location_world,
                    camera_rotation_world,
                )

        # For static cameras like BEV_static
        elif view_name.startswith("bev_"):
            ref_transform = self.get_bev_camera(
                follow_player=view_config["attach_to"] == "pedestrian",
                follow_walker=view_config["attach_to"] == "vehicle",
            )
        else:
            logger.error(f"View '{view_name}' not found")
            return carla.Transform()

        static_location = ref_transform.location
        rotation = ref_transform.rotation

        # Only add the relative transform if it's not the pedestrian POV
        if view_config["attach_to"] != "pedestrian":
            static_location += view_config["relative_transform"].location

        # Replace the rotation with the one given in CAMERA_CONFIGS
        if view_name.startswith("bev_"):
            static_location.z += view_config["relative_transform"].location.z
            rotation = view_config["relative_transform"].rotation

        return carla.Transform(static_location, rotation)

    def _get_head_world_transform(self) -> carla.Transform | None:
        """Retrieve the world transform of the pedestrian's head bone ('crl_Head__C').

        Returns:
            carla.Transform: The world transform of the head bone, or None if not found or bones
                cannot be retrieved.
        """
        try:
            bones_out = self.world.walker.get_bones()
            if not bones_out or not bones_out.bone_transforms:
                logger.warning(
                    f"Could not retrieve valid bones for pedestrian {self.world.walker.id}",
                )
                return None

            # Find the head bone transform in the list
            for bone_info in bones_out.bone_transforms:
                if bone_info.name == "crl_Head__C":
                    return bone_info.world  # Return the pre-computed world transform

            logger.warning(
                f"Head bone 'crl_Head__C' not found for pedestrian {self.world.walker.id}",
            )
        except Exception as e:
            logger.error(
                f"Error getting bones for pedestrian {self.world.walker.id}: {e}",
                exc_info=True,
            )
            return None
        else:
            return None

    def get_bev_camera(
        self,
        *,
        follow_player: bool = False,
        follow_walker: bool = False,
    ) -> carla.Transform:
        """Get a Bird's Eye View (BEV) camera.

        Args:
            follow_player (bool): If True, the camera will follow the vehicle. Default is False.
            follow_walker (bool): If True, the camera will follow the walker. Default is False.

        Returns:
            carla.Transform: The transform for the BEV camera.
        """
        if follow_player and follow_walker:
            logger.error(
                "Cannot follow both player and walker at the same time. "
                "Setting to follow vehicle only.",
            )
            follow_player = True

        if follow_walker:
            transform = self.world.walker.get_transform()

        if follow_player:
            transform = self.world.player.get_transform()

        # Using z=0.0 to replace it in _get_camera_transform using CAMERA_CONFIGS
        location = carla.Location(x=transform.location.x, y=transform.location.y, z=0)
        return carla.Transform(location, carla.Rotation(yaw=180.0, pitch=-90.0))

    def get_walker_pov_camera(self) -> carla.Transform:
        """Get a POV camera at the pedestrian's eye level."""
        # offset = carla.Location(x=0.0, y=0.0, z=1.7)
        eye_forward_offset = 0.4  # How far in front of the head bone origin
        eye_up_offset = 0.05  # How far above the head bone origin
        eye_right_offset = 0.0  # Sideways offset (usually 0)

        # --- Get Head Bone's World Transform ---
        head_transform = self._get_head_world_transform()

        if head_transform is None:
            # --- Fallback: Use the actor's base transform + fixed Z offset ---
            # This is less accurate but prevents errors if the head bone isn't found.
            logger.warning(
                f"Head bone 'crl_Head__C' not found for {self.world.walker.id}.",
            )
            pedestrian_transform = self.world.walker.get_transform()
            # Approx eye level from ground
            fallback_location = pedestrian_transform.location + carla.Location(z=1.7)
            final_transform = carla.Transform(fallback_location, pedestrian_transform.rotation)
        else:
            # --- Use Head Bone Transform for Position and Orientation Basis ---
            # 1. Calculate Final Camera ROTATION based on head's forward direction
            # Get the head bone's forward vector in world coordinates
            head_forward_vector = head_transform.get_forward_vector()

            # Calculate Yaw and Pitch from the forward vector
            # -90° makes the camera look forward instead of to the side
            yaw = math.degrees(math.atan2(head_forward_vector.y, head_forward_vector.x)) - 90
            # ************************************************************

            # Clamp asin argument for safety
            asin_arg = max(-1.0, min(1.0, head_forward_vector.z))
            pitch = math.degrees(math.asin(asin_arg)) - 15  # Make it look slightly down

            # Ensure camera is upright relative to the world
            final_rotation = carla.Rotation(pitch=pitch, yaw=yaw, roll=0.0)

            # Calculate Final Camera POSITION based on the calculated rotation and head origin
            head_origin = head_transform.location

            # Get the direction vectors *from the calculated final rotation*
            camera_forward = final_rotation.get_forward_vector()
            camera_up = final_rotation.get_up_vector()
            camera_right = final_rotation.get_right_vector()

            # Calculate the world offset vector based on the desired eye position relative to the
            # head origin
            world_offset = (
                camera_forward * eye_forward_offset
                + camera_up * eye_up_offset
                + camera_right * eye_right_offset
            )

            # Add the world offset to the head bone's origin
            final_location = head_origin + world_offset

            final_transform = carla.Transform(final_location, final_rotation)

        return final_transform

    def _spawn_camera(self, view_name: str, view_config: dict) -> carla.Actor | None:
        """Spawns a single camera based on its configuration."""
        try:
            camera_bp = self.carla_world.get_blueprint_library().find("sensor.camera.rgb")
            camera_bp.set_attribute("image_size_x", str(self.width))
            camera_bp.set_attribute("image_size_y", str(self.height))
            camera_bp.set_attribute("fov", view_config.get("fov", "105"))
            # Set sensor_tick *slightly* less than simulation step if possible,
            # or keep it 0 to sync with world ticks. Adjust if sync issues persist.
            # camera_bp.set_attribute('sensor_tick', str(0.0)) # Sync with world tick
            camera_bp.set_attribute(
                "sensor_tick",
                str(self.carla_world.get_settings().fixed_delta_seconds * 0.9),
            )

            transform = self._get_camera_transform(view_name, view_config)
            attach_target_name = view_config.get("attach_to")
            attachment_type = view_config.get("attachment_type", carla.AttachmentType.Rigid)

            attach_target_actor = None
            if attach_target_name == "spectator":
                pass  # Handled in tick
            elif attach_target_name == "vehicle":
                attach_target_actor = self.world.player
            elif attach_target_name == "pedestrian":
                attach_target_actor = self.world.walker
            elif attach_target_name is not None:
                logger.error(
                    f"Cannot attach camera '{view_name}': Actor '{attach_target_name}' not found.",
                )
                return None

            if attach_target_actor:
                camera = self.carla_world.try_spawn_actor(
                    camera_bp,
                    transform,
                    attach_to=attach_target_actor,
                    attachment_type=attachment_type,
                )
            else:
                camera = self.carla_world.try_spawn_actor(camera_bp, transform)

            if camera is None:
                logger.error(f"Failed to spawn camera actor {view_name}.")
                return None

        except Exception as e:
            logger.error(f"Error spawning camera {view_name}: {e}", exc_info=True)
            return None
        else:
            return camera

    def _process_frame(self, image: carla.Image, view_name: str) -> None:
        """Process frame and add to queue with view identifier."""
        # Check if the frame is newer than the last one processed for this camera
        if image.frame <= self.latest_frame_ids.get(view_name, -1):
            # logger.warning(f"Received old or duplicate frame {image.frame} for '{view_name}'.
            # Last was {self.latest_frame_ids.get(view_name)}. Skipping.")
            return  # Discard old/duplicate frames

        # Update the latest frame ID for this camera
        self.latest_frame_ids[view_name] = image.frame

        # Crucial: Create a copy of the raw data immediately.
        # image.raw_data might be a buffer that gets overwritten by the simulator.
        try:
            # Create a copy first
            data_copy = bytes(image.raw_data)
            # Then convert to numpy array from the copy
            array = np.frombuffer(data_copy, dtype=np.uint8)
            # Reshape and convert color format
            array = array.reshape((self.height, self.width, 4))[:, :, :3]
            # array = array[:, :, ::-1].copy()  # BGRA to BGR, ensure it's a new copy
        except Exception as e:
            logger.error(f"Error processing frame {image.frame} for '{view_name}': {e}")
            return

        # Save the image to disk if needed (for ffmpeg or explicit saving)
        if self._intermediate_frame_saving:
            frame_path = self.frames_base_path / view_name / f"frame_{image.frame:08d}.png"
            try:
                write_success = cv2.imwrite(
                    str(frame_path),
                    array,
                    [cv2.IMWRITE_PNG_COMPRESSION, 0],
                )
                if not write_success:
                    logger.error(f"cv2.imwrite failed for {frame_path}")
            except Exception as e:
                logger.error(f"Exception during cv2.imwrite for {frame_path}: {e}")

        # Add to queue ONLY if storing in memory (i.e., not using ffmpeg for final video)
        if self._store_in_memory:
            self.queue.put((image.frame, view_name, array))  # array is already a copy
            if self.debug:
                logger.debug(f"Put frame {image.frame} for '{view_name}' into queue.")

    def __clean_frames(self, verbose: int = 0) -> None:
        """Remove all frame subdirectories and their contents."""
        if not self.frames_base_path.exists():
            return

        cleaned_count = 0
        for view_dir in self.frames_base_path.iterdir():
            if view_dir.is_dir():
                png_files = list(view_dir.glob("*.png"))
                if png_files:
                    if verbose > 0 and cleaned_count == 0:  # Log only once
                        logger.warning(
                            f"Cleaning up previous frame directories in {self.frames_base_path}"
                        )
                    for f in png_files:
                        try:
                            f.unlink()
                        except OSError as e:  # noqa: PERF203
                            logger.error(f"Error removing frame file {f}: {e}")
                    cleaned_count += 1
                # Optionally remove the directory itself if empty and not needed for current run
                try:
                    view_dir.rmdir()
                except Exception as e:
                    logger.error(f"Error removing directory {view_dir}: {e}")

        if cleaned_count > 0 and verbose > 0:
            logger.info(f"Cleaned frame files from {cleaned_count} view directories.")

    def tick(self, world_frame: int) -> None:
        """Synchronize frames and update dynamic camera positions."""
        # 1. Update camera transforms
        for view_name, camera in self.cameras.items():
            if not camera or not camera.is_alive:
                logger.warning(f"Camera {view_name} is not valid or alive in tick.")
                continue

            view_config = CAMERA_CONFIGS.get(view_name)
            if not view_config:
                logger.warning(f"Camera view '{view_name}' not found in CAMERA_CONFIGS.")
                continue

            current_transform = self._get_camera_transform(view_name, view_config)
            try:
                self.cameras[view_name].set_transform(current_transform)
            except Exception as e:
                logger.error(f"Error updating transform for camera {view_name}: {e}")

        # 2. Process frame queue (only if storing frames in memory)
        if self._store_in_memory:
            frames_processed_this_tick = set()
            # Process all frames in the queue that are for the current world_frame or older
            # This helps catch up on any slightly delayed frames from previous ticks.
            temp_requeue = []  # To temporarily hold future frames
            try:
                while not self.queue.empty():
                    frame_id, view_name, frame_data = self.queue.get_nowait()

                    if view_name not in self.cameras:
                        continue

                    if frame_id <= world_frame:  # Process current or past frames
                        if (
                            view_name not in self.frames
                        ):  # Should not happen if initialized correctly
                            self.frames[view_name] = []

                        # Ensure we only add one frame per view for a given world_frame if strict
                        # sync is needed. However, for catching up, it's better to append if it's a
                        # new frame_id for that view
                        self.frames[view_name].append(frame_data)
                        frames_processed_this_tick.add(view_name)
                        if self.debug:
                            logger.debug(
                                f"Tick {world_frame}: "
                                f"Added frame {frame_id} for '{view_name}' to memory.",
                            )
                    else:  # frame_id > world_frame
                        temp_requeue.append((frame_id, view_name, frame_data))

                for item in temp_requeue:  # Put future frames back
                    self.queue.put(item)

            except Empty:
                logger.debug(f"Tick {world_frame}: Queue empty during processing.")

            if self.debug:
                mem_queue_size = (
                    len(self.frames.get(next(iter(self.cameras.keys())), []))
                    if self.cameras
                    else 0
                )
                logger.debug(
                    f"Tick End: World Frame={world_frame}, Queue Size={self.queue.qsize()}, "
                    f"Memory Frames={mem_queue_size}",
                )

    def _build_output_video_path(self, view_name: str) -> Path:
        """Build the specific output path for a single camera view's video."""
        filename = self.config["filename"] + "." + self.config["format"]
        if self.config.get("camera_in_filename", False):
            filename = view_name + "_" + filename
        return self.output_path_base / filename

    def save_videos(self) -> None:
        """Save the recorded frames as multiple MP4 videos, one per view."""
        if not self.cameras:
            logger.warning("No cameras were active. Skipping video save.")
            return

        if not self._store_in_memory and not self._intermediate_frame_saving:
            logger.warning("No frames recorded (neither memory nor disk). Skipping video save.")
            return

        active_views = list(self.cameras.keys())  # Views that were supposed to be recorded

        for view_name in active_views:
            output_video_path = self._build_output_video_path(view_name)
            logger.info(f"Processing video for view: {view_name} -> {output_video_path}")

            if self.use_ffmpeg:
                view_frames_path = self.frames_base_path / view_name
                # Check if the directory exists and contains PNG files
                if not view_frames_path.is_dir() or not list(view_frames_path.glob("*.png")):
                    logger.warning(
                        f"No frames found on disk for view '{view_name}' at {view_frames_path}. "
                        "Skipping ffmpeg.",
                    )
                    continue

                logger.info(f"Using ffmpeg for view '{view_name}' from {view_frames_path}")
                result = execute_ffmpeg_frames_to_video(
                    view_frames_path,
                    output_video_path,
                    self.video_fps,
                    skip_first_frames=5,  # Skip first 5 frames
                )

                if result and result.returncode == 0:
                    logger.success(
                        f"Video for view '{view_name}' saved successfully using ffmpeg."
                    )
                    if not self.save_frames_enabled:  # Clean up only if save_frames is false
                        logger.info(f"Cleaning up intermediate frames for view '{view_name}'...")
                        for f in view_frames_path.glob("*.png"):
                            try:
                                f.unlink()
                            except OSError as e:  # noqa: PERF203
                                logger.error(f"Error removing frame file {f}: {e}")
                else:
                    stderr_output = (
                        result.stderr.decode() if result and result.stderr else "No stderr output"
                    )
                    logger.error(f"ffmpeg failed for view '{view_name}'. Error: {stderr_output}")

            elif self._store_in_memory:  # Use cv2.VideoWriter with in-memory frames
                view_frames = self.frames.get(view_name, [])
                valid_frames = [f for f in view_frames if f is not None]

                if not valid_frames:
                    logger.warning(
                        f"No valid frames recorded in memory for view '{view_name}'. Skipping."
                    )
                    continue

                logger.info(
                    f"Using cv2.VideoWriter for view '{view_name}' with {len(valid_frames)} frames."
                )
                codec = get_best_codec()
                if not codec:
                    logger.error(f"Cannot write video for {view_name}, no suitable codec.")
                    continue
                fourcc = cv2.VideoWriter_fourcc(*codec)

                try:
                    video_writer = cv2.VideoWriter(
                        str(output_video_path), fourcc, self.video_fps, (self.width, self.height)
                    )
                    if not video_writer.isOpened():
                        logger.error(f"Failed to open VideoWriter for {output_video_path}")
                        continue
                except Exception as e:
                    logger.error(f"Error creating VideoWriter for {output_video_path}: {e}")
                    continue

                # quality_params = [cv2.IMWRITE_JPEG_QUALITY, 100, cv2.IMWRITE_PNG_COMPRESSION, 0]

                for frame_idx, frame in enumerate(view_frames):
                    if frame is None:
                        logger.warning(
                            f"Writing black frame for missing frame {frame_idx} in view {view_name}"
                        )
                        black_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                        video_writer.write(black_frame)
                        continue

                    # Encoding might slow things down
                    # _, buffer = cv2.imencode(".jpg", frame, quality_params)
                    # save_frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
                    video_writer.write(frame)  # Write the BGR frame directly

                video_writer.release()
                logger.success(
                    f"Video for view '{view_name}' saved at {output_video_path} using cv2.",
                )

            else:
                logger.warning(
                    f"Configuration error for view '{view_name}': "
                    "Neither ffmpeg nor in-memory storage enabled.",
                )

    def destroy(self) -> None:
        """Stop recording, destroy sensors, and save videos."""
        logger.info("Stopping multi-camera recording...")

        # Give a brief moment for ongoing sensor callbacks to finish
        # This helps get the very last frames into the queue or onto disk.
        time.sleep(5)  # Small delay

        active_cameras = list(self.cameras.keys())  # Get keys before potentially modifying dict

        for view_name in active_cameras:
            camera = self.cameras.get(view_name)
            try:
                if camera and camera.is_alive:
                    if camera.is_listening:
                        camera.stop()
                        logger.info(f"Stopped listening for camera '{view_name}'.")

                    # Short delay after stopping listener before destroying actor
                    time.sleep(0.1)

                    if not camera.destroy():
                        logger.warning(f"Could not destroy camera '{view_name}'.")
                    else:
                        logger.info(f"Camera '{view_name}' destroyed.")
            except Exception as e:
                logger.error(f"Error stopping/destroying camera {view_name}: {e}", exc_info=True)

        # Ensure the queue processing finishes (especially if using threads later)
        # Give a very short time for any final callbacks to potentially place items
        time.sleep(5)

        # Process any remaining items in the queue if storing in memory
        if self._store_in_memory:
            logger.info(f"Processing any remaining items in queue ({self.queue.qsize()})...")
            # Use the last known world frame or a slightly incremented one
            final_world_frame = self.carla_world.get_snapshot().frame + 1
            # Call tick one last time to ensure all queued frames up to this point are processed
            self.tick(final_world_frame)
            logger.info("Finished processing queue for in-memory frames.")

        # Now save the videos
        self.save_videos()

        # Attempt to remove from dict anyway
        for view_name in active_cameras:
            if view_name in self.cameras:
                del self.cameras[view_name]

        # Final cleanup of frame directories if needed
        if self.frames_base_path.exists() and self.use_ffmpeg and not self.save_frames_enabled:
            try:
                import shutil

                shutil.rmtree(self.frames_base_path)
                logger.info(f"Removed base frames directory: {self.frames_base_path}")
            except Exception as e:
                logger.error(
                    f"Could not remove base frames directory {self.frames_base_path}: {e}",
                )


class SpectatorRecorder:
    """Records the spectator's view in CARLA and saves it as a video."""

    def __init__(
        self,
        world: World,
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
        self.carla_world = self.world.world
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
        self.camera_bp = self.carla_world.get_blueprint_library().find("sensor.camera.rgb")
        self.camera_bp.set_attribute("image_size_x", str(self.width))
        self.camera_bp.set_attribute("image_size_y", str(self.height))
        self.camera_bp.set_attribute("fov", "75")  # Field of view in degrees

        # Get the spectator object
        self.spectator = self.carla_world.get_spectator()

        # Get initial transform of spectator
        # self.transform = self.spectator.get_transform()
        self.transform = self.world.player.get_transform()

        # Spawn the camera at the spectator's location
        self.camera = self.carla_world.spawn_actor(
            self.camera_bp,
            self.transform,
            attach_to=self.world.player,
            attachment_type=carla.AttachmentType.SpringArm,
        )

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

    def tick(self, world_frame: int, *, discard: bool = True) -> None:
        """Synchronize recorded frames with simulation.

        Args:
            world_frame (int): The current simulation frame.
            discard (bool): If True, discard frames that don't match the simulation tick.
        """
        try:
            # Ensure the camera follows the spectator exactly
            # Keep rotation & position in sync
            # self.camera.set_transform(self.spectator.get_transform())}
            offset = carla.Location(x=-0.25, y=-0.4, z=1.25)
            self.camera.set_transform(
                carla.Transform(
                    self.world.player.get_transform().location + offset,
                    self.world.player.get_transform().rotation,
                )
            )

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
