"""Recording module for CARLA simulator to capture the spectator's view."""

from pathlib import Path
from queue import Empty, Queue

import carla
import cv2
import numpy as np

from carla_icts2.config import logger


class SpectatorRecorder:
    """Records the spectator's view in CARLA and saves it as a video."""

    def __init__(
        self,
        world: carla.World,
        config: dict,
        width: int = 1920,
        height: int = 1080,
    ) -> None:
        """Initialize the SpectatorRecorder.

        Args:
            world (carla.World): The CARLA world instance.
            config (dict): Configuration dictionary containing output path and FPS.
            width (int): Width of the video frames.
            height (int): Height of the video frames.
        """
        self.world = world
        self.config = config
        self.frames: list[np.ndarray] = []
        self.queue: Queue[tuple[int, np.ndarray]] = Queue()
        self.video_fps = config["fps"]
        self.output_path = Path(config["output_path"])

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

        # As test, save the image to disk
        if self.config["save_frames"]:
            self.frames_path = self.output_path.parent / "frames"
            self.frames_path.mkdir(parents=True, exist_ok=True)

            # Remove the previous contents inside the folder
            for file in self.frames_path.glob("*.jpg"):
                file.unlink()

        logger.info(f"Spectator recording started: {self.output_path}")

    def _process_frame(self, image: carla.Image) -> None:
        """Process the camera frame and adds it to the queue for synchronization."""
        array = np.array(image.raw_data, dtype=np.uint8)
        array = array.reshape((self.height, self.width, 4))[:, :, :3]  # Convert to RGB

        # Save the image to disk
        if self.config["save_frames"]:
            cv2.imwrite(str(self.frames_path / f"image_{image.frame}.jpg"), array)

        self.queue.put((image.frame, array))  # Store frame in queue

    def tick(self, world_frame: int) -> None:
        """Synchronize recorded frames with simulation."""
        try:
            # Ensure the camera follows the spectator exactly
            # Keep rotation & position in sync
            self.camera.set_transform(self.spectator.get_transform())

            while not self.queue.empty():
                _, frame_data = self.queue.get(True, timeout=1.0)
                self.frames.append(frame_data)

                # if frame_id == world_frame:  # Ensure frames match simulation tick
                # break  # Only process one frame per tick
                # logger.warning(f"Frame {frame_id} discarded. Expected frame: {world_frame}")

        except Empty:
            logger.warning("Missing frame for spectator recording.")

    def save_video(self) -> None:
        """Save the recorded frames as an MP4 video."""
        if not self.frames:
            logger.warning("No frames recorded. Skipping video save.")
            return

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            self.output_path,
            fourcc,
            self.video_fps,
            (self.width, self.height),
        )

        for frame in self.frames:
            video_writer.write(frame)

        video_writer.release()
        logger.success(f"Spectator view video saved at {self.output_path}")

    def destroy(self) -> None:
        """Stop recording and releases resources."""
        self.camera.destroy()
        self.save_video()
