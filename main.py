"""Extract pedestrian and car data from the CARLA simulator using the GIDAS benchmark."""

import subprocess
import time as t
from multiprocessing import Process

import numpy as np
from cyclopts import App

from carla_icts2.benchmark.environment import GIDASBenchmark
from carla_icts2.config import VIDEOS_DIR, logger
from carla_icts2.scenarios_config import Config
from carla_icts2.utils.loading import load_yaml
from carla_icts2.utils.recording import (
    MultiCameraRecorder,
    stitch_videos_side_by_side,
    videos_in_folder,
)
from carla_icts2.utils.run import run_server_command

app = App()


def run(config: dict) -> None:
    """Run during the simulation (execution of CARLA)."""
    for scenario in config["scenarios"]:
        Config.scenarios = [scenario]

        # if args.int:
        #     # file = f"./P3VI/data/ICTS2_int_{datetime.today().strftime('%Y-%m-%d_%H-%M-%S')}.npy"
        #     file = f"./P3VI/data/dump/{Config.scenarios}.npy"
        #     car_file = f"./P3VI/data/dump/{Config.scenarios}_car.npy"
        #     # file = "./P3VI/data/int_new_prelim.npy"
        # else:
        # file = "./P3VI/data/01_non_int_prelim.npy"
        file = f"./P3VI/data/{Config.scenarios[0]}.npy"
        car_file = f"./P3VI/data/{Config.scenarios[0]}_car.npy"

        print(file)

        # Create environments
        env = GIDASBenchmark(port=Config.port)
        env.world.random = False
        env.world.dummy_car = True
        env.extract = True

        # Get the CARLA world instance
        carla_world = env.world.world

        # --- Initialize Multi-Camera Recorder ---
        multi_camera_recorder = None
        camera_views_to_record = config.get("camera", [])  # Get list from config
        if config.get("video", {}).get("save", False) and camera_views_to_record:
            multi_camera_recorder = MultiCameraRecorder(
                env.world,
                config["video"],
                camera_views=camera_views_to_record,
                width=Config.width,
                height=Config.height,
                scenario=scenario,
                debug=Config.debug,
            )
            # multi_camera_recorder = SpectatorRecorder(
            #     env.world,
            #     config["video"],
            #     width=Config.width,
            #     height=Config.height,
            #     scenario=scenario,
            #     camera_type=Config.camera,
            # )
        else:
            logger.info("Video recording disabled or no camera views specified.")

        data = []
        data_car = []

        # if args.int:
        #     iterations = 2 * len(env.episodes)
        # else:
        #     iterations = len(env.episodes)

        # Define the number of iterations based on the config if available
        if "iterations" in config["carla"]:
            iterations = config["carla"]["iterations"]
            logger.warning(
                f"The number of iterations is set in the config file to {iterations}. "
                "This will override the default behavior.",
            )
        else:
            # Default behavior: run through all episodes and test/validation episodes
            iterations = len(env.episodes) + len(env.test_episodes) + len(env.val_episodes)

        # Check if the number of iterations is greater than 1 and a spectator recorder is enabled
        if iterations > 1 and multi_camera_recorder is not None:
            logger.warning(
                "The number of iterations is greater than 1 and a spectator recorder is enabled. "
                "This has untested behavior and may not work as expected.",
            )

        logger.info(f"Running scenario: {scenario} for {iterations} iterations")

        t.sleep(10)  # Wait for the server to be ready
        for i in range(iterations):
            state = env.reset_extract()
            episode_length = 0

            ep_data = []
            ep_data_car = []
            prev_data = None

            try:
                while episode_length < Config.max_episode_length:
                    world_frame = carla_world.get_snapshot().frame  # Get current simulation frame

                    # x, y, icr, son = env.extract_step()
                    # ep_data.append((x, y, icr, son))
                    # print(episode_length, f"x = {x}, y = {y}, icr = {icr}, son = {son}")

                    data = env.extract_dbn_step(prev_data)
                    ep_data.append(data)
                    prev_data = data

                    # Synchronize spectator recording
                    if multi_camera_recorder is not None:
                        multi_camera_recorder.tick(world_frame)

                    # * Include radius of 50 m of perception
                    # * Videos (BEV and POV from car and pedestrian) of interactive scenario

                    # DIRECTLY AVAILABLE:
                    # Intention to claim the road for pedestrian (ICRped)
                    # Strategy of Negotiation (SN_ped) Avoiding, Yielding, Forcing
                    # Strategy of Negotiation (SN_car) Avoiding, Yielding, Forcing
                    # Acceleration (ACC) Discretized classes
                    # Speed (S) Discretized classes
                    # Distance (D) Discretized classes

                    # COULD BE DERIVED:
                    # Approaching (A) Yes, No
                    # Wheel stance (WS) Facing, Averting, Ignoring
                    # Car Body Orientation (CBO) Facing, Averting, Ignoring
                    # Head Orientation (HO) Facing, Averting, Ignoring
                    # Body Orientation (BO) Facing, Averting, Ignoring
                    # Hip Orientation (HIO) Neutral, Slightly leaning forward, Leaning forward

                    # TRICKIER / SUBJECTIVE:
                    # Sense of Security (SSEC) Very high, High, Medium, Low, Very low
                    # * Calculate this with ICR_ped in reverse
                    # Intention to claim the road for car (ICRcar)

                    # x_c, y_c = env.extract_car_pos()
                    # ep_data_car.append((x_c, y_c))
                    episode_length += 1

                # ep_data = np.array(ep_data)
                # ep_data_car = np.array(ep_data_car)
                # data.append(ep_data)
                # data_car.append(ep_data_car)
                # if i % 10 == 0:
                #     print("Episode:", i)
                #     print("time taken sofar: ", time.time() - start_time)
                # if i % 50 == 0 or i == iterations - 1:
                #     save_data = np.array(data)
                #     save_data_car = np.array(data_car)
                #     np.save(file, save_data, allow_pickle=True)
                #     np.save(car_file, save_data_car, allow_pickle=True)
                #     print("Saved", i)

                if multi_camera_recorder is not None:
                    logger.info("Episode finished. Running extra ticks for final frame capture...")
                    extra_ticks = 50
                    for m in range(extra_ticks):
                        carla_world.tick()  # Tick the CARLA world
                        current_snapshot_frame = carla_world.get_snapshot().frame

                        # Allow sensor callbacks to process and put frames on queue
                        t.sleep(carla_world.get_settings().fixed_delta_seconds * 1.1)  # Small wait

                        # Process frames for this tick
                        multi_camera_recorder.tick(current_snapshot_frame)
                        if m % 5 == 0:
                            logger.debug(
                                f"Extra tick {m + 1}/{extra_ticks}, "
                                f"World Frame: {current_snapshot_frame}",
                            )

                    logger.info("Finished extra ticks.")

            except KeyboardInterrupt:
                break

        # with open(file, "rb") as f:
        #     arr = np.load(f, allow_pickle=True)
        #     print(arr[0])
        #     print(len(arr))
        # with open(car_file, "rb") as f:
        #     arr = np.load(f, allow_pickle=True)
        #     print(arr[0])
        #     print(len(arr))

        t.sleep(2)  # Wait for the environment to stabilize

        # --- Cleanup after each scenario ---
        if multi_camera_recorder:
            logger.info(f"Destroying recorder for scenario {scenario}...")
            multi_camera_recorder.destroy()  # This now saves the videos
            multi_camera_recorder = None  # Ensure it's reset for the next scenario

        env.close()


def run_server(config: dict) -> subprocess.CompletedProcess:
    """Run the Carla server with the given `config`."""
    cmd = run_server_command(config)
    logger.info(f"Running CARLA server with command: {cmd}")
    return subprocess.run([cmd], shell=True, check=True)  # noqa: S602


def run_before() -> None:
    """Run before the main function."""
    kill_carla_server()


@app.command(name="postprocess")
def postprocessing(
    width: int | None = None,
    height: int | None = None,
    camera_views_to_stitch: list[str] | None = None,
    max_stitching_videos: int = 3,
    *,
    use_ffmpeg: bool = True,
    replace_existing: bool = False,
) -> None:
    """Run after the main function and stitches the videos together.

    Executes:
    1. Kill the Carla server;
    2. Stitch the recorded videos together inside the scenario folders;
    3. Save the stitched video as `all_views.mp4` in the same folder.

    Args:
        width (int | None): Width that each video should be resized to before stitching together.
            If None, it takes the `Config.width` value.
        height (int | None): Height that each video should be resized to before stitching together.
            If None, it takes the `Config.height` value.
        camera_views_to_stitch (list[str] | None): List of camera views to stitch together.
            If None, it defaults to the one in the `run_config.yaml` file.
        max_stitching_videos (int): Maximum number of videos to stitch together horizontally.
            Defaults to 3.
        use_ffmpeg (bool): Whether to use ffmpeg for stitching videos. Defaults to True.
        replace_existing (bool): If True, it will replace existing stitched videos.
            Defaults to False.
    """
    kill_carla_server()

    run_config = load_yaml("run_config.yaml")
    if camera_views_to_stitch is None:
        camera_views_to_stitch = run_config["stitched_video"]["views"]

    # Loop through the recorded videos and stitch them together
    for scenario_folder in VIDEOS_DIR.iterdir():
        if scenario_folder.is_dir():
            # If an all_views file already exists, delete it to re-generate it
            if (scenario_folder / "all_views.mp4").exists():
                if replace_existing:
                    logger.warning(
                        f"Deleting already existing {scenario_folder / 'all_views.mp4'}.",
                    )
                    (scenario_folder / "all_views.mp4").unlink()
                else:
                    continue

            # Get the list of videos in the folder, but only the views that we're interested in
            videos = videos_in_folder(scenario_folder, startswith=camera_views_to_stitch)
            if len(videos) > 1:
                if len(videos) > max_stitching_videos:
                    logger.warning(
                        f"Found {len(videos)} videos in {scenario_folder}, "
                        f"but only stitching the first {max_stitching_videos} videos.",
                    )
                    videos = videos[:max_stitching_videos]
                elif len(videos) < max_stitching_videos:
                    logger.warning(
                        f"Expected {max_stitching_videos} videos in {scenario_folder}, "
                        f"found {len(videos)}.",
                    )

                logger.info(
                    f"Stitching {len(videos)} videos: {[str(v) for v in videos]} "
                    f"in {scenario_folder}",
                )
                stitch_videos_side_by_side(
                    videos,
                    output_path=scenario_folder / "all_views.mp4",
                    width=Config.width if width is None else width,
                    height=Config.height if height is None else height,
                    use_ffmpeg=use_ffmpeg,
                )
                logger.success(f"Stitched video saved to {scenario_folder / 'all_views.mp4'}")


@app.command(name="kill")
def kill_carla_server() -> None:
    """Kill the Carla server process."""
    logger.info("Killing previous CarlaUE4-Linux-Shipping process")
    subprocess.run(  # noqa: S602
        ["kill -9 $(pidof CarlaUE4-Linux-Shipping) 2>/dev/null || true"],  # noqa: S607
        check=True,
        shell=True,
    )


@app.default()
def main(*, replace_existing: bool = False) -> None:
    """Run the complete script."""
    run_config = load_yaml("run_config.yaml")

    # Add necessary run config parameters to the Config class
    Config.port = run_config["carla"]["port"]
    Config.scenarios = run_config["scenarios"]
    Config.camera = run_config["camera"][0]
    Config.max_episode_length = run_config["carla"]["max_episode_length"]
    Config.width = run_config["carla"]["width"]
    Config.height = run_config["carla"]["height"]
    Config.load_complete_map = run_config["carla"]["load_complete_map"]
    Config.debug = run_config["carla"]["debug"]

    # Print the configuration
    logger.info(f"Env. port: {Config.port}")
    logger.info(f"Camera: {Config.camera}, {Config.width}x{Config.height}")
    logger.info(f"Scenarios: {Config.scenarios}")

    # Run commands before the execution of CARLA
    run_before()

    p = Process(target=run_server, args=(run_config,))
    p.start()
    logger.info("Waiting for CARLA server to start...")
    t.sleep(10)

    try:
        run(run_config)
    except Exception as e:
        logger.error(f"An error occurred in the main run function: {e}", exc_info=True)
        raise
    finally:
        # Ensure server is killed even if the script crashes
        postprocessing(replace_existing=replace_existing)
        if p.is_alive():
            p.terminate()
            p.join()
        logger.info("Main script finished.")


if __name__ == "__main__":
    app()
