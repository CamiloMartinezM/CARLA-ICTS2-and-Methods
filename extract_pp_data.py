"""Extract pedestrian and car data from the CARLA simulator using the GIDAS benchmark."""

import subprocess
import time
import time as t
from multiprocessing import Process
from tracemalloc import start

import numpy as np

from carla_icts2.benchmark.environment import GIDASBenchmark
from carla_icts2.config import VIDEOS_DIR, logger
from carla_icts2.scenarios_config import Config
from carla_icts2.utils.loading import load_yaml
from carla_icts2.utils.recording import (
    SpectatorRecorder,
    stitch_videos_side_by_side,
    videos_in_folder,
)
from carla_icts2.utils.run import run_server_command


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

        # Initialize the spectator recorder if enabled
        spectator_recorder = None
        if config["video"]["save"]:
            spectator_recorder = SpectatorRecorder(
                carla_world,
                config["video"],
                width=Config.width,
                height=Config.height,
                scenario=scenario,
                camera_type=Config.camera,
            )

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
        if iterations > 1 and spectator_recorder is not None:
            logger.warning(
                "The number of iterations is greater than 1 and a spectator recorder is enabled. "
                "This has untested behavior and may not work as expected.",
            )

        logger.info(f"Running scenario: {scenario} for {iterations} iterations")
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
                    if spectator_recorder:
                        spectator_recorder.tick(world_frame)

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

        # Stop recording when the simulation ends
        if spectator_recorder:
            spectator_recorder.destroy()

        env.close()

        # Run after the simulation ends
        run_afterwards()


def run_server(config: dict) -> subprocess.CompletedProcess:
    """Run the Carla server with the given `config`."""
    cmd = run_server_command(config)
    return subprocess.run([cmd], shell=True, check=True)  # noqa: S602


def run_before() -> None:
    """Run before the main function."""
    kill_carla_server()


def run_afterwards() -> None:
    """Run after the main function even if there is a `KeyboardInterrupt`."""
    kill_carla_server()

    # Loop through the recorded videos and stitch them together
    for scenario_folder in VIDEOS_DIR.iterdir():
        if scenario_folder.is_dir():
            # Get the list of videos in the folder, but only the views that we're interested in
            videos = videos_in_folder(
                scenario_folder,
                startswith=("bev_follow_vehicle", "pedestrian_pov_view", "vehicle_pov_view"),
            )
            if len(videos) > 1:
                if len(videos) != 3:
                    logger.warning(
                        f"Expected 3 videos in {scenario_folder}, found {len(videos)}.",
                    )

                logger.info(f"Stitching {len(videos)} videos: {videos} in {scenario_folder}")
                stitch_videos_side_by_side(videos, output_path=scenario_folder / "all_views.mp4")
                logger.success(f"Stitched video saved to {scenario_folder / 'all_views.mp4'}")


def kill_carla_server() -> None:
    """Kill the Carla server process."""
    logger.info("Killing previous CarlaUE4-Linux-Shipping process")
    subprocess.run(  # noqa: S602
        ["kill -9 $(pidof CarlaUE4-Linux-Shipping) 2>/dev/null || true"],  # noqa: S607
        check=True,
        shell=True,
    )


if __name__ == "__main__":
    run_config = load_yaml("run_config.yaml")

    # Add necessary run config parameters to the Config class
    Config.port = run_config["carla"]["port"]
    Config.scenarios = run_config["scenarios"]
    Config.camera = run_config["camera"][0]
    Config.max_episode_length = run_config["carla"]["max_episode_length"]

    logger.info(f"Env. port: {Config.port}")
    logger.info(f"Scenarios: {Config.scenarios}")

    # Run commands before the execution of CARLA
    run_before()

    p = Process(target=run_server, args=(run_config,))
    p.start()
    t.sleep(10)

    run(run_config)

    # run_afterwards()
