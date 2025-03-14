# extract_pp_data.py
import argparse
import os
import subprocess
import time
import time as t
from multiprocessing import Process

import numpy as np

from benchmark.environment import GIDASBenchmark
from config import Config
from utils.printing import display_iteration_data, display_simulation_data


def enum_to_str(enum_val):
    if enum_val is None:
        return "None"
    return enum_val.name if hasattr(enum_val, "name") else str(enum_val)


def run(args):
    pre_safe_scenarios = [
        # "01_int",
        "02_int",
        "03_int",
        "04_int",
        "05_int",
        "06_int",
        "01_non_int",
        "02_non_int",
        "03_non_int",
        "04_non_int",
        "05_non_int",
        "06_non_int",
    ]

    for scenario in pre_safe_scenarios:
        Config.scenarios = [scenario]
        print(Config.scenarios)

        # if args.int:
        #     # file = f"./P3VI/data/ICTS2_int_{datetime.today().strftime('%Y-%m-%d_%H-%M-%S')}.npy"
        #     file = f"./P3VI/data/dump/{Config.scenarios}.npy"
        #     car_file = f"./P3VI/data/dump/{Config.scenarios}_car.npy"
        #     # file = "./P3VI/data/int_new_prelim.npy"
        # else:
        # file = "./P3VI/data/01_non_int_prelim.npy"
        file = f"./P3VI/data/{Config.scenarios[0]}.npy"
        car_file = f"./P3VI/data/{Config.scenarios[0]}_car.npy"
        # file = f"./P3VI/data/ICTS2_non_int_{datetime.today().strftime('%Y-%m-%d_%H-%M-%S')}.npy"

        print(file)

        # Create environments.
        env = GIDASBenchmark(port=Config.port)

        # agent = SAC(env.world, env.map, env.scene)
        # env.reset_agent(agent)
        # test_env = GIDASBenchmark(port=Config.port + 100, setting="special")
        env.world.random = False
        env.world.dummy_car = True
        env.extract = True

        all_episodes_data = []
        dbn_variables = [
            "SN_car",
            "SN_ped",
            "ICR_car",
            "ICR_ped",
            "SSEC",
            "A_car",
            "WS_car",
            "CBO_car",
            "ACC_car",
            "S_car",
            "D",
            "HO_ped",
            "BO_ped",
            "HIO_ped",
            "A_ped",
            "ACC_ped",
            "S_ped",
        ]

        data = []
        data_car = []
        start_time = time.time()
        # if args.int:
        #     iterations = 2 * len(env.episodes)
        # else:
        #     iterations = len(env.episodes)
        iterations = len(env.episodes) + len(env.test_episodes) + len(env.val_episodes)

        print(iterations)
        for i in range(iterations):
            state = env.reset_extract()
            episode_length = 0

            ep_data = []
            ep_data_car = []
            prev_data = None

            while episode_length < Config.max_episode_length:
                # x, y, icr, son = env.extract_step()
                # ep_data.append((x, y, icr, son))
                # print(episode_length, f"x = {x}, y = {y}, icr = {icr}, son = {son}")

                data = env.extract_dbn_step(prev_data)
                ep_data.append(data)
                prev_data = data

                display_iteration_data(data, episode_length)

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

                x_c, y_c = env.extract_car_pos()
                ep_data_car.append((x_c, y_c))
                episode_length += 1

            ep_data = np.array(ep_data)
            ep_data_car = np.array(ep_data_car)
            data.append(ep_data)
            data_car.append(ep_data_car)
            if i % 10 == 0:
                print("Episode:", i)
                print("time taken sofar: ", time.time() - start_time)
            if i % 50 == 0 or i == iterations - 1:
                save_data = np.array(data)
                save_data_car = np.array(data_car)
                np.save(file, save_data, allow_pickle=True)
                np.save(car_file, save_data_car, allow_pickle=True)
                print("Saved", i)

        with open(file, "rb") as f:
            arr = np.load(f, allow_pickle=True)
            print(arr[0])
            print(len(arr))
        with open(car_file, "rb") as f:
            arr = np.load(f, allow_pickle=True)
            print(arr[0])
            print(len(arr))
        env.close()


def run_server():
    # train environment
    port = f"-carla-port={Config.port}"
    carla_p = "/home/camilo/Applications/carla-0-9-15-linux"
    if not Config.server:
        print("Running Carla in server mode")
        # carla_p = "your path to carla"
        # p = subprocess.run(['cd '+carla_p+' && ./CarlaUE4.sh your arguments' + port], shell=True)
        # -RenderOffScreen
        cmd = (
            "cd "
            + carla_p
            + " && DRI_PRIME=1 ./CarlaUE4.sh -quality-level=Epic -carla-server -benchmark -prevernvidia -fps=25"
            + port
        )
        # pro = subprocess.Popen(cmd, stdout=subprocess.PIPE,
        #                   shell=True, preexec_fn=os.setsid)
        p = subprocess.run([cmd], shell=True, check=False)
    else:
        # command = "unset SDL_VIDEODRIVER && ./CarlaUE4.sh  -quality-level="+ Config.qw  +" your arguments" + port # -quality-level=Low
        command = "unset SDL_VIDEODRIVER && ./CarlaUE4.sh  -quality-level=" + Config.qw + " -quality-level=Low " + port
        p = subprocess.run(["cd " + carla_p + " && " + command], shell=True, check=False)

    return p


def run_test_server():
    # test environment
    port = f"-carla-port={Config.port + 100}"
    carla_p = "your path to carla"
    command = (
        "unset SDL_VIDEODRIVER && ./CarlaUE4.sh  -quality-level=" + Config.qw + " your arguments" + port
    )  # -quality-level=Low
    p = subprocess.run(["cd " + carla_p + " && " + command], shell=True, check=False)
    return p


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join("SAC/sac_discrete/config", "sacd.yaml"),
    )
    parser.add_argument("--shared", action="store_true")
    parser.add_argument("--env_id", type=str, default="GIDASBenchmark")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--server", action="store_true")
    parser.add_argument("--qw", type=str, default="Low")
    parser.add_argument("--int", action="store_true")
    args = parser.parse_args()
    Config.server = args.server
    Config.port = args.port
    print(f"Env. port: {Config.port}")
    Config.port = args.port
    Config.qw = args.qw
    print(Config.scenarios)
    p = Process(target=run_server)
    p.start()
    t.sleep(20)

    # p2 = Process(target=run_test_server)
    # p2.start()
    # time.sleep(5)
    run(args)
    # subprocess.run(["kill -9 $(pidof CarlaUE4-Linux-Shipping)"], shell=True)
