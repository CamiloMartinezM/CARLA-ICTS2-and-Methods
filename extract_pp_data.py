import os
import yaml
import argparse
import subprocess
import time as t
import numpy as np
import sys
from datetime import datetime
from multiprocessing import Process
from SAC.sac_controller import SAC
import signal

from SAC.sac_discrete import SacdAgent
from SAC.sac_discrete.shared_sacd import SharedSacdAgent
from benchmark.environment import GIDASBenchmark
from config import Config
import time


def run(args):

    pre_safe_scenarios = [
        "01_int",
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
        "06_non_int"
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
        #file = "./P3VI/data/01_non_int_prelim.npy"
        file = f"./P3VI/data/{Config.scenarios[0]}.npy"
        car_file = f"./P3VI/data/{Config.scenarios[0]}_car.npy"
        # file = f"./P3VI/data/ICTS2_non_int_{datetime.today().strftime('%Y-%m-%d_%H-%M-%S')}.npy"


        print(file)
        # Create environments.
        env = GIDASBenchmark(port=Config.port)
        #agent = SAC(env.world, env.map, env.scene)
        #env.reset_agent(agent)
        #test_env = GIDASBenchmark(port=Config.port + 100, setting="special")
        env.world.random = False
        env.world.dummy_car = True

        env.extract = True
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

            while episode_length < Config.max_episode_length:

                x,y,icr,son = env.extract_step()
                ep_data.append((x,y,icr,son))

                x_c, y_c = env.extract_car_pos()
                ep_data_car.append((x_c, y_c))
                episode_length+=1

            ep_data = np.array(ep_data)
            ep_data_car = np.array(ep_data_car)
            data.append(ep_data)
            data_car.append(ep_data_car)
            if i % 10 == 0:
                print("Episode:", i)
                print("time taken sofar: ", time.time()-start_time)
            if i % 50 == 0 or i == iterations - 1:
                save_data = np.array(data)
                save_data_car = np.array(data_car)
                np.save(file, save_data, allow_pickle=True)
                np.save(car_file, save_data_car, allow_pickle=True)
                print("Saved",i)

        with open(file,'rb') as f:
            arr = np.load(f, allow_pickle=True)
            print(arr[0])
            print(len(arr))
        with open(car_file,'rb') as f:
            arr = np.load(f, allow_pickle=True)
            print(arr[0])
            print(len(arr))
        env.close()




def run_server():
    # train environment
    port = "-carla-port={}".format(Config.port)
    carla_p = "/home/jovyan/carla-0-9-15"
    if not Config.server:
        # carla_p = "your path to carla"
        # p = subprocess.run(['cd '+carla_p+' && ./CarlaUE4.sh your arguments' + port], shell=True)
        cmd = 'cd '+carla_p+' && ./CarlaUE4.sh -quality-level=Low -RenderOffScreen -carla-server -benchmark -fps=50' + port
        #pro = subprocess.Popen(cmd, stdout=subprocess.PIPE, 
        #                   shell=True, preexec_fn=os.setsid)
        p = subprocess.run([cmd], shell=True)
    else:
        # command = "unset SDL_VIDEODRIVER && ./CarlaUE4.sh  -quality-level="+ Config.qw  +" your arguments" + port # -quality-level=Low 
        command = "unset SDL_VIDEODRIVER && ./CarlaUE4.sh  -quality-level="+ Config.qw  +" -quality-level=Low " + port
        p = subprocess.run(['cd '+carla_p+' && ' + command ], shell=True)
        
    return p


def run_test_server():
    # test environment
    port = "-carla-port={}".format(Config.port + 100)
    carla_p = "your path to carla"
    command = "unset SDL_VIDEODRIVER && ./CarlaUE4.sh  -quality-level="+ Config.qw  +" your arguments" + port # -quality-level=Low 
    p = subprocess.run(['cd '+carla_p+' && ' + command ], shell=True)
    return p



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, default=os.path.join('SAC/sac_discrete/config', 'sacd.yaml'))
    parser.add_argument('--shared', action='store_true')
    parser.add_argument('--env_id', type=str, default='GIDASBenchmark')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--server', action='store_true')
    parser.add_argument("--qw", type=str, default="Low")
    parser.add_argument("--int", action='store_true')
    args = parser.parse_args()
    Config.server = args.server
    Config.port = args.port
    print('Env. port: {}'.format(Config.port))
    Config.port = args.port
    Config.qw = args.qw
    print(Config.scenarios)
    p = Process(target=run_server)
    p.start()
    t.sleep(20)

    #p2 = Process(target=run_test_server)
    #p2.start()
    #time.sleep(5)
    run(args)
    # subprocess.run(["kill -9 $(pidof CarlaUE4-Linux-Shipping)"], shell=True)
