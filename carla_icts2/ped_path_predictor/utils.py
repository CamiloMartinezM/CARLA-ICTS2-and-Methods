"""Author: Dikshant Gupta
Time: 22.01.22 12:08
"""

import glob

import numpy as np


def preprocess(path):
    data = np.genfromtxt(path, delimiter=",").T
    # numPeds = np.size(np.unique(data[1, :]))
    numPeds = np.unique(data[1, :])

    return data, numPeds


def get_obs_pred_like(data, observed_frame_num, predicting_frame_num):
    """Get input observed data and output predicted data
    """
    obs = []
    pred = []
    count = 0

    for pedIndex in range(len(data)):
        if len(data[pedIndex]) >= observed_frame_num + predicting_frame_num:
            seq = (
                int(
                    (len(data[pedIndex]) - (observed_frame_num + predicting_frame_num))
                    / observed_frame_num,
                )
                + 1
            )

            for k in range(seq):
                obs_pedIndex = []
                pred_pedIndex = []
                flag = False
                for i in range(observed_frame_num):
                    obs_pedIndex.append(data[pedIndex][i + k * observed_frame_num])
                    if (
                        abs(
                            data[pedIndex][k * observed_frame_num + i][1]
                            - data[pedIndex][k * observed_frame_num + i + 1][1],
                        )
                        > 1
                    ):
                        flag = True
                for j in range(predicting_frame_num):
                    pred_pedIndex.append(
                        data[pedIndex][k * observed_frame_num + j + observed_frame_num],
                    )
                    if (
                        abs(
                            data[pedIndex][k * observed_frame_num + j + observed_frame_num][1]
                            - data[pedIndex][k * observed_frame_num + j + observed_frame_num + 1][
                                1
                            ],
                        )
                        > 1
                    ):
                        flag = True
                obs_pedIndex = np.reshape(obs_pedIndex, [observed_frame_num, 4])
                pred_pedIndex = np.reshape(pred_pedIndex, [predicting_frame_num, 4])

                if not flag:
                    obs.append(obs_pedIndex)
                    pred.append(pred_pedIndex)
                    count += 1

    obs = np.reshape(obs, [count, observed_frame_num, 4])
    pred = np.reshape(pred, [count, predicting_frame_num, 4])

    return obs, pred


def get_traj_like(data, numPeds):
    """Reshape data format from [frame_ID, ped_ID, x,y]
    to pedestrian_num * [ped_ID, frame_ID, x,y]
    """
    traj_data = []
    for pedIndex in numPeds:
        traj = []
        for i in range(len(data[1])):
            if data[1][i] == pedIndex:
                traj.append([data[1][i], data[0][i], data[2][i], data[3][i]])
        traj = np.reshape(traj, [-1, 4])

        traj_data.append(traj)

    return traj_data


def get_raw_data(path, observed_frame_num, predicting_frame_num):
    total_obs = []
    total_pred = []
    paths = []

    for file in glob.glob(path + "*.csv"):
        raw_data, numPeds = preprocess(file)
        data = get_traj_like(raw_data, numPeds)
        obs, pred = get_obs_pred_like(data, observed_frame_num, predicting_frame_num)

        paths.append(file)
        total_obs.append(obs)
        total_pred.append(pred)

    return total_obs, total_pred, paths
