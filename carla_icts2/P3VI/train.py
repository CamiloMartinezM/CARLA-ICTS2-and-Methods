import os
import sys
from copy import deepcopy

sys.path.append("/workspace/data/CARLA-ICTS")
import time
from datetime import datetime as dt

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter

from P3VI.model import P3VI
from P3VI.utils import load_data

path_int = "./P3VI/data/ICTS2_int.npy"
path_non_int = "./P3VI/data/ICTS2_non_int.npy"

observed_frame_num = 15
predicting_frame_num = 20
batch_size = 512
train_samples = 1
test_samples = 1
epochs = 250
latent_dim = 24


def normalize(trajectories, shift_x=None, shift_y=None, scale=None):
    """This is the forward transformation function which transforms the
    trajectories to a square region around the origin such as
    [-1,-1] - [1,1] while the original trajectories could be in
    pixel unit or in metres. The transformation would be applied
    to the trajectories as follows:
      new_trajectories = (old_trajectories - shift) / scale
    :param trajectories: must be a 3D Python array of the form
           Ntrajectories x Ntime_step x 2D_x_y
    :param shift_x, shift_y: the amount of desired translation.
           If not given, the centroid of trajectory points
           would be used.
    :param scale: the desirable scale factor. If not given, the scale
          would be computed so that the trajectory points fall inside
          the [-1,-1] to [1,1] region.
    :return new_trajectories: the new trajectories after the
            transformation.
    :return shift_x, shift_y, scale: if these arguments were not
            supplied, then the function computes and returns them.
    The function assumes that either all the optional parameters
    are given or they are all not given.
    """
    if shift_x is not None:
        shift = np.array([shift_x, shift_y]).reshape(1, 2)
        new_trajectories = deepcopy((trajectories - shift) / scale)
        return new_trajectories
    shift = np.mean(trajectories, axis=(0, 1)).reshape(1, 2)
    new_trajectories = deepcopy(trajectories - shift)
    minxy = np.min(new_trajectories, axis=(0, 1))
    maxxy = np.max(new_trajectories, axis=(0, 1))
    scale = np.max(maxxy - minxy) / 2.0
    new_trajectories /= scale
    return new_trajectories, shift[0, 0], shift[0, 1], scale


# model = P3VI(40,30)


class P3VIWrapper:
    def __init__(
        self,
        model_path=None,
        observed_frame_num=observed_frame_num,
        predicting_frame_num=predicting_frame_num,
    ):
        self.model = P3VI(observed_frame_num, predicting_frame_num).cuda()
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        self.optim = torch.optim.Adam(lr=0.00005, params=self.model.parameters())
        log_dir = (
            "../_out/p3vi/"
            + f"new_{epochs}_{batch_size}_all_seed_0_{observed_frame_num}_{predicting_frame_num}_"
            + str(time.time())
            + "/summary"
        )

        export_dir = "./_out/weights/P3VI"
        self.save_path = (
            f"{export_dir}/P3VI_"
            f"{observed_frame_num}o_"
            f"{predicting_frame_num}p_"
            f"{epochs}e_"
            f"{batch_size}b_"
            f"{dt.today().strftime('%Y-%m-%d_%H-%M-%S')}.pth"
        )
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)

        if model_path is None:
            self.writer = SummaryWriter(log_dir=log_dir)
        self.observed_frame_num = observed_frame_num
        self.predicting_frame_num = predicting_frame_num

    def test(self, test=False, path=None):
        if not test:
            obs_train_int, pred_train_int = load_data(
                path_int, self.observed_frame_num, self.predicting_frame_num,
            )
            obs_train_non_int, pred_train_non_int = load_data(
                path_non_int, self.observed_frame_num, self.predicting_frame_num,
            )

            obs_train = np.concatenate((obs_train_int, obs_train_non_int))
            pred_train = np.concatenate((pred_train_int, pred_train_non_int))

            # print("Started loading")
            print(obs_train.shape)
            print(pred_train.shape)

            input_train = np.array(obs_train[:, :, :], dtype=np.float32)
            output_train = np.array(pred_train[:, :, :], dtype=np.float32)
            # input_train[:,:,0:2],_,_,_ = normalize(input_train[:,:,0:2])
            # output_train,_,_,_ = normalize(output_train)
            # input_train[:,:,0:2] = (input_train[:,:,0:2] - np.array([80,200],dtype=np.float32)) /100
            # output_train = (np.array(pred_train[:, :, :], dtype=np.float32) - np.array([80,200],dtype=np.float32))/100

            input_train, input_test, output_train, output_test = train_test_split(
                input_train, output_train, test_size=0.15, random_state=0,
            )

            # make output relative to the last observed frame
            i_t = input_train[:, self.observed_frame_num - 1, 0:2]
            i_t = np.expand_dims(i_t, axis=1)
            i_t = np.repeat(i_t, self.predicting_frame_num, axis=1)
            output_train = output_train - i_t
            print(np.mean(output_train[:, :, 0:2]))

            i_t = input_test[:, self.observed_frame_num - 1, 0:2]
            i_t = np.expand_dims(i_t, axis=1)
            i_t = np.repeat(i_t, self.predicting_frame_num, axis=1)
            output_test = output_test - i_t

            input_train = np.transpose(input_train, (1, 0, 2))
            output_train = np.transpose(output_train, (1, 0, 2))
            input_test = np.transpose(input_test, (1, 0, 2))
            output_test = np.transpose(output_test, (1, 0, 2))
            print("Input train shape =", input_train.shape)
            print("Output train shape =", output_train.shape)
        else:
            input_test, output_test = load_data(
                path, self.observed_frame_num, self.predicting_frame_num,
            )
            input_test = np.array(input_test[:, :, :], dtype=np.float32)
            output_test = np.array(output_test[:, :, :], dtype=np.float32)
            i_t = input_test[:, self.observed_frame_num - 1, 0:2]
            i_t = np.expand_dims(i_t, axis=1)
            i_t = np.repeat(i_t, self.predicting_frame_num, axis=1)
            output_test = output_test - i_t

            input_test = np.transpose(input_test, (1, 0, 2))
            output_test = np.transpose(output_test, (1, 0, 2))

        test_batches = int(np.floor(input_test.shape[1] / batch_size))
        eval_loss = 0
        fde_loss = 0
        for j in range(test_batches):
            x = input_test[:, j * batch_size : j * batch_size + batch_size, :]
            y = output_test[:, j * batch_size : j * batch_size + batch_size, :]
            x = torch.from_numpy(x).cuda()
            y = torch.from_numpy(y).cuda()

            x_traj = x[:, :, 0:2]
            x_cf = x[:, :, 2:]
            mse, fde = self.evaluate(x_traj, x_cf, y)
            eval_loss += mse
            fde_loss += fde
        # print("x", x[-1,0,:])
        # print("y", y[0,0,:])
        # print("y_pred", y_pred[0,0,:],"\n")
        eval_loss /= test_batches * self.predicting_frame_num * batch_size
        fde_loss /= test_batches * batch_size

        print("MSE", round(eval_loss, 2))
        print("FDE", round(fde_loss, 2))

        return round(eval_loss, 2), round(fde_loss, 2)

    def train(self):
        # train_dataset = SIMP3Dataset(path, observed_frame_num, predicting_frame_num, split = [0,0.8])
        # val_dataset = SIMP3Dataset(path, observed_frame_num, predicting_frame_num, [0.8,1.0])

        # train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        # val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
        # Get training data (past and future pedestrian bounding boxes)
        # obs_train, pred_train, train_paths = get_raw_data(train_annotations, observed_frame_num, predicting_frame_num)
        # print("Started loading")
        obs_train_int, pred_train_int = load_data(
            path_int, self.observed_frame_num, self.predicting_frame_num,
        )
        obs_train_non_int, pred_train_non_int = load_data(
            path_non_int, self.observed_frame_num, self.predicting_frame_num,
        )

        obs_train = np.concatenate((obs_train_int, obs_train_non_int))
        pred_train = np.concatenate((pred_train_int, pred_train_non_int))
        # print("Started loading")
        print(obs_train.shape)
        print(pred_train.shape)

        input_train = np.array(obs_train[:, :, :], dtype=np.float32)
        output_train = np.array(pred_train[:, :, :], dtype=np.float32)
        # input_train[:,:,0:2],_,_,_ = normalize(input_train[:,:,0:2])
        # output_train,_,_,_ = normalize(output_train)
        # input_train[:,:,0:2] = (input_train[:,:,0:2] - np.array([80,200],dtype=np.float32)) /100
        # output_train = (np.array(pred_train[:, :, :], dtype=np.float32) - np.array([80,200],dtype=np.float32))/100

        input_train, input_test, output_train, output_test = train_test_split(
            input_train, output_train, test_size=0.15, random_state=0,
        )

        # make output relative to the last observed frame
        i_t = input_train[:, self.observed_frame_num - 1, 0:2]
        i_t = np.expand_dims(i_t, axis=1)
        i_t = np.repeat(i_t, self.predicting_frame_num, axis=1)
        output_train = output_train - i_t
        print(np.mean(output_train[:, :, 0:2]))

        i_t = input_test[:, self.observed_frame_num - 1, 0:2]
        i_t = np.expand_dims(i_t, axis=1)
        i_t = np.repeat(i_t, self.predicting_frame_num, axis=1)
        output_test = output_test - i_t

        input_train = np.transpose(input_train, (1, 0, 2))
        output_train = np.transpose(output_train, (1, 0, 2))
        input_test = np.transpose(input_test, (1, 0, 2))
        output_test = np.transpose(output_test, (1, 0, 2))
        print("Input train shape =", input_train.shape)
        print("Output train shape =", output_train.shape)

        count = 0
        best_eval = np.inf
        best_eval_fde = np.inf
        for epoch in range(epochs):
            num_batches = int(np.floor(input_train.shape[1] / batch_size))
            ckp_loss = 0

            t_before = time.time()
            for i in range(num_batches):
                x = input_train[
                    :, i * batch_size : i * batch_size + batch_size, :,
                ]  # observed_frame_num x batch_size x 2
                y = output_train[:, i * batch_size : i * batch_size + batch_size, :]
                x = torch.from_numpy(x).cuda()
                y = torch.from_numpy(y).cuda()
                x_traj = x[:, :, 0:2]
                x_cf = x[:, :, 2:]
                # print(x_traj[0][0])

                y_pred = self.model(x_traj, x_cf)
                recons_loss = F.mse_loss(y_pred, y)
                # kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=2), dim=1)
                # print("gt",y[0][-1])
                # print("pred",y_pred[0][-1])
                loss = recons_loss

                self.optim.zero_grad()
                loss.backward()
                # <print(self.compute_policy_grad_norm())
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                self.optim.step()
                ckp_loss += loss.item()
                self.writer.add_scalar("loss", loss.item(), epoch * num_batches + i)
                if i % 100 == 0:
                    print(f"Time Taken for Batch: {(time.time() - t_before)}")
                    t_before = time.time()
                    print(f"Epoch: {epoch + 1}, batch: {i} Loss: {ckp_loss / 100:.4f}")

                    ckp_loss = 0

                    eval_loss = 0
                    fde_loss = 0
                    test_batches = int(np.floor(input_test.shape[1] / batch_size))
                    for j in range(test_batches):
                        x = input_test[:, j * batch_size : j * batch_size + batch_size, :]
                        y = output_test[:, j * batch_size : j * batch_size + batch_size, :]
                        x = torch.from_numpy(x).cuda()
                        y = torch.from_numpy(y).cuda()

                        x_traj = x[:, :, 0:2]
                        x_cf = x[:, :, 2:]
                        mse, fde = self.evaluate(x_traj, x_cf, y)
                        eval_loss += mse
                        fde_loss += fde

                    eval_loss /= test_batches * self.predicting_frame_num
                    fde_loss /= test_batches
                    if eval_loss < best_eval and fde_loss < best_eval_fde:
                        # "{}_{}_".format(observed_frame_num, predicting_frame_num)
                        # save_path = './_out/weights/new_{}_{}_all_seed_0_p3vi_{}_{}_{}.pth'.format(epochs, batch_size,"best",self.observed_frame_num, self.predicting_frame_num)
                        torch.save(self.model.state_dict(), self.save_path)
                        best_eval = eval_loss
                        best_eval_fde = fde_loss
                    self.writer.add_scalar("eval_loss", eval_loss, count)
                    self.writer.add_scalar("fde_loss", fde_loss, count)
                    count += 1

    def evaluate(self, x_traj, x_cf, y_test):
        with torch.no_grad():
            y_pred = self.model(x_traj, x_cf)
            return torch.square(y_pred - y_test).sum(2).sqrt().sum().item(), self.fde(
                y_pred, y_test,
            )
            # return torch.sum(torch.square(y_pred - y_test)).item(), self.fde(y_pred, y_test)

    def fde(self, y_pred, y_test):
        # print(100*"-")
        # for i in range(128):
        #    print(y_pred[i,-1,:],y_test[i,-1,:])

        return torch.square(y_pred[-1, :, :] - y_test[-1, :, :]).sum(1).sqrt().sum().item()
        # return torch.sum(torch.square((y_pred[-1,:,:] - y_test[-1,:,:]))).item()

    def compute_policy_grad_norm(self):
        total_norm = 0
        # print(self.model)
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1.0 / 2)
        return total_norm

    def get_single_prediction(self, x):
        x_traj = x[:, 0:2]
        x_cf = x[:, 2:]
        i_t = x_traj[self.observed_frame_num - 1, :]
        # x_traj = np.array(x_traj, dtype=np.float32)
        # x_traj = x_traj.reshape((observed_frame_num,1,2))
        # x_traj = torch.from_numpy(x_traj).cuda()
        x_traj = (
            torch.tensor(x_traj, dtype=torch.float32)
            .reshape((self.observed_frame_num, 1, 2))
            .cuda()
        )
        x_cf = (
            torch.tensor(x_cf, dtype=torch.float32).reshape((self.observed_frame_num, 1, 2)).cuda()
        )
        with torch.no_grad():
            path = self.model.forward(x_traj, x_cf)
        path = path.cpu().squeeze().numpy()
        i_t = np.expand_dims(i_t, axis=0)
        i_t = np.repeat(i_t, self.predicting_frame_num, axis=0)
        path = path + i_t
        return path


if __name__ == "__main__":
    p = P3VIWrapper()
    p.train()
