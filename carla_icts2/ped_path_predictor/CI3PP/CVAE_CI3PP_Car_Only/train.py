import logging
import os
import sys

from matplotlib import pyplot as plt

sys.path.append("/workspace/data/CARLA-ICTS")
import time
from datetime import datetime as dt

import numpy as np
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import MultiStepLR

from ped_path_predictor.CI3PP.CVAE_CI3PP_Car_Only.model import CVAE_CI3PP_Car_Only
from ped_path_predictor.new_util import getDataloaders, singleDatasets

path_int = "./ped_path_predictor/data/new_car/all_int.npy"
path_non_int = "./ped_path_predictor/data/new_car/all_non_int.npy"

path_int_car = "./ped_path_predictor/data/new_car/all_int_car.npy"
path_non_int_car = "./ped_path_predictor/data/new_car/all_non_int_car.npy"


n_obs = 15
n_pred = 20
batch_size = 512
lr = 0.001

epoch_limit = 1000


class CVAE_CI3PP_Car_Only_Wrapper:
    def __init__(self, path=None):
        start_time_str = dt.today().strftime("%Y-%m-%d_%H-%M-%S")
        obs_str = f"obs{n_obs}_pred{n_pred}"
        self.base_path = f"./_out/{self.__class__.__name__}/{obs_str}/{start_time_str}"
        os.makedirs(self.base_path, exist_ok=True)

        self.logger = logging.getLogger(self.__class__.__name__)
        logging.basicConfig(
            level=logging.INFO,
            filename=f"{self.base_path}/train.log",
            format="%(asctime)s %(name)s: %(levelname)-8s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self.logger.addHandler(logging.StreamHandler(sys.stdout))

        self.logger.info("Model: SocialCi3pWrapperV3CVAE_CAR_ONLY")
        self.logger.info(f"Observation Frames: {n_obs}")
        self.logger.info(f"Prediction Frames: {n_pred}")
        self.logger.info(f"Batch Size: {batch_size}")
        self.logger.info(f"Initital Learning Rate: {lr}")

        self.model = CVAE_CI3PP_Car_Only(n_obs, n_pred).cuda()

        self.optimiser = optim.Adam(self.model.parameters(), lr=lr, eps=1e-4)
        self.optimiser_scheduler = MultiStepLR(
            self.optimiser, milestones=[5, 10, 15, 20], gamma=0.5, verbose=True,
        )

        if path is not None:
            self.model.load_state_dict(torch.load(path))

        self.train_loader, self.test_loader, self.val_loader = getDataloaders(
            path_int,
            path_non_int,
            path_int_car,
            path_non_int_car,
            n_obs,
            n_pred,
            batch_size=batch_size,
        )

        self.logger.info(f"Train-Batches {len(self.train_loader)}")
        self.logger.info(f"Test-Batches {len(self.test_loader)}")

    def transform(self, x, y):
        ped_in = x[:, :, :2].cuda()
        cf_in = x[:, :, 2:4].cuda()
        car_in = x[:, :, 4:].cuda()
        ego_out = y.cuda()
        return ped_in, cf_in, car_in, ego_out

    def l2_loss_fde(self, pred, data):
        fde_loss = torch.norm(
            (pred[:, -1, :, :2].transpose(0, 1) - data[:, -1, :2].unsqueeze(1)), 2, dim=-1,
        )
        ade_loss = (
            torch.norm(
                (pred[:, :, :, :2].transpose(1, 2) - data[:, :, :2].unsqueeze(0)), 2, dim=-1,
            )
            .mean(dim=2)
            .transpose(0, 1)
        )
        loss, min_inds = (fde_loss + ade_loss).min(dim=1)
        return loss.mean()

    def _compute_ego_errors(self, ego_preds, ego_gt, ego_in=None):
        with torch.no_grad():
            ego_gt = ego_gt.transpose(0, 1).unsqueeze(0)
            ade_losses = (
                torch.mean(
                    torch.norm(ego_preds[:, :, :, :2] - ego_gt[:, :, :, :2], 2, dim=-1), dim=1,
                )
                .transpose(0, 1)
                .cpu()
                .numpy()
            )
            fde_losses = (
                torch.norm(ego_preds[:, -1, :, :2] - ego_gt[:, -1, :, :2], 2, dim=-1)
                .transpose(0, 1)
                .cpu()
                .numpy()
            )

            a, f = (
                torch.square(ego_preds[:, :, :, :2] - ego_gt[:, :, :, :2])
                .sum(-1)
                .sqrt()
                .sum()
                .item(),
                torch.square(ego_preds[:, -1:, :, :2] - ego_gt[:, -1:, :, :2])
                .sum(-1)
                .sqrt()
                .sum()
                .item(),
            )

            some = False
            if some == True:
                # # make output relative to the last observed frame
                i_t = ego_in[:, 60 - 1 :, 0:2].detach().cpu().numpy()
                i_t = np.expand_dims(i_t, axis=1)
                i_t = np.repeat(i_t, 80, axis=1)
                i_t = i_t.squeeze(2)

                ego_gt = ego_gt.squeeze(0).transpose(0, 1)
                ego_gt = ego_gt[:, :, :2].cpu().numpy() + i_t

                ego_preds = ego_preds.squeeze(0).transpose(0, 1)
                ego_preds = ego_preds[:, :, :2].cpu().numpy() + i_t

                plt.plot(ego_in[0, :, 0].cpu().numpy(), ego_in[0, :, 1].cpu().numpy())
                plt.plot(ego_gt[0, :, 0], ego_gt[0, :, 1])
                plt.plot(ego_preds[0, :, 0], ego_preds[0, :, 1])
                plt.show()

        return ade_losses, fde_losses, a, f

    def train(self):
        # eval variables
        best_eval = np.inf
        best_eval_fde = np.inf
        last_best_epoch = 0

        for epoch in range(1000):
            print(f"Epoch {epoch}")
            did_epoch_better = False

            self.model.train()

            t_before = time.time()
            for i, (x, y) in enumerate(self.train_loader):
                print(f"\rBatch {i}/{len(self.train_loader)}", end="")

                ped_in, cf_in, car_in, ego_out = self.transform(x, y)

                pred_obs, mean, log_var = self.model(ped_in, car_in, ego_out)

                loss = self.l2_loss_fde(pred_obs.unsqueeze(0).transpose(1, 2), ego_out)

                kld_loss = torch.mean(
                    -0.5 * torch.sum(1 + log_var - mean**2 - log_var.exp(), dim=2), dim=0,
                )

                loss += kld_loss[0]

                # pred: [1, 80, 512, 2]
                # data: [512, 80, 5]
                self.optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                self.optimiser.step()

                if i % 100 == 0:
                    self.logger.info(
                        f"Epoch: {epoch:4}, Batch: {i:4} Loss: {loss:.6f} Time: {(time.time() - t_before): 4.4f}",
                    )
                    t_before = time.time()

                    eval_loss, fde_loss = self.eval(self.val_loader)

                    if eval_loss < best_eval and fde_loss < best_eval_fde:
                        best_eval = eval_loss
                        best_eval_fde = fde_loss
                        did_epoch_better = True
                        self.logger.info(f"Saving Model with loss:{eval_loss:.4f},{fde_loss:.4f}")
                        torch.save(self.model.state_dict(), self.base_path + f"/model_{epoch}.pth")
                    self.model.train()

            if did_epoch_better:
                self.logger.info(
                    f"Epoch {epoch} was better than last best epoch({last_best_epoch})",
                )
                last_best_epoch = epoch
            if epoch - last_best_epoch > 10:
                self.logger.info(
                    f"Stopping training, no improvement in 10 epochs saved{last_best_epoch}",
                )
                break
            self.optimiser_scheduler.step()

    def eval(self, dataloader):
        eval_loss = 0
        fde_loss = 0
        self.model.eval()
        with torch.no_grad():
            for j, (x_val, y_val) in enumerate(dataloader):
                print(f"\rBatch {j}/{len(dataloader)}", end="")
                ped_in, cf_in, car_in, ego_out = self.transform(x_val, y_val)

                pred_obs = self.model.inference(ped_in, car_in)

                ade_losses, fde_losses, a, f = self._compute_ego_errors(
                    pred_obs.unsqueeze(0).transpose(1, 2), ego_out, ego_in=ped_in,
                )

                eval_loss += a / n_pred
                fde_loss += f
        self.model.train()
        eval_loss /= len(dataloader) * batch_size
        fde_loss /= len(dataloader) * batch_size
        return eval_loss, fde_loss


if __name__ == "__main__":
    if "--test" in sys.argv:
        abw_eval = CVAE_CI3PP_Car_Only_Wrapper(
            path="./ped_path_predictor/saved_models/15_20/c_ci3pp_car_only.pth",
        )

        dl = singleDatasets(
            (
                "./ped_path_predictor/data/new_car/01_int_cleaned.npy",
                "./ped_path_predictor/data/new_car/01_int_cleaned_car.npy",
            ),
            n_obs,
            n_pred,
            512,
        )
        int_1_a, int_1_f = abw_eval.eval(dl)
        print(f"\nINT-1 {int_1_a:.4f} {int_1_f:.4f}")

        dl = singleDatasets(
            (
                "./ped_path_predictor/data/new_car/02_int_cleaned.npy",
                "./ped_path_predictor/data/new_car/02_int_cleaned_car.npy",
            ),
            n_obs,
            n_pred,
            512,
        )
        int_2_a, int_2_f = abw_eval.eval(dl)
        print(f"\nINT-2 {int_2_a:.4f} {int_2_f:.4f}")

        dl = singleDatasets(
            (
                "./ped_path_predictor/data/new_car/03_int_cleaned.npy",
                "./ped_path_predictor/data/new_car/03_int_cleaned_car.npy",
            ),
            n_obs,
            n_pred,
            512,
        )
        int_3_a, int_3_f = abw_eval.eval(dl)
        print(f"\nINT-3 {int_3_a:.4f} {int_3_f:.4f}")

        dl = singleDatasets(
            (
                "./ped_path_predictor/data/new_car/04_int_cleaned.npy",
                "./ped_path_predictor/data/new_car/04_int_cleaned_car.npy",
            ),
            n_obs,
            n_pred,
            512,
        )
        int_4_a, int_4_f = abw_eval.eval(dl)
        print(f"\nINT-1 {int_4_a:.4f} {int_4_f:.4f}")

        dl = singleDatasets(
            (
                "./ped_path_predictor/data/new_car/05_int_cleaned.npy",
                "./ped_path_predictor/data/new_car/05_int_cleaned_car.npy",
            ),
            n_obs,
            n_pred,
            512,
        )
        int_5_a, int_5_f = abw_eval.eval(dl)
        print(f"\nINT-5 {int_5_a:.4f} {int_5_f:.4f}")

        dl = singleDatasets(
            (
                "./ped_path_predictor/data/new_car/06_int_cleaned.npy",
                "./ped_path_predictor/data/new_car/06_int_cleaned_car.npy",
            ),
            n_obs,
            n_pred,
            512,
        )
        int_6_a, int_6_f = abw_eval.eval(dl)
        print(f"\nINT-6 {int_6_a:.4f} {int_6_f:.4f}")

        # print("\npure test", abw_eval.eval(abw_eval.test_loader))

        dl = singleDatasets(
            (
                "./ped_path_predictor/data/new_car/01_non_int_cleaned.npy",
                "./ped_path_predictor/data/new_car/01_non_int_cleaned_car.npy",
            ),
            n_obs,
            n_pred,
            512,
        )
        non_int_1_a, non_int_1_f = abw_eval.eval(dl)
        print(f"\nNON-INT-1 {non_int_1_a:.4f} {non_int_1_f:.4f}")

        dl = singleDatasets(
            (
                "./ped_path_predictor/data/new_car/02_non_int_cleaned.npy",
                "./ped_path_predictor/data/new_car/02_non_int_cleaned_car.npy",
            ),
            n_obs,
            n_pred,
            512,
        )
        non_int_2_a, non_int_2_f = abw_eval.eval(dl)
        print(f"\nNON-INT-2 {non_int_2_a:.4f} {non_int_2_f:.4f}")

        dl = singleDatasets(
            (
                "./ped_path_predictor/data/new_car/03_non_int_cleaned.npy",
                "./ped_path_predictor/data/new_car/03_non_int_cleaned_car.npy",
            ),
            n_obs,
            n_pred,
            512,
        )
        non_int_3_a, non_int_3_f = abw_eval.eval(dl)
        print(f"\nNON-INT-3 {non_int_3_a:.4f} {non_int_3_f:.4f}")

        dl = singleDatasets(
            (
                "./ped_path_predictor/data/new_car/04_non_int_cleaned.npy",
                "./ped_path_predictor/data/new_car/04_non_int_cleaned_car.npy",
            ),
            n_obs,
            n_pred,
            512,
        )
        non_int_4_a, non_int_4_f = abw_eval.eval(dl)
        print(f"\nNON-INT-4 {non_int_4_a:.4f} {non_int_4_f:.4f}")

        dl = singleDatasets(
            (
                "./ped_path_predictor/data/new_car/05_non_int_cleaned.npy",
                "./ped_path_predictor/data/new_car/05_non_int_cleaned_car.npy",
            ),
            n_obs,
            n_pred,
            512,
        )
        non_int_5_a, non_int_5_f = abw_eval.eval(dl)
        print(f"\nNON-INT-5 {non_int_5_a:.4f} {non_int_5_f:.4f}")

        dl = singleDatasets(
            (
                "./ped_path_predictor/data/new_car/06_non_int_cleaned.npy",
                "./ped_path_predictor/data/new_car/06_non_int_cleaned_car.npy",
            ),
            n_obs,
            n_pred,
            512,
        )
        non_int_6_a, non_int_6_f = abw_eval.eval(dl)
        print(f"\nNON-INT-6 {non_int_6_a:.4f} {non_int_6_f:.4f}")

        print("\n\n\n")
        print("INT")
        print(
            f"&CI3P+(CAR+AS+CVAE) & {int_1_a:.4f}/{int_1_f:.4f} & {int_2_a:.4f}/{int_2_f:.4f} & {int_3_a:.4f}/{int_3_f:.4f} & {int_4_a:.4f}/{int_4_f:.4f} & {int_5_a:.4f}/{int_5_f:.4f} & {int_6_a:.4f}/{int_6_f:.4f} \\\\",
        )
        print("NON-INT")
        print(
            f"&CI3P+(CAR+AS+CVAE) & {non_int_1_a:.4f}/{non_int_1_f:.4f} & {non_int_2_a:.4f}/{non_int_2_f:.4f} & {non_int_3_a:.4f}/{non_int_3_f:.4f} & {non_int_4_a:.4f}/{non_int_4_f:.4f} & {non_int_5_a:.4f}/{non_int_5_f:.4f} & {non_int_6_a:.4f}/{non_int_6_f:.4f} \\\\",
        )

    else:
        abw = CVAE_CI3PP_Car_Only_Wrapper()
        abw.train()
