import numpy as np
import torch
from torch.utils.data import Dataset


def load_data(file, n_oberserved_frames, n_predict_frames):
    with open(file, "rb") as f:
        raw = np.load(f, allow_pickle=True)

    enum_conv = lambda t: t.value
    vfunc = np.vectorize(enum_conv)
    raw[:, :, 2] = vfunc(raw[:, :, 2])
    raw[:, :, 3] = vfunc(raw[:, :, 3])
    raw = raw.astype(np.float32)
    # print(raw[1][0:200])
    window = raw.shape[1] - n_oberserved_frames - n_predict_frames

    observed_data, predict_data = [], []
    for k in range(0, window, 2):
        observed = raw[:, k : n_oberserved_frames + k, :]
        pred = raw[:, k + n_oberserved_frames : n_predict_frames + n_oberserved_frames + k, 0:2]

        observed_data.append(observed)
        predict_data.append(pred)

    # rng = np.random.default_rng(seed=42)
    # observed_data = rng.permutation(observed_data)
    # rng = np.random.default_rng(seed=42)
    # predict_data = rng.permutation(predict_data)
    # print(observed_data[0])
    # print(predict_data[0])
    observed_data = np.concatenate(observed_data, axis=0)
    predict_data = np.concatenate(predict_data, axis=0)
    # print(observed_data.shape)
    # print(predict_data.shape)
    # torch.tensor(predict_data)
    # print(observed_data[10])

    return torch.tensor(observed_data), torch.tensor(predict_data)


class SIMP3Dataset(Dataset):
    def __init__(self, file, n_oberserved_frames, n_predict_frames, split=[0, 0.8]) -> None:
        super().__init__()
        self.observed_data, self.predict_data = load_data(
            file, n_oberserved_frames, n_predict_frames,
        )
        n = len(self.observed_data)
        bottom = int(n * split[0])
        top = int(n * split[1])
        self.observed_data = self.observed_data[bottom:top]
        self.predict_data = self.predict_data[bottom:top]

    def __len__(self):
        return len(self.observed_data)

    def __getitem__(self, idx):
        return self.observed_data[idx], self.predict_data[0]


# path = "./SIMP3/data/01_int_test.npy"
# train_dataset = SIMP3Dataset(path, 40, 30, split = [0,0.8])
# val_dataset = SIMP3Dataset(path, 40, 30, [0.8,1.0])

# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)

# for i,(input, output) in enumerate(train_loader):
#    print(i, input.shape, output.shape)

# for i,(input, output) in enumerate(val_loader):
#    print(i, input.shape, output.shape)

# load_data("test.npy",40,30)
