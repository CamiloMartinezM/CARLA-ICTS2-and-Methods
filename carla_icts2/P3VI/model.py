import torch
from torch import nn


class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        # print(type(x))
        # print(x.size())
        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)
        y = self.module(x_reshape)
        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)
        return y


class LinearReLu(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearReLu, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        return self.activation(x)


class P3VI(nn.Module):
    def __init__(self, n_oberserved_frames, n_predict_frames):
        super(P3VI, self).__init__()
        self.n_predict_frames = n_predict_frames

        self.attention_cf = TimeDistributed(
            LinearReLu(2, 50), batch_first=False,
        )  # input batch x timesteps x 2
        self.attention_traj = TimeDistributed(
            LinearReLu(2, 50), batch_first=False,
        )  # input batch x timesteps x 2

        self.encoder_cf = nn.GRU(input_size=50, hidden_size=128)
        self.encoder_traj = nn.GRU(input_size=50, hidden_size=128)

        self.decoder_linear = LinearReLu(256, 128)
        self.decoder_lstm = nn.GRU(input_size=128, hidden_size=128)

        self.prediction_head = TimeDistributed(nn.Linear(128, 2))

    def forward(self, x_tray, x_cf):
        att_cf = self.attention_cf(x_cf)
        att_tray = self.attention_traj(x_tray)
        _, enc_cf = self.encoder_cf(att_cf)
        _, enc_traj = self.encoder_traj(att_tray)

        # enc_cf = enc_cf[:,-1,:]
        # enc_traj = en c_traj[:,-1,:]
        # print(enc_cf.shape)
        stacked = torch.cat((enc_cf, enc_traj), dim=-1)[
            0
        ]  # torch.concatenate([enc_cf,enc_traj], axis=0)
        # print(stacked.shape)
        decoded_lin = self.decoder_linear(stacked).repeat(self.n_predict_frames, 1, 1)
        # print(decoded_lin.shape)
        decoded_lstm, _ = self.decoder_lstm(decoded_lin)

        pred = self.prediction_head(decoded_lstm)

        return pred
