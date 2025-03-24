import torch
from torch import nn

from P3VI.model import LinearReLu, TimeDistributed


class CI3PP_COG_Only(nn.Module):
    def __init__(self, n_observed_frames, n_predict_frames):
        super(CI3PP_COG_Only, self).__init__()
        self.n_pred = n_predict_frames
        self.n_obs = n_observed_frames

        embedded = 32

        # Trajectory Embedder
        self.embedder_traj = TimeDistributed(LinearReLu(2, embedded), batch_first=True)
        self.embedder_cf = TimeDistributed(LinearReLu(2, embedded), batch_first=True)

        # Cross Attention
        self.mha_traj_x_cf = nn.MultiheadAttention(
            embed_dim=embedded, num_heads=4, batch_first=True,
        )
        self.mha_cf_x_traj = nn.MultiheadAttention(
            embed_dim=embedded, num_heads=4, batch_first=True,
        )

        # ENCODER
        self.traj_gru = nn.GRU(input_size=embedded, hidden_size=128, batch_first=True)
        self.cf_gru = nn.GRU(input_size=embedded, hidden_size=128, batch_first=True)

        # DECODER
        self.decoder_linear = LinearReLu(2 * 128, 128)
        self.decoder_gru = nn.GRU(input_size=128, hidden_size=128, batch_first=True)

        self.prediction_head = TimeDistributed(nn.Linear(128, 2), batch_first=True)

    def forward(self, x_traj, x_cf):
        # Embedding
        x_traj = self.embedder_traj(x_traj)
        x_cf = self.embedder_cf(x_cf)

        # Cross Attention
        mh_traj_x_cf, _ = self.mha_traj_x_cf(x_traj, x_cf, x_cf)
        mh_cf_x_traj, _ = self.mha_cf_x_traj(x_cf, x_traj, x_traj)

        # Encoder
        _, traj_gru_f = self.traj_gru(mh_traj_x_cf)
        _, cf_gru_f = self.cf_gru(mh_cf_x_traj)

        # Decoder
        stacked = torch.cat([traj_gru_f, cf_gru_f], dim=-1).transpose(0, 1)

        decoded_lin = self.decoder_linear(stacked).repeat(1, self.n_pred, 1)
        decoded_gru, _ = self.decoder_gru(decoded_lin)

        pred = self.prediction_head(decoded_gru)

        return pred
