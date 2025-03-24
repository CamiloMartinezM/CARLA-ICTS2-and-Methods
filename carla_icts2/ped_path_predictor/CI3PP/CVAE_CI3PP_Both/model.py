import torch
from torch import nn

from P3VI.model import LinearReLu, TimeDistributed


class LinearTanh(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearTanh, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.linear(x)
        return self.activation(x)


class CVAE_CI3PP_Both(nn.Module):
    def __init__(self, n_observed_frames, n_predict_frames):
        super(CVAE_CI3PP_Both, self).__init__()
        self.n_pred = n_predict_frames
        self.n_obs = n_observed_frames

        embedded = 32
        self.latent_dim = 32

        # CVAE STUFF
        self.y_encoder = nn.Sequential(
            TimeDistributed(LinearTanh(2, 128)), nn.GRU(128, 256, batch_first=True),
        )
        self.mu = nn.Linear(256 + 128 * 3, self.latent_dim)
        self.var = nn.Linear(256 + 128 * 3, self.latent_dim)

        # Trajectory Embedder
        self.embedder_traj = TimeDistributed(LinearReLu(2, embedded), batch_first=True)
        self.embedder_cf = TimeDistributed(LinearReLu(2, embedded), batch_first=True)
        self.embedder_car = TimeDistributed(LinearReLu(2, embedded), batch_first=True)

        # Cross Attention
        self.mha_traj_x_cf = nn.MultiheadAttention(
            embed_dim=embedded, num_heads=4, batch_first=True,
        )
        self.mha_traj_x_car = nn.MultiheadAttention(
            embed_dim=embedded, num_heads=4, batch_first=True,
        )

        self.mha_cf_x_traj = nn.MultiheadAttention(
            embed_dim=embedded, num_heads=4, batch_first=True,
        )
        self.mha_cf_x_car = nn.MultiheadAttention(
            embed_dim=embedded, num_heads=4, batch_first=True,
        )

        self.mha_car_x_traj = nn.MultiheadAttention(
            embed_dim=embedded, num_heads=4, batch_first=True,
        )
        self.mha_car_x_cf = nn.MultiheadAttention(
            embed_dim=embedded, num_heads=4, batch_first=True,
        )

        # ENCODER
        self.traj_gru = nn.GRU(input_size=embedded * 2, hidden_size=128, batch_first=True)
        self.cf_gru = nn.GRU(input_size=embedded * 2, hidden_size=128, batch_first=True)
        self.car_gru = nn.GRU(input_size=embedded * 2, hidden_size=128, batch_first=True)

        # DECODER
        self.decoder_linear = LinearReLu(3 * 128 + self.latent_dim, 128)
        self.decoder_gru = nn.GRU(input_size=128, hidden_size=128, batch_first=True)

        self.prediction_head = TimeDistributed(nn.Linear(128, 2), batch_first=True)

    def forward(self, x_traj, x_cf, x_car, y_traj):
        # Embedding
        x_traj = self.embedder_traj(x_traj)
        x_cf = self.embedder_cf(x_cf)
        x_car = self.embedder_car(x_car)

        # Cross Attention
        mh_traj_x_cf, _ = self.mha_traj_x_cf(x_traj, x_cf, x_cf)
        mh_traj_x_car, _ = self.mha_traj_x_car(x_traj, x_car, x_car)

        mh_cf_x_traj, _ = self.mha_cf_x_traj(x_cf, x_traj, x_traj)
        mh_cf_x_car, _ = self.mha_cf_x_car(x_cf, x_car, x_car)

        mh_car_x_traj, _ = self.mha_car_x_traj(x_car, x_traj, x_traj)
        mh_car_x_cf, _ = self.mha_car_x_cf(x_car, x_cf, x_cf)

        # Stacked
        stacked_car = torch.cat([mh_cf_x_car, mh_traj_x_car], dim=-1)
        stacked_cf = torch.cat([mh_car_x_cf, mh_traj_x_cf], dim=-1)
        stacked_traj = torch.cat([mh_cf_x_traj, mh_car_x_traj], dim=-1)

        # Encoder
        _, traj_gru = self.traj_gru(stacked_traj)
        _, cf_gru = self.cf_gru(stacked_cf)
        _, car_gru = self.car_gru(stacked_car)

        _, y_enc = self.y_encoder(y_traj)

        # Decoder
        stacked = torch.cat([traj_gru, cf_gru, car_gru], dim=-1).transpose(0, 1)
        cat_x = torch.cat((stacked, y_enc.transpose(0, 1)), dim=-1)
        mean = self.mu(cat_x)
        log_var = self.var(cat_x)
        z = self.sample(mean, log_var)
        decoder_x = torch.cat((stacked, z), dim=-1)

        decoded_lin = self.decoder_linear(decoder_x).repeat(1, self.n_pred, 1)
        decoded_gru, _ = self.decoder_gru(decoded_lin)

        pred = self.prediction_head(decoded_gru)

        return pred, mean, log_var

    def inference(self, x_traj, x_cf, x_car):
        z = torch.normal(torch.zeros((self.latent_dim,)), torch.ones((self.latent_dim,))).cuda()

        # Embedding
        x_traj = self.embedder_traj(x_traj)
        x_cf = self.embedder_cf(x_cf)
        x_car = self.embedder_car(x_car)

        # Cross Attention
        mh_traj_x_cf, _ = self.mha_traj_x_cf(x_traj, x_cf, x_cf)
        mh_traj_x_car, _ = self.mha_traj_x_car(x_traj, x_car, x_car)

        mh_cf_x_traj, _ = self.mha_cf_x_traj(x_cf, x_traj, x_traj)
        mh_cf_x_car, _ = self.mha_cf_x_car(x_cf, x_car, x_car)

        mh_car_x_traj, _ = self.mha_car_x_traj(x_car, x_traj, x_traj)
        mh_car_x_cf, _ = self.mha_car_x_cf(x_car, x_cf, x_cf)

        # Stacked
        stacked_car = torch.cat([mh_cf_x_car, mh_traj_x_car], dim=-1)
        stacked_cf = torch.cat([mh_car_x_cf, mh_traj_x_cf], dim=-1)
        stacked_traj = torch.cat([mh_cf_x_traj, mh_car_x_traj], dim=-1)

        # Encoder
        _, traj_gru = self.traj_gru(stacked_traj)
        _, cf_gru = self.cf_gru(stacked_cf)
        _, car_gru = self.car_gru(stacked_car)

        # Decoder
        stacked = torch.cat([traj_gru, cf_gru, car_gru], dim=-1).transpose(0, 1)
        z = z.unsqueeze(dim=0).unsqueeze(dim=1).repeat(stacked.shape[0], 1, 1)
        decoder_x = torch.cat((stacked, z), dim=-1)

        decoded_lin = self.decoder_linear(decoder_x).repeat(1, self.n_pred, 1)
        decoded_gru, _ = self.decoder_gru(decoded_lin)

        pred = self.prediction_head(decoded_gru)
        return pred

    def sample(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
