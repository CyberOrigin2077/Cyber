"""
Aspect of the code inspired/taken from:
google-research/planet: https://github.com/google-research/planet
planet-torch: https://github.com/abhayraw1/planet-torch
"""

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812 # this is the convention used in PyTorch
from torch.distributions import Normal, kl_divergence

from typing import Callable, Dict

from cyber.models.world import DynamicModel
from cyber.models.world.dynamic.rssm.visual_encoder import VisualEncoder, VisualDecoder

import tqdm


class RecurrentStateSpaceModel(DynamicModel):
    """
    Deterministic and stochastic state model as described in
    Learning Latent Dynamics for Planning from Pixels [https://arxiv.org/abs/1811.04551]

    The deterministic hidden state is computed from the previous hidden state and action.

    The stochastic latent is computed from the hidden state at the same time
    step. If an observation is present, the posterior latent is compute from both
    the hidden state and the observation.

    All three models are parameterized by nueral networks.

    Prior:    Posterior:

       (a)       (a)
          \         \
          v         v
    [h]->[h]  [h]->[h]
        ^ |       ^ :
       /  v      /  v
    (s)  (s)  (s)  (s)
                    ^
                    :
                   (o)
    a: action, h: hidden state, s: latent state, o: observation

    """  # noqa: W605

    def __init__(
        self,
        hidden_size,
        action_size,
        latent_size,
        embed_size,
        layer_width,
        activation_function="relu",
        embed_num_layers=3,
        latent_num_layers=3,
        min_stddev=0.1,
        encoder_type="visual",
        encoder_kwargs=None,
        decoder_kwargs=None,
        **kwargs,
    ):
        """
        Args:
            hidden_size (int): Dimension of the hidden state.
            action_size (int): Dimension of the action space.
            latent_size (int): Dimension of the latent state.
            embed_size (int): Model input embedding size. i.e. the size of the input to the RNN.
            activation_function (str): Activation function to use for FC layers.
            embed_num_layers (int): Number of layers in the embedding network.
            latent_num_layers (int): Number of layers in the latent networks.
            min_stddev (float): Minimum standard deviation for the latent distributions.
            encoder_type (str): one of ['visual'].
            encoder_kwargs (dict): kwargs for the encoder.
            decoder_kwargs (dict): kwargs for the decoder.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.action_size = action_size
        self.latent_size = latent_size
        self.embed_size = embed_size
        self.act_fn = getattr(F, activation_function)
        self.min_stddev = min_stddev
        # TODO: transformer encoder decoder
        if encoder_kwargs is None:
            encoder_kwargs = {}
        if decoder_kwargs is None:
            decoder_kwargs = {}
        if encoder_type == "visual":
            self.encoder = VisualEncoder(embed_size, **encoder_kwargs)
            self.decoder = VisualDecoder(hidden_size + latent_size, **decoder_kwargs)

        # Deterministic hidden state model
        self.rnn_embed = nn.ModuleList(
            [nn.Linear(action_size + hidden_size, layer_width), *[nn.Linear(layer_width, layer_width) for _ in range(embed_num_layers - 1)]]
        )
        self.grucell: nn.GRUCell = nn.GRUCell(layer_width, hidden_size)

        # Stochastic prior latent state model
        self.fc_prior_z = nn.ModuleList([nn.Linear(hidden_size, layer_width)])
        for _ in range(latent_num_layers - 1):
            self.fc_prior_z.append(nn.Linear(layer_width, layer_width))
        self.fc_prior_mean = nn.Linear(layer_width, latent_size)
        self.fc_prior_std_dev = nn.Linear(layer_width, latent_size)

        # Stochastic posterior latent state model
        self.fc_posterior_z = nn.ModuleList([nn.Linear(hidden_size + embed_size, layer_width)])
        for _ in range(latent_num_layers - 1):
            self.fc_posterior_z.append(nn.Linear(layer_width, layer_width))
        self.fc_posterior_mean = nn.Linear(layer_width, latent_size)
        self.fc_posterior_std_dev = nn.Linear(layer_width, latent_size)

    def get_init_state(self, enc, h_t=None, s_t=None, a_t=None, use_mle=False):
        """Returns the initial posterior given the observation."""
        n, dev = enc.size(0), enc.device
        h_t = torch.zeros(n, self.hidden_size).to(dev) if h_t is None else h_t
        s_t = torch.zeros(n, self.latent_size).to(dev) if s_t is None else s_t
        a_t = torch.zeros(n, self.action_size).to(dev) if a_t is None else a_t
        h_tp1 = self.deterministic_state_fwd(h_t, s_t, a_t)
        if use_mle:
            s_tp1, _ = self.state_posterior(h_t, enc)
        else:
            s_tp1 = self.state_posterior(h_t, enc, sample=True)
        return h_tp1, s_tp1

    def deterministic_state_fwd(self, h_t, s_t, a_t):
        """
        Returns the deterministic state given the previous hidden state, latent state, and action.
        """
        gru_input = torch.cat([s_t, a_t], dim=-1)
        for layer in self.rnn_embed:
            gru_input = self.act_fn(layer(gru_input))
        return self.grucell.forward(gru_input, h_t)

    def state_prior(self, h_t, sample=False):
        """
        Returns the prior latent state given the hidden state.
        """
        z = h_t
        for layer in self.fc_prior_z:
            z = self.act_fn(layer(z))
        mean = self.fc_prior_mean(z)
        stddev = F.softplus(self.fc_prior_std_dev(z)) + self.min_stddev
        if sample:
            return mean + torch.randn_like(mean) * stddev
        return mean, stddev

    def state_posterior(self, h_t, e_t, sample=False):
        """
        Returns the posterior latent state given the hidden state and observation.
        """
        z = torch.cat([h_t, e_t], dim=-1)
        for layer in self.fc_posterior_z:
            z = self.act_fn(layer(z))
        mean = self.fc_posterior_mean(z)
        stddev = F.softplus(self.fc_posterior_std_dev(z)) + self.min_stddev
        if sample:
            return mean + torch.randn_like(mean) * stddev
        return mean, stddev

    def forward_method(self, inputs: Dict[str, torch.Tensor], frames_to_generate: int, use_mle=False, *args, **kwargs):
        """
        The inference method of a RSSM.
        Given the inputs (hidden states and actions), generate the next states.
        Note that the latent states are sampled from the prior distribution, so the generated states are not deterministic unless
        use_mle is set to True.

        Args:
            inputs(Dict[str, torch.Tensor]): dictionary containing the following items:
                'initial'(torch.Tensor): size(batchsize, hidden_size+latent_size) the initial hidden states and latent states
                'actions'(torch.Tensor): size(batchsize, frames_to_generate, action_size) the actions
            frames_to_generate(int): number of new frames to generate autoregressively
            use_mle(bool): whether to use maximum likelihood estimation to generate the next states i.e. use mean of the prior distribution

        Returns:
            torch.Tensor: shape(frames_to_generate, hidden_size+latent_size) generated frames
        """
        assert (
            inputs["initial"].size(1) == self.hidden_size + self.latent_size
        ), "The size of the initial hidden states should be the sum of the hidden size and latent size"
        assert inputs["actions"].size(2) == self.action_size, "The size of the actions should be the same as the action size of the model"
        hidden_states = inputs["initial"][:, : self.hidden_size]
        latent_states = inputs["initial"][:, self.hidden_size :]
        actions = inputs["actions"]
        assert actions.size(1) == frames_to_generate, "The number of frames to generate should be the same as the number of actions"
        assert hidden_states.size(0) == actions.size(0), "The batch size of the initial hidden states should be the same as the batch size of the actions"

        predictions = []
        for step in tqdm.tqdm(range(frames_to_generate), desc="Generating frames"):
            action = actions[:, step, :]
            hidden_states = self.deterministic_state_fwd(hidden_states, latent_states, action)
            if use_mle:
                latent_states, _ = self.state_prior(hidden_states, sample=False)
            else:
                latent_states = self.state_prior(hidden_states, sample=True)
            predictions.append(torch.cat([hidden_states, latent_states], dim=-1))
        return torch.stack(predictions, dim=1)

    def compute_training_loss(self, episodes: Dict[str, torch.Tensor], free_nats: int = 3, kl_scale=1.0, *args, **kwargs):
        """
        Compute the training loss as defined by the lower bound of the single-step predictive distribution.

        Args:
            episodes([dict]): dictionary containing the following items:
                'actions'(torch.Tensor): size(batchsize, ep_length, action_size) the actions
                'observations'(torch.Tensor): size(batchsize, ep_length, **observation_size) the observations
            free_nats(int): as described in the paper, the number of nats to consider the KL divergence as free
            kl_scale(float): scaling factor for the KL divergence. The original paper did not use this scaling factor and uses free_nats instead
        """
        # TODO: latent overshooting
        observations = episodes["observations"]
        actions = episodes["actions"]
        assert observations.size(0) == actions.size(0), "The batch size of the observations should be the same as the batch size of the actions"
        assert observations.size(1) == actions.size(1), "The number of frames in the observations should be the same as the number of frames in the actions"
        h, s = self.get_init_state(self.encoder(episodes["observations"][:, 0]))  # initial hidden and latent states
        loss = 0
        for t in range(observations.size(1) - 1):
            # forward step
            h = self.deterministic_state_fwd(h, s, actions[:, t])
            prior_mean, prior_std = self.state_prior(h)
            posterior_mean, posterior_std = self.state_posterior(h, self.encoder(observations[:, t + 1]))
            prior = Normal(prior_mean, prior_std)
            posterior = Normal(posterior_mean, posterior_std)
            kl_div = kl_divergence(prior, posterior).sum(-1)
            kl_div = torch.max(kl_div, torch.tensor(free_nats).to(kl_div.device))
            s = posterior.sample()
            rec_loss = F.mse_loss(
                self.decoder(torch.cat([h, s], dim=-1)), observations[:, t]
            )  # mse loss entails the assumption of gaussian with identity covariance
            loss += rec_loss + kl_div * kl_scale
        return loss

    def get_train_collator(self, *args, **kwargs) -> Callable:
        raise NotImplementedError()
