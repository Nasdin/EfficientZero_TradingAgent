import time
import math
import torch
import torch.nn as nn
import numpy as np

from omegaconf import open_dict
from ez.agents.base import Agent
from ez.utils.format import DiscreteSupport, symexp
from ez.agents.models import EfficientZero
from ez.agents.models.base_model import ImproveResidualBlock, PNorm


################################################################################
# 1) Representation Network: 1D CNN + LSTM for multi-timeframe data
#    We assume input shape [B, T, num_features], e.g. T time-steps, multiple features.
#    - A small Conv1D stack over time dimension
#    - Then an LSTM to capture longer temporal patterns
################################################################################
class TradingRepresentationNetwork(nn.Module):
    """
    Takes (batch_size, seq_len, num_features) -> hidden_size vector.
    Perfect for multi-timeframe data (short horizon T).
    If T is large, consider deeper CNN layers or strided convolutions.
    """

    def __init__(
        self,
        num_features: int,
        seq_len: int,
        hidden_size: int = 128,
        cnn_channels: int = 64,
        lstm_hidden_size: int = 128,
        num_cnn_layers: int = 2,
        kernel_size: int = 3,
        use_bn: bool = True,
        value_prefix: bool = False
    ):
        super().__init__()
        self.num_features = num_features
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.lstm_hidden_size = lstm_hidden_size
        self.use_bn = use_bn
        self.value_prefix = value_prefix

        # 1D CNN over time dimension, input shape to CNN: (B, in_channels, seq_len)
        # We'll treat "in_channels = num_features"
        cnn_layers = []
        in_channels = num_features
        for _ in range(num_cnn_layers):
            out_channels = cnn_channels
            conv = nn.Conv1d(
                in_channels=in_channels, 
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2
            )
            ln = nn.BatchNorm1d(out_channels) if self.use_bn else nn.Identity()
            cnn_layers += [conv, ln, nn.ReLU()]
            in_channels = out_channels
        self.conv_stack = nn.Sequential(*cnn_layers)

        # MLP to reduce the (cnn_channels) dimension if needed
        self.post_cnn_fc = nn.Linear(cnn_channels, hidden_size)
        self.post_cnn_ln = nn.LayerNorm(hidden_size)

        # LSTM to handle sequential data across time
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=lstm_hidden_size, batch_first=True)
        self.lstm_ln = nn.LayerNorm(lstm_hidden_size)
        self.final_fc = nn.Linear(lstm_hidden_size, hidden_size)

    def forward(self, x):
        """
        x: shape [B, T, num_features]
        Returns: shape [B, hidden_size]
        """
        B, T, F = x.shape
        # CNN expects shape [B, in_channels, seq_len] => [B, num_features, T]
        x_cnn = x.transpose(1, 2)  # shape [B, F, T]
        feat = self.conv_stack(x_cnn)  # [B, cnn_channels, T]

        # Average-pool or last step across time dimension
        # Alternatively, keep the entire sequence for LSTM. We'll do LSTM on the *spatial* dimension T
        feat = feat.transpose(1, 2)  # back to [B, T, cnn_channels]
        feat = self.post_cnn_fc(feat)
        feat = self.post_cnn_ln(feat)
        feat = torch.relu(feat)

        # Feed into LSTM
        lstm_out, _ = self.lstm(feat)  # shape [B, T, lstm_hidden_size]
        # Take the last time-step
        final_h = lstm_out[:, -1, :]  # shape [B, lstm_hidden_size]
        final_h = self.lstm_ln(final_h)
        final_h = torch.relu(self.final_fc(final_h))  # [B, hidden_size]

        return final_h


################################################################################
# 2) Dynamics Network: Next latent state from (current_state, action)
#    For MuZero/EZ style, we typically embed the action and do some residual MLP
################################################################################
class TradingDynamicsNetwork(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        action_size: int,
        dyn_hidden_size: int = 128
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.action_size = action_size

        # If discrete, we assume one-hot action
        # If action_size=9, we feed (hidden_size + 9) -> next hidden
        self.fc1 = nn.Linear(hidden_size + action_size, dyn_hidden_size)
        self.ln1 = nn.LayerNorm(dyn_hidden_size)
        self.fc2 = nn.Linear(dyn_hidden_size, hidden_size)
        self.resblock = ImproveResidualBlock(hidden_size, hidden_size)

    def forward(self, state, action):
        """
        state: [B, hidden_size]
        action: [B, action_size] (assume one-hot)
        """
        x = torch.cat([state, action], dim=-1)
        x = self.fc1(x)
        x = self.ln1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        # Combine with skip connection (and optionally a second residual block).
        next_state = state + x
        next_state = self.resblock(next_state)
        return next_state


################################################################################
# 3) Reward Network: Predict symlog reward or distribution from next state.
#    If value_prefix = True, we can use an LSTM approach. Here we do a simpler MLP.
################################################################################
class TradingRewardNetwork(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        reward_support_size: int,
        value_prefix: bool = False,
        lstm_hidden_size: int = 128
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.value_prefix = value_prefix
        self.resblock = ImproveResidualBlock(hidden_size, hidden_size)
        self.ln = nn.LayerNorm(hidden_size)

        # If you want a simple MLP for reward:
        self.reward_head = nn.Linear(hidden_size, reward_support_size)

        # If you want LSTM-based prefix prediction, do it here:
        if self.value_prefix:
            self.lstm = nn.LSTM(hidden_size, lstm_hidden_size, batch_first=False)
            self.lstm_ln = nn.LayerNorm(lstm_hidden_size)
            self.lstm_head = nn.Linear(lstm_hidden_size, reward_support_size)

    def forward(self, state, reward_hidden=None):
        """
        state: [B, hidden_size]
        reward_hidden: optional LSTM hidden for prefix mode
        returns: (reward, reward_hidden)
        """
        x = self.resblock(state)
        x = self.ln(x)
        if self.value_prefix:
            # pass through LSTM
            x = x.unsqueeze(0)  # [1, B, hidden_size]
            x, reward_hidden = self.lstm(x, reward_hidden)  # [1, B, lstm_hidden_size]
            x = x.squeeze(0)
            x = self.lstm_ln(x)
            reward = self.lstm_head(x)
        else:
            reward = self.reward_head(x)

        return reward, reward_hidden


################################################################################
# 4) Value + Policy Network: produces discrete symlog value distribution & a policy
################################################################################
class TradingValuePolicyNetwork(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        value_support_size: int,
        action_size: int,
        v_num: int = 1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.value_support_size = value_support_size
        self.action_size = action_size
        self.v_num = v_num

        # Value path
        self.val_resblock = ImproveResidualBlock(hidden_size, hidden_size)
        self.val_ln = nn.LayerNorm(hidden_size)
        self.value_heads = nn.ModuleList([
            nn.Linear(hidden_size, value_support_size) for _ in range(v_num)
        ])

        # Policy path
        self.pi_resblock = ImproveResidualBlock(hidden_size, hidden_size)
        self.pi_ln = nn.LayerNorm(hidden_size)
        self.policy_head = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        # Value
        v = self.val_resblock(state)
        v = self.val_ln(v)
        values = []
        for val_head in self.value_heads:
            values.append(val_head(v))  # each shape [B, value_support_size]
        # shape: [v_num, B, value_support_size]
        values = torch.stack(values, dim=0)

        # Policy
        p = self.pi_resblock(state)
        p = self.pi_ln(p)
        policy = self.policy_head(p)  # [B, action_size]

        return values, policy


################################################################################
# 5) Optional Projection Modules (for latent consistency or self-supervised regularization)
################################################################################
class ProjectionNetwork(nn.Module):
    def __init__(self, hidden_size, proj_hidden_size, proj_output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, proj_hidden_size),
            nn.LayerNorm(proj_hidden_size),
            nn.ReLU(),
            nn.Linear(proj_hidden_size, proj_hidden_size),
            nn.LayerNorm(proj_hidden_size),
            nn.ReLU(),
            nn.Linear(proj_hidden_size, proj_output_size),
            nn.LayerNorm(proj_output_size)
        )

    def forward(self, x):
        return self.net(x)


class ProjectionHeadNetwork(nn.Module):
    def __init__(self, proj_output_size, head_hidden_size, head_output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(proj_output_size, head_hidden_size),
            nn.LayerNorm(head_hidden_size),
            nn.ReLU(),
            nn.Linear(head_hidden_size, head_output_size)
        )

    def forward(self, x):
        return self.net(x)


################################################################################
# 6) TradingEfficientZero: Wrap the above modules into a full EZ model
################################################################################
class TradingEfficientZero(EfficientZero):
    """
    Extends the standard EfficientZero to incorporate the trading modules.
    Uses:
      - representation_net: 1D CNN + LSTM
      - dynamics_net
      - reward_net
      - value_policy_net
      - projection_net + projection_head_net (optional)
    """
    def __init__(
        self,
        representation_net,
        dynamics_net,
        reward_net,
        value_policy_net,
        projection_net,
        projection_head_net,
        config,
        state_norm=False,
        value_prefix=False
    ):
        super().__init__(
            representation_net,
            dynamics_net,
            reward_net,
            value_policy_net,
            projection_net,
            projection_head_net,
            config,
            state_norm=state_norm,
            value_prefix=value_prefix
        )
        # You can override additional logic here if needed for trading domain.


################################################################################
# 7) TradingAgent: final class that sets up config, builds the model, etc.
################################################################################
class TradingAgent(Agent):
    """
    Usage:
      1) Construct with a config that has relevant fields:
         - config.env.obs_shape = [channels, ...] or (?)
         - config.env.num_features, config.env.seq_len, config.env.action_space_size, etc.
         - config.model.*
         - config.train.*
      2) The environment or data loader should produce observations shaped like (B, T, num_features).
      3) The reward / value are symlog discrete distributions for big PnL ranges.
    """

    def __init__(self, config):
        super().__init__(config)
        self.update_config()
        self._update = True

        # pull from config
        self.num_features = self.config.env.num_features
        self.seq_len = self.config.env.seq_len  # multi-timeframe length
        self.action_space_size = self.config.env.action_space_size
        self.hidden_size = self.config.model.hidden_size
        self.lstm_hidden_size = self.config.model.lstm_hidden_size
        self.cnn_channels = self.config.model.cnn_channels
        self.num_cnn_layers = self.config.model.num_cnn_layers
        self.kernel_size = self.config.model.kernel_size
        self.value_prefix = self.config.model.value_prefix

        # discrete symlog support, e.g. range +/- large
        self.config.model.reward_support.size = 601
        self.config.model.reward_support.type = "symlog"
        self.config.model.value_support.size = 601
        self.config.model.value_support.type = "symlog"

        # Just an example: you can set these from config or tune
        self.reward_support_size = self.config.model.reward_support.size
        self.value_support_size = self.config.model.value_support.size

    def update_config(self):
        """
        We can discover environment shape, set synergy, and finalize paths.
        """
        assert not self._update, "update_config() should be called only once."
        localtime = time.strftime('%Y-%m-%d %H:%M:%S')
        tag = '{}-seed={}-{}/'.format(self.config.tag, self.config.env.base_seed, localtime)

        with open_dict(self.config):
            # Example environment shape definitions
            # config.env.num_features, config.env.seq_len, config.env.action_space_size, etc.
            self.config.save_path += tag

    def build_model(self):
        """
        Assembles the full EfficientZero for trading from our custom modules.
        """
        # Representation: 1D CNN + LSTM
        rep_net = TradingRepresentationNetwork(
            num_features=self.num_features,
            seq_len=self.seq_len,
            hidden_size=self.hidden_size,
            cnn_channels=self.cnn_channels,
            lstm_hidden_size=self.lstm_hidden_size,
            num_cnn_layers=self.num_cnn_layers,
            kernel_size=self.kernel_size,
            use_bn=self.config.model.use_bn,
            value_prefix=self.value_prefix
        )

        dyn_net = TradingDynamicsNetwork(
            hidden_size=self.hidden_size,
            action_size=self.action_space_size,
            dyn_hidden_size=128
        )

        rew_net = TradingRewardNetwork(
            hidden_size=self.hidden_size,
            reward_support_size=self.reward_support_size,
            value_prefix=self.value_prefix,
            lstm_hidden_size=self.lstm_hidden_size
        )

        val_pol_net = TradingValuePolicyNetwork(
            hidden_size=self.hidden_size,
            value_support_size=self.value_support_size,
            action_size=self.action_space_size,
            v_num=self.config.train.v_num
        )

        # Projection networks for consistency or representation learning
        proj_net = ProjectionNetwork(
            hidden_size=self.hidden_size,
            proj_hidden_size=self.config.model.proj_hidden_size,
            proj_output_size=self.config.model.proj_output_size
        )
        proj_head_net = ProjectionHeadNetwork(
            proj_output_size=self.config.model.proj_output_size,
            head_hidden_size=self.config.model.pred_hidden_size,
            head_output_size=self.config.model.pred_output_size
        )

        model = TradingEfficientZero(
            representation_net=rep_net,
            dynamics_net=dyn_net,
            reward_net=rew_net,
            value_policy_net=val_pol_net,
            projection_net=proj_net,
            projection_head_net=proj_head_net,
            config=self.config,
            state_norm=self.config.model.state_norm,
            value_prefix=self.value_prefix
        )
        return model

    # You can override or replicate your existing .train(), .update_weights(), etc.
    # referring to your Atari/DMC training flow, with synergy for synergy.


################################################################################
# End of TradingAgent. You can now plug this Agent into your training pipeline.
################################################################################

#
# Example config snippet (pseudocode):
#
# config = {
#   "tag": "TradingRun",
#   "env": {
#       "base_seed": 42,
#       "num_features": 64,
#       "seq_len": 16,
#       "action_space_size": 9
#   },
#   "model": {
#       "hidden_size": 128,
#       "lstm_hidden_size": 128,
#       "cnn_channels": 64,
#       "num_cnn_layers": 2,
#       "kernel_size": 3,
#       "use_bn": True,
#       "value_prefix": False,
#       "state_norm": False,
#       "proj_hidden_size": 128,
#       "proj_output_size": 64,
#       "pred_hidden_size": 64,
#       "pred_output_size": 64,
#       "reward_support": {"size": 601, "type": "symlog"},
#       "value_support": {"size": 601, "type": "symlog"},
#   },
#   "train": {
#       "v_num": 1,
#       ...
#   }
# }
#
# agent = TradingAgent(config)
# agent.train(...)  # adapt from your existing MuZero/EZ training pipeline
#
