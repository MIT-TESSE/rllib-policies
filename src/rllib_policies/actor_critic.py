from typing import Any, Dict, List, Tuple

import gym
import torch
import torch.nn as nn
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from torch import Tensor

from rllib_policies.base import RllibPolicy


class ActorCritic(TorchModelV2, RllibPolicy):
    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: Dict[str, Any],
        name: str,
        dense_layers=[
            512,
        ],
        **network_args: Dict[str, Any],
    ):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)
        RllibPolicy.__init__(self, obs_space, **network_args)
        dense_layers = [self.nets_out_feature_size] + dense_layers

        self.linear = nn.Sequential(
            *[
                nn.Linear(dense_layers[i], dense_layers[i + 1])
                for i in range(len(dense_layers) - 1)
            ]
        )

        self.policy = nn.Linear(dense_layers[-1], num_outputs)
        self.value_branch = nn.Linear(dense_layers[-1], 1)

    @override(TorchModelV2)
    def forward(
        self,
        input_dict: Dict[str, Tensor],
        state: List[Tensor],
        seq_len: Tensor,
    ) -> Tuple[Tensor, List[Tensor]]:
        linear_in = self.forward_nets(input_dict["obs"], rnn_input=False)
        self._features = self.linear(linear_in)
        return self.policy(self._features), state

    @override(ModelV2)
    def value_function(self):
        return torch.reshape(self.value_branch(self._features), [-1])


class RNNActorCritic(TorchRNN, RllibPolicy):
    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: Dict[str, Any],
        name: str,
        dense_layers: List[int] = [],
        hidden_size: int = 512,
        rnn_type: str = "LSTM",
        **network_args: Dict[str, Any],
    ):
        TorchRNN.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)
        RllibPolicy.__init__(self, obs_space, **network_args)
        dense_layers = [self.nets_out_feature_size] + dense_layers

        self.linear = nn.Sequential(
            *[
                nn.Linear(dense_layers[i], dense_layers[i + 1])
                for i in range(len(dense_layers) - 1)
            ]
        )

        rnn_input_size = dense_layers[-1]
        self.rnn_state_size = hidden_size
        self.rnn = getattr(nn, rnn_type)(
            input_size=rnn_input_size, hidden_size=self.rnn_state_size, batch_first=True
        )

        self.policy = nn.Linear(self.rnn_state_size, num_outputs)
        self.value_branch = nn.Linear(self.rnn_state_size, 1)
        self._features = None

    @override(TorchRNN)
    def forward_rnn(
        self, input: Tensor, state: List[Tensor], seq_len: Tensor
    ) -> Tuple[Tensor, List[Tensor]]:
        base_features = self.forward_nets(input, rnn_input=True)
        base_features = self.linear(base_features)

        # the CNN combines batch and time indices for inference
        # this needs to be decoupled for RNN
        base_features_time_ranked = torch.reshape(
            base_features,
            [input.shape[0], input.shape[1], base_features.shape[-1]],
        )

        # appears that it's necessary to move `seq_len` to cpu in rllib v1.x
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(
            base_features_time_ranked,
            seq_len.to("cpu"),
            batch_first=True,
            enforce_sorted=False,
        )

        self._features, state = self.rnn(
            packed_input, self._preprocess_rnn_state(state)
        )
        self._features = torch.nn.utils.rnn.pad_packed_sequence(
            self._features, batch_first=True
        )[0]

        return self.policy(self._features), self._postprocess_rnn_state(state)

    def _preprocess_rnn_state(self, state: List[Tensor]) -> List[Tensor]:
        """Reshape state as required by RNN inference."""
        return (
            torch.unsqueeze(state[0], 0)
            if isinstance(self.rnn, nn.GRU)
            else [torch.unsqueeze(state[0], 0), torch.unsqueeze(state[1], 0)]
        )

    def _postprocess_rnn_state(self, state: List[Tensor]) -> List[Tensor]:
        """Reshape state returned by RNN as required by rllib."""
        return (
            [state.squeeze(0)]
            if isinstance(self.rnn, nn.GRU)
            else [state[0].squeeze(0), state[1].squeeze(0)]
        )

    @override(ModelV2)
    def get_initial_state(self) -> List[Tensor]:
        """Get initial RNN state consisting of zero vectors."""
        if isinstance(self.rnn, nn.GRU):
            h = [self.policy.weight.new(1, self.rnn_state_size).zero_().squeeze(0)]
        elif isinstance(self.rnn, nn.LSTM):
            h = [
                self.policy.weight.new(1, self.rnn_state_size).zero_().squeeze(0),
                self.policy.weight.new(1, self.rnn_state_size).zero_().squeeze(0),
            ]
        else:
            raise ValueError(f"{self.rnn} not supported RNN type")

        return h

    @override(ModelV2)
    def value_function(self):
        return torch.reshape(self.value_branch(self._features), [-1])

    def get_weights(self):
        return None

    def set_weights(self, weights):
        pass
