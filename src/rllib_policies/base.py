###################################################################################################
# DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.
#
# This material is based upon work supported by the Under Secretary of Defense for Research and
# Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions
# or recommendations expressed in this material are those of the author(s) and do not necessarily
# reflect the views of the Under Secretary of Defense for Research and Engineering.
#
# (c) 2022 Massachusetts Institute of Technology.
#
# MIT Proprietary, Subject to FAR52.227-11 Patent Rights - Ownership by the contractor (May 2014)
#
# The software/firmware is provided to you on an As-Is basis
#
# Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013
# or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work
# are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other
# than as specifically authorized by the U.S. Government may violate any copyrights that exist in
# this work.
###################################################################################################

from typing import Any, Dict, List, Union

import gym
import numpy as np
import torch
import torch.nn as nn
from ray.rllib.models.modelv2 import restore_original_dimensions


class NetworkBase(nn.Module):
    def __init__(
        self,
        net: nn.Module,
        feature_length: int,
        fields: Union[Dict[str, str], List[str]],
    ) -> None:
        """Base class for networks used in Rllib policy.

        Network is defined via a pytorch module to process one or
        more observations. Observations are read from an rllib
        observation dictionary, with keys specified by `fields`.
        The expected output feature length must be specified, so the
        network can be incorporated in a policy.

        Parameters
        ----------
        net: nn.Module
            Pytorch network.

        feature_length: int
            Feature vector length produced by `net`.

        fields: List[str]
            Fields in observation dictionary that
            are used by `net`.
        """
        super().__init__()
        self.net = net
        self.feature_length = feature_length
        self.fields = fields

    def get_obs(
        self, obs: Dict[str, torch.Tensor], rnn_input: bool
    ) -> Union[List[torch.Tensor], Dict[str, torch.Tensor]]:
        """Read observations from input dictionary.

        Parameters
        ----------
        obs: Dict[str, torch.Tensor]
            Dictionary mapping modality names to
            data.

        rnn_input: bool
            True if policy has rnn.

        Returns
        -------
        torch.Tensor
            Concatenated tensors specified in `fields`.
        """
        x = []
        for k, data in obs.items():
            if k in self.fields:
                if rnn_input:  # combine time and batch inds
                    data = data.reshape((-1,) + tuple(data.shape[2:]))
                x.append(data)
        return x

    def preprocess_obs(self, x: torch.Tensor) -> torch.Tensor:
        """Perform any preprocessing needed.

        Parameters
        ----------
        x: torch.Tensor
            Input observations.

        Returns
        -------
        Preprocessed tensors.
        """
        return x

    def forward(self, obs: Dict, rnn_input) -> torch.tensor:
        """Pass observations through network.

        Parameters
        ----------
        obs: Dict[str, torch.Tensor]
            Dictionary of observations.

        rnn_input: bool
            True if policy has rnn.

        Returns
        -------
        torch.Tensor
            Features from network.
        """
        data = self.get_obs(obs, rnn_input)
        return self.net(self.preprocess_obs(data))


class RllibPolicy(nn.Module):
    def __init__(self, obs_space: gym.spaces.Space, **custom_net_args):
        """Base class for RLlib policies (e.g., actor-critic).

        Each policy is comprised of networks to process observations
        (e.g., images, pose). These networks produce feature
        representations of known length that are concatenated and
        given to a dense layer(s) for action and value prediction.


        Parameters
        ----------
        obs_space: gym.spaces.Space
            Model observation space.

        custom_network_args: Dict[str, Any]
            Arguments used for custom networks.
        """
        super().__init__()
        self.full_obs_space = obs_space
        self.nets = nn.ModuleList(self.init_nets(**custom_net_args))
        self.nets_out_feature_size = np.sum(n.feature_length for n in self.nets)

    def forward_nets(self, input_obs: torch.Tensor, rnn_input: bool = False):
        """Pass observations through networks. Features
        are concatenated and returned as single tensor.

        Parameters
        ----------
        input_obs: Dict[str, torch.Tensor]
            Dictionary of observations.

        rnn_input: bool
            True if policy has rnn.

        Returns
        -------
        torch.Tensor
            Concatenated features from network(s).
        """

        obs = restore_original_dimensions(
            input_obs, self.full_obs_space, tensorlib="torch"
        )
        return torch.cat([n(obs, rnn_input) for n in self.nets])

    def init_nets(self, **custom_net_args: Dict[str, Any]) -> List[NetworkBase]:
        """Initialize custom networks.

        Parameters
        ----------
        custom_network_args: Dict[str, Any]
            Any arguments used by networks.
        """
        raise NotImplementedError()
