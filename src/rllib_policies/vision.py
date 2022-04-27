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

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn

from rllib_policies.actor_critic import ActorCritic, RNNActorCritic
from rllib_policies.base import NetworkBase


class NatureCNN(nn.Module):
    def __init__(self, cnn_shape: Tuple[int, int, int]):
        """CNN from `Human-level control through
        deep reinforcement learning`
        See `https://www.nature.com/articles/nature14236`
        Args:
            cnn_shape (Tuple[int, int, int]): CHW of expected image.
        """
        nn.Module.__init__(self)

        self.cnn_shape = cnn_shape
        self.out_feature_length = 512
        self.base, n_outputs = self._get_base()

        self.linear = nn.Linear(n_outputs, self.out_feature_length)

        for layer in self.base:
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))

    def _get_base(self) -> Tuple[nn.Module, int]:
        """Get CNN portion of network.
        Returns:
            Tuple[nn.Module, int]
            - Module of CNN layers
            - Length of resultant feature vector
                based on input image
        """
        cnn_downsample_factor = 8

        activ = nn.ReLU
        base = nn.Sequential(
            nn.Conv2d(self.cnn_shape[0], 32, (8, 8), 4, padding=4),
            activ(),
            nn.Conv2d(32, 64, (4, 4), 2, padding=1),
            activ(),
            nn.Conv2d(64, 64, (3, 3), 1, padding=1),
            activ(),
        )
        n_outputs = (
            self.cnn_shape[1]
            // cnn_downsample_factor
            * self.cnn_shape[2]
            // cnn_downsample_factor
            * 64
        )
        return base, n_outputs

    def forward(self, input_imgs: torch.Tensor) -> torch.Tensor:
        """Call model on input tensor.
        Args:
            input_imgs (Tensor): flatted tensor of images
                and pose of shape [Batch, Time, obs_size].
        Returns:
            Tensor: Output of shape [Batch*Time, out_feature_size].
        """
        conv_out = self.base(input_imgs)
        conv_flattened = conv_out.reshape(conv_out.shape[0], -1)
        return self.linear(conv_flattened)


class NatureCNNBase(NetworkBase):
    """Process images with NatureCNN."""

    def __init__(self, fields: List[str], cnn_shape_chw: Tuple[int, int, int]):
        net = NatureCNN(cnn_shape_chw)
        super().__init__(net, net.out_feature_length, fields)

    def preprocess_obs(self, x: List[torch.Tensor]):
        images = torch.cat(x, dim=-1)
        images = images.permute(0, 3, 1, 2).float()  # hwc -> chw
        return images


class PoseBase(NetworkBase):
    """Passthrough network."""

    def __init__(self, fields: List[str], pose_length: int):
        super().__init__(nn.Identity(), pose_length, fields)


class NatureCNNRNNActorCritic(RNNActorCritic):
    """Actor critic network that processes image data with
    NatureCNN."""

    def init_nets(self, *, fields: List[str], cnn_shape_chw: Tuple[int, int, int]):
        return [NatureCNNBase(fields, cnn_shape_chw)]


class NatureCNNActorCritic(ActorCritic):
    """Actor critic with recurrent network that processes
    image data with NatureCNN."""

    def init_nets(self, *, fields: List[str], cnn_shape_chw: Tuple[int, int, int]):
        return [NatureCNNBase(fields, cnn_shape_chw)]
