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

from abc import ABC, abstractclassmethod
from typing import Dict, List, Optional

import numpy as np
import torch
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv

from rllib_policies.base import NetworkBase

from .actor_critic import ActorCritic


class GCN(torch.nn.Module, ABC):
    def __init__(
        self,
        in_features,
        graph_conv_features=[16, 32],
        embedding_size=[128],
    ):
        """Initialize Graph Convolutional Network

        ReLU is applied after each graph conv.
        Global mean pooling is applied at the end.

        Args:
            in_features (int): Features per node.
            graph_conv_features (List[int]): Graph convolution
                features size, defines size of the network.
            embedding_method (GCNEmbedding): Method of embedding
                learned graph node features into common feature
                space for policy. Options are [Mean, GlobalAverage,
                LinearMapping].
            n_nodes (int): Max number of nodes in graph. Only
                required if graph needes to be padded (e.g.,
                for a linear layer).
            embedding_size (int): Final embedding size, if
                applicable (e.g., for a linear layer).
        """
        super(GCN, self).__init__()
        if len(embedding_size) > 0:
            self.out_features = embedding_size[-1]
        else:
            self.out_features = graph_conv_features[-1]

        self.graph_convs = torch.nn.ModuleList()
        self.graph_convs.append(GCNConv(in_features, graph_conv_features[0]))
        self.dense_layers = None

        # graph conv layers
        for i in range(1, len(graph_conv_features)):
            self.graph_convs.append(
                GCNConv(graph_conv_features[i - 1], graph_conv_features[i])
            )

    def get_dense_layers(self, layers: List[int]) -> torch.nn.Module:
        """Get set of linear layers.

        Args:
            layers (List[int]): Dense layer sizes. Implies
                number of layers.
        """
        return torch.nn.Sequential(
            *[torch.nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)]
        )

    def reshape_nodes_flat_to_batch(
        self, h: torch.Tensor, batch_index: torch.Tensor
    ) -> torch.Tensor:
        """Reshape nodes flattened to shape (N*B, F) to
        (B, N, F) where `N` is the number of nodes, `B` is
        batch size, and `F` is node feature size.

        Args:
            h (torch.Tensor): Shape (N*B, F) tensor of nodes.
            batch_index (torch.Tensor): Shape (N*B, ) batch index
                of nodes in `h`.

        Returns:
            torch.Tensor: Shape (B, N, F) tensor of nodes
                organized by batch.
        """
        values, counts = torch.unique(batch_index, return_counts=True)

        # assume fixed number of nodes per graph
        assert torch.sum(counts.unsqueeze(0) - counts.unsqueeze(0).t()) == 0
        nodes_per_graph = counts[0]

        # reshape to tensor to (batch, nodes, features)
        if len(values) > 1:
            h = h.reshape(len(values), nodes_per_graph, -1)
        else:
            h = h.unsqueeze(0)
        return h

    @abstractclassmethod
    def postprocess_padded_nodes(
        cls, node_tensors: torch.Tensor, node_shapes: torch.Tensor
    ) -> torch.Tensor:
        """Postprocess padded node tensors, if needed.

        Args:
            node_tensors (torch.Tensor): (B, N, F) tensor of node
                features where `B` is batch, `N` is number of padded
                nodes, and `F` is node feature length.
            node_shapes (torch.Tensor): (B, 2) tensor of unpadded
                node shape.

        Returns:
            torch.Tensor: Shape (B, N', F) tensor where `N'` may
                be new node shape.
        """
        pass

    def get_datasets_from_batch(
        self,
        graph_nodes: torch.Tensor,
        graph_node_shapes: torch.Tensor,
        graph_edges: torch.Tensor,
        graph_edge_shapes: torch.Tensor,
    ) -> List[Data]:
        """Get a set of pytorch geometric Data objects from
        batch of graphs.

        Args:
            graph_nodes (torch.Tensor): (B, PN, F) shape tensor where
                `B` is batch size, `PN` is padded node size, `F` is
                node feature size.
            graph_nodes_shapes (torch.Tensor): (B, 2) shape tensor of
                node shapes used to denote padding.
            graph_edges (torch.Tensor): (B, 2, PE) shape tensor where
                `B` is batch size, `PE` is padded edge size.
            graph_edge_shapes (torch.Tensor): (B, 2) shape tensor
                of edge shapes used to denote padding.
            poses (torch.Tensor): (B, 3) shape pose tensor.

        Returns:
            List[Data]: List of pytorch geometric dataset objects.
        """
        datasets = []
        valid_inds = []
        for i in range(graph_nodes.shape[0]):
            # If needed, rllib will 0 pad observations. This is fine for
            # image and vector processing, but 0 valued edge indices will break
            # a gcn model. Thus, if a graph is 0 padded (as denoted by the
            # expected shape), skip and add 0 to the resultant feature vector.
            if (graph_node_shapes[i] > 0).all():
                gn = self.postprocess_padded_nodes(graph_nodes[i], graph_node_shapes[i])
                ge = graph_edges[
                    i, : graph_edge_shapes[i][0], : graph_edge_shapes[i][1]
                ].type(torch.long)
                datasets.append(Data(x=gn, edge_index=ge))
                valid_inds.append(i)

        return datasets, valid_inds

    def batched_graph_features_to_tensor(
        self,
        batched_graph_features: List[torch.Tensor],
        graph_node_shapes: torch.Tensor,
        valid_inds: torch.Tensor,
    ) -> torch.Tensor:
        """Convert list of graph features from batch to single tensor.

        Args:
            batched_graph_features (List[torch.Tensor]): Learned
                graph features per dataset, may include padding.
            graph_node_shapes (torch.Tensor): Original graph node
            shape.
        """
        batched_graph_features = torch.cat(batched_graph_features, dim=0)
        device = batched_graph_features.device
        graph_features = torch.zeros(
            (graph_node_shapes[0], batched_graph_features.shape[-1])
        ).to(device)
        graph_features[valid_inds] = batched_graph_features
        return graph_features

    @abstractclassmethod
    def get_policy_features(
        self,
        h: torch.Tensor,
        batch_index: torch.Tensor,
    ) -> torch.Tensor:
        """Get learned graph features for policy.

        Args:
            h (Tensor): Shape (N, F) tensor where `N` is number
                of nodes and `F` is feature length.
            batch_index (Tensor): Shape (N, ) tensor mapping
                node to graph ID.

        Returns:
            Tensor: Shape (F, ) tensor containing features
                for policy depending on embedding method.
        """
        pass

    def get_out_features(self) -> int:
        """Get GCN output features."""
        return self.out_features

    def forward_graph(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Pass graph, defined by `x` and `edge_index`, through network.

        Args:
            x (Tensor): Shape (N, F) tensor, `N` is the number of nodes
                `F` is the node feature length.
            edge_index (Tensor): Shape (2, E) tensor, `E` is the number
                of directed edges.
            batch_index (Tensor): Shape (N, ) tensor of batch indices.
        """
        h = x
        for conv in self.graph_convs:
            h = conv(h.float(), edge_index).relu()

        # embed learned node features to single vector
        h = self.get_policy_features(h, batch_index)
        return h

    def forward_batch(self, dataloader: DataLoader) -> List[torch.Tensor]:
        """Run inference on batch of graphs.

        Args:
            dataloader (DataLoader): PyG dataloader
                containing batch of graphs.
            poses (torch.Tensor): Corresponding agent
                pose for each graph in `dataloader`.
        """
        batched_graph_features = []
        # # run inference
        for batch in dataloader:
            pass
            batched_graph_features.append(
                self.forward_graph(batch.x, batch.edge_index, batch_index=batch.batch)
            )

        return batched_graph_features

    def postprocess_batch(
        self, batched_graph_features, graph_node_shapes, valid_inds
    ) -> torch.Tensor:
        """Perform any needed postprocessing to learned
        graph features. Default behavior is identity.
        """
        return batched_graph_features

    def forward(
        self,
        nodes: torch.Tensor,
        node_shapes: torch.Tensor,
        edges: torch.Tensor,
        edge_shapes: torch.Tensor,
    ):
        """Forward a batch of graphs through the network

        Args:
            graph_nodes (torch.Tensor): Shape (B, N, X) tensor
                `of  `N` nodes with `X` features. `B` is the
                batch dimension. Note that nodes may be padded.
            graph_node_shapes (torch.Tensor): Shape (B, n, x)
                tensor describing the pre-padded shape of
                `graph_nodes`. `B` is the batch dimension.
            graph_edges (torch.Tensor): Shape (B, 2, E) tensor
                of `B` `E` (from_node, to_node) edges. `B`
                is the batch dimension.
            graph_edge_shapes (torch.Tensor): Shape (B, 2, e)
                tensor describing the original shape of
                `graph_edges`. `B` is the batch dimension.

        Returns:
            torch.Tensor: Shape (B, F): `F` length feature
                vector for each graph in the batch.
        """
        # rllib sends a batch of 0 values observations on initialization
        # skip this batch to avoid breaking gcn graph padding
        if (node_shapes == 0).all():
            graph_features = torch.zeros((nodes.shape[0], self.out_features)).to(
                nodes.device
            )

        else:
            datasets, valid_inds = self.get_datasets_from_batch(
                nodes, node_shapes, edges, edge_shapes
            )
            loader = DataLoader(datasets, batch_size=len(datasets))
            batched_graph_features = self.forward_batch(loader)
            graph_features = self.postprocess_batch(
                batched_graph_features, node_shapes, valid_inds
            )
            graph_features = torch.cat(graph_features, axis=0)

        return graph_features


class AgentFrameGCN(GCN):
    def __init__(
        self,
        in_features: torch.Tensor,
        graph_conv_features: List[int],
        embedding_size: List[int],
        n_frame_nodes: int,
    ):
        super().__init__(in_features, graph_conv_features, embedding_size)
        self.n_agent_frame_nodes = n_frame_nodes
        if len(embedding_size) == 0:
            self.out_features *= n_frame_nodes * graph_conv_features[-1]
        else:
            layers = [n_frame_nodes * graph_conv_features[-1]] + embedding_size
            self.dense_layers = self.get_dense_layers(layers)

    def postprocess_padded_nodes(
        cls, node_tensors: torch.Tensor, node_shapes: torch.Tensor
    ) -> torch.Tensor:
        return node_tensors

    def get_policy_features(
        self, h: torch.Tensor, batch_index: torch.Tensor
    ) -> torch.Tensor:
        # if no batch index is given, assume all data is from same graph
        if batch_index is None:
            batch_index = torch.zeros(h.shape[0], dtype=torch.int64).to(h.device)

        h = self.reshape_nodes_flat_to_batch(h, batch_index)
        h = h[:, : self.n_agent_frame_nodes]
        h = h.reshape(h.shape[0], -1)

        if self.dense_layers is not None:
            h = self.dense_layers(h)
        return h


class GNNBase(NetworkBase):
    def __init__(
        self,
        fields: Dict[str, str],
        in_features,
        graph_conv_features,
        embedding_size,
        n_frame_nodes,
    ):
        net = AgentFrameGCN(
            in_features, graph_conv_features, embedding_size, n_frame_nodes
        )
        super().__init__(net, net.out_features, fields.keys())
        self.fields = fields

    def get_obs(self, obs: Dict[str, torch.Tensor], rnn_input: bool) -> torch.Tensor:
        graph_obs = {}

        for field, obs_key in self.fields.items():
            data = obs[obs_key]
            if rnn_input:  # combine time and batch inds
                data = data.reshape((-1,) + tuple(data.shape[2:]))
            if field in ("node_shapes", "edge_shapes"):
                # if isinstance(data, torch.Tensor):
                data = data.type(torch.int32)
                # elif isinstance(data, np.ndarray):
                # data = data.astype(np.int32)
            graph_obs[field] = data

        return graph_obs

    def forward(self, obs, rnn_input):
        data = self.get_obs(obs, rnn_input)
        return self.net(**data)


class ActionLayerGNNActorCritic(ActorCritic):
    """Actor critic with recurrent network that processes
    image data with NatureCNN."""

    def init_nets(
        self,
        *,
        fields: List[str],
        in_features,
        graph_conv_features,
        embedding_size,
        n_layer_nodes
    ):
        return [
            GNNBase(
                fields, in_features, graph_conv_features, embedding_size, n_layer_nodes
            )
        ]
