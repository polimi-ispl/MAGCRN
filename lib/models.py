import torch
from einops import repeat
from torch import Tensor
from torch_geometric.typing import OptTensor
from tsl.nn.blocks.decoders import LinearReadout
from typing import Optional
from tsl.nn.blocks.encoders.recurrent.base import RNNBase
from tsl.nn.layers import AGCRNCell, NodeEmbedding, AdaptiveGraphConv, DiffConv, Norm, DenseGraphConvOrderK
from tsl.nn.models.base_model import BaseModel
from tsl.nn.utils import maybe_cat_exog
from torch import nn as nn
from torch.nn import Module
from torch.nn import functional as F
from tsl.nn.utils import get_layer_activation
from tsl.ops.connectivity import adj_to_edge_index


class ConditionalBlockMod(nn.Module):
    r"""
    The ConditionalBlockMod module used in the MAGCRN model and inspired by the ConditionalBlock module
    of Torch Spatiotemporal, 2022.
    Reference:
    https://torch-spatiotemporal.readthedocs.io/en/latest/modules/nn_blocks.html#tsl.nn.blocks.encoders.ConditionalBlock

    Args:
        input_size (int): The size of the input data.
        conditioning_size (int): The size of the conditioning tensor.
        output_size (int): The size of the output.
        dropout (float, optional): Dropout probability. Default is 0.
        skip_connection (bool, optional): Whether to include a skip connection. Default is False.
        activation (str, optional): The activation function to use. Default is 'relu'.
    """

    def __init__(self,
                 input_size,
                 conditioning_size,
                 output_size,
                 dropout=0.,
                 skip_connection=False,
                 activation='relu'):
        """
        Initializes a ConditionalBlockMod module.

        Parameters:
        - input_size (int): The size of the input data.
        - conditioning_size (int): The size of the conditioning tensor.
        - output_size (int): The size of the output.
        - dropout (float, optional): Dropout probability. Default is 0.
        - skip_connection (bool, optional): Whether to include a skip connection. Default is False.
        - activation (str, optional): The activation function to use. Default is 'relu'.
        """
        super().__init__()

        self.d_in = input_size
        self.d_cond = conditioning_size
        self.d_out = output_size
        self.activation = get_layer_activation(activation)()
        self.dropout = nn.Dropout(dropout)

        # Inputs module
        self.input_affinity = nn.Linear(self.d_in, self.d_out)
        self.condition_affinity = nn.Linear(self.d_cond, self.d_out)

        self.out_inputs_affinity = nn.Linear(self.d_out, self.d_out)
        self.out_cond_affinity = nn.Linear(self.d_out, self.d_out, bias=False)

        if skip_connection:
            self.skip_conn = nn.Linear(self.d_in, self.d_out)
        else:
            self.register_parameter('skip_conn', None)

    def forward(self, x: Tensor, cond: Tensor):
        """
        Forward pass of the ConditionalBlockMod module.

        Parameters:
        - x (Tensor): Input data.
        - cond (Tensor): Conditioning tensor.

        Returns:
        - out (Tensor): Output tensor.
        """
        # Inputs block
        out = self.activation(self.input_affinity(x))
        # Conditions block
        conditions = self.activation(self.condition_affinity(cond))

        # Combine processed results
        out = self.out_inputs_affinity(out) + self.out_cond_affinity(conditions)
        out = self.dropout(self.activation(out))

        # Add skip connection if specified
        if self.skip_conn is not None:
            out = self.skip_conn(x) + out

        return out

class MAGCRN(BaseModel):
    """
    Modified AGCRN (MAGCRN), with the possibility to condition on future covariates.
    Reference:
    Giganti, A. et al., Back to the Future: GNN-based NO2 Forecasting via Future Covariates
    IGARSS 2024

    Attributes:
        return_type: Type of the return value (Tensor).

    Args:
        input_size (int): Size of the input time series.
        output_size (int): Size of the output (forecast).
        horizon (int): Number of time steps to forecast into the future.
        exog_size (int): Size of the exogenous variables.
        n_nodes (int): Number of nodes in the input graph.
        hidden_size (int, optional): Hidden size for internal representations. Default is 64.
        emb_size (int, optional): Size of the input node embeddings. Default is 10.
        past_cov_size (int, optional): Size of the past covariates. Default is 0.
        fut_cov_size (int, optional): Size of the future covariates. Default is 0.
        n_layers (int, optional): Number of recurrent layers. Default is 3.
        alpha (float, optional): Weight for the future conditioning branch. Default is 0.5.
        dropout (float, optional): Dropout probability. Default is 0.0.
        skip_connection (bool, optional): Whether to use skip connections. Default is False.
    """

    return_type = Tensor

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 horizon: int,
                 exog_size: int,
                 n_nodes: int,
                 hidden_size: int = 64,
                 emb_size: int = 10,
                 past_cov_size: int = 0,
                 fut_cov_size: int = 0,
                 n_layers: int = 3,
                 alpha: float = 0.5,
                 dropout: float = 0.0,
                 skip_connection: bool = False):
        """
        Initializes a MAGCRN model.

        Parameters:
        - input_size (int): Size of the input time series.
        - output_size (int): Size of the output (forecast).
        - horizon (int): Number of time steps to forecast into the future.
        - exog_size (int): Size of the exogenous variables.
        - n_nodes (int): Number of nodes in the input graph.
        - hidden_size (int, optional): Hidden size for internal representations. Default is 64.
        - emb_size (int, optional): Size of the input node embeddings. Default is 10.
        - past_cov_size (int, optional): Size of the past covariates. Default is 0.
        - fut_cov_size (int, optional): Size of the future covariates. Default is 0.
        - n_layers (int, optional): Number of recurrent layers. Default is 3.
        - alpha (float, optional): Weight for the future conditioning branch. Default is 0.5.
        - dropout (float, optional): Dropout probability. Default is 0.0.
        - skip_connection (bool, optional): Whether to use skip connections. Default is False.
        """
        super(MAGCRN, self).__init__()

        # Set alpha attribute
        self.alpha = alpha

        # Conditioning on future covariates
        self.input_encoder_fut_cond = ConditionalBlockMod(input_size=input_size,
                                                          conditioning_size=fut_cov_size,
                                                          output_size=hidden_size,
                                                          dropout=dropout,
                                                          skip_connection=skip_connection)

        # Conditioning on past covariates and exogenous variables
        self.input_encoder_additional_cond = ConditionalBlockMod(input_size=input_size,
                                                                 conditioning_size=past_cov_size + exog_size,
                                                                 output_size=hidden_size,
                                                                 dropout=dropout,
                                                                 skip_connection=skip_connection)

        # AGCRN modules
        self.agrn_past = AGCRNMod(input_size=hidden_size,
                                  emb_size=emb_size,
                                  num_nodes=n_nodes,
                                  hidden_size=hidden_size,
                                  n_layers=n_layers,
                                  return_only_last_state=True)

        self.agrn_fut = AGCRNMod(input_size=hidden_size,
                                 emb_size=emb_size,
                                 num_nodes=n_nodes,
                                 hidden_size=hidden_size,
                                 n_layers=n_layers,
                                 return_only_last_state=True)

        # Linear readout layer
        self.readout = LinearReadout(input_size=hidden_size, output_size=output_size, horizon=horizon)

    def forward(self, x: Tensor, fut_cov: Tensor, u: Tensor, past_cov: Tensor) -> Tensor:
        """
        Forward pass of the MAGCRN model.

        Parameters:
        - x (Tensor): Input time series.
        - fut_cov (Tensor): Future covariates.
        - u (Tensor): Exogenous variables.
        - past_cov (Tensor): Past covariates.

        Returns:
        - forecast (Tensor): Forecasted time series.
        """
        # Concatenate past covariates and exogenous variables
        _x = maybe_cat_exog(past_cov, u)

        # Condition on past and exogenous variables
        x1 = self.input_encoder_additional_cond(x, cond=_x)

        # Condition on future covariates
        x2 = self.input_encoder_fut_cond(x, cond=fut_cov)

        # AGCRN forward pass for both branches
        out1 = self.agrn_past(x1)
        out2 = self.agrn_fut(x2)

        # Combine results with weighted sum
        out = (1 - self.alpha) * out1 + self.alpha * out2

        # Linear readout for forecasting
        forecast = self.readout(out)

        return forecast


class GraphWaveNetModelMod(BaseModel):
    """
    Graph WaveNet Model with a modified forward method to access the learned adjacency matrix.

    Reference:
    Wu, Z., et al., Graph wavenet for deep spatial-temporal graph modeling.
    IJCAI 2019.

    https://www.ijcai.org/proceedings/2019/0264.pdf
    https://torch-spatiotemporal.readthedocs.io/en/latest/modules/nn_models.html#tsl.nn.models.stgn.GraphWaveNetModel

    Attributes:
        return_type: Type of the return value (Tensor).

    Args:
        input_size (int): Size of the input time series.
        output_size (int): Size of the output (forecast).
        horizon (int): Number of time steps to forecast into the future.
        exog_size (int, optional): Size of the exogenous variables. Default is 0.
        hidden_size (int, optional): Hidden size for internal representations. Default is 32.
        ff_size (int, optional): Size of the fully connected layers. Default is 256.
        n_layers (int, optional): Number of temporal convolutional layers. Default is 8.
        temporal_kernel_size (int, optional): Kernel size for temporal convolutions. Default is 2.
        spatial_kernel_size (int, optional): Kernel size for spatial convolutions. Default is 2.
        learned_adjacency (bool, optional): Whether to use a learned adjacency matrix. Default is True.
        n_nodes (int, optional): Number of nodes in the graph. Required if learned_adjacency is True.
        emb_size (int, optional): Size of the node embeddings. Default is 10.
        dilation (int, optional): Dilation factor for temporal convolutions. Default is 2.
        dilation_mod (int, optional): Dilation modification factor for temporal convolutions. Default is 2.
        norm (str, optional): Type of normalization layer ('batch' or 'layer'). Default is 'batch'.
        dropout (float, optional): Dropout probability. Default is 0.3.
    """

    return_type = Tensor

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 horizon: int,
                 exog_size: int = 0,
                 hidden_size: int = 32,
                 ff_size: int = 256,
                 n_layers: int = 8,
                 temporal_kernel_size: int = 2,
                 spatial_kernel_size: int = 2,
                 learned_adjacency: bool = True,
                 n_nodes: Optional[int] = None,
                 emb_size: int = 10,
                 dilation: int = 2,
                 dilation_mod: int = 2,
                 norm: str = 'batch',
                 dropout: float = 0.3):
        """
        Initializes a GraphWaveNetModelMod model.

        Parameters:
        - input_size (int): Size of the input time series.
        - output_size (int): Size of the output (forecast).
        - horizon (int): Number of time steps to forecast into the future.
        - exog_size (int, optional): Size of the exogenous variables. Default is 0.
        - hidden_size (int, optional): Hidden size for internal representations. Default is 32.
        - ff_size (int, optional): Size of the fully connected layers. Default is 256.
        - n_layers (int, optional): Number of temporal convolutional layers. Default is 8.
        - temporal_kernel_size (int, optional): Kernel size for temporal convolutions. Default is 2.
        - spatial_kernel_size (int, optional): Kernel size for spatial convolutions. Default is 2.
        - learned_adjacency (bool, optional): Whether to use a learned adjacency matrix. Default is True.
        - n_nodes (int, optional): Number of nodes in the graph. Required if learned_adjacency is True.
        - emb_size (int, optional): Size of the node embeddings. Default is 10.
        - dilation (int, optional): Dilation factor for temporal convolutions. Default is 2.
        - dilation_mod (int, optional): Dilation modification factor for temporal convolutions. Default is 2.
        - norm (str, optional): Type of normalization layer ('batch' or 'layer'). Default is 'batch'.
        - dropout (float, optional): Dropout probability. Default is 0.3.
        """
        super(GraphWaveNetModelMod, self).__init__()

        if learned_adjacency:
            assert n_nodes is not None
            self.source_embeddings = NodeEmbedding(n_nodes, emb_size)
            self.target_embeddings = NodeEmbedding(n_nodes, emb_size)
        else:
            self.register_parameter('source_embedding', None)
            self.register_parameter('target_embedding', None)

        self.input_encoder = nn.Linear(input_size + exog_size, hidden_size)

        temporal_conv_blocks = []
        spatial_convs = []
        skip_connections = []
        norms = []
        receptive_field = 1
        for i in range(n_layers):
            d = dilation ** (i % dilation_mod)
            temporal_conv_blocks.append(
                TemporalConvNet(input_channels=hidden_size,
                                hidden_channels=hidden_size,
                                kernel_size=temporal_kernel_size,
                                dilation=d,
                                exponential_dilation=False,
                                n_layers=1,
                                causal_padding=False,
                                gated=True))

            spatial_convs.append(
                DiffConv(in_channels=hidden_size,
                         out_channels=hidden_size,
                         k=spatial_kernel_size))

            skip_connections.append(nn.Linear(hidden_size, ff_size))
            norms.append(Norm(norm, hidden_size))
            receptive_field += d * (temporal_kernel_size - 1)
        self.tconvs = nn.ModuleList(temporal_conv_blocks)
        self.sconvs = nn.ModuleList(spatial_convs)
        self.skip_connections = nn.ModuleList(skip_connections)
        self.norms = nn.ModuleList(norms)
        self.dropout = nn.Dropout(dropout)

        self.receptive_field = receptive_field

        dense_sconvs = []
        if learned_adjacency:
            for _ in range(n_layers):
                dense_sconvs.append(
                    DenseGraphConvOrderK(input_size=hidden_size,
                                         output_size=hidden_size,
                                         support_len=1,
                                         order=spatial_kernel_size,
                                         include_self=False,
                                         channel_last=True))
        self.dense_sconvs = nn.ModuleList(dense_sconvs)
        self.readout = nn.Sequential(
            nn.ReLU(),
            MLPDecoder(input_size=ff_size,
                       hidden_size=2 * ff_size,
                       output_size=output_size,
                       horizon=horizon,
                       activation='relu'))

    def get_learned_adj(self) -> Tensor:
        """
        Computes the learned adjacency matrix.

        Returns:
        - adj (Tensor): Learned adjacency matrix.
        """
        logits = F.relu(self.source_embeddings() @ self.target_embeddings().T)
        adj = torch.softmax(logits, dim=1)
        return adj

    def forward(self,
                x: Tensor,
                edge_index: OptTensor = None,
                edge_weight: OptTensor = None,
                u: OptTensor = None) -> Tensor:
        """
        Forward pass of the GraphWaveNetModelMod model.

        Parameters:
        - x (Tensor): Input time series.
        - edge_index (OptTensor, optional): Edge indices for graph connectivity. Default is None.
        - edge_weight (OptTensor, optional): Edge weights for graph connectivity. Default is None.
        - u (OptTensor, optional): Exogenous variables. Default is None.

        Returns:
        - forecast (Tensor): Forecasted time series.
        """
        # x: [b t n f]

        # Calculate the learned adjacency matrix
        adj = self.get_learned_adj()
        edge_index, edge_weight = adj_to_edge_index(adj)[0], adj_to_edge_index(adj)[1]

        if u is not None:
            if u.dim() == 3:
                u = repeat(u, 'b t f -> b t n f', n=x.size(-2))
            x = torch.cat([x, u], -1)

        if self.receptive_field > x.size(1):
            # pad temporal dimension
            x = F.pad(x, (0, 0, 0, 0, self.receptive_field - x.size(1), 0))

        if len(self.dense_sconvs):
            adj_z = self.get_learned_adj()

        x = self.input_encoder(x)

        out = torch.zeros(1, x.size(1), 1, 1, device=x.device)
        for i, (tconv, sconv, skip_conn, norm) in enumerate(
                zip(self.tconvs, self.sconvs, self.skip_connections,
                    self.norms)):
            res = x
            # temporal conv
            x = tconv(x)
            # residual connection -> out
            out = skip_conn(x) + out[:, -x.size(1):]
            # spatial conv
            xs = sconv(x, edge_index, edge_weight)
            if len(self.dense_sconvs):
                x = xs + self.dense_sconvs[i](x, adj_z)
            else:
                x = xs
            x = self.dropout(x)
            # residual connection -> next layer
            x = x + res[:, -x.size(1):]
            x = norm(x)

        return self.readout(out)
