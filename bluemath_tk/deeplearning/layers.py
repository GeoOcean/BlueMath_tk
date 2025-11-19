"""
Project: BlueMath_tk
Sub-Module: deeplearning.layers
Author: GeoOcean Research Group, Universidad de Cantabria
Repository: https://github.com/GeoOcean/BlueMath_tk.git
Status: Under development (Working)

Custom PyTorch layers for deep learning models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Double convolution block: (convolution => [BN] => activation) * 2

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    mid_channels : int, optional
        Number of intermediate channels. If None, uses out_channels.
    activation : nn.Module, optional
        Activation function. Default is SiLU.
    """

    def __init__(self, in_channels, out_channels, mid_channels=None, activation=None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        if activation is None:
            activation = nn.SiLU(inplace=True)

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            activation,
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            activation,
        )

    def forward(self, x):
        return self.double_conv(x)


class DoubleConv3D(nn.Module):
    """
    Double 3D convolution block: (convolution => [BN] => activation) * 2

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    mid_channels : int, optional
        Number of intermediate channels. If None, uses out_channels.
    activation : nn.Module, optional
        Activation function. Default is SiLU.
    """

    def __init__(self, in_channels, out_channels, mid_channels=None, activation=None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        if activation is None:
            activation = nn.SiLU(inplace=True)

        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
            activation,
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            activation,
        )

    def forward(self, x):
        return self.double_conv(x)


class TripleConv(nn.Module):
    """
    Triple convolution with separable spatial convolutions.
    Uses (1, kernel_size) and (kernel_size, 1) convolutions then combines.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    mid_channels : int, optional
        Number of intermediate channels. If None, uses out_channels.
    kernel_size : int, optional
        Kernel size for separable convolutions. Must be 3, 5, or 7. Default is 7.
    """

    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=7):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels

        assert kernel_size in (3, 5, 7), "kernel size must be 3, 5, or 7"
        padding = kernel_size // 2

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                mid_channels,
                kernel_size=(1, kernel_size),
                padding=(0, padding),
            ),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                mid_channels,
                kernel_size=(kernel_size, 1),
                padding=(padding, 0),
            ),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True),
        )

        self.conv_out = nn.Sequential(
            nn.Conv2d(mid_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x = self.conv_out(torch.cat([x1, x2], dim=1))
        return x


class Down(nn.Module):
    """
    Downscaling with maxpool then double conv.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Down3D(nn.Module):
    """
    Downscaling with maxpool then double 3D conv.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2), DoubleConv3D(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Upscaling then double conv.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    bilinear : bool, optional
        Whether to use bilinear upsampling. Default is True.
    """

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(
                in_channels, out_channels, mid_channels=in_channels // 2
            )
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Up3D(nn.Module):
    """
    Upscaling then double 3D conv.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    trilinear : bool, optional
        Whether to use trilinear upsampling. Default is True.
    """

    def __init__(self, in_channels, out_channels, trilinear=True):
        super().__init__()

        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
            self.conv = DoubleConv3D(
                in_channels, out_channels, mid_channels=in_channels // 2
            )
        else:
            self.up = nn.ConvTranspose3d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv3D(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CDHW
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]

        x1 = F.pad(
            x1,
            [
                diffX // 2,
                diffX - diffX // 2,
                diffY // 2,
                diffY - diffY // 2,
                diffZ // 2,
                diffZ - diffZ // 2,
            ],
        )
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """
    Output convolution layer.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class OutConv3D(nn.Module):
    """
    Output 3D convolution layer.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """

    def __init__(self, in_channels, out_channels):
        super(OutConv3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class LatentDecorr(nn.Module):
    """
    Latent pass-through layer that adds covariance decorrelation loss.

    This layer encourages the latent representations to be decorrelated
    by penalizing off-diagonal elements of the covariance matrix.

    Parameters
    ----------
    strength : float, optional
        Strength of the decorrelation penalty, by default 1e-2.
    """

    def __init__(self, strength: float = 1e-2):
        super().__init__()
        self.strength = strength

    def forward(self, z):
        """
        Forward pass with decorrelation loss.

        Parameters
        ----------
        z : torch.Tensor
            Latent representations, shape (batch, k).

        Returns
        -------
        torch.Tensor
            Unchanged latent representations.
        """
        # z: (batch, k)
        zc = z - z.mean(dim=0, keepdim=True)  # center
        B = zc.size(0)
        cov = torch.matmul(zc.t(), zc) / (B - 1.0)  # (k, k)
        diag_mask = torch.eye(cov.size(0), device=cov.device, dtype=cov.dtype)
        offdiag = cov * (1 - diag_mask)  # zero diag
        loss = self.strength * torch.sum(offdiag**2)

        # Add loss to computation graph
        z = z + 0 * loss  # Trick to add loss to graph without changing z

        # Store current loss for retrieval during training
        self._loss = loss

        return z


class PositionalEmbedding(nn.Module):
    """
    Learnable positional embedding layer for transformer models.

    Parameters
    ----------
    n_tokens : int
        Number of tokens/patches.
    d_model : int
        Model dimension.
    """

    def __init__(self, n_tokens: int, d_model: int):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, n_tokens, d_model))

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tokens, shape (B, N, D).

        Returns
        -------
        torch.Tensor
            Tokens with positional embeddings added.
        """
        return x + self.pos


class Patchify(nn.Module):
    """
    Patchify layer that splits images into patches.

    Parameters
    ----------
    patch_size : int
        Size of each patch (patch_size x patch_size).
    """

    def __init__(self, patch_size: int):
        super().__init__()
        self.p = patch_size

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input images, shape (B, C, H, W), where H and W are multiples of p.

        Returns
        -------
        torch.Tensor
            Patches, shape (B, N, p*p*C).
        """
        B, C, H, W = x.shape
        p = self.p
        Hp, Wp = H // p, W // p

        # Unfold into patches
        x = x.unfold(2, p, p).unfold(3, p, p)  # (B, C, Hp, Wp, p, p)
        x = x.contiguous().view(B, C, Hp, Wp, p * p)
        x = x.permute(0, 2, 3, 1, 4).contiguous()  # (B, Hp, Wp, C, p*p)
        x = x.view(B, Hp * Wp, C * p * p)  # (B, N, p*p*C)

        return x


class Unpatchify(nn.Module):
    """
    Unpatchify layer that reconstructs images from patches.

    Parameters
    ----------
    patch_size : int
        Size of each patch.
    Hp : int
        Number of patches in height dimension.
    Wp : int
        Number of patches in width dimension.
    C : int
        Number of channels.
    """

    def __init__(self, patch_size: int, Hp: int, Wp: int, C: int):
        super().__init__()
        self.p = patch_size
        self.Hp = Hp
        self.Wp = Wp
        self.C = C

    def forward(self, tokens):
        """
        Forward pass.

        Parameters
        ----------
        tokens : torch.Tensor
            Patches, shape (B, N=Hp*Wp, p*p*C).

        Returns
        -------
        torch.Tensor
            Reconstructed images, shape (B, C, H, W).
        """
        p, Hp, Wp, C = self.p, self.Hp, self.Wp, self.C
        B = tokens.size(0)

        x = tokens.view(B, Hp, Wp, C, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()  # (B, C, Hp, p, Wp, p)
        x = x.view(B, C, Hp * p, Wp * p)  # (B, C, H, W)

        return x


class TimePositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for temporal sequences.

    Adds 1D sinusoidal time positions to per-timestep embeddings.
    """

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input sequences, shape (B, L, D).

        Returns
        -------
        torch.Tensor
            Sequences with positional encodings added.
        """
        # x: (B, L, D)
        B, L, D = x.shape
        device = x.device

        pos = torch.arange(L, device=device, dtype=torch.float32).unsqueeze(1)  # (L, 1)
        i = torch.arange(D, device=device, dtype=torch.float32).unsqueeze(0)  # (1, D)
        angles = pos / (10000.0 ** (2 * (i // 2) / D))

        pe = torch.zeros(L, D, device=device)
        pe[:, 0::2] = torch.sin(angles[:, 0::2])
        pe[:, 1::2] = torch.cos(angles[:, 1::2])

        pe = pe.unsqueeze(0)  # (1, L, D)
        return x + pe


class ConvLSTMCell(nn.Module):
    """
    ConvLSTM Cell implementation.

    Parameters
    ----------
    input_dim : int
        Number of channels of input tensor.
    hidden_dim : int
        Number of channels of hidden state.
    kernel_size : int or tuple, optional
        Size of the convolutional kernel. Default is 3.
    bias : bool, optional
        Whether to add bias. Default is True.
    """

    def __init__(self, input_dim, hidden_dim, kernel_size=3, bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size)
        )
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)
        self.bias = bias

        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat(
            [input_tensor, h_cur], dim=1
        )  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (
            torch.zeros(
                batch_size,
                self.hidden_dim,
                height,
                width,
                device=self.conv.weight.device,
            ),
            torch.zeros(
                batch_size,
                self.hidden_dim,
                height,
                width,
                device=self.conv.weight.device,
            ),
        )


class ConvLSTM(nn.Module):
    """
    ConvLSTM module.

    Parameters
    ----------
    input_dim : int
        Number of channels of input tensor.
    hidden_dim : int or list
        Number of channels of hidden state(s).
    kernel_size : int or tuple, optional
        Size of the convolutional kernel. Default is 3.
    num_layers : int, optional
        Number of ConvLSTM layers. Default is 1.
    batch_first : bool, optional
        If True, input and output tensors are provided as (batch, seq, channel, height, width).
        Default is False.
    bias : bool, optional
        Whether to add bias. Default is True.
    return_all_layers : bool, optional
        If True, returns all layers' outputs. Default is False.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        kernel_size=3,
        num_layers=1,
        batch_first=False,
        bias=True,
        return_all_layers=False,
    ):
        super().__init__()

        self._check_kernel_size_consistency(kernel_size)

        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError("Inconsistent list length.")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(
                ConvLSTMCell(
                    input_dim=cur_input_dim,
                    hidden_dim=self.hidden_dim[i],
                    kernel_size=self.kernel_size[i],
                    bias=self.bias,
                )
            )

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](
                    input_tensor=cur_layer_input[:, t, :, :, :], cur_state=[h, c]
                )
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (
            isinstance(kernel_size, tuple)
            or (
                isinstance(kernel_size, list)
                and all([isinstance(elem, tuple) for elem in kernel_size])
            )
        ):
            raise ValueError("`kernel_size` must be tuple or list of tuples")

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class LinearSelfAttention(nn.Module):
    """
    Softmax-free, Performer-style linear attention on the time axis.

    Provides O(B * L * D * H) scaling, good for large sequence lengths.

    Parameters
    ----------
    d_model : int
        Model dimension.
    num_heads : int, optional
        Number of attention heads, by default 4.
    """

    def __init__(self, d_model: int, num_heads: int = 4):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)

    def _phi(self, x):
        """Positive feature map (ELU + 1) as in linear transformers."""
        return F.elu(x) + 1.0

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input sequences, shape (B, L, D).

        Returns
        -------
        torch.Tensor
            Output sequences, shape (B, L, D).
        """
        # x: (B, L, D)
        B, L, D = x.shape
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)  # (B, L, D)

        # split heads
        Qh = Q.view(B, L, self.num_heads, self.d_head).transpose(1, 2)  # (B, H, L, Dh)
        Kh = K.view(B, L, self.num_heads, self.d_head).transpose(1, 2)  # (B, H, L, Dh)
        Vh = V.view(B, L, self.num_heads, self.d_head).transpose(1, 2)  # (B, H, L, Dh)

        Qf, Kf = self._phi(Qh), self._phi(Kh)  # (B, H, L, Dh)

        # Precompute Kf^T Vh and Kf^T 1 for normalization (linear time in L)
        Kv = torch.einsum("bhlm,bhln->bhmn", Kf, Vh)  # (B, H, Dh, Dh)
        K1 = torch.einsum(
            "bhlm,bhl->bhm",
            Kf,
            torch.ones((B, self.num_heads, L), device=Kf.device, dtype=Kf.dtype),
        )  # (B, H, Dh)

        # Numerator: Qf @ (Kf^T V)
        num = torch.einsum("bhlm,bhmn->bhln", Qf, Kv)  # (B, H, L, Dh)
        # Denominator: Qf @ (Kf^T 1)  (broadcast over Dh)
        den = torch.einsum("bhlm,bhm->bhl", Qf, K1)  # (B, H, L)
        den = (den + 1e-6).unsqueeze(-1)  # (B, H, L, 1)

        out = num / den  # (B, H, L, Dh)
        # merge heads
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)  # (B, L, D)

        return self.Wo(out)
