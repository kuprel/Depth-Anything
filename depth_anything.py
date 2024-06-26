from torch import nn
import torch.nn.quantized
import torch.nn.functional as F
from torch import Tensor
from functools import partial
from vision_transformer import DinoVisionTransformer
from layers import NestedTensorBlock, MemEffAttention

class ResidualConvUnit(nn.Module):
    """Residual convolution module.
    """

    def __init__(self, features, activation, bn):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn

        self.groups=1

        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups
        )

        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups
        )

        if self.bn==True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation

        self.skip_add = torch.nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """

        out = self.activation(x)
        out = self.conv1(out)
        if self.bn==True:
            out = self.bn1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.bn==True:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):
    """Feature fusion block.
    """

    def __init__(self, features, activation, deconv=False, bn=False, expand=False, align_corners=True, size=None):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups=1

        self.expand = expand
        out_features = features
        if self.expand==True:
            out_features = features//2

        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)

        self.resConfUnit1 = ResidualConvUnit(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn)

        self.skip_add = torch.nn.quantized.FloatFunctional()

        self.size=size

    def forward(self, *xs, size=None):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        output = nn.functional.interpolate(
            output, **modifier, mode="bilinear", align_corners=self.align_corners
        )

        output = self.out_conv(output)

        return output


class ScratchBlock(nn.Module):
    def __init__(self, in_shape, out_shape, groups=1, expand=False):
        super().__init__()

        out_shape1 = out_shape
        out_shape2 = out_shape
        out_shape3 = out_shape
        if len(in_shape) >= 4:
            out_shape4 = out_shape

        if expand:
            out_shape1 = out_shape
            out_shape2 = out_shape*2
            out_shape3 = out_shape*4
            if len(in_shape) >= 4:
                out_shape4 = out_shape*8

        self.layer1_rn = nn.Conv2d(
            in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
        )
        self.layer2_rn = nn.Conv2d(
            in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
        )
        self.layer3_rn = nn.Conv2d(
            in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
        )
        if len(in_shape) >= 4:
            self.layer4_rn = nn.Conv2d(
                in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
            )

        self.stem_transpose = None

        self.refinenet1 = FeatureFusionBlock(
            features=out_shape,
            activation=nn.ReLU(False),
            deconv=False,
            bn=False,
            expand=False,
            align_corners=True,
            size=None
        )
        self.refinenet2 = FeatureFusionBlock(
            features=out_shape,
            activation=nn.ReLU(False),
            deconv=False,
            bn=False,
            expand=False,
            align_corners=True,
            size=None
        )
        self.refinenet3 = FeatureFusionBlock(
            features=out_shape,
            activation=nn.ReLU(False),
            deconv=False,
            bn=False,
            expand=False,
            align_corners=True,
            size=None
        )
        self.refinenet4 = FeatureFusionBlock(
            features=out_shape,
            activation=nn.ReLU(False),
            deconv=False,
            bn=False,
            expand=False,
            align_corners=True,
            size=None
        )

        head_features_1 = out_shape
        head_features_2 = 32

        self.output_conv1 = nn.Conv2d(
            head_features_1,
            head_features_1 // 2,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Identity(),
        )

    def forward(self,
        layer_1: Tensor, layer_2: Tensor, layer_3: Tensor, layer_4: Tensor,
        patch_h: int, patch_w: int
    ) -> Tensor:
        layer_1_rn = self.layer1_rn.forward(layer_1)
        layer_2_rn = self.layer2_rn.forward(layer_2)
        layer_3_rn = self.layer3_rn.forward(layer_3)
        layer_4_rn = self.layer4_rn.forward(layer_4)

        path_4 = self.refinenet4.forward(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.refinenet3.forward(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.refinenet2.forward(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.refinenet1.forward(path_2, layer_1_rn)

        out = self.output_conv1.forward(path_1)
        out = F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
        out = self.output_conv2.forward(out)
        return out


class DPTHead(nn.Module):
    def __init__(self, nclass, in_channels, features=256, out_channels=[256, 512, 1024, 1024]):
        super(DPTHead, self).__init__()

        self.nclass = nclass

        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])

        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])

        self.scratch = ScratchBlock(out_channels, features, groups=1, expand=False)


    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            x = x[0]

            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))

            x = self.projects[i](x)
            x = self.resize_layers[i](x)

            out.append(x)

        out = self.scratch.forward(out[0], out[1], out[2], out[3], patch_h, patch_w)

        return out


class DepthAnything(nn.Module):
    def __init__(self, encoder: str):
        super().__init__()

        embed_dim, depth, num_heads, features, out_channels = {
            'vits': (384, 12, 6, 64, [48, 96, 192, 384]),
            'vitb': (768, 12, 12, 128, [96, 192, 384, 768]),
            'vitl': (1024, 24, 16, 256, [256, 512, 1024, 1024])
        }[encoder]

        self.pretrained = DinoVisionTransformer(
            img_size=518,
            patch_size=14,
            block_chunks=0,
            init_values=1,
            ffn_layer='mlp',
            num_register_tokens=0,
            interpolate_antialias=False,
            interpolate_offset=0.1,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            block_fn=partial(NestedTensorBlock, attn_class=MemEffAttention)
        )

        self.depth_head = DPTHead(
            nclass=1,
            in_channels=embed_dim,
            features=features,
            out_channels=out_channels
        )

    def forward(self, x: Tensor) -> Tensor:
        h, w = x.shape[-2:]
        features = self.pretrained.get_intermediate_layers(x, 4, return_class_token=True)
        depth = self.depth_head.forward(features, h // 14, w // 14)
        depth = F.interpolate(depth, size=(h, w), mode="bilinear", align_corners=True)
        depth = F.relu(depth)

        return depth.squeeze(1)