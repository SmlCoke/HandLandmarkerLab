"""Hand Landmarker v2 with ReLU and deploy-time branch fusion.

The training graph uses Conv+BN branches to improve optimization.  Every
``Rep*`` layer can be folded into one biased convolution for deployment, so
the exported graph contains no BatchNormalization or auxiliary branches.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Conv2D,
    DepthwiseConv2D,
    Input,
    Layer,
    MaxPooling2D,
    Permute,
    ReLU,
)
from tensorflow.keras.models import Model


BN_EPSILON = 1.0e-5
BN_MOMENTUM = 0.9


def _fuse_conv_bn(kernel, bn: BatchNormalization, depthwise: bool = False):
    gamma, beta, moving_mean, moving_variance = bn.get_weights()
    scale = gamma / np.sqrt(moving_variance + float(bn.epsilon))
    if depthwise:
        shape = (1, 1, int(kernel.shape[2]), int(kernel.shape[3]))
    else:
        shape = (1, 1, 1, int(kernel.shape[3]))
    return kernel * scale.reshape(shape), beta - moving_mean * scale


def _pad_kernel(kernel, target_shape):
    pad_h = int(target_shape[0]) - int(kernel.shape[0])
    pad_w = int(target_shape[1]) - int(kernel.shape[1])
    if pad_h < 0 or pad_w < 0 or pad_h % 2 or pad_w % 2:
        raise ValueError("Auxiliary convolution kernel cannot be centered in deploy kernel")
    return np.pad(
        kernel,
        ((pad_h // 2, pad_h // 2), (pad_w // 2, pad_w // 2), (0, 0), (0, 0)),
    )


class RepConv2D(Layer):
    """Conv2D whose training branches fuse into one deploy Conv2D."""

    def __init__(self, filters: int, kernel_size=1, padding="valid", deploy=False, **kwargs):
        super().__init__(**kwargs)
        self.filters = int(filters)
        self.kernel_size = int(kernel_size)
        if self.kernel_size != 1:
            raise ValueError("RepConv2D is intentionally limited to fuseable 1x1 pointwise convolutions")
        self.padding = str(padding)
        self.deploy = bool(deploy)
        self.input_channels = None

    def build(self, input_shape):
        self.input_channels = int(input_shape[-1])
        if self.deploy:
            self.deploy_conv = Conv2D(
                self.filters,
                self.kernel_size,
                strides=1,
                padding=self.padding,
                use_bias=True,
                name="deploy_conv",
            )
        else:
            self.main_conv = Conv2D(
                self.filters,
                self.kernel_size,
                strides=1,
                padding=self.padding,
                use_bias=False,
                name="main_conv",
            )
            self.main_bn = BatchNormalization(
                momentum=BN_MOMENTUM, epsilon=BN_EPSILON, name="main_bn"
            )
            self.aux_conv = Conv2D(
                self.filters,
                1,
                strides=1,
                padding=self.padding if self.kernel_size == 1 else "same",
                use_bias=False,
                name="aux_conv",
            )
            self.aux_bn = BatchNormalization(
                momentum=BN_MOMENTUM, epsilon=BN_EPSILON, name="aux_bn"
            )
            self.identity_bn = None
            if self.input_channels == self.filters:
                self.identity_bn = BatchNormalization(
                    momentum=BN_MOMENTUM, epsilon=BN_EPSILON, name="identity_bn"
                )
        super().build(input_shape)

    def call(self, inputs, training=None):
        if self.deploy:
            return self.deploy_conv(inputs)
        values = [
            self.main_bn(self.main_conv(inputs), training=training),
            self.aux_bn(self.aux_conv(inputs), training=training),
        ]
        if self.identity_bn is not None:
            values.append(self.identity_bn(inputs, training=training))
        return tf.add_n(values)

    def equivalent_weights(self):
        if self.deploy:
            raise ValueError("equivalent_weights must be called on the training graph")
        main_kernel = self.main_conv.get_weights()[0]
        kernel, bias = _fuse_conv_bn(main_kernel, self.main_bn)
        aux_kernel, aux_bias = _fuse_conv_bn(
            self.aux_conv.get_weights()[0], self.aux_bn
        )
        kernel += _pad_kernel(aux_kernel, main_kernel.shape)
        bias += aux_bias
        if self.identity_bn is not None:
            identity = np.zeros_like(main_kernel)
            center_h = int(main_kernel.shape[0]) // 2
            center_w = int(main_kernel.shape[1]) // 2
            for channel in range(self.input_channels):
                identity[center_h, center_w, channel, channel] = 1.0
            identity, identity_bias = _fuse_conv_bn(identity, self.identity_bn)
            kernel += identity
            bias += identity_bias
        return [kernel, bias]


class RepDepthwiseConv2D(Layer):
    """DepthwiseConv2D whose 3x3, 1x1 and identity branches fuse."""

    def __init__(self, kernel_size=3, padding="same", deploy=False, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = int(kernel_size)
        self.padding = str(padding)
        self.deploy = bool(deploy)
        self.input_channels = None

    def build(self, input_shape):
        self.input_channels = int(input_shape[-1])
        if self.deploy:
            self.deploy_conv = DepthwiseConv2D(
                self.kernel_size,
                strides=1,
                padding=self.padding,
                depth_multiplier=1,
                use_bias=True,
                name="deploy_conv",
            )
        else:
            self.main_conv = DepthwiseConv2D(
                self.kernel_size,
                strides=1,
                padding=self.padding,
                depth_multiplier=1,
                use_bias=False,
                name="main_conv",
            )
            self.main_bn = BatchNormalization(
                momentum=BN_MOMENTUM, epsilon=BN_EPSILON, name="main_bn"
            )
            self.aux_conv = DepthwiseConv2D(
                1,
                strides=1,
                padding="same",
                depth_multiplier=1,
                use_bias=False,
                name="aux_conv",
            )
            self.aux_bn = BatchNormalization(
                momentum=BN_MOMENTUM, epsilon=BN_EPSILON, name="aux_bn"
            )
            self.identity_bn = BatchNormalization(
                momentum=BN_MOMENTUM, epsilon=BN_EPSILON, name="identity_bn"
            )
        super().build(input_shape)

    def call(self, inputs, training=None):
        if self.deploy:
            return self.deploy_conv(inputs)
        return tf.add_n(
            [
                self.main_bn(self.main_conv(inputs), training=training),
                self.aux_bn(self.aux_conv(inputs), training=training),
                self.identity_bn(inputs, training=training),
            ]
        )

    def equivalent_weights(self):
        if self.deploy:
            raise ValueError("equivalent_weights must be called on the training graph")
        main_kernel = self.main_conv.get_weights()[0]
        kernel, bias = _fuse_conv_bn(main_kernel, self.main_bn, depthwise=True)
        aux_kernel, aux_bias = _fuse_conv_bn(
            self.aux_conv.get_weights()[0], self.aux_bn, depthwise=True
        )
        kernel += _pad_kernel(aux_kernel, main_kernel.shape)
        bias += aux_bias
        identity = np.zeros_like(main_kernel)
        identity[
            int(main_kernel.shape[0]) // 2,
            int(main_kernel.shape[1]) // 2,
            :,
            0,
        ] = 1.0
        identity, identity_bias = _fuse_conv_bn(
            identity, self.identity_bn, depthwise=True
        )
        kernel += identity
        bias += identity_bias
        return [kernel, bias]


def _normalize_stage_iterations(num_iterations) -> Tuple[int, ...]:
    if isinstance(num_iterations, int):
        if num_iterations < 1:
            raise ValueError("num_iterations must be >= 1")
        return (num_iterations,) * 7
    if isinstance(num_iterations, (list, tuple)) and len(num_iterations) == 7:
        values = tuple(int(value) for value in num_iterations)
        if any(value < 1 for value in values):
            raise ValueError("Each stage num_iterations must be >= 1")
        return values
    raise ValueError("num_iterations must be an int or a list/tuple of 7 positive ints")


def _residual_blocks(x, filters: int, iterations: int, prefix: str, deploy: bool):
    for index in range(iterations):
        block = "{}_{}".format(prefix, index + 1)
        x = ReLU(name=block + "_input_relu")(x)
        shortcut = x
        x = RepConv2D(
            filters // 2, kernel_size=1, padding="valid", deploy=deploy,
            name=block + "_reduce",
        )(x)
        x = ReLU(name=block + "_reduce_relu")(x)
        x = RepDepthwiseConv2D(deploy=deploy, name=block + "_depthwise")(x)
        x = ReLU(name=block + "_depthwise_relu")(x)
        x = RepConv2D(
            filters, kernel_size=1, padding="valid", deploy=deploy,
            name=block + "_expand",
        )(x)
        x = Add(name=block + "_add")([shortcut, x])
    return x


def _downsample_block(x, filters: int, channel_align: bool, prefix: str, deploy: bool):
    x = ReLU(name=prefix + "_input_relu")(x)
    shortcut = MaxPooling2D(
        pool_size=2, strides=2, padding="valid", name=prefix + "_shortcut_pool"
    )(x)
    if channel_align:
        shortcut = RepConv2D(
            filters, kernel_size=1, padding="valid", deploy=deploy,
            name=prefix + "_shortcut_align",
        )(shortcut)
    # The stride-2 2x2 convolution stays single-branch because its sampling
    # alignment is not exactly equivalent to a stride-2 1x1 auxiliary branch.
    x = Conv2D(
        filters // 2, kernel_size=2, strides=2, padding="valid", use_bias=True,
        name=prefix + "_downsample",
    )(x)
    x = ReLU(name=prefix + "_downsample_relu")(x)
    x = RepDepthwiseConv2D(deploy=deploy, name=prefix + "_depthwise")(x)
    x = ReLU(name=prefix + "_depthwise_relu")(x)
    x = RepConv2D(
        filters, kernel_size=1, padding="valid", deploy=deploy,
        name=prefix + "_expand",
    )(x)
    return Add(name=prefix + "_add")([x, shortcut])


def hand_landmark_2d_model(
    input_size=(1, 256, 256), num_iterations=8, deploy: bool = False
):
    """Build the unchanged NCHW-in / three-head-out Hand interface."""

    iterations: Sequence[int] = _normalize_stage_iterations(num_iterations)
    inputs = Input(input_size)
    x = Permute((2, 3, 1), name="input_nchw_to_nhwc")(inputs)
    x = Conv2D(
        16, kernel_size=3, strides=2, padding="same", use_bias=True,
        name="stem_conv",
    )(x)

    channels = (16, 32, 64, 256, 256, 256, 256)
    for stage, (filters, repeat) in enumerate(zip(channels, iterations), start=1):
        x = _residual_blocks(x, filters, repeat, "stage{}".format(stage), deploy)
        if stage < 7:
            next_filters = channels[stage]
            x = _downsample_block(
                x,
                next_filters,
                channel_align=next_filters != filters,
                prefix="stage{}_down".format(stage),
                deploy=deploy,
            )

    x = ReLU(name="head_relu")(x)
    hand_flag = Conv2D(
        1, kernel_size=2, padding="valid", use_bias=True, name="conv_handflag"
    )(x)
    hand_flag = Activation("sigmoid", name="activation_handflag")(hand_flag)
    handedness = Conv2D(
        1, kernel_size=2, padding="valid", use_bias=True, name="conv_handedness"
    )(x)
    handedness = Activation("sigmoid", name="activation_handedness")(handedness)
    landmarks = Conv2D(
        42, kernel_size=2, padding="valid", use_bias=True, name="convld_21_2d"
    )(x)
    return Model(inputs, [landmarks, hand_flag, handedness], name="hand_landmarker_v2")


def reparameterize_for_deploy(training_model, num_iterations=8):
    """Return a branch-fused deploy model with numerically equivalent outputs."""

    deploy_model = hand_landmark_2d_model(
        input_size=(1, 256, 256), num_iterations=num_iterations, deploy=True
    )
    for source in training_model.layers:
        target = deploy_model.get_layer(source.name)
        if isinstance(source, (RepConv2D, RepDepthwiseConv2D)):
            target.deploy_conv.set_weights(source.equivalent_weights())
        elif source.weights:
            target.set_weights(source.get_weights())
    return deploy_model
