"""Stable Hand Landmarker v2 with deploy-time Conv+BN folding.

The training graph uses batch-normalized convolutions and zero-initialized
residual tails.  Every ``Rep*`` layer folds into one biased convolution for
deployment, so the exported graph contains no BatchNormalization and keeps
the fixed board interface and operator set.
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
STAGE_CHANNELS = (24, 32, 64, 128, 192, 256, 384)
DEFAULT_STAGE_ITERATIONS = (2, 2, 3, 4, 4, 6, 6)
MAX_DEPTHWISE_CHANNELS = 128


def _fuse_conv_bn(kernel, bn: BatchNormalization, depthwise: bool = False):
    gamma, beta, moving_mean, moving_variance = bn.get_weights()
    scale = gamma / np.sqrt(moving_variance + float(bn.epsilon))
    if depthwise:
        shape = (1, 1, int(kernel.shape[2]), int(kernel.shape[3]))
    else:
        shape = (1, 1, 1, int(kernel.shape[3]))
    return kernel * scale.reshape(shape), beta - moving_mean * scale


class RepConv2D(Layer):
    """Batch-normalized Conv2D that folds into one deploy Conv2D."""

    def __init__(
        self,
        filters: int,
        kernel_size=1,
        strides=1,
        padding="valid",
        deploy=False,
        zero_gamma=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.filters = int(filters)
        self.kernel_size = int(kernel_size)
        self.strides = int(strides)
        self.padding = str(padding)
        self.deploy = bool(deploy)
        self.zero_gamma = bool(zero_gamma)

    def build(self, input_shape):
        if self.deploy:
            self.deploy_conv = Conv2D(
                self.filters,
                self.kernel_size,
                strides=self.strides,
                padding=self.padding,
                use_bias=True,
                name="deploy_conv",
            )
        else:
            self.train_conv = Conv2D(
                self.filters,
                self.kernel_size,
                strides=self.strides,
                padding=self.padding,
                use_bias=False,
                name="train_conv",
            )
            self.train_bn = BatchNormalization(
                momentum=BN_MOMENTUM,
                epsilon=BN_EPSILON,
                gamma_initializer="zeros" if self.zero_gamma else "ones",
                name="train_bn",
            )
        super().build(input_shape)

    def call(self, inputs, training=None):
        if self.deploy:
            return self.deploy_conv(inputs)
        return self.train_bn(self.train_conv(inputs), training=training)

    def equivalent_weights(self):
        if self.deploy:
            raise ValueError("equivalent_weights must be called on the training graph")
        kernel = self.train_conv.get_weights()[0]
        fused_kernel, fused_bias = _fuse_conv_bn(kernel, self.train_bn)
        return [fused_kernel, fused_bias]


class RepDepthwiseConv2D(Layer):
    """Batch-normalized depthwise convolution folded for deployment."""

    def __init__(
        self,
        kernel_size=3,
        strides=1,
        padding="same",
        deploy=False,
        zero_gamma=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.kernel_size = int(kernel_size)
        self.strides = int(strides)
        self.padding = str(padding)
        self.deploy = bool(deploy)
        self.zero_gamma = bool(zero_gamma)

    def build(self, input_shape):
        if self.deploy:
            self.deploy_conv = DepthwiseConv2D(
                self.kernel_size,
                strides=self.strides,
                padding=self.padding,
                depth_multiplier=1,
                use_bias=True,
                name="deploy_conv",
            )
        else:
            self.train_conv = DepthwiseConv2D(
                self.kernel_size,
                strides=self.strides,
                padding=self.padding,
                depth_multiplier=1,
                use_bias=False,
                name="train_conv",
            )
            self.train_bn = BatchNormalization(
                momentum=BN_MOMENTUM,
                epsilon=BN_EPSILON,
                gamma_initializer="zeros" if self.zero_gamma else "ones",
                name="train_bn",
            )
        super().build(input_shape)

    def call(self, inputs, training=None):
        if self.deploy:
            return self.deploy_conv(inputs)
        return self.train_bn(self.train_conv(inputs), training=training)

    def equivalent_weights(self):
        if self.deploy:
            raise ValueError("equivalent_weights must be called on the training graph")
        kernel = self.train_conv.get_weights()[0]
        fused_kernel, fused_bias = _fuse_conv_bn(
            kernel, self.train_bn, depthwise=True
        )
        return [fused_kernel, fused_bias]


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


def _bottleneck_channels(filters: int) -> int:
    """Keep depthwise groups within the previously converted A1 envelope."""

    return min(max(8, int(filters) // 2), MAX_DEPTHWISE_CHANNELS)


def _residual_blocks(x, filters: int, iterations: int, prefix: str, deploy: bool):
    for index in range(iterations):
        block = "{}_{}".format(prefix, index + 1)
        x = ReLU(name=block + "_input_relu")(x)
        shortcut = x
        x = RepConv2D(
            _bottleneck_channels(filters),
            kernel_size=1,
            padding="valid",
            deploy=deploy,
            name=block + "_reduce",
        )(x)
        x = ReLU(name=block + "_reduce_relu")(x)
        x = RepDepthwiseConv2D(deploy=deploy, name=block + "_depthwise")(x)
        x = ReLU(name=block + "_depthwise_relu")(x)
        x = RepConv2D(
            filters, kernel_size=1, padding="valid", deploy=deploy,
            zero_gamma=True,
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
    x = RepConv2D(
        _bottleneck_channels(filters),
        kernel_size=2,
        strides=2,
        padding="valid",
        deploy=deploy,
        name=prefix + "_downsample",
    )(x)
    x = ReLU(name=prefix + "_downsample_relu")(x)
    x = RepDepthwiseConv2D(deploy=deploy, name=prefix + "_depthwise")(x)
    x = ReLU(name=prefix + "_depthwise_relu")(x)
    x = RepConv2D(
        filters, kernel_size=1, padding="valid", deploy=deploy,
        zero_gamma=True,
        name=prefix + "_expand",
    )(x)
    return Add(name=prefix + "_add")([x, shortcut])


def hand_landmark_2d_model(
    input_size=(1, 256, 256),
    num_iterations=DEFAULT_STAGE_ITERATIONS,
    deploy: bool = False,
):
    """Build the unchanged NCHW-in / three-head-out Hand interface."""

    iterations: Sequence[int] = _normalize_stage_iterations(num_iterations)
    inputs = Input(input_size)
    x = Permute((2, 3, 1), name="input_nchw_to_nhwc")(inputs)
    x = RepConv2D(
        STAGE_CHANNELS[0], kernel_size=3, strides=2, padding="same", deploy=deploy,
        name="stem_conv",
    )(x)

    channels = STAGE_CHANNELS
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
        1,
        kernel_size=2,
        padding="valid",
        use_bias=True,
        kernel_initializer=tf.keras.initializers.RandomNormal(
            mean=0.0, stddev=0.001, seed=20260715
        ),
        bias_initializer="zeros",
        name="conv_handflag",
    )(x)
    hand_flag = Activation("sigmoid", name="activation_handflag")(hand_flag)
    handedness = Conv2D(
        1,
        kernel_size=2,
        padding="valid",
        use_bias=True,
        kernel_initializer=tf.keras.initializers.RandomNormal(
            mean=0.0, stddev=0.001, seed=20260716
        ),
        bias_initializer="zeros",
        name="conv_handedness",
    )(x)
    handedness = Activation("sigmoid", name="activation_handedness")(handedness)
    landmarks = Conv2D(
        42,
        kernel_size=2,
        padding="valid",
        use_bias=True,
        kernel_initializer="zeros",
        bias_initializer=tf.keras.initializers.Constant(0.5),
        name="convld_21_2d",
    )(x)
    return Model(inputs, [landmarks, hand_flag, handedness], name="hand_landmarker_v2")


def reparameterize_for_deploy(
    training_model, num_iterations=DEFAULT_STAGE_ITERATIONS
):
    """Return a branch-fused deploy model with numerically equivalent outputs."""

    deploy_model = hand_landmark_2d_model(
        input_size=(1, 256, 256), num_iterations=num_iterations, deploy=True
    )
    for source in training_model.layers:
        if isinstance(source, (RepConv2D, RepDepthwiseConv2D)):
            target = deploy_model.get_layer(source.name)
            target.deploy_conv.set_weights(source.equivalent_weights())
        elif source.weights:
            target = deploy_model.get_layer(source.name)
            target.set_weights(source.get_weights())
    return deploy_model
