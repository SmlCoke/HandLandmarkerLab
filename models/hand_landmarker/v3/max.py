"""Iris v3-max with a materially larger, exactly fusible training graph.

Each deploy convolution is expanded into four Conv+BN branches during
training.  Eligible blocks also add an identity-BN branch; every 3x3
depthwise block adds a 1x1 depthwise branch as well.  All branches are linear
inside the block and therefore sum into one biased deploy convolution without
changing the v2 deployment topology or the fixed model I/O contract.
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

from ..v2 import (
    BN_EPSILON,
    BN_MOMENTUM,
    DEFAULT_STAGE_ITERATIONS,
    STAGE_CHANNELS,
    _bottleneck_channels,
    _fuse_conv_bn,
    _normalize_stage_iterations,
)


TRAIN_CONV_BRANCHES = 4


def _bn(
    name: str,
    zero_gamma: bool = False,
) -> BatchNormalization:
    return BatchNormalization(
        momentum=BN_MOMENTUM,
        epsilon=BN_EPSILON,
        gamma_initializer="zeros" if zero_gamma else "ones",
        name=name,
    )


def _pad_kernel(kernel: np.ndarray, target_size: int) -> np.ndarray:
    source_size = int(kernel.shape[0])
    if source_size == target_size:
        return kernel
    if source_size > target_size or (target_size - source_size) % 2:
        raise ValueError(
            "Cannot center-pad {}x{} kernel to {}x{}".format(
                source_size, source_size, target_size, target_size
            )
        )
    margin = (target_size - source_size) // 2
    return np.pad(kernel, ((margin, margin), (margin, margin), (0, 0), (0, 0)))


def _identity_kernel(
    channels: int,
    kernel_size: int,
    depthwise: bool,
) -> np.ndarray:
    center = kernel_size // 2
    if depthwise:
        kernel = np.zeros((kernel_size, kernel_size, channels, 1), dtype=np.float32)
        kernel[center, center, :, 0] = 1.0
        return kernel
    kernel = np.zeros(
        (kernel_size, kernel_size, channels, channels), dtype=np.float32
    )
    indices = np.arange(channels)
    kernel[center, center, indices, indices] = 1.0
    return kernel


def _fuse_identity_bn(
    bn: BatchNormalization,
    channels: int,
    kernel_size: int,
    depthwise: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    gamma, beta, moving_mean, moving_variance = bn.get_weights()
    scale = gamma / np.sqrt(moving_variance + float(bn.epsilon))
    kernel = _identity_kernel(channels, kernel_size, depthwise)
    if depthwise:
        kernel = kernel * scale.reshape((1, 1, channels, 1))
    else:
        kernel = kernel * scale.reshape((1, 1, 1, channels))
    return kernel, beta - moving_mean * scale


class RepMultiBranchConv2D(Layer):
    """Four Conv+BN branches plus an optional identity-BN branch."""

    def __init__(
        self,
        filters: int,
        kernel_size=1,
        strides=1,
        padding="valid",
        deploy=False,
        zero_gamma=False,
        branch_count=TRAIN_CONV_BRANCHES,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.filters = int(filters)
        self.kernel_size = int(kernel_size)
        self.strides = int(strides)
        self.padding = str(padding)
        self.deploy = bool(deploy)
        self.zero_gamma = bool(zero_gamma)
        self.branch_count = int(branch_count)
        if self.branch_count < 2:
            raise ValueError("v3-max requires at least two train-time convolution branches")

    def build(self, input_shape):
        input_channels = int(input_shape[-1])
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
            self.train_convs = []
            self.train_bns = []
            for index in range(self.branch_count):
                suffix = str(index + 1)
                self.train_convs.append(
                    Conv2D(
                        self.filters,
                        self.kernel_size,
                        strides=self.strides,
                        padding=self.padding,
                        use_bias=False,
                        name="train_conv_{}".format(suffix),
                    )
                )
                self.train_bns.append(
                    _bn("train_bn_{}".format(suffix), self.zero_gamma)
                )
            self.identity_enabled = (
                self.strides == 1
                and self.padding == "same"
                and input_channels == self.filters
                and self.kernel_size % 2 == 1
            )
            self.identity_bn = (
                _bn("identity_bn", self.zero_gamma)
                if self.identity_enabled
                else None
            )
        super().build(input_shape)

    def call(self, inputs, training=None):
        if self.deploy:
            return self.deploy_conv(inputs)
        outputs = [
            bn(conv(inputs), training=training)
            for conv, bn in zip(self.train_convs, self.train_bns)
        ]
        if self.identity_bn is not None:
            outputs.append(self.identity_bn(inputs, training=training))
        return tf.add_n(outputs)

    def equivalent_weights(self):
        if self.deploy:
            raise ValueError("equivalent_weights must be called on the training graph")
        kernels = []
        biases = []
        for conv, bn in zip(self.train_convs, self.train_bns):
            kernel, bias = _fuse_conv_bn(conv.get_weights()[0], bn)
            kernels.append(kernel)
            biases.append(bias)
        if self.identity_bn is not None:
            channels = int(self.train_convs[0].kernel.shape[2])
            kernel, bias = _fuse_identity_bn(
                self.identity_bn,
                channels,
                self.kernel_size,
                depthwise=False,
            )
            kernels.append(kernel)
            biases.append(bias)
        return [np.sum(kernels, axis=0), np.sum(biases, axis=0)]


class RepMultiBranchDepthwiseConv2D(Layer):
    """Four 3x3 branches plus 1x1 and identity branches, fused to one DWConv."""

    def __init__(
        self,
        kernel_size=3,
        strides=1,
        padding="same",
        deploy=False,
        zero_gamma=False,
        branch_count=TRAIN_CONV_BRANCHES,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.kernel_size = int(kernel_size)
        self.strides = int(strides)
        self.padding = str(padding)
        self.deploy = bool(deploy)
        self.zero_gamma = bool(zero_gamma)
        self.branch_count = int(branch_count)
        if self.branch_count < 2:
            raise ValueError("v3-max requires at least two train-time depthwise branches")

    def build(self, input_shape):
        self.input_channels = int(input_shape[-1])
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
            self.train_convs = []
            self.train_bns = []
            for index in range(self.branch_count):
                suffix = str(index + 1)
                self.train_convs.append(
                    DepthwiseConv2D(
                        self.kernel_size,
                        strides=self.strides,
                        padding=self.padding,
                        depth_multiplier=1,
                        use_bias=False,
                        name="train_conv_{}".format(suffix),
                    )
                )
                self.train_bns.append(
                    _bn("train_bn_{}".format(suffix), self.zero_gamma)
                )
            self.scale_enabled = (
                self.kernel_size > 1
                and self.kernel_size % 2 == 1
                and self.strides == 1
                and self.padding == "same"
            )
            if self.scale_enabled:
                self.scale_conv = DepthwiseConv2D(
                    1,
                    strides=1,
                    padding="same",
                    depth_multiplier=1,
                    use_bias=False,
                    name="scale_conv",
                )
                self.scale_bn = _bn("scale_bn", self.zero_gamma)
                self.identity_bn = _bn("identity_bn", self.zero_gamma)
            else:
                self.scale_conv = None
                self.scale_bn = None
                self.identity_bn = None
        super().build(input_shape)

    def call(self, inputs, training=None):
        if self.deploy:
            return self.deploy_conv(inputs)
        outputs = [
            bn(conv(inputs), training=training)
            for conv, bn in zip(self.train_convs, self.train_bns)
        ]
        if self.scale_conv is not None:
            outputs.append(
                self.scale_bn(self.scale_conv(inputs), training=training)
            )
            outputs.append(self.identity_bn(inputs, training=training))
        return tf.add_n(outputs)

    def equivalent_weights(self):
        if self.deploy:
            raise ValueError("equivalent_weights must be called on the training graph")
        kernels = []
        biases = []
        for conv, bn in zip(self.train_convs, self.train_bns):
            kernel, bias = _fuse_conv_bn(
                conv.get_weights()[0], bn, depthwise=True
            )
            kernels.append(kernel)
            biases.append(bias)
        if self.scale_conv is not None:
            kernel, bias = _fuse_conv_bn(
                self.scale_conv.get_weights()[0],
                self.scale_bn,
                depthwise=True,
            )
            kernels.append(_pad_kernel(kernel, self.kernel_size))
            biases.append(bias)
            kernel, bias = _fuse_identity_bn(
                self.identity_bn,
                self.input_channels,
                self.kernel_size,
                depthwise=True,
            )
            kernels.append(kernel)
            biases.append(bias)
        return [np.sum(kernels, axis=0), np.sum(biases, axis=0)]


def _residual_blocks(x, filters: int, iterations: int, prefix: str, deploy: bool):
    for index in range(iterations):
        block = "{}_{}".format(prefix, index + 1)
        x = ReLU(name=block + "_input_relu")(x)
        shortcut = x
        x = RepMultiBranchConv2D(
            _bottleneck_channels(filters),
            kernel_size=1,
            padding="valid",
            deploy=deploy,
            name=block + "_reduce",
        )(x)
        x = ReLU(name=block + "_reduce_relu")(x)
        x = RepMultiBranchDepthwiseConv2D(
            deploy=deploy, name=block + "_depthwise"
        )(x)
        x = ReLU(name=block + "_depthwise_relu")(x)
        x = RepMultiBranchConv2D(
            filters,
            kernel_size=1,
            padding="valid",
            deploy=deploy,
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
        shortcut = RepMultiBranchConv2D(
            filters,
            kernel_size=1,
            padding="valid",
            deploy=deploy,
            name=prefix + "_shortcut_align",
        )(shortcut)
    x = RepMultiBranchConv2D(
        _bottleneck_channels(filters),
        kernel_size=2,
        strides=2,
        padding="valid",
        deploy=deploy,
        name=prefix + "_downsample",
    )(x)
    x = ReLU(name=prefix + "_downsample_relu")(x)
    x = RepMultiBranchDepthwiseConv2D(
        deploy=deploy, name=prefix + "_depthwise"
    )(x)
    x = ReLU(name=prefix + "_depthwise_relu")(x)
    x = RepMultiBranchConv2D(
        filters,
        kernel_size=1,
        padding="valid",
        deploy=deploy,
        zero_gamma=True,
        name=prefix + "_expand",
    )(x)
    return Add(name=prefix + "_add")([x, shortcut])


def hand_landmark_2d_model(
    input_size=(1, 256, 256),
    num_iterations=DEFAULT_STAGE_ITERATIONS,
    deploy: bool = False,
):
    iterations: Sequence[int] = _normalize_stage_iterations(num_iterations)
    inputs = Input(input_size)
    x = Permute((2, 3, 1), name="input_nchw_to_nhwc")(inputs)
    x = RepMultiBranchConv2D(
        STAGE_CHANNELS[0],
        kernel_size=3,
        strides=2,
        padding="same",
        deploy=deploy,
        name="stem_conv",
    )(x)

    for stage, (filters, repeat) in enumerate(
        zip(STAGE_CHANNELS, iterations), start=1
    ):
        x = _residual_blocks(x, filters, repeat, "stage{}".format(stage), deploy)
        if stage < 7:
            next_filters = STAGE_CHANNELS[stage]
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
    return Model(
        inputs,
        [landmarks, hand_flag, handedness],
        name="hand_landmarker_v3_max",
    )


def reparameterize_for_deploy(
    training_model, num_iterations=DEFAULT_STAGE_ITERATIONS
):
    deploy_model = hand_landmark_2d_model(
        input_size=(1, 256, 256),
        num_iterations=num_iterations,
        deploy=True,
    )
    branch_types = (RepMultiBranchConv2D, RepMultiBranchDepthwiseConv2D)
    for source in training_model.layers:
        if isinstance(source, branch_types):
            target = deploy_model.get_layer(source.name)
            target.deploy_conv.set_weights(source.equivalent_weights())
        elif source.weights:
            target = deploy_model.get_layer(source.name)
            target.set_weights(source.get_weights())
    return deploy_model
