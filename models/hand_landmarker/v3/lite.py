"""Iris v3-lite with a smaller deploy-time channel schedule.

The block topology, nonlinearities, downsampling schedule and three output
heads remain identical to v2.  Only backbone widths are reduced, so the board
interface and A1 operator set are unchanged while deploy parameters and MACs
fall materially.
"""

from __future__ import annotations

from typing import Sequence

import tensorflow as tf
from tensorflow.keras.layers import Activation, Add, Conv2D, Input, MaxPooling2D, Permute, ReLU
from tensorflow.keras.models import Model

from ..v2 import (
    DEFAULT_STAGE_ITERATIONS,
    RepConv2D,
    RepDepthwiseConv2D,
    _bottleneck_channels,
    _normalize_stage_iterations,
)


STAGE_CHANNELS = (16, 24, 40, 80, 112, 160, 224)


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
        pool_size=2,
        strides=2,
        padding="valid",
        name=prefix + "_shortcut_pool",
    )(x)
    if channel_align:
        shortcut = RepConv2D(
            filters,
            kernel_size=1,
            padding="valid",
            deploy=deploy,
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
    x = RepConv2D(
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
    return Model(inputs, [landmarks, hand_flag, handedness], name="hand_landmarker_v3_lite")


def reparameterize_for_deploy(training_model, num_iterations=DEFAULT_STAGE_ITERATIONS):
    deploy_model = hand_landmark_2d_model(
        input_size=(1, 256, 256),
        num_iterations=num_iterations,
        deploy=True,
    )
    for source in training_model.layers:
        if isinstance(source, (RepConv2D, RepDepthwiseConv2D)):
            target = deploy_model.get_layer(source.name)
            target.deploy_conv.set_weights(source.equivalent_weights())
        elif source.weights:
            target = deploy_model.get_layer(source.name)
            target.set_weights(source.get_weights())
    return deploy_model
