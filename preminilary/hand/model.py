# File purpose: Define the 2D hand-landmark Keras network architecture in NHWC execution style.
# Hand landmark 2d model - NHWC version
# Uses default Keras data format (channels_last).
from tensorflow.keras.layers import Activation, Add, Conv2D, DepthwiseConv2D, Input, LeakyReLU, MaxPooling2D, Permute
from tensorflow.keras.models import Model


LEAKY_RELU_ALPHA = 0.1


def conv_blocks(x, num_filter, num_iterations=1):
    for _ in range(0, num_iterations):
        x = LeakyReLU(alpha=LEAKY_RELU_ALPHA)(x)

        shortcut = x
        x = Conv2D(
            int(num_filter / 2),
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="valid",
            use_bias=True,
        )(x)
        x = LeakyReLU(alpha=LEAKY_RELU_ALPHA)(x)
        x = DepthwiseConv2D(
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            use_bias=True,
        )(x)
        x = Conv2D(
            num_filter,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="valid",
            use_bias=True,
        )(x)
        x = Add()([shortcut, x])
    return x


def conv_blocks_with_pooling(x, num_filter, channel_align=False):
    x = LeakyReLU(alpha=LEAKY_RELU_ALPHA)(x)

    shortcut = x
    shortcut = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(shortcut)

    if channel_align:
        shortcut = Conv2D(
            num_filter,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="valid",
            use_bias=True,
        )(shortcut)

    x = Conv2D(
        int(num_filter / 2),
        kernel_size=(2, 2),
        strides=(2, 2),
        padding="valid",
        use_bias=True,
    )(x)
    x = LeakyReLU(alpha=LEAKY_RELU_ALPHA)(x)
    x = DepthwiseConv2D(
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        use_bias=True,
    )(x)
    x = Conv2D(
        num_filter,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="valid",
        use_bias=True,
    )(x)
    x = Add()([x, shortcut])
    return x


def _normalize_stage_iterations(num_iterations):
    if isinstance(num_iterations, int):
        if num_iterations < 1:
            raise ValueError("num_iterations must be >= 1")
        return (num_iterations,) * 7

    if isinstance(num_iterations, (list, tuple)):
        if len(num_iterations) != 7:
            raise ValueError("num_iterations list/tuple must have 7 elements")
        normalized = []
        for value in num_iterations:
            value = int(value)
            if value < 1:
                raise ValueError("Each stage num_iterations must be >= 1")
            normalized.append(value)
        return tuple(normalized)

    raise TypeError("num_iterations must be int or list/tuple of 7 ints")


def hand_landmark_2d_model(input_size=(1, 256, 256), num_iterations=8):
    it1, it2, it3, it4, it5, it6, it7 = _normalize_stage_iterations(num_iterations)

    inputs = Input(input_size)
    x = Permute((2, 3, 1), name="input_nchw_to_nhwc")(inputs)
    x = Conv2D(16, kernel_size=(3, 3), strides=(2, 2), padding="same", use_bias=True)(x)

    # block 1 ~ 9
    x = conv_blocks(x, 16, num_iterations=it1)
    x = conv_blocks_with_pooling(x, 32, channel_align=True)

    # block 10 ~ 18
    x = conv_blocks(x, 32, num_iterations=it2)
    x = conv_blocks_with_pooling(x, 64, channel_align=True)

    # block 19 ~ 27
    x = conv_blocks(x, 64, num_iterations=it3)
    x = conv_blocks_with_pooling(x, 256, channel_align=True)

    # block 28 ~ 36
    x = conv_blocks(x, 256, num_iterations=it4)
    x = conv_blocks_with_pooling(x, 256, channel_align=False)

    # block 37 ~ 45
    x = conv_blocks(x, 256, num_iterations=it5)
    x = conv_blocks_with_pooling(x, 256, channel_align=False)

    # block 46 ~ 54
    x = conv_blocks(x, 256, num_iterations=it6)
    x = conv_blocks_with_pooling(x, 256, channel_align=False)

    # block 55 ~ 63
    x = conv_blocks(x, 256, num_iterations=it7)

    # Last layer
    x = LeakyReLU(alpha=LEAKY_RELU_ALPHA)(x)
    hand_flag = Conv2D(
        1,
        kernel_size=(2, 2),
        strides=(1, 1),
        padding="valid",
        use_bias=True,
        name="conv_handflag",
    )(x)
    hand_flag = Activation("sigmoid", name="activation_handflag")(hand_flag)

    handedness = Conv2D(
        1,
        kernel_size=(2, 2),
        strides=(1, 1),
        padding="valid",
        use_bias=True,
        name="conv_handedness",
    )(x)
    handedness = Activation("sigmoid", name="activation_handedness")(handedness)

    landmarks = Conv2D(
        42,
        kernel_size=(2, 2),
        strides=(1, 1),
        padding="valid",
        use_bias=True,
        name="convld_21_2d",
    )(x)

    model = Model(inputs, [landmarks, hand_flag, handedness])
    return model
