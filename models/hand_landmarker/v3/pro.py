"""Iris v3-pro: the unchanged v2 architecture under the v3 model family."""

from __future__ import annotations

from ..v2 import DEFAULT_STAGE_ITERATIONS
from ..v2 import hand_landmark_2d_model as _build_v2
from ..v2 import reparameterize_for_deploy as _reparameterize_v2


def hand_landmark_2d_model(
    input_size=(1, 256, 256),
    num_iterations=DEFAULT_STAGE_ITERATIONS,
    deploy: bool = False,
):
    """Build the byte-for-byte v2 layer topology for the v3-pro experiment."""

    return _build_v2(
        input_size=input_size,
        num_iterations=num_iterations,
        deploy=deploy,
    )


def reparameterize_for_deploy(training_model, num_iterations=DEFAULT_STAGE_ITERATIONS):
    """Use the unchanged v2 Conv/DepthwiseConv plus BN folding path."""

    return _reparameterize_v2(
        training_model, num_iterations=num_iterations
    )
