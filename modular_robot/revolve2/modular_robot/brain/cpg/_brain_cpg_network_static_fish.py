from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ...body.base import ActiveHinge
from .._brain import Brain
from .._brain_instance import BrainInstance
from ._brain_cpg_instance_fish import BrainInstanceFish
from ._cpg_network_structure import CpgNetworkStructure


class BrainCpgNetworkStaticFish(Brain):
    """
    A CPG brain with cpgs and connections defined by the user.

    A state vector is integrated over time using a weight matrix which multiplication with the state vector sum defines the derivative of the state vector.
    I.e X' = WX

    The first `num_output_neurons` in the state vector are the outputs for the controller created by this brain.
    """

    _initial_state: npt.NDArray[np.float_]
    _weight_matrix: npt.NDArray[np.float_]
    _output_mapping: list[tuple[int, ActiveHinge]]
    # new parameters to adapt the individual joints
    _A: npt.NDArray[np.float_]# amplitude
    _W: npt.NDArray[np.float_]# intrinsic frequency
    _LAMBDA: np.float_

    def __init__(
        self,
        initial_state: npt.NDArray[np.float_],
        weight_matrix: npt.NDArray[np.float_],
        output_mapping: list[tuple[int, ActiveHinge]],
        A: npt.NDArray[np.float_],
        W: npt.NDArray[np.float_],
        LAMBDA: np.float_
    ) -> None:
        """
        Initialize this object.

        :param initial_state: The initial state of the neural network.
        :param weight_matrix: The weight matrix used during integration.
        :param output_mapping: Marks neurons as controller outputs and map them to the correct active hinge.
        """
        self._initial_state = initial_state
        self._weight_matrix = weight_matrix
        self._output_mapping = output_mapping
        self._A = A
        self._W = W
        self._LAMBDA = LAMBDA

    @classmethod
    def uniform_from_params(
        cls,
        params: npt.NDArray[np.float_],
        cpg_network_structure: CpgNetworkStructure,
        initial_state_uniform: float,
        output_mapping: list[tuple[int, ActiveHinge]],
    ) -> BrainCpgNetworkStaticFish:
        """
        Create and initialize an instance of this brain from the provided parameters, assuming uniform initial state.

        :param params: Parameters for the weight matrix to be created.
        :param cpg_network_structure: The cpg network structure.
        :param initial_state_uniform: Initial state to use for all neurons.
        :param output_mapping: Marks neurons as controller outputs and map them to the correct active hinge.
        :returns: The created brain.
        """

        initial_state = cpg_network_structure.make_uniform_state(initial_state_uniform)
        weight_matrix = (
            cpg_network_structure.make_connection_weights_matrix_from_params(
                list(params[:len(cpg_network_structure.cpgs)]),
            )
        )

        amplitudes = cpg_network_structure.make_A(params[len(cpg_network_structure.cpgs):len(cpg_network_structure.cpgs)*2])
        intrinsic_frequencies = cpg_network_structure.make_W(params[len(cpg_network_structure.cpgs*2):])

        return BrainCpgNetworkStaticFish(
            initial_state=initial_state,
            weight_matrix=weight_matrix,
            output_mapping=output_mapping,
            A=amplitudes,
            W=intrinsic_frequencies,
            LAMBDA=0.1
        )

    def make_instance(self) -> BrainInstance:
        """
        Create an instance of this brain.

        :returns: The created instance.
        """
        return BrainInstanceFish(
            initial_state=self._initial_state,
            weight_matrix=self._weight_matrix,
            output_mapping=self._output_mapping,
            A=self._A,
            W=self._W,
            LAMBDA=self._LAMBDA
        )