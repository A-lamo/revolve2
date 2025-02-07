import math
from abc import abstractmethod

import numpy as np
import numpy.typing as npt

from ...body.base import ActiveHinge, Body
from .._brain import Brain
from .._brain_instance import BrainInstance
from ._brain_cpg_instance_fish import BrainInstanceFish
from ._make_cpg_network_structure_neighbor import (
    active_hinges_to_cpg_network_structure_neighbor,
)

class BrainFish(Brain):
    """
    This is our custom brain.

    It stores references to each hinge of the robot body so they can be controlled individually.
    A brain has a function `make_instance`, which creates the actual object that controls a robot.
    """
    _initial_state: npt.NDArray[np.float_]
    _output_mapping: list[tuple[int, ActiveHinge]]
    _weight_matrix: npt.NDArray[np.float_]
    _A: npt.NDArray[np.float_]
    _W: npt.NDArray[np.float_]
    _LAMBDA: np.float_

    def __init__(
        self,
        body : Body,
        _W=0.5,
        
    ) -> None:
        """
        Initialize this object.
        :param body: The body to create the cpg network and brain for.
        """
        active_hinges = body.find_modules_of_type(ActiveHinge)

        # self._W = self._make_W(active_hinges, body)
        self._W = np.asarray([_W for i in range(len(active_hinges))])

        (cpg_network_structure,self._output_mapping,) = active_hinges_to_cpg_network_structure_neighbor(active_hinges)
        # print('CPG-NETWORK-STRUCTURE-CONNECTIONS : ', cpg_network_structure.connections)
        
        connections = [
            (
                active_hinges[pair.cpg_index_lowest.index],
                active_hinges[pair.cpg_index_highest.index],
            )
            for pair in cpg_network_structure.connections
        ]

        self._initial_state = cpg_network_structure.make_uniform_state(math.sqrt(2) / 2.0)

        (internal_weights,external_weights,) = self._make_weights(active_hinges, connections, body)
        internal_weights = np.asarray(internal_weights)
        external_weights  = np.asarray(external_weights)

        # print('INTERNAL-WEIGHTS : ', internal_weights)
        # print('EXTERNAL-WEIGHTS : ', external_weights)

        self._weight_matrix = cpg_network_structure.make_connection_weights_matrix(
            {
                cpg: weight
                for cpg, weight in zip(cpg_network_structure.cpgs, internal_weights)
            },
            {
                pair: weight
                for pair, weight in zip(
                    cpg_network_structure.connections, external_weights
                )
            },
        )

        self._A = np.asarray([0.5 for i in range(len(active_hinges))])
        self._LAMBDA = 0.1

    def make_instance(self) -> BrainInstance:
        """
        Create an instance of this brain.

        :returns: The created instance.
        """
        return BrainInstanceFish(
            initial_state=self._initial_state,
            output_mapping=self._output_mapping,
            weight_matrix=self._weight_matrix,
            A=self._A,
            W=self._W,
            LAMBDA=self._LAMBDA
        )

    @abstractmethod
    def _make_weights(
        self,
        active_hinges: list[ActiveHinge],
        connections: list[tuple[ActiveHinge, ActiveHinge]],
        body: Body,
    ) -> tuple[list[float], list[float]]:
        """
        Define the weights between neurons.

        :param active_hinges: The active hinges corresponding to each cpg.
        :param connections: Pairs of active hinges corresponding to pairs of cpgs that are connected.
                            Connection is from hinge 0 to hinge 1.
                            Opposite connection is not provided as weights are assumed to be negative.
        :param body: The body that matches this brain.
        :returns: Two lists. The first list contains the internal weights in cpgs, corresponding to `active_hinges`
                 The second list contains the weights between connected cpgs, corresponding to `connections`
                 The lists should match the order of the input parameters.
        """
    