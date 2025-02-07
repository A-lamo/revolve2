from typing import cast

import multineat
import numpy as np

from revolve2.modular_robot.body.base import ActiveHinge, Body
from revolve2.modular_robot.body.v2 import ActiveHingeV2, BrickV2
# ! changed import !
from revolve2.modular_robot.brain.cpg import BrainFish


# ARON'S ADDITION
class BrainCpgNetworkNeighborV1Fish(BrainFish):
    """
    A CPG brain based on `BrainFish` that creates weights from a CPPNWIN network.

    Weights are determined by querying the CPPN network with inputs:
    (hinge1_posx, hinge1_posy, hinge1_posz, hinge2_posx, hinge2_posy, hinge3_posz)
    If the weight in internal, hinge1 and hinge2 position will be the same.
    """

    _genotype: multineat.Genome

    def __init__(self, genotype: multineat.Genome, body: Body, include_bias: bool) -> None:
        """
        Initialize this object.

        :param genotype: A multineat genome used for determining weights.
        :param body: The body of the robot.
        """
        self._genotype = genotype
        self.include_bias = include_bias
        super().__init__(body)

    def _make_weights(
        self,
        active_hinges: list[ActiveHinge],
        connections: list[tuple[ActiveHinge, ActiveHinge]],
        body: Body,
    ) -> tuple[list[float], list[float]]:
        brain_net = multineat.NeuralNetwork()
        self._genotype.BuildPhenotype(brain_net)

        # Get the grid and core grid position
        grid, core_grid_position, id_string = body.to_grid(ActiveHingeV2, BrickV2)
        self.id_string = id_string

        # Get positions of joints
        if len(active_hinges) > 0:
            positions = []
            for hinge in active_hinges:
                position = np.where(grid == hinge)
                position = np.array([position[0], position[1], position[2]]).flatten()
                position = position - core_grid_position
                positions.append(position)
        else:
            positions = []
        
        # Internal weights
        if self.include_bias:
            internal_weights = [self._evaluate_network(brain_net,[1.0,
                        float(pos[0]), float(pos[1]), float(pos[2]), float(pos[0]), float(pos[1]), float(pos[2]),],)
                for pos in positions]
        else:
            internal_weights = [self._evaluate_network(brain_net,[
                float(pos[0]), float(pos[1]), float(pos[2]), float(pos[0]), float(pos[1]), float(pos[2]),],)
                for pos in positions]

        # External weights
        connecting_positions = []
        for connection in connections:
            idx1 = active_hinges.index(connection[0])
            idx2 = active_hinges.index(connection[1])
            connecting_positions.append((positions[idx1], positions[idx2]))

        if self.include_bias:
            external_weights = [
                self._evaluate_network(brain_net,[1.0,
                        float(pos1[0]), float(pos1[1]), float(pos1[2]), float(pos2[0]), float(pos2[1]), float(pos2[2]),],)
                for (pos1, pos2) in connecting_positions]
        else:
            external_weights = [
                self._evaluate_network(brain_net,[
                float(pos1[0]), float(pos1[1]), float(pos1[2]), float(pos2[0]), float(pos2[1]), float(pos2[2]),],)
                for (pos1, pos2) in connecting_positions]

        return (internal_weights, external_weights)
    

    @staticmethod
    def _evaluate_network(
        network: multineat.NeuralNetwork, inputs: list[float]
    ) -> float:
        network.Input(inputs)
        network.ActivateAllLayers()
        return cast(float, network.Output()[0])