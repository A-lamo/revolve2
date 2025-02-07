from __future__ import annotations

from dataclasses import dataclass

import multineat
import numpy as np
from typing_extensions import Self

from revolve2.modular_robot.body.base import Body

from .._multineat_genotype_pickle_wrapper import MultineatGenotypePickleWrapper
from .._multineat_rng_from_random import multineat_rng_from_random
from .._random_multineat_genotype import random_multineat_genotype
# ! changed import !
from ._brain_cpg_network_neighbor_fish import BrainCpgNetworkNeighborV1Fish
from ._multineat_params import get_multineat_params

_MULTINEAT_PARAMS = get_multineat_params()

@dataclass
class BrainGenotypeCpgFish:
    """An SQLAlchemy model for a CPPNWIN cpg brain genotype."""

    _NUM_INITIAL_MUTATIONS = 5

    brain: MultineatGenotypePickleWrapper

    @classmethod
    def random_brain(
        cls,
        innov_db: multineat.InnovationDatabase,
        rng: np.random.Generator,
        include_bias: bool
    ) -> BrainGenotypeCpgFish:
        """
        Create a random genotype.

        :param innov_db: Multineat innovation database. See Multineat library.
        :param rng: Random number generator.
        :returns: The created genotype.
        """
        multineat_rng = multineat_rng_from_random(rng)

        brain = MultineatGenotypePickleWrapper(
            random_multineat_genotype(
                innov_db=innov_db,
                rng=multineat_rng,
                multineat_params=_MULTINEAT_PARAMS,
                output_activation_func=multineat.ActivationFunction.SIGNED_SINE,
                num_inputs=7,  # bias(always 1), x1, y1, z1, x2, y2, z2
                num_outputs=1,  # weight
                num_initial_mutations=cls._NUM_INITIAL_MUTATIONS,
            )
        )

        return BrainGenotypeCpgFish(brain)

    def mutate_brain(
        self,
        innov_db: multineat.InnovationDatabase,
        rng: np.random.Generator,
    ) -> BrainGenotypeCpgFish:
        """
        Mutate this genotype.

        This genotype will not be changed; a mutated copy will be returned.

        :param innov_db: Multineat innovation database. See Multineat library.
        :param rng: Random number generator.
        :returns: A mutated copy of the provided genotype.
        """
        multineat_rng = multineat_rng_from_random(rng)

        return BrainGenotypeCpgFish(
            MultineatGenotypePickleWrapper(
                self.brain.genotype.MutateWithConstraints(
                    False,
                    multineat.SearchMode.BLENDED,
                    innov_db,
                    _MULTINEAT_PARAMS,
                    multineat_rng,
                )
            )
        )

    @classmethod
    def crossover_brain(
        cls,
        parent1: Self,
        parent2: Self,
        rng: np.random.Generator,
    ) -> BrainGenotypeCpgFish:
        """
        Perform crossover between two genotypes.

        :param parent1: The first genotype.
        :param parent2: The second genotype.
        :param rng: Random number generator.
        :returns: A newly created genotype.
        """
        multineat_rng = multineat_rng_from_random(rng)

        return BrainGenotypeCpgFish(
            MultineatGenotypePickleWrapper(
                parent1.brain.genotype.MateWithConstraints(
                    parent2.brain.genotype,
                    False,
                    False,
                    multineat_rng,
                    _MULTINEAT_PARAMS,
                )
            )
        )

    def develop_brain(self, body: Body, include_bias: bool) -> BrainCpgNetworkNeighborV1Fish:
        """
        Develop the genotype into a modular robot.

        :param body: The body to develop the brain for.
        :returns: The created robot.
        """
        return BrainCpgNetworkNeighborV1Fish(genotype=self.brain.genotype, body=body, include_bias=include_bias)