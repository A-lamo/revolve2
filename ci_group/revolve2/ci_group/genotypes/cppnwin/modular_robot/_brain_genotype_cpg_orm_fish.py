from __future__ import annotations

import multineat
import numpy as np
import sqlalchemy.orm as orm
from sqlalchemy import event
from sqlalchemy.engine import Connection
from typing_extensions import Self

from revolve2.modular_robot.body.base import Body

from .._multineat_rng_from_random import multineat_rng_from_random
from .._random_multineat_genotype import random_multineat_genotype
# ! changed import !
from ._brain_cpg_network_neighbor_fish import BrainCpgNetworkNeighborV1Fish
from ._multineat_params import get_multineat_params


# Multineat parameters
_MULTINEAT_PARAMS = get_multineat_params()

class BrainGenotypeCpgOrmFish(orm.MappedAsDataclass, kw_only=True):
    """Goal:
        An SQLAlchemy model for a CPPNWIN cpg brain genotype."""

    # Initial number of mutations
    _NUM_INITIAL_MUTATIONS = 10

    # Brain genotype
    brain: multineat.Genome

    # Serialized brain
    _serialized_brain: orm.Mapped[str] = orm.mapped_column(
        "serialized_brain", init=False, nullable=False
    )
    

    @classmethod
    def random_brain(
        cls,
        innov_db: multineat.InnovationDatabase,
        rng: np.random.Generator,
        include_bias: bool,
    ) -> BrainGenotypeCpgOrmFish:
        """
        Goal:
            Create a random genotype.
        -------------------------------------------------------------------------------------------
        Input:
            innov_db: Multineat innovation database. See Multineat library.
            rng: Random number generator.
            include_bias: Whether to include the bias as input for CPPN.
        -------------------------------------------------------------------------------------------
        Output:
            The created genotype.
        """
        # Create a multineat rng and seed it with the numpy rng state
        multineat_rng = multineat_rng_from_random(rng)

        # Number of Inputs
        ninputs = 6 # x1, y1, z1, x2, y2, z2
        if include_bias: # bias(always 1) 
            ninputs += 1

        # Create a random brain
        brain = random_multineat_genotype(
            innov_db = innov_db,
            rng = multineat_rng,
            multineat_params = _MULTINEAT_PARAMS,
            output_activation_func = multineat.ActivationFunction.SIGNED_SINE,
            num_inputs = ninputs,
            num_outputs = 1, # weight
            num_initial_mutations = cls._NUM_INITIAL_MUTATIONS,
        )

        return BrainGenotypeCpgOrmFish(brain=brain)

    def mutate_brain(
        self,
        innov_db: multineat.InnovationDatabase,
        rng: np.random.Generator,
    ) -> BrainGenotypeCpgOrmFish:
        """
        Goal:
            Mutate this genotype. This genotype will not be changed; a mutated copy will be returned.
        -------------------------------------------------------------------------------------------
        Input:
            innov_db: Multineat innovation database. See Multineat library.
            rng: Random number generator.
        -------------------------------------------------------------------------------------------
        Output:
            A mutated copy of the provided genotype.
        """
        multineat_rng = multineat_rng_from_random(rng)

        return BrainGenotypeCpgOrmFish(
            brain = self.brain.MutateWithConstraints(
                False,
                multineat.SearchMode.BLENDED,
                innov_db,
                _MULTINEAT_PARAMS,
                multineat_rng,
            )
        )

    @classmethod
    def crossover_brain(
        cls,
        parent1: Self,
        parent2: Self,
        rng: np.random.Generator,
    ) -> BrainGenotypeCpgOrmFish:
        """
        Goal:
            Perform crossover between two genotypes.
        -------------------------------------------------------------------------------------------
        Input:
            parent1: The first genotype.
            parent2: The second genotype.
            rng: Random number generator.
        -------------------------------------------------------------------------------------------
        Output:
            A newly created genotype.
        """
        multineat_rng = multineat_rng_from_random(rng)

        return BrainGenotypeCpgOrmFish(
            brain = parent1.brain.MateWithConstraints(
                parent2.brain,
                False,
                False,
                multineat_rng,
                _MULTINEAT_PARAMS,
            )
        )

    def develop_brain(self, body: Body, include_bias: bool) -> BrainCpgNetworkNeighborV1Fish:
        """
        Goal:
            Develop the genotype into a modular robot.
        -------------------------------------------------------------------------------------------
        Input:
            body: The body to develop the brain
            include_bias: Whether to include the bias as input for CPPN.
        -------------------------------------------------------------------------------------------
        Output:
            The created robot brain
        """
        return BrainCpgNetworkNeighborV1Fish(genotype = self.brain, body = body, include_bias=include_bias)


@event.listens_for(BrainGenotypeCpgOrmFish, "before_update", propagate=True)
@event.listens_for(BrainGenotypeCpgOrmFish, "before_insert", propagate=True)
def _serialize_brain(
    mapper: orm.Mapper[BrainGenotypeCpgOrmFish],
    connection: Connection,
    target: BrainGenotypeCpgOrmFish,
) -> None:
    target._serialized_brain = target.brain.Serialize()


@event.listens_for(BrainGenotypeCpgOrmFish, "load", propagate=True)
def _deserialize_brain(target: BrainGenotypeCpgOrmFish, context: orm.QueryContext) -> None:
    brain = multineat.Genome()
    brain.Deserialize(target._serialized_brain)
    target.brain = brain
