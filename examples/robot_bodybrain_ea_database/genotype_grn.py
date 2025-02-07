"""Genotype class."""

from __future__ import annotations

import multineat
import numpy as np
import os
from base import Base

# change BrainGenotypeCpgOrm to BrainGenotypeCpgOrmFish if you want to use BrainGenotypeCpgOrmFish
from revolve2.ci_group.genotypes.cppnwin.modular_robot import BrainGenotypeCpgOrmFish as BrainGenotypeCpgOrm
if os.environ["ALGORITHM"] == "GRN":
    from revolve2.ci_group.genotypes.cppnwin.modular_robot.v2 import BodyGenotypeOrmV2GRN
elif os.environ["ALGORITHM"] in ["GRN_system"]:
    from revolve2.ci_group.genotypes.cppnwin.modular_robot.v2 import BodyGenotypeOrmV2GRN_system as BodyGenotypeOrmV2GRN
elif os.environ["ALGORITHM"] in ["GRN_system_adv"]:
    from revolve2.ci_group.genotypes.cppnwin.modular_robot.v2 import BodyGenotypeOrmV2GRN_system_adv as BodyGenotypeOrmV2GRN
    
from revolve2.experimentation.database import HasId
from revolve2.modular_robot import ModularRobot

class Genotype(Base, HasId, BodyGenotypeOrmV2GRN, BrainGenotypeCpgOrm):
    """SQLAlchemy model for a genotype for a modular robot body and brain."""

    __tablename__ = "genotype"

    @classmethod
    def random(
        cls,
        innov_db_brain: multineat.InnovationDatabase,
        rng: np.random.Generator, include_bias: bool,
    ) -> Genotype:
        """
        Goal:
            Create a random genotype.
        -------------------------------------------------------------------------------------------
        Input:
            innov_db_brain: Multineat innovation database for the brain. See Multineat library.
            rng: Random number generator for CPPN.
            include_bias: Whether to include the bias  as input for CPPN.
        -------------------------------------------------------------------------------------------
        Output:
            The created genotype: (base class sqlalchemy, hashid sqlalchemy, body genotype, brain
                genotype).
        """
        # Set random body and brain
        body = cls.random_body(rng)
        brain = cls.random_brain(innov_db_brain, rng, include_bias)

        return Genotype(body = body.body, brain = brain.brain)

    def mutate(
        self,
        innov_db_brain: multineat.InnovationDatabase,
        rng: np.random.Generator,
        mutation_prob: float,
    ) -> Genotype:
        """
        Goal:
            Mutate the genotype. Note: The genotype will not be changed; a mutated 
            copy will be returned.
        -------------------------------------------------------------------------------------------
        Input:
            self: The genotype to mutate (base class sqlalchemy, hashid sqlalchemy, body genotype,
                brain genotype).
            innov_db_brain: Multineat innovation database for the brain. See Multineat library.
            rng: Random number generator for CPPN.
            mutation_prob: The probability of mutation.
        -------------------------------------------------------------------------------------------
        Output:
            A mutated copy of the provided genotype: (base class sqlalchemy, hashid sqlalchemy, 
                body genotype, brain genotype).
        """
        # Get random number
        random_number = rng.uniform(0, 1)

        if random_number > mutation_prob:
            return self
        else:
            # Mutate body and brain
            body = self.mutate_body(rng)
            brain = self.mutate_brain(innov_db_brain, rng)

            return Genotype(body = body.body, brain = brain.brain)

    @classmethod
    def crossover(
        cls,
        parent1: Genotype,
        parent2: Genotype,
        rng: np.random.Generator,
        crossover_prob: float,
    ) -> Genotype:
        """
        Goal:
            Perform crossover between two genotypes.
        -------------------------------------------------------------------------------------------
        Input:
            parent1: The first genotype.
            parent2: The second genotype.
            rng: Random number generator for CPPN.
            crossover_prob: The probability of crossover.
        -------------------------------------------------------------------------------------------
        Output:
            A newly created genotype: (base class sqlalchemy, hashid sqlalchemy, body genotype, 
                brain genotype).
        """
        # Get random number
        random_number = rng.uniform(0, 1)
        if random_number > crossover_prob:
            return parent1
        else:
            # Perform crossover for body (not brain in case of GRN --> CPPN not desired to have crossover)
            body = cls.crossover_body(parent1, parent2, rng)

            return Genotype(body=body.body, brain = parent1.brain)

    def develop(self, include_bias, max_parts, mode_core_mult) -> ModularRobot:
        """
        Goal:
            Develop the genotype into a modular robot.
        -------------------------------------------------------------------------------------------
        Input:
            self: The genotype to develop (base class sqlalchemy, hashid sqlalchemy, body genotype,
                brain genotype).
            include_bias: Whether to include the bias as input for CPPN.
            max_parts: The maximum number of parts for the robot.
            mode_core_mult: Whether to use 3 x 3 core block
        -------------------------------------------------------------------------------------------
        Output:
            The created robot: ModularRobot.
        """
        # Develop body and brain
        body = self.develop_body(max_parts, mode_core_mult)
        brain = self.develop_brain(body = body, include_bias = include_bias)

        return ModularRobot(body = body, brain = brain)
