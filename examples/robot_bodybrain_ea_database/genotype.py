"""Genotype class."""

from __future__ import annotations

import multineat
import numpy as np
from base import Base

# change BrainGenotypeCpgOrm to BrainGenotypeCpgOrmFish if you want to use BrainGenotypeCpgOrmFish
from revolve2.ci_group.genotypes.cppnwin.modular_robot import BrainGenotypeCpgOrmFish as BrainGenotypeCpgOrm
from revolve2.ci_group.genotypes.cppnwin.modular_robot.v2 import BodyGenotypeOrmV2
from revolve2.ci_group.genotypes.cppnwin.modular_robot.v2 import BodyMappingSeedOrmV2
from revolve2.experimentation.database import HasId
from revolve2.modular_robot import ModularRobot

class Genotype(Base, HasId, BodyGenotypeOrmV2, BrainGenotypeCpgOrm, BodyMappingSeedOrmV2):
    """SQLAlchemy model for a genotype for a modular robot body and brain."""

    __tablename__ = "genotype"
    
    @classmethod
    def random(
        cls,
        innov_db_body: multineat.InnovationDatabase,
        innov_db_brain: multineat.InnovationDatabase,
        rng: np.random.Generator,
        zdirection: bool, include_bias: bool, include_chain_length: bool,
        include_empty: bool,
    ) -> Genotype:
        """
        Goal:
            Create a random genotype.
        -------------------------------------------------------------------------------------------
        Input:
            innov_db_body: Multineat innovation database for the body. See Multineat library.
            innov_db_brain: Multineat innovation database for the brain. See Multineat library.
            rng: Random number generator for CPPN.
            zdirection: Whether to include the z direction  as input for CPPN.
            include_bias: Whether to include the bias  as input for CPPN.
            include_chain_length: Whether to include the chain length as input for CPPN.
            include_empty: Whether to include the empty module output  as input for CPPN
        -------------------------------------------------------------------------------------------
        Output:
            The created genotype: (base class sqlalchemy, hashid sqlalchemy, body genotype, brain
                genotype, mapping seed).
        """
        # Set random body and brain
        body = cls.random_body(innov_db = innov_db_body, rng = rng, zdirection = zdirection, 
                               include_bias = include_bias, include_chain_length =  include_chain_length,
                               include_empty = include_empty)
        
        brain = cls.random_brain(innov_db = innov_db_brain, rng = rng, include_bias = include_bias)

        # Set random mapping seed
        mapping_seed = rng.integers(0, 2 ** 32)

        return Genotype(body = body.body, brain = brain.brain, mapping_seed = mapping_seed)

    def mutate(
        self,
        innov_db_body: multineat.InnovationDatabase,
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
                brain genotype, mapping seed).
            innov_db_body: Multineat innovation database for the body. See Multineat library.
            innov_db_brain: Multineat innovation database for the brain. See Multineat library.
            rng: Random number generator for CPPN.
            mutation_prob: The probability of mutation.
        -------------------------------------------------------------------------------------------
        Output:
            A mutated copy of the provided genotype: (base class sqlalchemy, hashid sqlalchemy, 
                body genotype, brain genotype, mapping seed).
        """
        # Get random number
        random_number = rng.uniform(0, 1)

        if random_number > mutation_prob:
            self.mapping_seed = np.frombuffer(self.mapping_seed, dtype=np.uint64)[0]
            return self
        else:
            # Mutate body and brain
            body = self.mutate_body(innov_db = innov_db_body, rng = rng)
            brain = self.mutate_brain(innov_db = innov_db_brain, rng = rng)

            # Convert mapping seed to integer
            if type(self.mapping_seed) == bytes:
                self.mapping_seed = np.frombuffer(self.mapping_seed, dtype=np.uint64)[0]

            return Genotype(body = body.body, brain = brain.brain, mapping_seed = self.mapping_seed)

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
                brain genotype, mapping seed).
        """
        # Get random number
        random_number = rng.uniform(0, 1)
        if random_number > crossover_prob:
            return parent1
        else:
            # Perform crossover for body and brain
            body = cls.crossover_body(parent1 = parent1, parent2 = parent2, rng = rng)
            brain = cls.crossover_brain(parent1 = parent1, parent2 = parent2, rng = rng)

            # Set mapping seed to the first parent's mapping seed
            mapping_seed = parent1.mapping_seed

            return Genotype(body = body.body, brain = brain.brain, mapping_seed = mapping_seed)

    def develop(self, zdirection, include_bias, include_chain_length, include_empty,
                max_parts, mode_collision, mode_core_mult, mode_slots4face,
            mode_slots4face_all, mode_not_vertical) -> ModularRobot:
        """
        Goal:
            Develop the genotype into a modular robot.
        -------------------------------------------------------------------------------------------
        Input:
            self: The genotype to develop (base class sqlalchemy, hashid sqlalchemy, body genotype,
                brain genotype, mapping seed).
            zdirection: Whether to include the z direction as input for CPPN.
            include_bias: Whether to include the bias as input for CPPN.
            include_chain_length: Whether to include the chain length as input for CPPN.
            include_empty: Whether to include the empty module output as input for CPPN.
            max_parts: The maximum number of parts.
            mode_collision: Whether to stop if collision occurs.
            mode_core_mult: Whether to allow multiple core slots.
            mode_slots4face: Whether multiple slots can be used for a single face for the core module.
            mode_slots4face_all: Whether slots can be set for all 9 attachments, or only 3, 4, 5.
            mode_not_vertical: Whether to disable vertical expansion of the body.
        -------------------------------------------------------------------------------------------
        Output:
            The created robot: ModularRobot.
        """
        # Develop body and brain
        body = self.develop_body(querying_seed = self.mapping_seed, zdirection = zdirection, 
                                 include_bias = include_bias, include_chain_length = include_chain_length,
                                 include_empty = include_empty, max_parts = max_parts,
                                 mode_collision = mode_collision, mode_core_mult = mode_core_mult,
                                 mode_slots4face = mode_slots4face, mode_slots4face_all = mode_slots4face_all,
                                 mode_not_vertical = mode_not_vertical)
        brain = self.develop_brain(body = body, include_bias = include_bias)

        return ModularRobot(body = body, brain = brain)
