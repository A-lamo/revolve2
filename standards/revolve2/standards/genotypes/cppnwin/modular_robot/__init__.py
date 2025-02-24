"""CPPNWIN genotypes for modular robots."""

from ._brain_cpg_network_neighbor import BrainCpgNetworkNeighbor
from ._brain_cpg_network_neighbor_fish import BrainCpgNetworkNeighborFish

from ._brain_genotype_cpg import BrainGenotypeCpg
from _brain_genotype_cpg_fish import BrainGenotypeCpgFish

from _brain_genotype_cpg_orm import BrainGenotypeCpgOrm
from _brain_genotype_cpg_orm_fish import BrainGenotypeCpgOrmFish

__all__ = [
    "BrainCpgNetworkNeighbor",
    "BrainGenotypeCpg",
    "BrainGenotypeCpgOrm",
    "BrainCpgNetworkNeighborFish",
    "BrainGenotypeCpgFish",
    "BrainGenotypeCpgOrmFish",
]
