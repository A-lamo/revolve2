from typing import TYPE_CHECKING, Any, ClassVar, ForwardRef, Generic, Type, TypeVar

import sqlalchemy
import sqlalchemy.orm as orm
from typing_extensions import Self

from ..._util.init_subclass_get_generic_args import init_subclass_get_generic_args
from ...database import HasId

TGenotype = TypeVar("TGenotype")


class Individual(HasId, orm.MappedAsDataclass, Generic[TGenotype]):
    """
    Generic SQLAlchemy model for an individual.

    Inherit from this to create your own individual type.

    This class is designed to be part of a `Population`.
    The metaclass argument `population_table` must contain the name of the population table.

    The generic parameter `TGenotype` refers to the user-defined genotype type,
    which should have an `id` field that will be used as a foreign key reference.
    This parameter cannot be a forward reference.

    For example::

        class MyIndividual(Base, Individual[MyGenotype], population_table="my_population"):
            __tablename__ = "my_individual"
    """

    # -------------------------------------
    # Class members interesting to the user
    # -------------------------------------
    if TYPE_CHECKING:
        population_id: orm.Mapped[int] = orm.mapped_column(nullable=False, init=False)
        population_index: orm.Mapped[int] = orm.mapped_column(nullable=False, init=False)
        genotype_id: orm.Mapped[int] = orm.mapped_column(nullable=False, init=False)
        genotype: orm.Mapped[TGenotype] = orm.relationship()
        fitness: orm.Mapped[float] = orm.mapped_column(nullable=False)

    # ----------------------
    # Implementation details
    # ----------------------
    else:

        @orm.declared_attr
        def population_id(cls) -> orm.Mapped[int]:  # noqa
            return cls.__population_id_impl()

        @orm.declared_attr
        def population_index(cls) -> orm.Mapped[int]:  # noqa
            return cls.__population_index_impl()

        @orm.declared_attr
        def genotype_id(cls) -> orm.Mapped[int]:  # noqa
            return cls.__genotype_id_impl()

        @orm.declared_attr
        def genotype(cls) -> orm.Mapped[TGenotype]:  # noqa
            return cls.__genotype_impl()

        @orm.declared_attr
        def fitness(cls) -> orm.Mapped[float]:  # noqa
            return cls.__fitness_impl()
        
        @orm.declared_attr
        def energy_used(cls) -> orm.Mapped[float]:
            return cls.__behavioral_measures_impl() 
        
        @orm.declared_attr
        def x_distance(cls) -> orm.Mapped[float]:
            return cls.__behavioral_measures_impl()

        @orm.declared_attr
        def tot_xdistance(cls) -> orm.Mapped[float]:
            return cls.__behavioral_measures_impl()
        
        @orm.declared_attr
        def xmax(cls) -> orm.Mapped[float]:
            return cls.__behavioral_measures_impl()

        @orm.declared_attr
        def y_distance(cls) -> orm.Mapped[float]:
            return cls.__behavioral_measures_impl()

        @orm.declared_attr
        def tot_ydistance(cls) -> orm.Mapped[float]:
            return cls.__behavioral_measures_impl()

        @orm.declared_attr
        def min_dx(cls) -> orm.Mapped[float]:
            return cls.__behavioral_measures_impl()

        @orm.declared_attr
        def dx25(cls) -> orm.Mapped[float]:
            return cls.__behavioral_measures_impl()

        @orm.declared_attr
        def mean_dx(cls) -> orm.Mapped[float]:
            return cls.__behavioral_measures_impl()

        @orm.declared_attr
        def median_dx(cls) -> orm.Mapped[float]:
            return cls.__behavioral_measures_impl()

        @orm.declared_attr
        def dx75(cls) -> orm.Mapped[float]:
            return cls.__behavioral_measures_impl()

        @orm.declared_attr
        def max_dx(cls) -> orm.Mapped[float]:
            return cls.__behavioral_measures_impl()

        @orm.declared_attr
        def std_dx(cls) -> orm.Mapped[float]:
            return cls.__behavioral_measures_impl()

        @orm.declared_attr
        def min_dy(cls) -> orm.Mapped[float]:
            return cls.__behavioral_measures_impl()

        @orm.declared_attr
        def dy25(cls) -> orm.Mapped[float]:
            return cls.__behavioral_measures_impl()

        @orm.declared_attr
        def mean_dy(cls) -> orm.Mapped[float]:
            return cls.__behavioral_measures_impl()

        @orm.declared_attr
        def median_dy(cls) -> orm.Mapped[float]:
            return cls.__behavioral_measures_impl()

        @orm.declared_attr
        def dy75(cls) -> orm.Mapped[float]:
            return cls.__behavioral_measures_impl()

        @orm.declared_attr
        def max_dy(cls) -> orm.Mapped[float]:
            return cls.__behavioral_measures_impl()

        @orm.declared_attr
        def std_dy(cls) -> orm.Mapped[float]:
            return cls.__behavioral_measures_impl()

        @orm.declared_attr
        def energy_used_min(cls) -> orm.Mapped[float]:
            return cls.__behavioral_measures_impl()

        @orm.declared_attr
        def energy_used_25(cls) -> orm.Mapped[float]:
            return cls.__behavioral_measures_impl()

        @orm.declared_attr
        def energy_used_mean(cls) -> orm.Mapped[float]:
            return cls.__behavioral_measures_impl()

        @orm.declared_attr
        def energy_used_median(cls) -> orm.Mapped[float]:
            return cls.__behavioral_measures_impl()

        @orm.declared_attr
        def energy_used_75(cls) -> orm.Mapped[float]:
            return cls.__behavioral_measures_impl()

        @orm.declared_attr
        def energy_used_max(cls) -> orm.Mapped[float]:
            return cls.__behavioral_measures_impl()

        @orm.declared_attr
        def energy_used_std(cls) -> orm.Mapped[float]:
            return cls.__behavioral_measures_impl()

        @orm.declared_attr
        def force_std_motor_min(cls) -> orm.Mapped[float]:
            return cls.__behavioral_measures_impl()

        @orm.declared_attr
        def force_std_motor_25(cls) -> orm.Mapped[float]:
            return cls.__behavioral_measures_impl()

        @orm.declared_attr
        def force_std_motor_mean(cls) -> orm.Mapped[float]:
            return cls.__behavioral_measures_impl()

        @orm.declared_attr
        def force_std_motor_median(cls) -> orm.Mapped[float]:
            return cls.__behavioral_measures_impl()

        @orm.declared_attr
        def force_std_motor_75(cls) -> orm.Mapped[float]:
            return cls.__behavioral_measures_impl()

        @orm.declared_attr
        def force_std_motor_max(cls) -> orm.Mapped[float]:
            return cls.__behavioral_measures_impl()

        @orm.declared_attr
        def force_std_motor_std(cls) -> orm.Mapped[float]:
            return cls.__behavioral_measures_impl()

        @orm.declared_attr
        def force_std_all_min(cls) -> orm.Mapped[float]:
            return cls.__behavioral_measures_impl()

        @orm.declared_attr
        def force_std_all_25(cls) -> orm.Mapped[float]:
            return cls.__behavioral_measures_impl()

        @orm.declared_attr
        def force_std_all_mean(cls) -> orm.Mapped[float]:
            return cls.__behavioral_measures_impl()

        @orm.declared_attr
        def force_std_all_median(cls) -> orm.Mapped[float]:
            return cls.__behavioral_measures_impl()

        @orm.declared_attr
        def force_std_all_75(cls) -> orm.Mapped[float]:
            return cls.__behavioral_measures_impl()

        @orm.declared_attr
        def force_std_all_max(cls) -> orm.Mapped[float]:
            return cls.__behavioral_measures_impl()

        @orm.declared_attr
        def force_std_all_std(cls) -> orm.Mapped[float]:
            return cls.__behavioral_measures_impl()

        @orm.declared_attr
        def efficiency(cls) -> orm.Mapped[float]:
            return cls.__behavioral_measures_impl()

        @orm.declared_attr
        def efficiency_min(cls) -> orm.Mapped[float]:
            return cls.__behavioral_measures_impl()

        @orm.declared_attr
        def efficiency_25(cls) -> orm.Mapped[float]:
            return cls.__behavioral_measures_impl()

        @orm.declared_attr
        def efficiency_mean(cls) -> orm.Mapped[float]:
            return cls.__behavioral_measures_impl()

        @orm.declared_attr
        def efficiency_median(cls) -> orm.Mapped[float]:
            return cls.__behavioral_measures_impl()

        @orm.declared_attr
        def efficiency_75(cls) -> orm.Mapped[float]:
            return cls.__behavioral_measures_impl()

        @orm.declared_attr
        def efficiency_max(cls) -> orm.Mapped[float]:
            return cls.__behavioral_measures_impl()

        @orm.declared_attr
        def efficiency_std(cls) -> orm.Mapped[float]:
            return cls.__behavioral_measures_impl()
        
        @orm.declared_attr
        def balance(cls) -> orm.Mapped[float]:
            return cls.__behavioral_measures_impl()
        
        @orm.declared_attr
        def body_id(cls) -> orm.Mapped[str]:
            return cls.__behavioral_measures_impl()
        
        #Morhological measures
        @orm.declared_attr
        def proportion_Niels(cls) -> orm.Mapped[float]:
            return cls.__morphological_measures_impl()
        
        @orm.declared_attr
        def proportion_2d(cls) -> orm.Mapped[float]:
            return cls.__morphological_measures_impl()
        
        @orm.declared_attr
        def single_neighbor_brick_ratio(cls) -> orm.Mapped[float]:
            return cls.__morphological_measures_impl()
        
        @orm.declared_attr
        def single_neighbour_ratio(cls) -> orm.Mapped[float]:
            return cls.__morphological_measures_impl()
        
        @orm.declared_attr
        def double_neigbour_brick_and_active_hinge_ratio(cls) -> orm.Mapped[float]:
            return cls.__morphological_measures_impl()
                
        @orm.declared_attr
        def attachment_length_max(cls) -> orm.Mapped[float]:
            return cls.__morphological_measures_impl()
        
        @orm.declared_attr
        def attachment_length_mean(cls) -> orm.Mapped[float]:
            return cls.__morphological_measures_impl()
        
        @orm.declared_attr
        def attachment_length_std(cls) -> orm.Mapped[float]:
            return cls.__morphological_measures_impl()
        
        @orm.declared_attr
        def joint_brick_ratio(cls) -> orm.Mapped[float]:
            return cls.__morphological_measures_impl()
        
        @orm.declared_attr
        def symmetry_incl_sum(cls) -> orm.Mapped[float]:
            return cls.__morphological_measures_impl()
        
        @orm.declared_attr
        def symmetry_excl_sum(cls) -> orm.Mapped[float]:
            return cls.__morphological_measures_impl()
        
        @orm.declared_attr
        def coverage(cls) -> orm.Mapped[float]:
            return cls.__morphological_measures_impl()
        
        @orm.declared_attr
        def branching(cls) -> orm.Mapped[float]:
            return cls.__morphological_measures_impl()
        
        @orm.declared_attr
        def surface(cls) -> orm.Mapped[float]:
            return cls.__morphological_measures_impl()

    __type_tgenotype: ClassVar[Type[TGenotype]]  # type: ignore[misc]
    __population_table: ClassVar[str]

    def __init_subclass__(
        cls: Type[Self], population_table: str, **kwargs: dict[str, Any]
    ) -> None:
        """
        Initialize a version of this class when it is subclassed.

        Gets the actual type of `TGenotype` and value of `population_table` and stores them for later use.

        :param population_table: Name of the population table.
        :param kwargs: Remaining arguments passed to super.
        """
        generic_types = init_subclass_get_generic_args(cls, Individual)
        assert len(generic_types) == 1
        cls.__type_tgenotype = generic_types[0]
        assert not isinstance(
            cls.__type_tgenotype, ForwardRef
        ), "TGenotype generic argument cannot be a forward reference."
        cls.__population_table = population_table
        assert isinstance(
            cls.__population_table, str
        ), "population_table argument must be a string."

        super().__init_subclass__(**kwargs)  # type: ignore[arg-type]

    @classmethod
    def __population_id_impl(cls) -> orm.Mapped[int]:
        return orm.mapped_column(
            sqlalchemy.ForeignKey(f"{cls.__population_table}.id"),
            nullable=False,
            init=False,
        )

    @classmethod
    def __population_index_impl(cls) -> orm.Mapped[int]:
        return orm.mapped_column(nullable=False, init=False)

    @classmethod
    def __genotype_id_impl(cls) -> orm.Mapped[int]:
        return orm.mapped_column(
            sqlalchemy.ForeignKey(f"{cls.__type_tgenotype.__tablename__}.id"),
            nullable=False,
            init=False,
        )

    @classmethod
    def __genotype_impl(cls) -> orm.Mapped[TGenotype]:
        return orm.relationship(cls.__type_tgenotype)

    @classmethod
    def __fitness_impl(cls) -> orm.Mapped[float]:
        return orm.mapped_column(nullable=False)
        
    @classmethod
    def __behavioral_measures_impl(cls) -> orm.Mapped[float]:
        return orm.mapped_column(nullable=False)    

    @classmethod
    def __morphological_measures_impl(cls) -> orm.Mapped[float]:
        return orm.mapped_column(nullable=False)