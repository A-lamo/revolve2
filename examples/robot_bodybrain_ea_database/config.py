"""Configuration parameters for this example."""
import os 

DATABASE_FILE = os.environ["DATABASE_FILE"]
NUM_REPETITIONS = 10 #20#5
NUM_SIMULATORS = os.cpu_count()
POPULATION_SIZE = 10
OFFSPRING_SIZE = 5
if os.environ["MODE"] == "random search":
    NUM_GENERATIONS = 50
elif os.environ["MODE"] == "evolution":
    NUM_GENERATIONS = 150
else:
    raise ValueError("MODE must be either random search or evolution")

NPARENTS = 2
PARENT_TOURNAMENT_SIZE = 4
SURVIVOR_TOURNAMENT_SIZE = 4

if os.environ["ALGORITHM"] in ["GRN", "GRN_system", "GRN_system_adv"]:
    CROSSOVER_PROBABILITY = 1
    MUTATION_PROBABILITY = 0.9
elif os.environ["ALGORITHM"] == "CPPN":
    CROSSOVER_PROBABILITY = 0
    MUTATION_PROBABILITY = 0.9
else:
    raise ValueError("ALGORITHM must be either GRN or CPPN")

TERRAIN = "flat" # "tilted
FITNESS_FUNCTION = "xy_displacement" # "x_efficiency"# "xy_displacement"


ZDIRECTION = False # Whether to evolve in the z-direction.
CPPNBIAS = False # Whether BIAS is an Input for the CPPN.
CPPNCHAINLENGTH = False # Whether CHAINLENGTH is an Input for the CPPN.
CPPNEMPTY = False # Whether EMPTY Module is an Output for the CPPN.

MAX_PARTS = 30 # Maximum number of parts in the body --> better pass as parameter???? 
MODE_COLLISION = False # Whether to stop if collision occurs
MODE_CORE_MULT = True # Whether to allow multiple core slots
MODE_SLOTS4FACE = False # Whether multiple slots can be used for a single face for the core module
MODE_SLOTS4FACE_ALL = False # Whether slots can be set for all 9 attachments, or only 3, 4, 5
MODE_NOT_VERTICAL = True # Whether to disable vertical expansion of the body

SIMULATION_TIME = 30
SAMPLING_FREQUENCY = 5
SIMULATION_TIMESTEP = 0.001
CONTROL_FREQUENCY = 20


# Assertions
assert SIMULATION_TIME == 30, "SIMULATION_TIME must be 30"
if os.environ["MODE"] == "random search":
    assert NUM_GENERATIONS == 50, "NUM_GENERATIONS must be 50 if MODE is random search"
elif os.environ["MODE"] == "evolution":
    assert NUM_GENERATIONS == 150, "NUM_GENERATIONS must be 150 if MODE is evolution"

assert POPULATION_SIZE == 10, "POPULATION_SIZE must be 100"
assert OFFSPRING_SIZE == 5, "OFFSPRING_SIZE must be 50"
assert NPARENTS == 2, "NPARENTS must be 2"
assert PARENT_TOURNAMENT_SIZE == 4, "PARENT_TOURNAMENT_SIZE must be 4"
assert SURVIVOR_TOURNAMENT_SIZE == 4, "SURVIVOR_TOURNAMENT_SIZE must be 4"

if os.environ["ALGORITHM"] in ["GRN", "GRN_system", "GRN_system_adv"]:
    assert MUTATION_PROBABILITY == 0.9, "MUTATION_PROBABILITY must be 0.9 if ALGORITHM is GRN"
    assert CROSSOVER_PROBABILITY == 1, "CROSSOVER_PROBABILITY must be 1 if ALGORITHM is GRN"
elif os.environ["ALGORITHM"]== "CPPN":
    assert MUTATION_PROBABILITY == 0.9, "MUTATION_PROBABILITY must be 0.9 if ALGORITHM is CPPN"
    assert CROSSOVER_PROBABILITY == 0, "CROSSOVER_PROBABILITY must be 0 if ALGORITHM is CPPN"
else:
    raise ValueError("ALGORITHM must be either GRN or CPPN")


if MODE_CORE_MULT and (os.environ["ALGORITHM"] in ["GRN", "GRN_system", "GRN_system_adv"]):
    pass
    #print("For GRN MODE_CORE_MULT only will provide a 3 x 3 grid during querying. No additional attachments!")
else:
    assert not MODE_SLOTS4FACE, "MODE_SLOTS4FACE must be False if MODE_CORE_MULT is True"
    assert not MODE_SLOTS4FACE_ALL, "MODE_SLOTS4FACE_ALL must be False if MODE_CORE_MULT is True"

if MODE_SLOTS4FACE_ALL:
    assert MODE_SLOTS4FACE, "MODE_SLOTS4FACE must be True if MODE_SLOTS4FACE is True"




