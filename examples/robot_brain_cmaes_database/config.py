"""Configuration parameters for this example."""

from revolve2.ci_group.modular_robots_v2 import gecko_v2

DATABASE_FILE = "/home/aronf/thesis/please-for-fucks-sake/revolve2/examples/robot_brain_cmaes_database/ARON_DB/database.sqlite"
NUM_REPETITIONS = 2
NUM_SIMULATORS = 16
INITIAL_STD = 0.002
NUM_GENERATIONS = 100
BODY = gecko_v2()
