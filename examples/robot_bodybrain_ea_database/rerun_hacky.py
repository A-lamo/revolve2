"""Rerun the best robot between all experiments."""
import logging
import os
import sys
from matplotlib import pyplot as plt
import numpy as np
import math
import pandas as pd
from collections import defaultdict


# Set algorithm, mode and file name from command line arguments.
algo = sys.argv[1]
mode = sys.argv[2]
file_name = sys.argv[3]
headless = sys.argv[4]
writefiles = sys.argv[5]
writevideos = sys.argv[6]

assert headless in ["True", "False"], "HEADLESS must be either True or False"
if writefiles == "True":
    assert writevideos == "False", "WRITEVIDEOS must be False if WRITEFILES is True"
    assert headless == "True", "HEADLESS must be True if WRITEFILES is True"
assert writefiles in ["True", "False"], "WRITEFILES must be either True or False"
assert writevideos in ["True", "False"], "WRITEVIDEOS must be either True or False"
assert algo in ["GRN", "GRN_system", "GRN_system_adv", "CPPN"], "ALGORITHM must be either GRN, 'GRN_system' or CPPN"
assert mode in ["random search", "evolution"], "MODE must be either random search or evolution"
assert type(file_name) == str, "FILE_NAME must be a string"
# assert file_name.endswith(".sqlite"), "FILE_NAME must end with sqlite"
os.environ["ALGORITHM"] = algo
os.environ["MODE"] = mode
os.environ["DATABASE_FILE"] = file_name
os.environ["HEADLESS"] = headless
os.environ["WRITEFILES"] = writefiles
os.environ["WRITEVIDEOS"] = writevideos

if os.environ["WRITEFILES"] == "True":
    os.environ["RERUN"] = "True"
else:
    os.environ["RERUN"] = "False"

# Import parameters
import config
os.environ['MAXPARTS'] = str(config.MAX_PARTS)

# Import the genotype
if os.environ["ALGORITHM"] == "CPPN":
    from genotype import Genotype
elif os.environ["ALGORITHM"] in ["GRN", "GRN_system", "GRN_system_adv"]:
    from genotype_grn import Genotype
else:
    raise ValueError("ALGORITHM must be either GRN or CPPN")

# Import other modules
from evaluator import Evaluator
from experiment import Experiment
from generation import Generation
from individual import Individual
from population import Population

import shutil
from scipy.spatial.distance import euclidean
from sqlalchemy import func, select
from sqlalchemy.orm import Session, aliased

from revolve2.experimentation.database import OpenMethod, open_database_sqlite
from revolve2.experimentation.logging import setup_logging


def polar(x, y):
    """returns rho, theta (degrees)"""
    return np.hypot(x, y), np.degrees(np.arctan2(y, x))

def get_polar_vectors(x, y):
    return np.array([polar(_x, _y) for _x, _y in zip(x, y)])

def parse_body(serialized_body):
    return list(map(float, serialized_body.split(',')))

def calculate_diversity(individuals):
    parsed_bodies = [(ind.id, parse_body(ind._serialized_body)) for ind in individuals]
    length_groups = defaultdict(list)
    for ind_id, body in parsed_bodies:
        length_groups[len(body)].append((ind_id, body))

    sum_distances = {}
    for length, group in length_groups.items():
        for batch_start in range(0, len(group), 5):
            batch = group[batch_start:batch_start+5]
            batch_ids = [ind_id for ind_id, _ in batch]
            batch_bodies = [body for _, body in batch]
            max_length = max(len(body) for body in batch_bodies)
            padded_bodies = [body + [0.0] * (max_length - len(body)) for body in batch_bodies]

            num_batch = len(padded_bodies)
            distances = np.zeros((num_batch, num_batch))
            for i in range(num_batch):
                for j in range(num_batch):
                    if i != j:
                        distances[i][j] = euclidean(padded_bodies[i], padded_bodies[j])

            for i in range(num_batch):
                sum_distances[batch_ids[i]] = np.sum(distances[i])

    sorted_individuals = sorted(sum_distances.items(), key=lambda x: x[1], reverse=True)
    top_10_individuals = sorted_individuals[:10]
    return [ind_id for ind_id, _ in top_10_individuals]

def main():
    setup_logging()
    all_individuals = []

    for db_file in os.listdir(config.DATABASE_FILE):
        if db_file.endswith(".sqlite"):
            db_file_path = os.path.join(config.DATABASE_FILE, db_file)
            dbengine = open_database_sqlite(db_file_path, open_method=OpenMethod.OPEN_IF_EXISTS)
            with Session(dbengine) as ses:
                last_gen_index = ses.execute(
                    select(func.max(Generation.generation_index))
                    .join_from(Generation, Experiment, Generation.experiment_id == Experiment.id)
                ).scalar()

                if last_gen_index is None:
                    logging.error("No generations found in the database.")
                    continue

                individuals = ses.execute(
                    select(Individual.id, Genotype._serialized_body)
                    .join_from(Generation, Individual, Generation.population_id == Individual.population_id)
                    .join_from(Individual, Genotype, Individual.genotype_id == Genotype.id)
                    .where(Generation.generation_index == last_gen_index)
                ).all()

                all_individuals.extend(individuals)

    if not all_individuals:
        logging.error("No individuals found across all databases.")
        return

    top_10_ids = calculate_diversity(all_individuals)

    for db_file in os.listdir(config.DATABASE_FILE):
        if db_file.endswith(".sqlite"):
            db_file_path = os.path.join(config.DATABASE_FILE, db_file)
            dbengine = open_database_sqlite(db_file_path, open_method=OpenMethod.OPEN_IF_EXISTS)
            body_ids = []
            with Session(dbengine) as ses:
                GenotypeAlias = aliased(Genotype)
                IndividualAlias = aliased(Individual)
                rows = ses.execute(
                    select(
                        GenotypeAlias,
                        IndividualAlias.fitness,
                        IndividualAlias.energy_used,
                        IndividualAlias.efficiency,
                        IndividualAlias.x_distance,
                        IndividualAlias.y_distance,
                        Generation.experiment_id,
                        Generation.generation_index,
                        IndividualAlias.body_id,
                        GenotypeAlias._serialized_body
                    )
                    .join_from(Experiment, Generation, Experiment.id == Generation.experiment_id)
                    .join_from(Generation, Population, Generation.population_id == Population.id)
                    .join_from(Population, IndividualAlias, Population.id == IndividualAlias.population_id)
                    .join_from(IndividualAlias, GenotypeAlias, IndividualAlias.genotype_id == GenotypeAlias.id)
                    .where(Generation.generation_index == last_gen_index)
                    .where(IndividualAlias.id.in_(top_10_ids))
                ).all()

                for row in rows:
                    genotype = row[0]
                    fitness = row[1]
                    energy_used = row[2]
                    efficiency = row[3]
                    x_distance = row[4]
                    y_distance = row[5]
                    exp_id = row[6]
                    gen_index = row[7]
                    body_id = row[8]
                    logging.info(f"Database: {db_file_path}")
                    logging.info(f"Experiment ID: {exp_id}")
                    logging.info(f"Generation Index: {gen_index}")

                    if os.environ["ALGORITHM"] == "CPPN":
                        modular_robot = genotype.develop(
                            zdirection=config.ZDIRECTION, include_bias=config.CPPNBIAS,
                            include_chain_length=config.CPPNCHAINLENGTH, include_empty=config.CPPNEMPTY,
                            max_parts=config.MAX_PARTS, mode_collision=config.MODE_COLLISION,
                            mode_core_mult=config.MODE_CORE_MULT, mode_slots4face=config.MODE_SLOTS4FACE,
                            mode_slots4face_all=config.MODE_SLOTS4FACE_ALL, mode_not_vertical=config.MODE_NOT_VERTICAL)
                    else:
                        modular_robot = genotype.develop(
                            include_bias=config.CPPNBIAS, max_parts=config.MAX_PARTS, mode_core_mult=config.MODE_CORE_MULT)

                    logging.info(f"Fitness: {fitness}")
                    logging.info(f"Energy used: {energy_used}")
                    logging.info(f"Efficiency: {efficiency}")
                    logging.info(f"X distance: {x_distance}")
                    logging.info(f"Y distance: {y_distance}")
                    logging.info(f"Body ID: {body_id}")
                    body_ids.append(body_id)
                    evaluator = Evaluator(
                        headless=True, num_simulators=1, terrain=config.TERRAIN, fitness_function=config.FITNESS_FUNCTION,
                        simulation_time=config.SIMULATION_TIME, sampling_frequency=config.SAMPLING_FREQUENCY,
                        simulation_timestep=config.SIMULATION_TIMESTEP, control_frequency=config.CONTROL_FREQUENCY,
                        writefiles=writefiles, record=writevideos, video_path=os.path.join(os.getcwd(), f"MuJoCo_videos/MuJoCo_{os.path.basename(db_file_path)}"))

                    fitnesses, behavioral_measures, ids, x, y, morphological_measures = evaluator.evaluate([modular_robot])
                    logging.info(f"Fitness Measured: {fitnesses[0]}")
                    logging.info(f"X_distance Measured: {behavioral_measures[0]['x_distance']}")
                    assert ids[0] == body_id, "Body ID measured does not match the one in the database"
                    logging.info(f"Body ID Measured: {ids[0]}")
                    logging.info(f"XY-DISPLACEMENT: {x[-1], y[-1]}")
                print(body_ids)

if __name__ == "__main__":
    # run with arguments <algo> <mode> <file_name> <headless> <writefiles> <writevideos>!!!
    # --- Create/Empty directories for XMLs and PKLs
    for directory_path in ["MuJoCo_videos"]: # "RERUN\\XMLs", "RERUN\\PKLs", 
        if not os.path.exists(directory_path):
            # Create the directory and its parents if they don't exist
                os.makedirs(directory_path)
        else:
            for filename in os.listdir(directory_path):
                # Construct the full path
                file_path = os.path.join(directory_path, filename)

                # Check if it's a file
                if os.path.isfile(file_path):
                    # Remove the file
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    # Remove the directory
                    shutil.rmtree(file_path)
    # --- Rerun
    main()