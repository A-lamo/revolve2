"""Rerun the best robot between all experiments."""
import logging
import os
import sys
from matplotlib import pyplot as plt
import numpy as np
import math
import pandas as pd


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


# stack overflow : https://stackoverflow.com/questions/20924085/python-conversion-between-coordinates
def polar(x, y) -> tuple:
    """returns rho, theta (degrees)"""
    return np.hypot(x, y), np.degrees(np.arctan2(y, x))

def get_polar_vectors(x, y):
    polar_vecs = np.array([polar(_x, _y) for _x, _y in zip(x, y)])
    # get average angle change between each point
    return polar_vecs


def main() -> None:
    """Perform the rerun."""
    setup_logging()
    # Process each database file in the directory specified by config.DATABASE_FILE
    for file_name in os.listdir(config.DATABASE_FILE):
        if file_name.endswith(".sqlite"):
            db_file_path = os.path.join(config.DATABASE_FILE, file_name)

            dbengine = open_database_sqlite(db_file_path, open_method=OpenMethod.OPEN_IF_EXISTS)
            with Session(dbengine) as ses:
                # Find the last generation index
                last_gen_index = ses.execute(
                    select(func.max(Generation.generation_index))
                    .join_from(Generation, Experiment, Generation.experiment_id == Experiment.id)
                ).scalar()

                if last_gen_index is None:
                    logging.error("No generations found in the database.")
                    return


                # Step 1: Load all individuals from the final generation
                all_individuals = ses.execute(
                    select(
                        Individual.id,
                        Individual.body_id,
                        Individual.fitness,
                        Individual.proportion_Niels,
                        Individual.proportion_2d,
                        Individual.single_neighbor_brick_ratio,
                        Individual.single_neighbour_ratio,
                        Individual.double_neigbour_brick_and_active_hinge_ratio,
                        Individual.attachment_length_max,
                        Individual.attachment_length_mean,
                        Individual.attachment_length_std,
                        Individual.joint_brick_ratio,
                        Individual.symmetry_incl_sum,
                        Individual.symmetry_excl_sum,
                        Individual.coverage,
                        Individual.branching,
                        Individual.surface
                    )
                    .join_from(Generation, Individual, Generation.population_id == Individual.population_id)
                    .where(Generation.generation_index == last_gen_index)
                ).all()

                # Step 2: Extract and convert the morphological metrics into a list of floats
                def extract_metrics(individual):
                    return [
                        float(individual.proportion_Niels),
                        float(individual.proportion_2d),
                        float(individual.single_neighbor_brick_ratio),
                        float(individual.single_neighbour_ratio),
                        float(individual.double_neigbour_brick_and_active_hinge_ratio),
                        float(individual.attachment_length_max),
                        float(individual.attachment_length_mean),
                        float(individual.attachment_length_std),
                        float(individual.joint_brick_ratio),
                        float(individual.symmetry_incl_sum),
                        float(individual.symmetry_excl_sum),
                        float(individual.coverage),
                        float(individual.branching),
                        float(individual.surface)
                    ]

                metrics = [extract_metrics(ind) for ind in all_individuals]
                num_individuals = len(metrics)

                # Step 3: Calculate pairwise Euclidean distances
                distances = np.zeros((num_individuals, num_individuals))

                for i in range(num_individuals):
                    for j in range(num_individuals):
                        if i != j:
                            distances[i][j] = euclidean(metrics[i], metrics[j])

                # Step 4: Measure diversity as the average pairwise distance
                diversity = np.mean(distances, axis=1)

                # Step 5: Order individuals by diversity (from highest to lowest)
                sorted_indices = np.argsort(-diversity)

                # Step 6: Select the top 10 most diverse individuals
                top_30_indices = sorted_indices[:30]
                top_30_individuals = [all_individuals[i] for i in top_30_indices]

                # Step 7: Filter out individuals that are too similar to each other
                def is_too_similar(ind1, ind2, threshold=0.5):
                    """Check if the Euclidean distance between two individuals is below a given threshold."""
                    return euclidean(ind1, ind2) < threshold

                filtered_top_30 = []
                used_indices = set()

                for i in range(len(top_30_individuals)):
                    if i in used_indices:
                        continue
                    current_individual = extract_metrics(top_30_individuals[i])
                    filtered_top_30.append(top_30_individuals[i])
                    for j in range(i + 1, len(top_30_individuals)):
                        if is_too_similar(current_individual, extract_metrics(top_30_individuals[j])):
                            used_indices.add(j)

                # Ensure we do not exceed 10 individuals if there are fewer after filtering
                final_top_30_individual_ids = [ind.id for ind in filtered_top_30[:10]]

                # Step 8: Fetch the details for these filtered top 10 individuals
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
                    .where(IndividualAlias.id.in_(final_top_30_individual_ids))
                ).all()


            body_ids = []
            for i, row in enumerate(rows):
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

                evaluator = Evaluator(
                    headless=False, num_simulators=1, terrain=config.TERRAIN, fitness_function=config.FITNESS_FUNCTION,
                    simulation_time=config.SIMULATION_TIME, sampling_frequency=config.SAMPLING_FREQUENCY,
                    simulation_timestep=config.SIMULATION_TIMESTEP, control_frequency=config.CONTROL_FREQUENCY,
                    writefiles=writefiles, record=writevideos, video_path=os.path.join(os.getcwd(), f"MuJoCo_videos/MuJoCo_{os.path.basename(db_file_path)}"))


                fitnesses, behavioral_measures, ids, x, y, morphological_measures = evaluator.evaluate([modular_robot])
                print("i : ", i)
                print("len morph measures :", len(morphological_measures))
                logging.info(f"Fitness Measured: {fitnesses[0]}")
                logging.info(f"X_distance Measured: {behavioral_measures[0]['x_distance']}")
                assert ids[0] == body_id, "Body ID measured does not match the one in the database"
                logging.info(f"Body ID Measured: {ids[0]}")
                logging.info(f"XY-DISPLACEMENT: {x[-1], y[-1]}")
                total_distance = 0.0
                for j in range(1, len(x)):
                    dx = x[j] - x[j - 1]
                    dy = y[j] - y[j - 1]
                    total_distance += math.sqrt(dx**2 + dy**2)
                
                logging.info(f"Total distance: {total_distance}")

                # Create a figure and subplots
                # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

                # # # First subplot: scatter plot
                # ax1.plot(x, y, alpha=0.4, color='blue')
                # ax1.scatter(x[0], y[0], color='red')  # Mark the first point in red
                # ax1.text(x[0], y[0], 'start', color='red')  # Label the first point as 'start'
                # ax1.scatter(x[-1], y[-1], color='green')  # Mark the last point in green
                # ax1.text(x[-1], y[-1], 'end', color='green')  # Label the last point as 'end'
                # ax1.set_title("Agent's journey")
                # ax1.set_xlabel('x-coordinate')
                # ax1.set_ylabel('y-coordinate')

                # # # Second subplot: polar vectors plot
                polar_vectors = get_polar_vectors(x, y)
                # ax2.plot([i for i in range(len(polar_vectors))], polar_vectors[:, 1])
                # ax2.set_title(f"Polar coordinate theta for {config.DATABASE_FILE}")
                # ax2.set_yticks(np.arange(-180, 181, 30))

                # # Adjust layout for better display
                # plt.tight_layout()
                # plt.show()


                df=pd.DataFrame({'x':x, 'y':y, 'theta':polar_vectors[:, 1], 'energy_used_mean':behavioral_measures[0]["energy_used_mean"],'energy_used_max':behavioral_measures[0]["energy_used_max"],'efficiency_mean':behavioral_measures[0]["efficiency_mean"],'efficiency_max':behavioral_measures[0]["efficiency_max"],'fitness':fitnesses[0], 'attachment_length_max':morphological_measures[0]["attachment_length_max"], "attachment_length_mean":morphological_measures[0]["attachment_length_mean"], "attachment_length_std":morphological_measures[0]["attachment_length_std"], "joint_brick_ratio":morphological_measures[0]["joint_brick_ratio"], "symmetry_incl_sum":morphological_measures[0]["symmetry_incl_sum"], "symmetry_excl_sum":morphological_measures[0]["symmetry_excl_sum"], "coverage":morphological_measures[0]["coverage"], "branching":morphological_measures[0]["branching"], "surface":morphological_measures[0]["surface"]})
                df.to_csv(f'ANALYSIS/{str(file_name.split(".")[0])}/sAngleAndMagnitudeForAllCPGs/{final_top_30_individual_ids[i]}_0_input_xy_theta_morphMeasures.csv', index=False)
                print("-----------------------------------------------")
                body_ids.append(body_id)
            print('BODY-NAME : ', final_top_30_individual_ids)
            print("BODY IDS : ", body_ids)

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