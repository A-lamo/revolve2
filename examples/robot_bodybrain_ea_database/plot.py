"""Plot fitness over generations for all experiments, averaged."""

# Set algorithm, mode and file name from command line arguments.
import os
import sys
import seaborn as sns
algo = sys.argv[1]
mode = sys.argv[2]
file_name = "nan.sqlite"
folder_path = sys.argv[3]

assert algo in ["GRN", "GRN_system", "GRN_system_adv", "CPPN"], "ALGORITHM must be either GRN, GRN_system or CPPN"
assert mode in ["random search", "evolution"], "MODE must be either random search or evolution"
assert type(file_name) == str, "FILE_NAME must be a string"
assert file_name.endswith(".sqlite"), "FILE_NAME must end with sqlite"
os.environ["ALGORITHM"] = algo
os.environ["MODE"] = mode
os.environ["DATABASE_FILE"] = file_name

# Import parameters
import config

# Import other modules
import matplotlib.pyplot as plt
import pandas as pd
from experiment import Experiment
from generation import Generation
from individual import Individual
from population import Population
from sqlalchemy import select
from sqlalchemy.orm import Session

from revolve2.experimentation.database import OpenMethod, open_database_sqlite
from revolve2.experimentation.logging import setup_logging


def main(column, path) -> None:
    """Run the program."""
    setup_logging()

    df = pd.DataFrame([])
    n_exp = 0
    for file in os.listdir(folder_path):
        if file.endswith(".sqlite"):
            print(file)
            # Open database
            dbengine = open_database_sqlite(folder_path + "/" + file, open_method=OpenMethod.OPEN_IF_EXISTS)

            # Get pandas data
            df_sub = select_data(dbengine, column)
            df_sub["experiment_id"] = ((df_sub["experiment_id"] - df_sub["experiment_id"].min()) + n_exp).astype(int).values
            # Print ids and max fitness
            ids = df_sub["experiment_id"].unique()
            for id in ids:
                print(f"Experiment {id}: {df_sub.loc[df_sub['experiment_id'] == id, column].max()}")
            
            print(df_sub["experiment_id"].shape)
            n_exp = df_sub["experiment_id"].max() + 1

            # Concat data
            df = pd.concat([df, df_sub], axis = 0)

    # Select even generations
    first_generation = df.loc[df["generation_index"] == 0, :]
    df = df.loc[(df.loc[:, "generation_index"] % 2) == 0, :]
    df = pd.concat([first_generation, df])
    df["generation_index"] = (df.loc[:, "generation_index"] / 2).astype(int).values

    # # Plot mean fitness per experiment over generations
    # for exp_id in df["experiment_id"].unique():
    #     df_exp = df.loc[df["experiment_id"] == exp_id, :]
    #     df_exp = df_exp.groupby("generation_index").agg({column: "mean"}).reset_index()
    #     plt.plot(df_exp["generation_index"], df_exp[column], label = f"Experiment {exp_id}")
    

    # Get max and mean fitness per experiment per generation
    agg_per_experiment_per_generation = (
        df.groupby(["experiment_id", "generation_index"])
        .agg({column: ["max", "mean"]})
        .reset_index()
    )

    # Aggregate over experiments
    agg_per_experiment_per_generation.columns = [
        "experiment_id",
        "generation_index",
        f"max_{column}",
        f"mean_{column}",
    ]

    agg_per_generation = (
        agg_per_experiment_per_generation.groupby("generation_index")
        .agg({f"max_{column}": ["mean", "std"], f"mean_{column}": ["mean", "std"]})
        .reset_index()
    )

    # Set columns
    agg_per_generation.columns = [
        "generation_index",
        f"max_{column}_mean",
        f"max_{column}_std",
        f"mean_{column}_mean",
        f"mean_{column}_std", ]
    
    # Set up the color palette
    palette = sns.color_palette("bright", n_colors=len(df["experiment_id"].unique()))
    color_max = palette[0]
    color_mean = palette[-1]

    plt.figure()

    # Plot max
    plt.plot(
        agg_per_generation["generation_index"],
        agg_per_generation[f"max_{column}_mean"],
        label=f"Max {column}",
        color="b",
    )
    plt.fill_between(
        agg_per_generation["generation_index"],
        agg_per_generation[f"max_{column}_mean"] - agg_per_generation[f"max_{column}_std"],
        agg_per_generation[f"max_{column}_mean"] + agg_per_generation[f"max_{column}_std"],
        color="b",
        alpha=0.2,
    )

    # Plot mean
    plt.plot(
        agg_per_generation["generation_index"],
        agg_per_generation[f"mean_{column}_mean"],
        label=f"Mean {column}",
        color="r",
    )
    plt.fill_between(
        agg_per_generation["generation_index"],
        agg_per_generation[f"mean_{column}_mean"]
        - agg_per_generation[f"mean_{column}_std"],
        agg_per_generation[f"mean_{column}_mean"]
        + agg_per_generation[f"mean_{column}_std"],
        color="r",
        alpha=0.2,
    )

    plt.xlabel("Generation Index", fontweight = "bold", size = 16)
    plt.ylabel(column.title(), fontweight = "bold", size = 16)
    # plt.title(f"{column.title()}", fontweight = "bold", size = 16)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(path + f"//{column}.png")
    plt.show()

    if column == "fitness":
        for exp_id, color in zip(df["experiment_id"].unique(), palette):
            df_exp = df.loc[df["experiment_id"] == exp_id, :]
            df_exp = df_exp.groupby("generation_index").agg({column: "mean"}).reset_index()
            plt.plot(df_exp["generation_index"], df_exp[column], color = color)
        plt.xlabel("Generation Index", fontweight = "bold", size = 16)
        plt.ylabel(column.title(), fontweight = "bold", size = 16)
        # plt.title(f"{column.title()}", fontweight = "bold", size = 16)
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(path + f"//{column}_experiment.png")
        plt.show()


def select_data(dbengine, column: str) -> pd.DataFrame:
    """Goal:
        Select the data of the column
    -------------------------------------------------------------
    Input:
        dbengine: ?
        column: The column that needs to be selected
    --------------------------------------------------------------
    Output:
        df: pd.Dataframe"""
    df = pd.read_sql(
        select(Experiment.id.label("experiment_id"), Generation.generation_index, getattr(Individual, column),)
        .join_from(Experiment, Generation, Experiment.id == Generation.experiment_id)
        .join_from(Generation, Population, Generation.population_id == Population.id)
        .join_from(Population, Individual, Population.id == Individual.population_id),
        dbengine,
    )

    return df

if __name__ == "__main__":
    # Columns
    columns = ["fitness", "energy_used", 
               
               "x_distance", "tot_xdistance", "xmax", "y_distance", "tot_ydistance",
               
               "min_dx", "dx25", "mean_dx", "median_dx", "dx75", "max_dx", "std_dx",
               
               "min_dy", "dy25", "mean_dy", "median_dy", "dy75", "max_dy", "std_dy",
                
                "energy_used_min", "energy_used_25", "energy_used_mean", "energy_used_median",
                "energy_used_75", "energy_used_max", "energy_used_std",

                "force_std_motor_min", "force_std_motor_25", "force_std_motor_mean", "force_std_motor_median",
                "force_std_motor_75", "force_std_motor_max", "force_std_motor_std",

                "force_std_all_min", "force_std_all_25", "force_std_all_mean", "force_std_all_median",
                "force_std_all_75", "force_std_all_max", "force_std_all_std",

                "efficiency", "efficiency_min", "efficiency_25", "efficiency_mean", "efficiency_median",
                "efficiency_75", "efficiency_max", "efficiency_std",

                "balance", "proportion_2d", "proportion_Niels", "single_neighbor_brick_ratio",
                "single_neighbour_ratio", "double_neigbour_brick_and_active_hinge_ratio",
                "attachment_length_max", "attachment_length_mean", "attachment_length_std",
                "joint_brick_ratio", "symmetry_incl_sum", "symmetry_excl_sum", "coverage",
                "branching", "surface"]
    
    # If folder does not exist, create it
    path = f"C:\\Users\\niels\\OneDrive\\Documenten\\GitHub\\revolve2\\Test\\{os.environ['ALGORITHM']}\\BehavioralMeasures\\plots"
    if not os.path.exists(path):
        os.makedirs(path)

    # Plot
    for column in columns:
        main(column = column, path = path)


# Planning
# Alle variabelen even doorlopen met plots en kijken of het is zoals gewenst
# 2D check opnieuw maken
# Paar plaatjes controleren
# Probleem met balance
# Crossover voor brein indien GRN?
# Moet ik de offspring ook niet opslaan???