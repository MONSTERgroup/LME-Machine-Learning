# import os
# import pandas as pd

# tome_path = f"{os.environ.get('LME_TRAINING_DATA')}\\The Tome 3-22-21.xlsx"

# the_tome = pd.read_excel(
#    tome_path, sheet_name="No Dupes (simple)", header=0, usecols="A:FF"
# )

# for row in the_tome.itertuples(index=False, name='UnparsedTomeEntry'):
#    print(row)


# print(the_tome.columns.values.tolist())
from numpy import array, tile
import pygad
import os
import pandas as pd
from lme.AllVAll2 import AllVAll2
from lme.Alloy import Alloy
from pymatgen.core.periodic_table import Element
from lme.OneVOne import OneVOne

from lme.Tome import Tome


def test_Alloy_calc():
    comp_m = Alloy({Element("Te"): 1}, init_with_wt=True)

    ar = comp_m.atomic_radius()
    en = comp_m.electronegativity()
    bp = comp_m.boiling_point()

    print(comp_m)
    print(f"{ar}")
    print(f"{en}")
    print(f"{bp}")

    liquid_comp = sorted(comp_m.items(), key=lambda x: x[1], reverse=True)
    print(liquid_comp[0][0])


def import_excel_tome():
    DataPath = os.environ.get("LME_TRAINING_DATA")
    path = f"{DataPath}\\The Tome 3-22-21.xlsx"
    sheet = "No Dupes (simple)"

    t = Tome(path=path, sheet=sheet, verbose=False)

    for e in t.list_of_entries:
        print(f"{e}")


def optimize_params():

    initial_population = tile(
        array(
            [
                [
                    60,
                    3,
                    0.4,
                    0.1,
                    0.1,
                    20,
                    20,
                    7,
                    0.2,
                    0.1,
                    0.3,
                    11,
                    30,
                    5,
                    0.8,
                    0.1,
                    0.5,
                    7,
                    10,
                    2,
                    0.2,
                    0.2,
                    1,
                    9,
                    100,
                    2,
                    0.2,
                    0.3,
                    0.1,
                    5,
                    30,
                    6,
                    0.5,
                    0.1,
                    0.1,
                    5,
                ]
            ]
        ).transpose(),
        (1, 8),
    )

    gene_type = [
        int,
        int,
        float,
        float,
        float,
        int,
        int,
        int,
        float,
        float,
        float,
        int,
        int,
        int,
        float,
        float,
        float,
        int,
        int,
        int,
        float,
        float,
        float,
        int,
        int,
        int,
        float,
        float,
        float,
        int,
        int,
        int,
        float,
        float,
        float,
        int,
    ]

    init_range_low = 1
    init_range_high = 2

    def fitness_func(solution, solution_idx):
        params = {}

        params["01"] = {
            "n_estimators": solution[0],
            "max_depth": solution[1],
            "min_samples_split": solution[2],
            "min_samples_leaf": solution[3],
            "learning_rate": solution[4],
            "max_features": solution[5],
            "subsample": 1,
            "loss": "deviance",
        }
        params["02"] = {
            "n_estimators": solution[6],
            "max_depth": solution[7],
            "min_samples_split": solution[8],
            "min_samples_leaf": solution[9],
            "learning_rate": solution[10],
            "max_features": solution[11],
            "subsample": 1,
            "loss": "deviance",
        }
        params["03"] = {
            "n_estimators": solution[12],
            "max_depth": solution[13],
            "min_samples_split": solution[14],
            "min_samples_leaf": solution[15],
            "learning_rate": solution[16],
            "max_features": solution[17],
            "subsample": 1,
            "loss": "deviance",
        }
        params["12"] = {
            "n_estimators": solution[18],
            "max_depth": solution[19],
            "min_samples_split": solution[20],
            "min_samples_leaf": solution[21],
            "learning_rate": solution[22],
            "max_features": solution[23],
            "subsample": 1,
            "loss": "deviance",
        }
        params["13"] = {
            "n_estimators": solution[24],
            "max_depth": solution[25],
            "min_samples_split": solution[26],
            "min_samples_leaf": solution[27],
            "learning_rate": solution[28],
            "max_features": solution[29],
            "subsample": 1,
            "loss": "deviance",
        }
        params["23"] = {
            "n_estimators": solution[30],
            "max_depth": solution[31],
            "min_samples_split": solution[32],
            "min_samples_leaf": solution[33],
            "learning_rate": solution[34],
            "max_features": solution[35],
            "subsample": 1,
            "loss": "deviance",
        }
        DataPath = os.environ.get("LME_TRAINING_DATA")
        path = f"{DataPath}\\The Tome 3-22-21.txt"
        t = Tome(source="pickle", path=path)
        trainable = t.prepare_for_gb_classifier(path=path)

        classificator = AllVAll(trainable, verbose=True, genetic=True, params=params)

        return classificator.how_accurate()

    fitness_function = fitness_func
    num_generations = 50
    num_parents_mating = 4

    sol_per_pop = 8
    num_genes = 36

    parent_selection_type = "sss"
    keep_parents = 1

    crossover_type = "single_point"

    mutation_type = "random"
    mutation_percent_genes = 10

    def on_generation(ga_instance):
        print("New Generation")
        ga_instance.plot_fitness()

    ga_instance = pygad.GA(
        num_generations=num_generations,
        num_parents_mating=num_parents_mating,
        # initial_population=initial_population,
        fitness_func=fitness_function,
        sol_per_pop=sol_per_pop,
        num_genes=num_genes,
        gene_type=gene_type,
        init_range_low=init_range_low,
        init_range_high=init_range_high,
        parent_selection_type=parent_selection_type,
        keep_parents=keep_parents,
        crossover_type=crossover_type,
        mutation_type=mutation_type,
        mutation_percent_genes=mutation_percent_genes,
        on_generation=on_generation,
    )

    ga_instance.run()

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Parameters of the best solution : {solution}".format(solution=solution))
    print(
        "Fitness value of the best solution = {solution_fitness}".format(
            solution_fitness=solution_fitness
        )
    )


def get_plot():
    DataPath = os.environ.get("LME_TRAINING_DATA")
    path = f"{DataPath}\\The Tome 3-22-21.txt"
    t = Tome(source="pickle", path=path)
    trainable = t.prepare_for_gb_classifier(path=path)

    classificator = AllVAll2(
        trainable,
        verbose=True,
    )

def test_onevone():
    DataPath = os.environ.get("LME_TRAINING_DATA")
    path = f"{DataPath}\\The Tome 3-22-21.txt"
    t = Tome(source="pickle", path=path)
    trainable = t.prepare_for_gb_classifier(path=path)

    classificator = OneVOne(trainable, '0','1', verbose=True)


if __name__ == "__main__":

    get_plot()
    #test_onevone()
