from sklearn import preprocessing

from numpy import array, tile
import os
import numpy as np
from src.lme.Classifer_KNN_Multi import Classifier_KNN_Multi
from src.lme.Classifier_RandomForestEnsemble import Classifier_RandomForestEnsemble
from src.lme.Classifier_GradientBoostedTrees_1v1 import Classifier_GradientBoostedTrees_1v1
from src.lme.Classifier_KNN_1v1 import Classifier_KNN_1v1
from pymatgen.core.periodic_table import Element

from src.lme.Tome import Tome


# def test_Alloy_calc():
#     comp_m = Alloy({Element("Te"): 1}, init_with_wt=True)

#     ar = comp_m.atomic_radius()
#     en = comp_m.electronegativity()
#     bp = comp_m.boiling_point()

#     print(comp_m)
#     print(f"{ar}")
#     print(f"{en}")
#     print(f"{bp}")

#     liquid_comp = sorted(comp_m.items(), key=lambda x: x[1], reverse=True)
#     print(liquid_comp[0][0])


def import_excel_tome():
    DataPath = os.environ.get("LME_TRAINING_DATA")
    path = f"{DataPath}\\Tome_2023_04_08.xlsx"
    sheet = "No Dupes (simple)"

    t = Tome(path=path, sheet=sheet, verbose=True)

    fID = open(f"{DataPath}\\Tome_2023_04_08_HumanReadable.txt", "a")

    for e in t.list_of_entries:
        fID.write(f"{e}")
    
    fID.close()

    fID = open(f"{DataPath}\\Tome_2023_04_08_HumanReadable.txt", "r")
    print(fID.read())

def make_readable_tome():
    DataPath = os.environ.get("LME_TRAINING_DATA")
    path = f"{DataPath}\\Tome_2023_04_08.txt"

    t = Tome(source='pickle', path=path, verbose=True)

    fID = open(f"{DataPath}\\Tome_2023_04_08_HumanReadable.txt", "a")

    for e in t.list_of_entries:
        fID.write(f"{e}")
    
    fID.close()

    fID = open(f"{DataPath}\\Tome_2023_04_08_HumanReadable.txt", "r")
    print(fID.read())

# def get_plot():
#     DataPath = os.environ.get("LME_TRAINING_DATA")
#     path = f"{DataPath}\\The Tome 3-22-21.txt"
#     t = Tome(source="pickle", path=path)
#     trainable, feature_names = t.prepare_for_gb_classifier(path=path)

#     classificator = AllVAll2(
#         trainable,
#         verbose=True,
#         features = feature_names
#     )

# def test_onevone():
#     DataPath = os.environ.get("LME_TRAINING_DATA")
#     path = f"{DataPath}\\The Tome 3-22-21.txt"
#     t = Tome(source="pickle", path=path)
#     trainable, feature_names = t.prepare_for_gb_classifier(path=path)

#     classificator = OneVOne(trainable, '2','3', verbose=True, features=feature_names)

def make_training_tome():
    DataPath = os.environ.get("LME_TRAINING_DATA")
    path = f"{DataPath}\\Tome_2023_04_08.txt"
    t = Tome(source='pickle', path=path, verbose=False)
    trainable, feature_names = t.prepare_for_gb_classifier(path=path)
    return t, trainable, feature_names

# def make_predictable_tome():
#     DataPath = os.environ.get("LME_TRAINING_DATA")
#     path = f"{DataPath}\\tome_for_predictions.txt"
#     p = Tome(source='pickle', path=path, verbose=False)
#     predictable, feature_names = p.prepare_for_gb_classifier(path=path)
#     return p, predictable, feature_names

def get_scaler(trainable):

    x0 = trainable["0"]["feat"]
    x1 = trainable["1"]["feat"]
    x2 = trainable["2"]["feat"]
    x3 = trainable["3"]["feat"]
    x = np.concatenate((x0, x1, x2, x3))

    scaler = preprocessing.StandardScaler().fit(x)

    return scaler

def run_KNNMulti(trainable, feature_names, scaler):
    KNN_classificator = Classifier_KNN_Multi(
        trainable=trainable,
        scaler=scaler,
        verbose=True,
        features=feature_names
    )
    return KNN_classificator

def run_KNN_1v1(trainable, feature_names, scaler):
    KNN_classificator = Classifier_KNN_1v1(
        trainable=trainable,
        scaler=scaler,
        verbose=True,
        features=feature_names
    )
    return KNN_classificator

def run_RandomForestEnsemble(trainable, feature_names, scaler):
    RFE_classificator = Classifier_RandomForestEnsemble(
        trainable=trainable,
        scaler=scaler,
        verbose=True,
        features=feature_names
    )
    return RFE_classificator

def run_GBTrees_1v1(trainable, feature_names, scaler):
    GBT_classificator = Classifier_GradientBoostedTrees_1v1(
        trainable=trainable,
        scaler=scaler,
        verbose=True,
        features=feature_names
    )
    return GBT_classificator

if __name__ == "__main__":

    t, trainable, feature_names_training = make_training_tome()

    scaler = get_scaler(trainable)

    #run_RandomForestEnsemble(trainable, feature_names_training, scaler)
    run_GBTrees_1v1(trainable, feature_names_training, scaler)
    # run_KNNMulti(trainable, feature_names_training, scaler)
    # run_KNN_1v1(trainable, feature_names_training, scaler)
