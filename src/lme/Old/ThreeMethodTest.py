import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import RandomizedSearchCV, train_test_split

from scipy.stats import uniform as sp_randFloat
from scipy.stats import randint as sp_randInt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler

class ThreeMethodTest:
    def __init__(self, trainable, predictable, **kwargs) -> None:

        if "features" in kwargs:
            feature_names = kwargs.get("features")

        x = np.concatenate((trainable["0"]["feat"], trainable["1"]["feat"], trainable["2"]["feat"], trainable["3"]["feat"]))
        y = np.concatenate((trainable["0"]["mech"], trainable["1"]["mech"], trainable["2"]["mech"], trainable["3"]["mech"]))

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

        #################################################################################################################################
        print("\n\n\nK-nearest Neighbors\n\n\n")
        tic = time.perf_counter()

        model = KNeighborsClassifier(weights="distance")

        parameters = {
            "n_neighbors": sp_randInt(5, 20),
        }

        print(f'Performing Random Forest Classifier variable search')

        randm_NN = RandomizedSearchCV(
            estimator=model,
            param_distributions=parameters,
            cv = 5,
            n_iter=1000,
            n_jobs=-1,
            verbose=1,
            refit = True
        )

        randm_NN.fit(x_train, y_train)

        print(" Results from Random Search ")
        print("The best estimator across ALL searched params:", randm_NN.best_estimator_)
        print("The best score across ALL searched params:", randm_NN.best_score_)
        print("The best parameters across ALL searched params:", randm_NN.best_params_)

        toc = time.perf_counter()
        print(f"Calculated in {toc - tic:0.4f} seconds")

        #################################################################################################################################
        print("Random Forest\n\n\n")
        tic = time.perf_counter()

        model = RandomForestClassifier()

        parameters = {
            "n_estimators": sp_randInt(10, 500),
            "criterion": ["gini", "entropy"],
            "max_depth": sp_randInt(1, 100),
            #"min_samples_split": sp_randInt(2, 10),
            "min_samples_leaf": sp_randInt(1, 10),
            "max_features": sp_randInt(1, len(feature_names)),
        }

        print(f'Performing Random Forest Classifier variable search')

        randm_RF = RandomizedSearchCV(
            estimator=model,
            param_distributions=parameters,
            cv = 5,
            n_iter=1000,
            n_jobs=-1,
            verbose=1,
            refit = True
        )

        randm_RF.fit(x_train, y_train)

        print(" Results from Random Search ")
        print("The best estimator across ALL searched params:", randm_RF.best_estimator_)
        print("The best score across ALL searched params:", randm_RF.best_score_)
        print("The best parameters across ALL searched params:", randm_RF.best_params_)

        toc = time.perf_counter()
        print(f"Calculated in {toc - tic:0.4f} seconds")

        #################################################################################################################################
        print("\n\n\nNeural Network\n\n\n")

        tic = time.perf_counter()

        scaled_x_train = MinMaxScaler().fit_transform(x_train)
        max_iter = 1000
        
        model = MLPClassifier()

        parameters = {
            "hidden_layer_sizes": (50, sp_randInt(0,25))
        }

        print(f'Performing Random Forest Classifier variable search')

        randm_KNN = RandomizedSearchCV(
            estimator=model,
            param_distributions=parameters,
            cv = 5,
            n_iter=1000,
            n_jobs=-1,
            verbose=1,
            refit = True
        )

        randm_KNN.fit(x_train, y_train)

        print(" Results from Random Search ")
        print("The best estimator across ALL searched params:", randm_KNN.best_estimator_)
        print("The best score across ALL searched params:", randm_KNN.best_score_)
        print("The best parameters across ALL searched params:", randm_KNN.best_params_)

        toc = time.perf_counter()
        print(f"Calculated in {toc - tic:0.4f} seconds")

