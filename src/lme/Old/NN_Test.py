import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.tree import ExtraTreeClassifier

from scipy.stats import uniform as sp_randFloat
from scipy.stats import randint as sp_randInt

class NN_Test:
    def __init__(self, trainable, predictable, **kwargs) -> None:

        if kwargs.get("verbose", False):
            print("setting parameters...")

        if "features" in kwargs:
            feature_names = kwargs.get("features")

        if kwargs.get("verbose", False):
            print("building classifiers...")

        # different learning rate schedules and momentum parameters
        params = [
            {
                "solver": "sgd",
                "learning_rate": "constant",
                "momentum": 0,
                "learning_rate_init": 0.2,
            },
            {
                "solver": "sgd",
                "learning_rate": "constant",
                "momentum": 0.9,
                "nesterovs_momentum": False,
                "learning_rate_init": 0.2,
            },
            {
                "solver": "sgd",
                "learning_rate": "constant",
                "momentum": 0.9,
                "nesterovs_momentum": True,
                "learning_rate_init": 0.2,
            },
            {
                "solver": "sgd",
                "learning_rate": "invscaling",
                "momentum": 0,
                "learning_rate_init": 0.2,
            },
            {
                "solver": "sgd",
                "learning_rate": "invscaling",
                "momentum": 0.9,
                "nesterovs_momentum": True,
                "learning_rate_init": 0.2,
            },
            {
                "solver": "sgd",
                "learning_rate": "invscaling",
                "momentum": 0.9,
                "nesterovs_momentum": False,
                "learning_rate_init": 0.2,
            },
            {"solver": "adam", "learning_rate_init": 0.01},
        ]

        labels = [
            "constant learning-rate",
            "constant with momentum",
            "constant with Nesterov's momentum",
            "inv-scaling learning-rate",
            "inv-scaling with momentum",
            "inv-scaling with Nesterov's momentum",
            "adam",
        ]

        plot_args = [
            {"c": "red", "linestyle": "-"},
            {"c": "green", "linestyle": "-"},
            {"c": "blue", "linestyle": "-"},
            {"c": "red", "linestyle": "--"},
            {"c": "green", "linestyle": "--"},
            {"c": "blue", "linestyle": "--"},
            {"c": "black", "linestyle": "-"},
        ]

        def plot_on_dataset(X, y, ax, name, **kwargs):
            # for each dataset, plot learning for each learning strategy
            print("\nlearning on dataset %s" % name)
            ax.set_title(name)

            mlps = []
            X = MinMaxScaler().fit_transform(X)
            max_iter = 1000

            for label, param in zip(labels, params):
                print("training: %s" % label)
                
                mlp = MLPClassifier(
                    random_state=0, 
                    max_iter=max_iter,
                    hidden_layer_sizes=(40,10), 
                    **param)
                mlp.fit(X, y)  # apply scaling on training data

                mlps.append(mlp)
                print("Training set score: %f" % mlp.score(X, y))
                print("Training set loss: %f" % mlp.loss_)
                if "X_test" in kwargs:
                    X_test = kwargs.get("X_test")
                    y_test = kwargs.get("y_test")
                    print("Test set score: %f" % mlp.score(X_test, y_test))
            for mlp, label, args in zip(mlps, labels, plot_args):
                ax.plot(mlp.loss_curve_, label=label, **args)

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        # load / generate some toy datasets
        x = np.concatenate((trainable["0"]["feat"], trainable["1"]["feat"], trainable["2"]["feat"], trainable["3"]["feat"]))
        y = np.concatenate((trainable["0"]["mech"], trainable["1"]["mech"], trainable["2"]["mech"], trainable["3"]["mech"]))

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

        data_sets = [
            (x_train, y_train),
        ]

        for ax, data, name in zip(
            axes.ravel(), data_sets, ["train"]
        ):
            plot_on_dataset(*data, ax=ax, name=name, X_test=x_test, y_test=y_test)

        fig.legend(ax.get_lines(), labels, ncol=3, loc="upper center")
        plt.show()