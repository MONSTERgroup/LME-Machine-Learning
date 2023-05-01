import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

class PCA_KNN_demo:
    def __init__(self, trainable, predictable, **kwargs) -> None:

        if kwargs.get("verbose", False):
            print("setting parameters...")

        if "features" in kwargs:
            feature_names = kwargs.get("features")

        if kwargs.get("verbose", False):
            print("building classifiers...")

        x = np.concatenate((trainable["0"]["feat"], trainable["1"]["feat"], trainable["2"]["feat"], trainable["3"]["feat"]))
        y = np.concatenate((trainable["0"]["mech"], trainable["1"]["mech"], trainable["2"]["mech"], trainable["3"]["mech"]))

        random_state = 0
        n_neighbors = 3

        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            x, y, test_size=0.5, stratify=y, random_state=random_state
        )

        dim = len(x[0])
        n_classes = len(np.unique(y))

        # Reduce dimension to 2 with PCA
        pca = make_pipeline(StandardScaler(), PCA(n_components=10, random_state=random_state))

        # Reduce dimension to 2 with LinearDiscriminantAnalysis
        lda = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(n_components=10))

        # Reduce dimension to 2 with NeighborhoodComponentAnalysis
        nca = make_pipeline(
            StandardScaler(),
            NeighborhoodComponentsAnalysis(n_components=10, random_state=random_state),
        )

        # Use a nearest neighbor classifier to evaluate the methods
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)

        # Make a list of the methods to be compared
        dim_reduction_methods = [("PCA", pca), ("LDA", lda), ("NCA", nca)]

        # plt.figure()
        for i, (name, model) in enumerate(dim_reduction_methods):
            plt.figure()
            # plt.subplot(1, 3, i + 1, aspect=1)

            # Fit the method's model
            model.fit(X_train, y_train)

            # Fit a nearest neighbor classifier on the embedded training set
            knn.fit(model.transform(X_train), y_train)

            # Compute the nearest neighbor accuracy on the embedded test set
            acc_knn = knn.score(model.transform(X_test), y_test)

            # Embed the data set in 2 dimensions using the fitted model
            X_embedded = model.transform(x)

            # Plot the projected points and show the evaluation score
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            ax.scatter(X_embedded[:, 0], X_embedded[:, 1], X_embedded[:, 2], c=y)#, s=30, cmap="Set1")
            plt.title(
                "{}, KNN (k={})\nTest accuracy = {:.2f}".format(name, n_neighbors, acc_knn)
            )
        plt.show()

