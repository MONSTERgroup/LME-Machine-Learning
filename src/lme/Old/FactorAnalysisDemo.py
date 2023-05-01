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

from sklearn.decomposition import FactorAnalysis
from sklearn.covariance import ShrunkCovariance, LedoitWolf
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from scipy import linalg

class FactorAnalysisDemo:
    def __init__(self, trainable, predictable, **kwargs) -> None:

        if kwargs.get("verbose", False):
            print("setting parameters...")

        if "features" in kwargs:
            feature_names = kwargs.get("features")

        if kwargs.get("verbose", False):
            print("building classifiers...")

        n_components = np.arange(0, len(feature_names), 5)
        rank = 4

        X = np.concatenate((trainable["0"]["feat"], trainable["1"]["feat"], trainable["2"]["feat"], trainable["3"]["feat"]))
        y = np.concatenate((trainable["0"]["mech"], trainable["1"]["mech"], trainable["2"]["mech"], trainable["3"]["mech"]))

        ax = plt.axes()

        im = ax.imshow(np.corrcoef(X.T), cmap="RdBu_r", vmin=-1, vmax=1)

        ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        ax.set_xticklabels(list(feature_names), rotation=90)
        ax.set_yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        ax.set_yticklabels(list(feature_names))

        plt.colorbar(im).ax.set_ylabel("$r$", rotation=0)
        ax.set_title("Iris feature correlation matrix")
        plt.tight_layout()

        n_comps = 2

        methods = [
            ("PCA", PCA()),
            ("Unrotated FA", FactorAnalysis()),
            ("Varimax FA", FactorAnalysis(rotation="varimax")),
        ]
        fig, axes = plt.subplots(ncols=len(methods), figsize=(10, 8), sharey=True)

        for ax, (method, fa) in zip(axes, methods):
            fa.set_params(n_components=n_comps)
            fa.fit(X)

            components = fa.components_.T
            print("\n\n %s :\n" % method)
            print(components)

            vmax = np.abs(components).max()
            ax.imshow(components, cmap="RdBu_r", vmax=vmax, vmin=-vmax)
            ax.set_yticks(np.arange(len(feature_names)))
            ax.set_yticklabels(feature_names)
            ax.set_title(str(method))
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Comp. 1", "Comp. 2"])
        fig.suptitle("Factors")
        plt.tight_layout()
        plt.show()

        def compute_scores(X):
            pca = PCA(svd_solver="full")
            fa = FactorAnalysis()

            pca_scores, fa_scores = [], []
            for n in n_components:
                pca.n_components = n
                fa.n_components = n
                pca_scores.append(np.mean(cross_val_score(pca, X)))
                fa_scores.append(np.mean(cross_val_score(fa, X)))

            return pca_scores, fa_scores


        def shrunk_cov_score(X):
            shrinkages = np.logspace(-2, 0, 30)
            cv = GridSearchCV(ShrunkCovariance(), {"shrinkage": shrinkages})
            return np.mean(cross_val_score(cv.fit(X).best_estimator_, X))


        def lw_score(X):
            return np.mean(cross_val_score(LedoitWolf(), X))


        for X, title in [(X, "LME Data")]:
            pca_scores, fa_scores = compute_scores(X)
            n_components_pca = n_components[np.argmax(pca_scores)]
            n_components_fa = n_components[np.argmax(fa_scores)]

            pca = PCA(svd_solver="full", n_components="mle")
            pca.fit(X)
            n_components_pca_mle = pca.n_components_

            print("best n_components by PCA CV = %d" % n_components_pca)
            print("best n_components by FactorAnalysis CV = %d" % n_components_fa)
            print("best n_components by PCA MLE = %d" % n_components_pca_mle)

            plt.figure()
            plt.plot(n_components, pca_scores, "b", label="PCA scores")
            plt.plot(n_components, fa_scores, "r", label="FA scores")
            plt.axvline(rank, color="g", label="TRUTH: %d" % rank, linestyle="-")
            plt.axvline(
                n_components_pca,
                color="b",
                label="PCA CV: %d" % n_components_pca,
                linestyle="--",
            )
            plt.axvline(
                n_components_fa,
                color="r",
                label="FactorAnalysis CV: %d" % n_components_fa,
                linestyle="--",
            )
            plt.axvline(
                n_components_pca_mle,
                color="k",
                label="PCA MLE: %d" % n_components_pca_mle,
                linestyle="--",
            )

            # compare with other covariance estimators
            plt.axhline(
                shrunk_cov_score(X),
                color="violet",
                label="Shrunk Covariance MLE",
                linestyle="-.",
            )
            plt.axhline(
                lw_score(X),
                color="orange",
                label="LedoitWolf MLE" % n_components_pca_mle,
                linestyle="-.",
            )

            plt.xlabel("nb of components")
            plt.ylabel("CV scores")
            plt.legend(loc="lower right")
            plt.title(title)

        plt.show()

