import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import time
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance

from scipy.stats import uniform as sp_randFloat
from scipy.stats import randint as sp_randInt

from datetime import datetime

class Classifier_RandomForestEnsemble:
    def __init__(self, trainable, scaler, **kwargs) -> None:

        start_date_time = datetime.fromtimestamp(datetime.now().timestamp(), tz=None)

        fID = open("Classifier_RandomForestEnsemble_Logs.txt", "a")
        fID.write(f"\nRun date and time: {start_date_time}\n")

        if "features" in kwargs:
            feature_names = kwargs.get("features")
            print(f"\nFeature Names:\n")
            fID.write(f"\nFeature Names:\n")
            for f in feature_names:
                print(f"\t{f}\n")
                fID.write(f"\t{f}\n")

        if kwargs.get("verbose", False):
            print(f"\nConcatenating full feature set and mechanism set...\n")
            fID.write(f"\nConcatenating full feature set and mechanism set...\n")

        x = np.concatenate((trainable["0"]["feat"], trainable["1"]["feat"], trainable["2"]["feat"], trainable["3"]["feat"]))
        y = np.concatenate((trainable["0"]["mech"], trainable["1"]["mech"], trainable["2"]["mech"], trainable["3"]["mech"]))

        if kwargs.get("verbose", False):
            print(f"\nScaling x in feature set...\n")
            fID.write(f"\nScaling x in feature set...\n")

        x_scaled = scaler.transform(x)

        test_size = 0.3
        validate_size = 0.2
        random_state = 42
        if kwargs.get("verbose", False):
            to_write = (f"\nMaking test-train split:\n"
                    + f"\tTest set size: {test_size*100}% of full set\n"
                    + f"\tRandom state: {random_state}\n"
                    + f"\tValidation set size: {validate_size*100}% of training set ({validate_size*(1-test_size)*100}% of full set)\n")
            print(to_write)
            fID.write(to_write)

        x_train_valid, x_test, y_train_valid, y_test = train_test_split(x_scaled, y, test_size=test_size, random_state=random_state)
        x_train, x_validate, y_train, y_validate = train_test_split(x_train_valid, y_train_valid, test_size=validate_size, random_state=random_state)

        cv = 10
        n_iter = 1000
        if kwargs.get("verbose", False):
            to_write = (f"\nStarting parameter search for Random Forest Classifier:\n"
                    + f"\tStart time: {datetime.fromtimestamp(datetime.now().timestamp(), tz=None)}"
                    + f"\tCross validation:\n"
                    + f"\t\t{cv} steps\n"
                    + f"\t\t{n_iter} iterations\n"
                    + f"\t\t{cv*n_iter} total fits\n")
            print(to_write)
            fID.write(to_write)

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

        randm = RandomizedSearchCV(
            estimator=model,
            param_distributions=parameters,
            cv = 10,
            n_iter=1000,
            n_jobs=-1,
            verbose=1,
            refit = True
        )
        randm.fit(x_train, y_train)

        toc = time.perf_counter()

        if kwargs.get("verbose", False):
            to_write = (f"\nResults from Random Search:\n"
                    + f"\tThe best estimator across ALL searched params: {randm.best_estimator_}\n"
                    + f"\tThe best score across ALL searched params: {randm.best_score_}\n"
                    + f"\tThe best parameters across ALL searched params: {randm.best_params_}\n"
                    + f"Calculated in {toc - tic:0.4f} seconds.\n")
            print(to_write)
            fID.write(to_write)

        if kwargs.get("verbose", False):
            print(f"\nApplying sigmoid calibration...\n")
            fID.write(f"\nApplying sigmoid calibration...\n")

        clf = randm.best_estimator_
        cal_clf = CalibratedClassifierCV(clf, method="sigmoid", cv="prefit")
        cal_clf.fit(x_validate, y_validate)

        score = cal_clf.score(x_test,y_test)
        
        if kwargs.get("verbose", False):
            print(f"\nScore after sigmoid calibration: {score}\n")
            fID.write(f"\nScore after sigmoid calibration: {score}\n")

        feature_importance = clf.feature_importances_
        if kwargs.get("verbose", False):
            fID.write(f"\nFeature Importances:\n")
            for name, importance in zip(feature_names, feature_importance):
                print(f"\t{name}: {importance}\n")
                fID.write(f"\t{name}: {importance}\n")

        if kwargs.get("verbose", False):
            print(f"\nPlotting feature importance...\n")
            fID.write(f"\nPlotting feature importance...\n")

        sorted_idx = np.argsort(feature_importance)
        pos = np.arange(sorted_idx.shape[0]) + 0.5
        fig = plt.figure(figsize=(12, 6))
        ax1 = fig.add_subplot(1, 2, 1)
        feat_imp_plot = ax1.barh(pos, feature_importance[sorted_idx], align="center")
        ax1.set_yticks(pos, np.array(feature_names)[sorted_idx], fontsize=16)
        #ax1.set_title("Feature Importance (MDI)")
        ax1.set_facecolor("white")
        ax1.tick_params(top=False,
                        bottom=False,
                        left=False,
                        right=False,
                        labelleft=True,
                        labelbottom=True,
                        labelsize=24)

        if kwargs.get("verbose", False):
            print(f"\nCalculating permutation importance...\n")
            fID.write(f"\nCalculating permutation importance...\n")

        result = permutation_importance(
            cal_clf, x_test, y_test, n_repeats=100, random_state=random_state, n_jobs=2
        )

        sorted_idx = result.importances_mean.argsort()
        ax2 = fig.add_subplot(1, 2, 2)
        perm_imp_plot = ax2.boxplot(
            result.importances[sorted_idx].T,
            vert=False,
            labels=np.array(feature_names)[sorted_idx],
        )
        #ax2.set_title("Permutation Importance (test set)")
        ax2.set_facecolor("white")
        ax2.tick_params(top=False,
                        bottom=False,
                        left=False,
                        right=False,
                        labelleft=True,
                        labelbottom=True,
                        labelsize=24)
        fig.tight_layout()
        plt.savefig(f"FeatureImportances_Original_RandomForestEnsemble_{cv}-{n_iter}_ForPaper.png", 
                    bbox_inches='tight',
                    transparent=True)
        
        if kwargs.get("verbose", False):
            print(f"\nPlotting feature importance, take 2...\n")
            fID.write(f"\nPlotting feature importance, take 2...\n")

        sorted_idx = np.flip(np.argsort(feature_names))
        pos = np.arange(sorted_idx.shape[0]) + 0.5
        fig = plt.figure(figsize=(12, 6))
        ax1 = fig.add_subplot(1, 2, 1)
        feat_imp_plot = ax1.barh(pos, feature_importance[sorted_idx], align="center")
        ax1.set_yticks(pos, np.array(feature_names)[sorted_idx], fontsize=16)
        ax1.set_title("Feature Importance (MDI)")
        ax1.set_facecolor("white")
        ax1.tick_params(top=False,
                        bottom=True,
                        left=False,
                        right=False,
                        labelleft=True,
                        labelbottom=True,
                        labelsize=24)
        
        if kwargs.get("verbose", False):
            print(f"\nCalculating permutation importance...\n")
            fID.write(f"\nCalculating permutation importance...\n")

        # result = permutation_importance(
        #     randm, x_test_scaled, y_test, n_repeats=100, random_state=42, n_jobs=2
        # )

        #sorted_idx = result.importances_mean.argsort()
        ax2 = fig.add_subplot(1, 2, 2)
        perm_imp_plot = ax2.boxplot(
            result.importances[sorted_idx].T,
            vert=False,
            labels=np.array(feature_names)[sorted_idx],
        )
        ax2.set_title("Permutation Importance (test set)")
        ax2.set_facecolor("white")
        ax2.tick_params(top=False,
                        bottom=True,
                        left=False,
                        right=False,
                        labelleft=False,
                        labelbottom=True,
                        labelsize=24)
        fig.tight_layout()
        plt.savefig(f"FeatureImportances_Unriginal_RandomForestEnsemble_{cv}-{n_iter}_ForPaper.png", 
                    bbox_inches='tight',
                    transparent=True)
        
        if kwargs.get("verbose", False):
            print(f"\nMaking predictions on test set...\n")
            fID.write(f"\nMaking predictions on test set...\n")

        pred = cal_clf.predict(x_test)
        actual = pd.DataFrame(data=y_test, columns=["Actual"])

        ConfusionMatrixDisplay.from_predictions(y_test, 
                                                pred, 
                                                display_labels=clf.classes_,
                                                cmap='RdYlGn',
                                                normalize="true")
        plt.savefig(f'ConfusionMatrix_NormTrue_RandomForestEnsemble_{cv}-{n_iter}_ForPaper.png',
                    bbox_inches='tight',
                    transparent=True)
        
        ConfusionMatrixDisplay.from_predictions(y_test, 
                                                pred, 
                                                display_labels=clf.classes_,
                                                cmap='RdYlGn',
                                                normalize="pred")
        plt.savefig(f'ConfusionMatrix_NormPred_RandomForestEnsemble_{cv}-{n_iter}_ForPaper.png',
                    bbox_inches='tight',
                    transparent=True)
        
        ConfusionMatrixDisplay.from_predictions(y_test, 
                                                pred, 
                                                display_labels=clf.classes_,
                                                cmap='RdYlGn',
                                                normalize="all")
        plt.savefig(f'ConfusionMatrix_NormAll_RandomForestEnsemble_{cv}-{n_iter}_ForPaper.png',
                    bbox_inches='tight',
                    transparent=True)
        
        ConfusionMatrixDisplay.from_predictions(y_test, 
                                                pred, 
                                                display_labels=clf.classes_,
                                                cmap='RdYlGn',
                                                normalize=None)
        plt.savefig(f'ConfusionMatrix_NormNone_RandomForestEnsemble_{cv}-{n_iter}_ForPaper.png',
                    bbox_inches='tight',
                    transparent=True)
        
        if kwargs.get("verbose", False):
            print(f"\nWriting reports...\n")
            fID.write(f"\nWriting reports...\n")

        self.report = classification_report(actual,pred, output_dict=True)

        print(self.report)
        fID.write(json.dumps(self.report))

        fID.close()
