from datetime import datetime
import json
import time
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
# from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.inspection import permutation_importance
from scipy.stats import uniform as sp_randFloat
from scipy.stats import randint as sp_randInt


class Classifier_GradientBoostedTrees_1v1:
    def __init__(self, trainable, scaler, **kwargs) -> None:

        start_date_time = datetime.fromtimestamp(datetime.now().timestamp(), tz=None)

        self.fID = open("Classifier_GradientBoostedTrees_1v1_Logs.txt", "a")
        self.fID.write(f"\nRun date and time: {start_date_time}\n")

        if "features" in kwargs:
            feature_names = kwargs.get("features")
            print(f"\nFeature Names:\n")
            self.fID.write(f"\nFeature Names:\n")
            for f in feature_names:
                print(f"\t{f}\n")
                self.fID.write(f"\t{f}\n")

        if kwargs.get("verbose", False):
            print(f"\nConcatenating 1v1 feature sets and mechanism set...\n")
            self.fID.write(f"\nConcatenating 1v1 feature sets and mechanism set...\n")


        self.clf = {}
        self.x_test = {}
        self.y_test = {}

        cv=10
        n_iter=1000
        for k in trainable.keys():
            for l in trainable.keys():
                if int(l) > int(k):
                    #for m in range(1,5):
                        self.clf[f"{k}{l}"], self.x_test[f"{k}{l}"], self.y_test[f"{k}{l}"] = self.getBestClassifier(
                            trainable, k, l, cv=cv, nIter=n_iter, scaler=scaler, verbose=True, features=feature_names
                        )

        if kwargs.get("verbose", False):
            print(f"\nConcatenating test set...\n")
            self.fID.write(f"\nConcatenating test set...\n")

        feat_test = np.concatenate((self.x_test['01'], self.x_test['23']))
        mech_test = np.concatenate((self.y_test['01'], self.y_test['23']))

        if kwargs.get("verbose", False):
            print(f"\nMaking predictions on test set...\n")
            self.fID.write(f"\nMaking predictions on test set...\n")

        self.prob = {}

        for k in trainable.keys():
            for l in trainable.keys():
                if int(l) > int(k):
                    self.prob[f"{k}{l}"] = self.clf[f"{k}{l}"].predict_proba(feat_test)

        self.prob_all = np.concatenate(
            [
                self.prob[f"{k}{l}"]
                for k in trainable.keys()
                for l in trainable.keys()
                if int(l) > int(k)
            ],
            axis=1,
        )

        displayable_prob_all = pd.DataFrame(
            data=self.prob_all,
            columns=[
                f"{k}{l}-{k if i == 1 else l}"
                for k in trainable.keys()
                for l in trainable.keys()
                for i in range(1, 3)
                if int(l) > int(k)
            ],
        )
        pd.set_option("display.max_rows", self.prob_all.shape[0] + 1)
        print(displayable_prob_all)

        preds = [0.0] * feat_test.shape[0] * 4
        preds = np.reshape(preds, (feat_test.shape[0], 4))

        for ii in [0, 2, 4]:
            for jj in range(0, feat_test.shape[0]):
                preds[jj, 0] = preds[jj, 0] + self.prob_all[jj, ii]
        for ii in [1, 6, 8]:
            for jj in range(0, feat_test.shape[0]):
                preds[jj, 1] = preds[jj, 1] + self.prob_all[jj, ii]
        for ii in [3, 7, 10]:
            for jj in range(0, feat_test.shape[0]):
                preds[jj, 2] = preds[jj, 2] + self.prob_all[jj, ii]
        for ii in [5, 9, 11]:
            for jj in range(0, feat_test.shape[0]):
                preds[jj, 3] = preds[jj, 3] + self.prob_all[jj, ii]

        displayable_preds = pd.DataFrame(data=preds, columns=["0", "1", "2", "3"])
        pd.set_option("display.max_rows", preds.shape[0] + 1)
        print(displayable_preds)

        modelpred = displayable_preds.idxmax(axis=1)
        pred = pd.DataFrame(data=modelpred, columns=["Prediction"])
        actual = pd.DataFrame(data=mech_test, columns=["Actual"])
        # display_side_by_side(pred, actual)

        compare_predictions = pd.concat([pred, actual], axis=1)
        print(compare_predictions)

        modelpredn = np.array([0] * feat_test.shape[0])
        for ii in range(0, feat_test.shape[0] - 1):
            modelpredn[ii] = int(pred.iloc[ii])

        ConfusionMatrixDisplay.from_predictions(actual, 
                                                modelpredn, 
                                                display_labels=["0","1","2","3"],
                                                cmap='RdYlGn',
                                                normalize="true")
        plt.savefig(f'ConfusionMatrix_NormTrue_GBT-1v1_{cv}-{n_iter}_ForPaper.png',
                    bbox_inches='tight',
                    transparent=True)
        
        ConfusionMatrixDisplay.from_predictions(actual, 
                                                modelpredn, 
                                                display_labels=["0","1","2","3"],
                                                cmap='RdYlGn',
                                                normalize="pred")
        plt.savefig(f'ConfusionMatrix_NormPred_GBT-1v1_{cv}-{n_iter}_ForPaper.png',
                    bbox_inches='tight',
                    transparent=True)
        
        ConfusionMatrixDisplay.from_predictions(actual, 
                                                modelpredn, 
                                                display_labels=["0","1","2","3"],
                                                cmap='RdYlGn',
                                                normalize="all")
        plt.savefig(f'ConfusionMatrix_NormAll_GBT-1v1_{cv}-{n_iter}_ForPaper.png',
                    bbox_inches='tight',
                    transparent=True)
        
        ConfusionMatrixDisplay.from_predictions(actual, 
                                                modelpredn, 
                                                display_labels=["0","1","2","3"],
                                                cmap='RdYlGn',
                                                normalize=None)
        plt.savefig(f'ConfusionMatrix_NormNone_GBT-1v1_{cv}-{n_iter}_ForPaper.png',
                    bbox_inches='tight',
                    transparent=True)
        
        if kwargs.get("verbose", False):
            print(f"\nWriting reports...\n")
            self.fID.write(f"\nWriting reports...\n")

        # sns.set()
        # cm = confusion_matrix(actual, modelpredn, normalize="all")

        # ax = plt.subplot()
        # sns.heatmap(cm / np.sum(cm), annot=True, fmt=".2%", cmap="Blues")

        # # labels, title and ticks
        # ax.set_xlabel("Predicted labels")
        # ax.set_ylabel("True labels")
        # ax.set_title("Confusion Matrix")
        # ax.xaxis.set_ticklabels(["0", "1", "2", "3"])
        # ax.yaxis.set_ticklabels(["0", "1", "2", "3"])
        # plt.savefig(f'Confusion_{10}_{1000}_LikelyFinal.png', bbox_inches='tight')
        # #plt.show()

        self.report = classification_report(actual, modelpredn, output_dict=True)
        print(self.report)

        self.fID.write(json.dumps(self.report))

        self.fID.close()

    def how_accurate(self) -> float:
        return self.report["accuracy"]

    def getBestClassifier(
        self, trainable, first, second, cv, nIter, scaler, **kwargs
    ) -> GradientBoostingClassifier:

        if "features" in kwargs:
            feature_names = kwargs.get("features")

        if kwargs.get("verbose", False):
            print(f"\nConcatenating {first}v{second} feature and mechanism sets...\n")
            self.fID.write(f"\nConcatenating {first}v{second} feature and mechanism sets...\n")

        x_first = trainable[first]["feat"]
        x_second = trainable[second]["feat"]
        x = np.concatenate((x_first, x_second))

        y_first = trainable[first]["mech"]
        y_second = trainable[second]["mech"]
        y = np.concatenate((y_first, y_second))
        if kwargs.get("verbose", False):
            self.fID.write(f"\n")
            self.fID.write(''.join(str(e) for e in y))
            self.fID.write(f"\n")
            print(y)

        y[y == int(first)] = 0
        y[y == int(second)] = 1
        if kwargs.get("verbose", False):
            self.fID.write(f"\n")
            self.fID.write(''.join(str(e) for e in y))
            self.fID.write(f"\n")
            print(y)

        test_size = 0.3
        random_state = 42
        if kwargs.get("verbose", False):
            to_write = (f"\nMaking test-train split:\n"
                    + f"\tTest set size: {test_size*100}% of full set\n"
                    + f"\tRandom state: {random_state}\n")
            print(to_write)
            self.fID.write(to_write)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

        if kwargs.get("verbose", False):
            print(f"\nScaling x in feature set...\n")
            self.fID.write(f"\nScaling x in feature set...\n")

        x_train_scaled = scaler.transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        if kwargs.get("verbose", False):
            to_write = (f"\nStarting parameter search for 1v1 Gradient Boosted Trees Classifier:\n"
                    + f"\tStart time: {datetime.fromtimestamp(datetime.now().timestamp(), tz=None)}"
                    + f"\tCross validation:\n"
                    + f"\t\t{cv} steps\n"
                    + f"\t\t{nIter} iterations\n"
                    + f"\t\t{cv*nIter} total fits\n")
            print(to_write)
            self.fID.write(to_write)

        tic = time.perf_counter()

        model = GradientBoostingClassifier()
        parameters = {
            "learning_rate": sp_randFloat(loc=0, scale=10),
            "subsample": sp_randFloat(),
            "max_depth": sp_randInt(1, 32),
            "max_features": sp_randInt(1, 13),
            "min_samples_leaf": sp_randFloat(loc=0.001, scale=0.499),
            "min_samples_split": sp_randFloat(),
            "n_estimators": sp_randInt(1, 1000),
        }

        randm = RandomizedSearchCV(
            estimator=model,
            param_distributions=parameters,
            cv = cv,
            n_iter= nIter,
            n_jobs=-1,
            verbose=1,
        )
        randm.fit(x_train_scaled, y_train)

        toc = time.perf_counter()

        if kwargs.get("verbose", False):
            to_write = (f"\nResults from Random Search:\n"
                    + f"\tThe best estimator across ALL searched params: {randm.best_estimator_}\n"
                    + f"\tThe best score across ALL searched params: {randm.best_score_}\n"
                    + f"\tThe best parameters across ALL searched params: {randm.best_params_}\n"
                    + f"Calculated in {toc - tic:0.4f} seconds.\n")
            print(to_write)
            self.fID.write(to_write)

        feature_importance = randm.best_estimator_.feature_importances_
        if kwargs.get("verbose", False):
            self.fID.write(f"\nFeature Importances {first}v{second}:\n")
            for name, importance in zip(feature_names, feature_importance):
                print(f"\t{name}: {importance}\n")
                self.fID.write(f"\t{name}: {importance}\n")

        if kwargs.get("verbose", False):
            print(f"\nPlotting feature importance...\n")
            self.fID.write(f"\nPlotting feature importance...\n")

        sorted_idx = np.argsort(feature_importance)
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
            self.fID.write(f"\nCalculating permutation importance...\n")

        result = permutation_importance(
            randm, x_test_scaled, y_test, n_repeats=100, random_state=42, n_jobs=2
        )

        sorted_idx = result.importances_mean.argsort()
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
                        labelleft=True,
                        labelbottom=True,
                        labelsize=24)
        fig.tight_layout()
        plt.savefig(f"FeatureImportances_Original_{first}v{second}_GBT_1v1_{cv}-{nIter}_ForPaper.png", 
                    bbox_inches='tight',
                    transparent=True)
        
        if kwargs.get("verbose", False):
            print(f"\nPlotting feature importance, take 2...\n")
            self.fID.write(f"\nPlotting feature importance, take 2...\n")

        sorted_idx = np.flip(np.argsort(feature_names))
        pos = np.arange(sorted_idx.shape[0]) + 0.5
        fig = plt.figure(figsize=(12, 6))
        ax1 = fig.add_subplot(1, 2, 1)
        feat_imp_plot = ax1.barh(pos, feature_importance[sorted_idx], align="center")
        ax1.set_yticks(pos, np.array(feature_names)[sorted_idx], fontsize=16)
        #ax1.set_title("Feature Importance (MDI)")
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
            self.fID.write(f"\nCalculating permutation importance...\n")

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
        #ax2.set_title("Permutation Importance (test set)")
        ax2.set_facecolor("white")
        ax2.tick_params(top=False,
                        bottom=True,
                        left=False,
                        right=False,
                        labelleft=False,
                        labelbottom=True,
                        labelsize=24)
        fig.tight_layout()
        plt.savefig(f"FeatureImportances_Unoriginal_{first}v{second}_GBT_1v1_{cv}-{nIter}_ForPaper.png", 
                    bbox_inches='tight',
                    transparent=True)

        if (int(first) == 1 and (int(second) == 2 or int(second) == 3)):
            y_test[y_test == 1] = int(second)
            y_test[y_test == 0] = int(first)
            if kwargs.get("verbose", False):
                self.fID.write(f"\n")
                self.fID.write(''.join(str(e) for e in y_test))
                self.fID.write(f"\n")
                print(y_test)
        else:
            y_test[y_test == 0] = int(first)
            y_test[y_test == 1] = int(second)
            if kwargs.get("verbose", False):
                self.fID.write(f"\n")
                self.fID.write(''.join(str(e) for e in y_test))
                self.fID.write(f"\n")
                print(y_test)

        return randm.best_estimator_, x_test_scaled, y_test
