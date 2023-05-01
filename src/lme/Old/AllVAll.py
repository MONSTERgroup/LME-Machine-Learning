from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


class AllVAll:
    def __init__(self, trainable, **kwargs) -> None:

        if kwargs.get("verbose", False):
            print("setting parameters...")

        if not kwargs.get("genetic", False):
            self.params = {}

            self.params["01"] = {'learning_rate': 0.14460964141681998, 'max_depth': 9, 'max_features': 5, 'min_samples_leaf': 0.027127267243118, 'min_samples_split': 0.3725546395367668, 'n_estimators': 542, 'subsample': 0.7945532068183083}
            self.params["02"] = {'learning_rate': 0.17528554658298034, 'max_depth': 19, 'max_features': 61, 'min_samples_leaf': 0.21373876568924605, 'min_samples_split': 0.17830077198075767, 'n_estimators': 230, 'subsample': 0.6958616319367862}
            self.params["03"] = {'learning_rate': 0.25451251289455024, 'max_depth': 11, 'max_features': 27, 'min_samples_leaf': 0.0061240494986215645, 'min_samples_split': 0.931005132652906, 'n_estimators': 226, 'subsample': 0.9542023462571922}
            self.params["12"] = {'learning_rate': 0.6054325708217553, 'max_depth': 10, 'max_features': 60, 'min_samples_leaf': 0.2684584421879727, 'min_samples_split': 0.2617892399884044, 'n_estimators': 393, 'subsample': 0.8169639788630859}
            self.params["13"] = {'learning_rate': 0.9074038030926245, 'max_depth': 17, 'max_features': 41, 'min_samples_leaf': 0.09967639155410575, 'min_samples_split': 0.3029448258185029, 'n_estimators': 792, 'subsample': 0.5672707768083121}
            self.params["23"] = {'learning_rate': 0.04407844492043744, 'max_depth': 2, 'max_features': 27, 'min_samples_leaf': 0.25375798912904596, 'min_samples_split': 0.583263975885689, 'n_estimators': 261, 'subsample': 0.8195810025495328}
        else:
            self.params = kwargs.get("params")

        if kwargs.get("verbose", False):
            print("building classifiers...")

        self.clf = {}

        for k in trainable.keys():
            for l in trainable.keys():
                if int(l) > int(k):
                    self.clf[f"{k}{l}"] = GradientBoostingClassifier(
                        random_state=0, **self.params[f"{k}{l}"]
                    )

        if kwargs.get("verbose", False):
            print("creating test splits")

        self.feat_train = {}
        self.feat_test = {}
        self.mech_train = {}
        self.mech_test = {}

        for k in trainable.keys():
            (
                self.feat_train[int(k)],
                self.feat_test[int(k)],
                self.mech_train[int(k)],
                self.mech_test[int(k)],
            ) = train_test_split(
                trainable[k]["feat"],
                np.asarray(trainable[k]["mech"]),
                test_size=0.25,
                random_state=42,
            )

        if kwargs.get("verbose", False):
            print("Concatenate all the things")

        self.feat_train_pair = {}
        self.mech_train_pair = {}
        self.feat_test_pair = {}
        self.mech_test_pair = {}

        for k in trainable.keys():
            for l in trainable.keys():
                if int(l) > int(k):
                    self.feat_train_pair[f"{k}{l}"] = np.concatenate(
                        (self.feat_train[int(k)], self.feat_train[int(l)])
                    )
                    self.mech_train_pair[f"{k}{l}"] = np.concatenate(
                        (self.mech_train[int(k)], self.mech_train[int(l)])
                    )
                    self.feat_test_pair[f"{k}{l}"] = np.concatenate(
                        (self.feat_test[int(k)], self.feat_test[int(l)])
                    )
                    self.mech_test_pair[f"{k}{l}"] = np.concatenate(
                        (self.mech_test[int(k)], self.mech_test[int(l)])
                    )

        if kwargs.get("verbose", False):
            print("Training...maybe?...")

        for k in trainable.keys():
            for l in trainable.keys():
                if int(l) > int(k):
                    self.clf[f"{k}{l}"].fit(
                        self.feat_train_pair[f"{k}{l}"], self.mech_train_pair[f"{k}{l}"]
                    )

        feat_test = np.concatenate([self.feat_test[int(k)] for k in trainable.keys()])
        mech_test = np.concatenate([self.mech_test[int(k)] for k in trainable.keys()])

        if kwargs.get("verbose", False):
            print("Predicting...maybe?...")

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

        sns.set()
        cm = confusion_matrix(actual, modelpredn, normalize="all")

        ax = plt.subplot()
        sns.heatmap(cm / np.sum(cm), annot=True, fmt=".2%", cmap="Blues")

        # labels, title and ticks
        ax.set_xlabel("Predicted labels")
        ax.set_ylabel("True labels")
        ax.set_title("Confusion Matrix")
        ax.xaxis.set_ticklabels(["0", "1", "2", "3"])
        ax.yaxis.set_ticklabels(["0", "1", "2", "3"])
        plt.show()

        self.report = classification_report(actual, modelpredn, output_dict=True)
        print(self.report)

    def how_accurate(self) -> float:
        return self.report["accuracy"]
