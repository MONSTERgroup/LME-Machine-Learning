from datetime import datetime
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import time
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

from sklearn.model_selection import RandomizedSearchCV, train_test_split

from scipy.stats import randint as sp_randInt
from sklearn.neighbors import KNeighborsClassifier

class Classifier_KNN_Multi:
    def __init__(self, trainable, scaler, **kwargs) -> None:

        start_date_time = datetime.fromtimestamp(datetime.now().timestamp(), tz=None)

        fID = open("Classifier_KNN_Multi_Logs.txt", "a")
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
        random_state = 42
        if kwargs.get("verbose", False):
            to_write = (f"\nMaking test-train split:\n"
                    + f"\tTest set size: {test_size*100}% of full set\n"
                    + f"\tRandom state: {random_state}\n")
            print(to_write)
            fID.write(to_write)

        x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=test_size, random_state=random_state)

        cv = 10
        n_iter = 1000
        if kwargs.get("verbose", False):
            to_write = (f"\nStarting parameter search for Multi-Output KNN Classifier:\n"
                    + f"\tStart time: {datetime.fromtimestamp(datetime.now().timestamp(), tz=None)}"
                    + f"\tCross validation:\n"
                    + f"\t\t{cv} steps\n"
                    + f"\t\t{n_iter} iterations\n"
                    + f"\t\t{cv*n_iter} total fits\n")
            print(to_write)
            fID.write(to_write)

        tic = time.perf_counter()

        model = KNeighborsClassifier(weights="distance")

        parameters = {
            "n_neighbors": sp_randInt(2, 100),
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

        clf = randm.best_estimator_
        score = clf.score(x_test,y_test)

        if kwargs.get("verbose", False):
            print(f"\nScore after training: {score}\n")
            fID.write(f"\nScore after training: {score}\n")

        pred = clf.predict(x_test)
        actual = pd.DataFrame(data=y_test, columns=["Actual"])

        ConfusionMatrixDisplay.from_predictions(y_test, 
                                                pred, 
                                                display_labels=clf.classes_,
                                                cmap='RdYlGn',
                                                normalize="true")
        plt.savefig(f'ConfusionMatrix_NormTrue_KNNMulti_{cv}-{n_iter}_ForPaper.png',
                    bbox_inches='tight',
                    transparent=True)
        
        ConfusionMatrixDisplay.from_predictions(y_test, 
                                                pred, 
                                                display_labels=clf.classes_,
                                                cmap='RdYlGn',
                                                normalize="pred")
        plt.savefig(f'ConfusionMatrix_NormPred_KNNMulti_{cv}-{n_iter}_ForPaper.png',
                    bbox_inches='tight',
                    transparent=True)
        
        ConfusionMatrixDisplay.from_predictions(y_test, 
                                                pred, 
                                                display_labels=clf.classes_,
                                                cmap='RdYlGn',
                                                normalize="all")
        plt.savefig(f'ConfusionMatrix_NormAll_KNNMulti_{cv}-{n_iter}_ForPaper.png',
                    bbox_inches='tight',
                    transparent=True)
        
        ConfusionMatrixDisplay.from_predictions(y_test, 
                                                pred, 
                                                display_labels=clf.classes_,
                                                cmap='RdYlGn',
                                                normalize=None)
        plt.savefig(f'ConfusionMatrix_NormNone_KNNMulti_{cv}-{n_iter}_ForPaper.png',
                    bbox_inches='tight',
                    transparent=True)
        
        if kwargs.get("verbose", False):
            print(f"\nWriting reports...\n")
            fID.write(f"\nWriting reports...\n")

        self.report = classification_report(actual,pred, output_dict=True)

        print(self.report)
        fID.write(json.dumps(self.report))

        fID.close()

