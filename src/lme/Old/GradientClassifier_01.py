import os
import sklearn
import pandas as pd

from sklearn.model_selection import train_test_split

DataPath = os.environ.get("LME_TRAINING_DATA")

dfx = pd.read_excel(f"{DataPath}\\X.xlsx")
dfy = pd.read_excel(f"{DataPath}\\Y.xlsx")
dfmnc = pd.read_excel(f"{DataPath}\\AvValNC3.xlsx")
dfm = f"{DataPath}\\SectionsNC3.xlsx"
df0 = pd.read_excel(dfm, "Zero")
df1 = pd.read_excel(dfm, "One")
df2 = pd.read_excel(dfm, "Two")
df3 = pd.read_excel(dfm, "Three")

df1b = df1.iloc[0:200, :]

X = pd.DataFrame.to_numpy(dfx)
y = pd.DataFrame.to_numpy(dfy)
Xnc = pd.DataFrame.to_numpy(dfmnc.loc[:, dfmnc.columns != "Mechanisms"])
ync = pd.DataFrame.to_numpy(dfmnc.loc[dfmnc.columns == "Mechanisms"])
X0 = pd.DataFrame.to_numpy(df0.loc[:, df0.columns != "Mechanisms"])
y0 = pd.DataFrame.to_numpy(df0["Mechanisms"])
X1 = pd.DataFrame.to_numpy(df1b.loc[:, df1b.columns != "Mechanisms"])
y1 = pd.DataFrame.to_numpy(df1b["Mechanisms"])
X2 = pd.DataFrame.to_numpy(df2.loc[:, df2.columns != "Mechanisms"])
y2 = pd.DataFrame.to_numpy(df2["Mechanisms"])
X3 = pd.DataFrame.to_numpy(df3.loc[:, df3.columns != "Mechanisms"])
y3 = pd.DataFrame.to_numpy(df3["Mechanisms"])

X_train0, X_test0, y_train0, y_test0 = train_test_split(
    X0, y0.ravel(), test_size=0.3, random_state=42
)

X_train1, X_test1, y_train1, y_test1 = train_test_split(
    X1, y1.ravel(), test_size=0.3, random_state=42
)

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X2, y2.ravel(), test_size=0.3, random_state=42
)

X_train3, X_test3, y_train3, y_test3 = train_test_split(
    X3, y3.ravel(), test_size=0.3, random_state=42
)

# X_train, X_test, y_train, y_test = train_test_split(X, y.ravel(), test_size=0.3, random_state=42)

Xnc_train, Xnc_test, ync_train, ync_test = train_test_split(
    Xnc, ync.ravel(), test_size=0.3, random_state=42
)
