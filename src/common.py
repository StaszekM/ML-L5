from typing import List
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def create_base_classifier():
    return DecisionTreeClassifier(max_depth=1)


def get_score(X, y, clf=create_base_classifier(), cross_val=False):
    if cross_val:
        return cross_val_score(
            clf, X, y, cv=StratifiedKFold(shuffle=True, random_state=42)
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    clf.fit(X_train, y_train)

    return np.array(f1_score(y_test, clf.predict(X_test), average="weighted"))


def plot_results(
    data: pd.DataFrame,
    x: str,
    y: List[str],
    suptitle: str,
    size_inches=(14, 5),
    plot_fn=sns.lineplot,
):
    fig, ax = plt.subplots(ncols=len(y))
    for index, ycol in enumerate(y):
        plot_fn(data=data, ax=ax[index], x=x, y=ycol)
    fig.set_size_inches(size_inches)
    fig.suptitle(suptitle)
    fig.tight_layout()
