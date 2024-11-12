import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from matplotlib import cm

from typing import List


def plot_linear_classification_coefs(
    pipe: Pipeline, feat_cols: List[str], nb_to_show: int = 25
):

    assert nb_to_show > 0
    model = (
        pipe[-1].best_estimator_ if hasattr(pipe[-1], "best_estimator_") else pipe[-1]
    )
    importances = model.coef_
    for idx, label in enumerate(pipe.classes_):
        sort_idx = np.argsort(np.abs(importances[idx]))[::-1]
        plt.figure(figsize=(10, 8))
        x = np.array(feat_cols)[sort_idx[:nb_to_show]][::-1]
        y = importances[idx][sort_idx[:nb_to_show]][::-1]
        color = ["lightgreen" if v > 0 else "salmon" for v in y]
        plt.barh(x, np.abs(y), color=color)
        plt.title(label)
        plt.show()


def plot_learning_curve(
    estimator,
    title: str,
    X,
    y,
    ylim=None,
    cv: int = None,
    n_jobs: int = None,
    train_sizes=np.linspace(0.1, 1.0, 5),
    scoring=None,
    **kwargs,
):

    fig = plt.figure(figsize=(20, 10))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        scoring=scoring,
        **kwargs,
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid(True)

    plt.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    plt.fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    plt.plot(train_sizes, train_scores_mean, "o-", color="r", label="Training score")
    plt.plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )

    plt.legend(loc="best")
    return fig


def plot_confusion_matrix(
    y_true,
    y_pred,
    classes=None,
    normalize=False,
    title=None,
    cmap="Blues",  # plt.cm.Blues,
    ax=None,
):
    """Plot the confusion matrix.

    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = "Normalized confusion matrix"
        else:
            title = "Confusion matrix, without normalization"

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    # Only use the labels that appear in the data
    if classes is None: classes = unique_labels(y_true, y_pred)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    if ax is None:
        fig, ax = plt.subplots()
    ax.grid(False)
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        # ... and label them with the respective list entries
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    return ax
