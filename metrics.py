"""
This module contains useful functions and classes for computing quality evaluation metrics.
"""

from typing import Tuple

import numpy as np
import pandas as pd
import sklearn.metrics as skmetrics
from matplotlib import pyplot as plt


class BinaryPerformance:
    """
    Evaluate binary classification model performance and visualize results.
    """

    def __init__(self, y_true, y_score=None, threshold=0.5):  # pylint: disable=too-many-arguments
        """
        Initialize the BinaryPerformance instance.

        Parameters
        ----------
        y_score : np.ndarray
            Predicted scores from the model.
        y_true : np.ndarray
            True labels.
        threshold : float, optional
            Classification threshold. Defaults to 0.5.
        """
        self.given_y_score = y_score
        self.y_true = y_true
        self.threshold = threshold
        self.predictions = self.get_binary_predictions()

    def get_binary_predictions(self) -> pd.DataFrame:
        """
        Generate binary predictions and scores for a given model on a given dataset.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing binary predictions and scores.
        """
        if self.given_y_score is not None:
            y_score = self.given_y_score
            y_pred = (self.given_y_score >= self.threshold).astype(int)
        else:
            raise (ValueError("You need to provide y_score so I can generate binary predictions."))

        predictions = pd.DataFrame({"y_pred": y_pred, "y_score": y_score})

        return predictions

    def evaluate_binary_performance(
        self,
        label: str = "",
        add_prefix: bool = True,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Evaluate binary classification performance and return metrics.

        Parameters
        ----------
        label : str, optional
            A label for the performance report.
        add_prefix : bool, optional
            Whether to add the label as a prefix to metric keys.
        verbose : bool, optional
            Whether to print the performance report.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing evaluation metrics.
        """
        ap_score = skmetrics.average_precision_score(self.y_true, self.predictions["y_score"])
        aucroc = skmetrics.roc_auc_score(self.y_true, self.predictions["y_score"])
        recall = skmetrics.recall_score(self.y_true, self.predictions["y_pred"])
        precision = skmetrics.precision_score(self.y_true, self.predictions["y_pred"])
        f_score = skmetrics.f1_score(self.y_true, self.predictions["y_pred"])
        acc = skmetrics.accuracy_score(self.y_true, self.predictions["y_pred"])

        # EER
        fpr, tpr, threshold = skmetrics.roc_curve(
            self.y_true, self.predictions["y_score"], pos_label=1
        )
        fnr = 1 - tpr
        abs_diffs = np.abs(fpr - fnr)
        min_index = np.argmin(abs_diffs)
        eer = np.mean((fpr[min_index], fnr[min_index]))
        eer_threshold = threshold[min_index]

        prefix = label + "_" if add_prefix and len(label) > 0 else ""

        if verbose:
            print(
                f"""Performance metrics - {label}:
                Total samples: {len(self.y_true)}
                Positive samples: {np.sum(self.y_true)}
                Positive proportion: {np.mean(self.y_true)}
                ROC AUC: {aucroc:.3f}
                Average Precision (PR AUC): {ap_score:.3f}
                EER: {eer:.3f} (with threshold: {eer_threshold})

                With classification threshold = {self.threshold}:
                Precision: {precision:.3f};
                Recall: {recall:.3f};
                F-score: {f_score:.3f};
                Accuracy: {acc:.3f}
                """
            )

        results_dict = {
            f"{prefix}aucpr": [ap_score],
            f"{prefix}aucroc": [aucroc],
            f"{prefix}precision": [precision],
            f"{prefix}recall": [recall],
            f"{prefix}f_score": [f_score],
            f"{prefix}accuracy": [acc],
            f"{prefix}eer": [eer],
        }

        results_dataframe = pd.DataFrame(results_dict)

        return results_dataframe

    def set_threshold(self, new_threshold: float):
        """
        Set the threshold value for the object.

        Parameters
        ----------
        new_threshold : float
            The new threshold value to be set.

        Raises
        ------
        ValueError
            If the new threshold is less than 0 or greater than 1.

        Notes
        -----
        This method updates the object's threshold value.

        """
        if new_threshold < 0 or new_threshold > 1:
            raise ValueError("The threshold value must be between 0 and 1")

        self.threshold = new_threshold
        self.predictions = self.get_binary_predictions()

    def get_best_threshold(
        self,
        step: float = 0.01,
        metric: str = "f_score",
        aux_condition: dict = None,
        verbose: bool = True,
    ) -> float:
        """
        Get the best threshold value based on a given metric.

        Parameters
        ----------
        step : float, optional
            Step size for threshold iteration (default is 0.01).

        metric : str, optional
            The evaluation metric to optimize for.
            Supported metrics: "precision", "recall", "f_score", "accuracy"
            (default is "f_score").

        aux_condition : dict, optional
            A dictionary containing an auxiliary metric and a floor value.
            If specified, the best threshold will be the one that maximizes the
            specified metric, while the auxiliary metric is greater than or equal
            to the specified floor value.
            Supported metrics: "precision", "recall", "f_score", "accuracy"
            (default is None).

        verbose : bool, optional
            If True, print the best metric and threshold values (default is True).

        Returns
        -------
        float
            The best threshold value based on the specified metric.

        Notes
        -----
        This method iterates over threshold values and returns the one
        that maximizes the specified evaluation metric.
        """
        assert step < 1
        assert metric in ["precision", "recall", "f_score", "accuracy"]
        if aux_condition is not None:
            assert aux_condition["metric"] in ["precision", "recall", "f_score", "accuracy"]
            assert aux_condition["floor"] is not None
            aux_metric = aux_condition["metric"]
            aux_metric_floor = aux_condition["floor"]

        current_threshold = self.threshold
        best_metric = 0
        best_threshold = 0
        for i in range(int(1 / step) + 1):
            candidate_threshold = i * step

            self.set_threshold(candidate_threshold)
            threshold_results = self.evaluate_binary_performance(verbose=False)

            candidate_metric = threshold_results[metric].values[0]
            if best_metric < candidate_metric:
                if aux_condition is not None:
                    if threshold_results[aux_metric].values[0] >= aux_metric_floor:
                        best_metric = candidate_metric
                        best_threshold = candidate_threshold
                else:
                    best_metric = candidate_metric
                    best_threshold = candidate_threshold

        if verbose:
            print(f"best_{metric}: {best_metric}")
            print(f"best_threshold: {best_threshold}")

        self.set_threshold(current_threshold)

        return best_threshold

    def plot_confusion_matrix(self, label: str = None, figsize: Tuple = (20, 5)):
        """
        Plot confusion matrices with different normalization criteria.

        Parameters
        ----------
        label : str, optional
            A label for the confusion matrix.
        figsize : Tuple, optional
            Figure size.
        """
        fig = plt.figure(figsize=figsize)
        plt.suptitle(
            f"Confusion Matrices - {label}, threshold = {self.threshold:.3f}",
            fontsize=24,
        )
        colorbar = False
        cmap = "YlGnBu"

        # confusion matrix with totals
        axes = plt.subplot(1, 4, 1)
        plt.title("Totals")
        skmetrics.ConfusionMatrixDisplay.from_predictions(
            self.y_true, self.predictions["y_pred"], colorbar=colorbar, cmap=cmap, ax=axes
        )

        # confusion matrix normalized by total samples
        axes = plt.subplot(1, 4, 2)
        plt.title("Normalized by total")
        skmetrics.ConfusionMatrixDisplay.from_predictions(
            self.y_true,
            self.predictions["y_pred"],
            normalize="all",
            colorbar=colorbar,
            cmap=cmap,
            ax=axes,
        )

        # confusion matrix normalized by ground truth
        axes = plt.subplot(1, 4, 3)
        plt.title("Normalized by ground truth (rows add to 1)")
        skmetrics.ConfusionMatrixDisplay.from_predictions(
            self.y_true,
            self.predictions["y_pred"],
            normalize="true",
            colorbar=colorbar,
            cmap=cmap,
            ax=axes,
        )

        # confusion matrix normalized by prediction
        axes = plt.subplot(1, 4, 4)
        plt.title("Normalized by prediction (columns add to 1)")
        skmetrics.ConfusionMatrixDisplay.from_predictions(
            self.y_true,
            self.predictions["y_pred"],
            normalize="pred",
            colorbar=colorbar,
            cmap=cmap,
            ax=axes,
        )

        return fig

    def plot_classification_performance(
        self, label: str = None, figsize: Tuple = (14, 6), add_annotation_values: bool = True
    ) -> plt.figure:
        """
        Plot classification performance curves.

        Parameters
        ----------
        label : str, optional
            A label for the plots.
        figsize : Tuple, optional
            Figure size, specified as (width, height).
        add_annotation_values : bool, optional
            Whether to add annotation values to the curves.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The matplotlib Figure containing the classification performance plots.
        """
        fig = plt.figure(figsize=figsize)
        plt.suptitle(f"Classification Performance Curves ({label})", fontsize=24)

        plt.subplot(1, 2, 1)
        plt.title("ROC curve")
        fpr, tpr, _ = skmetrics.roc_curve(self.y_true, self.predictions["y_score"])
        aucroc_score = skmetrics.roc_auc_score(self.y_true, self.predictions["y_score"])
        plt.plot(fpr, tpr, marker=".", label=f"Classifier (AUC = {aucroc_score})")
        plt.legend()
        if add_annotation_values:
            for x_value, y_value in zip(fpr, tpr):
                label = f"{y_value:.2f}"
                plt.annotate(
                    label,  # this is the text
                    (x_value, y_value),  # these are the coordinates to position the label
                    textcoords="offset points",  # how to position the text
                    xytext=(0, 10),  # distance from text to points (x,y)
                    ha="center",
                )  # horizontal alignment can be left, right or center

        plt.subplot(1, 2, 2)
        plt.title("Precision/Recall curve")
        precision, recall, _ = skmetrics.precision_recall_curve(
            self.y_true, self.predictions["y_score"]
        )
        ap_score = skmetrics.average_precision_score(self.y_true, self.predictions["y_score"])
        plt.plot(recall, precision, marker=".", label=f"Classifier (AP = {ap_score})")
        plt.legend()
        if add_annotation_values:
            for x_value, y_value in zip(precision, recall):
                label = f"{y_value:.2f}"
                plt.annotate(
                    label,  # this is the text
                    (x_value, y_value),  # these are the coordinates to position the label
                    textcoords="offset points",  # how to position the text
                    xytext=(0, 10),  # distance from text to points (x,y)
                    ha="center",
                )  # horizontal alignment can be left, right or center

        return fig
