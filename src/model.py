"""Random forest classifier for the Iris dataset."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.utils import Bunch

SEED = 42
FOLDS = 5


class IrisModel:
    """A RandomForestClassifier wrapper specific to the Iris dataset."""

    def __init__(self) -> None:
        """Create an untrained model with a fixed seed so results repeat."""
        self.clf = RandomForestClassifier(random_state=SEED)
        self.iris: Bunch | None = None
        self.is_trained = False

    @staticmethod
    @st.cache_data
    def _load_data():
        """Load the Iris dataset from scikit-learn (cached across reruns)."""
        return datasets.load_iris()

    def train(self) -> None:
        """Fit the classifier on the full dataset. Does nothing if already fitted."""
        if self.is_trained:
            return
        self.iris = self._load_data()
        data: Bunch = self.iris
        self.clf.fit(data.data, data.target)
        self.is_trained = True

    def predict(self, input_data: pd.DataFrame):
        """Predict the class index for a one-row frame of measurements.

        Args:
            input_data: One row of sepal and petal measurements, in cm.

        Returns:
            Array holding the predicted class index.
        """
        self.train()
        return self.clf.predict(input_data.values)

    def predict_proba(self, input_data: pd.DataFrame):
        """Return the probability of each class for a one-row frame.

        Args:
            input_data: One row of sepal and petal measurements, in cm.

        Returns:
            Array of shape (1, 3) with one probability per species.
        """
        self.train()
        return self.clf.predict_proba(input_data.values)

    def get_target_names(self):
        """Return the species names, for example ``['setosa', ...]``."""
        data: Bunch = self.iris if self.iris is not None else self._load_data()
        self.iris = data
        return data.target_names

    def cv_accuracy(self) -> float:
        """Cross-validated accuracy of a fresh forest on the full dataset.

        Returns:
            Mean accuracy across ``FOLDS`` folds, between 0 and 1.
        """
        data = self._load_data()
        fresh = RandomForestClassifier(random_state=SEED)
        return float(cross_val_score(fresh, data.data, data.target, cv=FOLDS).mean())

    def dataset_frame(self) -> pd.DataFrame:
        """Return the 150 reference flowers with a species name column."""
        data = self._load_data()
        frame = pd.DataFrame(
            data.data,
            columns=["sepal_length", "sepal_width", "petal_length", "petal_width"],
        )
        frame["species"] = [str(data.target_names[i]) for i in data.target]
        return frame


@st.cache_resource
def get_model() -> IrisModel:
    """Return one trained model shared across reruns and sessions."""
    model = IrisModel()
    model.train()
    return model


@st.cache_data
def get_accuracy() -> float:
    """Return the cached cross-validated accuracy of the classifier."""
    return IrisModel().cv_accuracy()
