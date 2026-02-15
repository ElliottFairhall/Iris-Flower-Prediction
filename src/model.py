import pandas as pd
import streamlit as st
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier


class IrisModel:
    """
    A wrapper class for the Scikit-Learn RandomForestClassifier
    specifically for the Iris dataset.
    """

    def __init__(self):
        """Initializes the model and state flags."""
        self.clf = RandomForestClassifier()
        self.iris = None
        self.is_trained = False

    @staticmethod
    @st.cache_data
    def _load_data():
        """
        Loads the Iris dataset from sklearn datasets.
        Cached to optimize performance across reruns.
        """
        return datasets.load_iris()

    def train(self):
        """
        Trains the model if it hasn't been trained yet.
        Loads data, fits the classifier, and sets the trained flag.
        """
        self.iris = self._load_data()
        feature_data = self.iris.data
        target_labels = self.iris.target
        self.clf.fit(feature_data, target_labels)
        self.is_trained = True

    def predict(self, input_data: pd.DataFrame):
        """
        Predicts the class label for the given input data.

        Args:
            input_data (pd.DataFrame): 1-row DataFrame of measurements.

        Returns:
            np.ndarray: Predicted class index.
        """
        if not self.is_trained:
            self.train()
        return self.clf.predict(input_data.values)

    def predict_proba(self, input_data: pd.DataFrame):
        """
        Calculates the probability distribution for each class.

        Args:
            input_data (pd.DataFrame): 1-row DataFrame of measurements.

        Returns:
            np.ndarray: Array of probabilities.
        """
        if not self.is_trained:
            self.train()
        return self.clf.predict_proba(input_data.values)

    def get_target_names(self):
        """
        Retrieves the species names from the dataset.

        Returns:
            np.ndarray: Array of class names (e.g., ['setosa', ...]).
        """
        if self.iris is None:
            self.iris = self._load_data()
        return self.iris.target_names
