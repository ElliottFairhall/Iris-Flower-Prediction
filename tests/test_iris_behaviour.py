"""Behaviour tests for the Iris classifier: the model and the live UI."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.model import IrisModel

ROOT = Path(__file__).resolve().parent.parent
SETOSA = pd.DataFrame({"a": [5.1], "b": [3.5], "c": [1.4], "d": [0.2]})
VERSICOLOR = pd.DataFrame({"a": [6.0], "b": [2.9], "c": [4.5], "d": [1.5]})
VIRGINICA = pd.DataFrame({"a": [6.8], "b": [3.0], "c": [5.8], "d": [2.1]})


@pytest.fixture(scope="module")
def model() -> IrisModel:
    """A trained model shared by the model tests."""
    trained = IrisModel()
    trained.train()
    return trained


@pytest.mark.parametrize(
    ("flower", "expected"),
    [(SETOSA, 0), (VERSICOLOR, 1), (VIRGINICA, 2)],
    ids=["setosa", "versicolor", "virginica"],
)
def test_model_names_the_right_species(
    model: IrisModel, flower: pd.DataFrame, expected: int
) -> None:
    """Typical flowers of each species are classified correctly."""
    assert int(model.predict(flower)[0]) == expected


def test_probabilities_sum_to_one(model: IrisModel) -> None:
    """Class probabilities form a distribution."""
    proba = model.predict_proba(VERSICOLOR)
    assert proba.shape == (1, 3)
    assert np.isclose(proba.sum(), 1.0)


def test_predictions_repeat_between_runs() -> None:
    """A fixed seed makes two separately trained models agree exactly."""
    first, second = IrisModel(), IrisModel()
    assert np.allclose(
        first.predict_proba(VERSICOLOR), second.predict_proba(VERSICOLOR)
    )


def test_cross_validated_accuracy_is_high(model: IrisModel) -> None:
    """The accuracy quoted on the page is a real, high figure."""
    assert 0.9 <= model.cv_accuracy() <= 1.0


def test_dataset_frame_has_150_labelled_flowers(model: IrisModel) -> None:
    """The reference chart is drawn from all 150 flowers."""
    frame = model.dataset_frame()
    assert len(frame) == 150
    assert set(frame["species"]) == {"setosa", "versicolor", "virginica"}


def _metric_values(at: AppTest) -> dict[str, str]:
    return {metric.label: metric.value for metric in at.metric}


def test_default_sliders_predict_setosa() -> None:
    """The opening state shows Iris setosa with a confidence figure."""
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run()
    values = _metric_values(at)
    assert values["Predicted species"] == "Iris setosa"
    assert values["Confidence"].endswith("%")


def test_moving_the_sliders_changes_the_prediction() -> None:
    """The prediction updates live, with no button to press."""
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run()
    for slider, value in zip(at.slider, (6.0, 2.9, 4.5, 1.5), strict=True):
        slider.set_value(value)
    at.run()
    assert not at.exception
    assert _metric_values(at)["Predicted species"] == "Iris versicolor"
