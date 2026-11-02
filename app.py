"""Iris species classifier: entry point."""

import streamlit as st

from src.model import FOLDS, get_accuracy, get_model
from src.ui import (
    render_config,
    render_context,
    render_footer,
    render_intro,
    render_probabilities,
    render_result,
    render_sidebar,
    render_topbar,
)


def main() -> None:
    """Render the page. Predictions update live as the sliders move."""
    render_config()
    render_topbar()

    model = get_model()
    reference = model.dataset_frame()
    render_intro(get_accuracy(), len(reference), FOLDS)

    flower = render_sidebar()
    proba = model.predict_proba(flower)
    names = model.get_target_names()
    best = int(proba[0].argmax())

    render_result(str(names[best]), float(proba[0][best]))

    chart_left, chart_right = st.columns(2, gap="large")
    with chart_left:
        st.caption("Probability by species")
        render_probabilities(names, proba)
    with chart_right:
        st.caption("Where your flower sits among 150 measured irises")
        render_context(reference, flower)

    render_footer()


if __name__ == "__main__":
    main()
