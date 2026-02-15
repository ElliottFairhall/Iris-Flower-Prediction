import streamlit as st

from src.model import IrisModel
from src.ui import (
    render_config,
    render_header,
    render_prediction_result,
    render_sidebar,
)


def main():
    """
    Main entry point for the Iris Dynamics application.
    Orchestrates the UI rendering and model inference.
    """
    # 1. Page Configuration and CSS
    render_config()

    # 2. Premium Header
    render_header()

    # 3. Sidebar Input Workspace
    input_df = render_sidebar()

    # 4. Tabs for Project Clarity
    tab1, tab2 = st.tabs(["Project Overview", "Analysis Workspace"])

    with tab1:
        st.markdown(
            """
        ### Project Outline
        This project utilizes machine learning to classify Iris flowers into one of three species
        (*Setosa*, *Versicolor*, or *Virginica*) based on morphological measurements.

        ### Methodology
        - **Model:** Random Forest Classifier trained on the classic Iris dataset.
        - **Interface:** Real-time parameter adjustment via sidebar sliders.
        - **Output:** Instant classification with probability confidence metrics.
        """
        )

    with tab2:
        st.markdown("### 📝 Analysis Workspace")
        st.info(
            "Adjust the Bio-Signals in the sidebar and click "
            "'Process Intelligence' to generate botanical insights."
        )

        if st.button("Process Intelligence"):
            with st.spinner("Analyzing botanical signatures..."):
                model = IrisModel()
                model.train()  # Ensures model is loaded and trained

                prediction = model.predict(input_df)
                prediction_proba = model.predict_proba(input_df)
                target_names = model.get_target_names()

                # Render Results
                render_prediction_result(prediction, prediction_proba, target_names)
                st.success("Intelligence processing complete.")
        else:
            st.markdown("---")
            st.caption("Waiting for signal input...")


if __name__ == "__main__":
    main()
