import pandas as pd
import plotly.express as px
import streamlit as st

from src.utils import get_project_root, load_css


def render_config():
    """
    Sets page config and loads custom CSS.
    Configures the application layout and theme.
    """
    page_title = "Iris Dynamics"
    page_icon = ":iris:"
    st.set_page_config(page_title=page_title, page_icon=page_icon, layout="wide")

    root = get_project_root()
    css_file = root / "styles" / "main.css"
    load_css(css_file)


def render_header():
    """
    Renders the premium header section of the app.
    Uses custom CSS classes defined in main.css.
    """
    st.markdown(
        """
        <div class='header-container'>
            <h1 class='main-title'>Iris Dynamics</h1>
            <p class='sub-title'>PRECISION BOTANICAL PREDICTION</p>
        </div>
    """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> pd.DataFrame:
    """
    Renders the sidebar for user input parameters.

    Returns:
        pd.DataFrame: A single-row DataFrame containing the user-selected
        measurements.
    """
    with st.sidebar:
        st.markdown(
            "<h2 style='font-family:Outfit; margin-bottom:0;'>Bio-Signals</h2>",
            unsafe_allow_html=True,
        )
        st.caption("Configuring botanical dimensions.")
        st.markdown("---")

        sepal_length = st.slider("Sepal Length (cm)", 4.3, 7.9, 5.4)
        sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.4, 3.4)
        petal_length = st.slider("Petal Length (cm)", 1.0, 6.9, 1.3)
        petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

        st.markdown("---")
        st.caption("Intelligence: Iris-ML V2.0")
        st.caption("Aesthetic: Rose / Sky / Glass")

        with st.expander("📖 Botanical Guide"):
            st.markdown(
                """
            **Sepal**: The outer parts of the flower (often green and leaf-like)
            that enclose a developing bud.

            **Petal**: The colourful parts of the flower that attract insects.

            Measurements are used by the model to differentiate between
            *Setosa*, *Versicolor*, and *Virginica* species.
            """
            )

    data = {
        "Sepal Length": sepal_length,
        "Sepal Width": sepal_width,
        "Petal Length": petal_length,
        "Petal Width": petal_width,
    }
    return pd.DataFrame(data, index=[0])


def render_prediction_result(prediction, prediction_proba, target_names):
    """
    Renders the prediction results using modern UI elements.

    Args:
        prediction (int): The predicted class index.
        prediction_proba (np.ndarray): Array of probabilities for each class.
        target_names (np.ndarray): Array of class names.
    """
    t1, t2 = st.tabs(["🎯 Classification Snapshot", "📊 Confidence Metrics"])

    with t1:
        st.markdown("### 🎯 Model Accuracy")
        col1, col2 = st.columns([1, 1])

        # Scalar extraction for robust formatting
        pred_index = int(prediction[0])
        species_name = str(target_names[pred_index])
        conf_value = float(prediction_proba[0][pred_index]) * 100

        with col1:
            st.metric("Predicted Species", species_name)
            st.metric("Confidence Level", f"{conf_value:.1f}%")

        with col2:
            root = get_project_root()
            # Dynamic image selection based on predicted species
            species_image_name = f"{species_name.lower()}.png"
            image_path = root / "assets" / "images" / species_image_name

            # Fallback to generic Iris image if specific one is missing
            if not image_path.exists():
                image_path = root / "assets" / "images" / "Iris.jpg"

            if image_path.exists():
                st.image(str(image_path), width="stretch")

        st.markdown("---")
        st.markdown(
            """
        **Intelligence Snapshot**:
        Based on the provided biometrical data, the machine learning engine
        has identified this specimen as belonging to the species highlighted
        above.
        """
        )

    with t2:
        st.markdown("### 📊 Species Probability Distribution")

        # Create a DataFrame for Plotly
        prob_df = pd.DataFrame(
            {"Species": list(target_names), "Probability": prediction_proba[0]}
        )

        fig = px.bar(
            prob_df,
            x="Species",
            y="Probability",
            color="Probability",
            color_continuous_scale="Sunsetdark",
            template="plotly_dark",
        )

        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=20, b=20, l=20, r=20),
            font={"family": "Outfit"},
        )

        st.plotly_chart(fig, width="stretch")

        st.markdown(
            """
        **Metric Explained**:
        Prediction probability represents the likelihood of the model's
        classification being correct, ranging from 0 to 1.
        """
        )
