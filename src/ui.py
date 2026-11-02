"""Page layout for the Iris species classifier."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.utils import get_project_root, load_css

SITE_URL = "https://www.data-flakes.dev"

# Data Flakes palette: one colour per species, rose reserved for the user's flower.
SPECIES_COLOURS = {
    "setosa": "#5E8FBA",
    "versicolor": "#C06C84",
    "virginica": "#F8B195",
}
USER_COLOUR = "#F67280"
CHART_FONT = {"family": "Inter, sans-serif", "size": 13}


def render_config() -> None:
    """Set the page configuration and load the stylesheet."""
    root = get_project_root()
    st.set_page_config(
        page_title="Iris species classifier",
        page_icon=str(root / "assets" / "images" / "favicon.png"),
        layout="wide",
    )
    load_css(root / "styles" / "main.css")


def render_topbar() -> None:
    """Show a plain breadcrumb back to the Data Flakes site."""
    st.markdown(
        f"""
        <nav class="site-bar">
            <a href="{SITE_URL}/" target="_blank">Data Flakes</a>
            <span>/</span>
            <a href="{SITE_URL}/projects/" target="_blank">Projects</a>
            <span>/</span>
            <span>Iris classifier</span>
        </nav>
        """,
        unsafe_allow_html=True,
    )


def render_intro(accuracy: float, sample_size: int, folds: int) -> None:
    """Explain what the app does, using the model's measured accuracy.

    Args:
        accuracy: Cross-validated accuracy between 0 and 1.
        sample_size: Number of reference flowers the model learned from.
        folds: Number of cross-validation folds behind the accuracy figure.
    """
    st.title("Iris species classifier")
    st.write(
        f"Move the sliders to describe a flower. A random forest trained on "
        f"{sample_size} measured irises names the most likely species. "
        f"On flowers it hadn't seen, it was right {accuracy:.1%} of the time "
        f"({folds}-fold cross-validation)."
    )


def render_sidebar() -> pd.DataFrame:
    """Render the measurement sliders.

    Returns:
        A one-row DataFrame with the four measurements in centimetres.
    """
    with st.sidebar:
        st.header("Measurements")
        st.caption("The prediction updates as you move a slider.")
        sepal_length = st.slider("Sepal length (cm)", 4.3, 7.9, 5.4)
        sepal_width = st.slider("Sepal width (cm)", 2.0, 4.4, 3.4)
        petal_length = st.slider("Petal length (cm)", 1.0, 6.9, 1.3)
        petal_width = st.slider("Petal width (cm)", 0.1, 2.5, 0.2)
        with st.expander("Sepal or petal?"):
            st.write(
                "The petal is the coloured part of the flower. The sepal is the "
                "green leaf-like part that protected the bud. Both are measured "
                "at their longest and widest points."
            )
    return pd.DataFrame(
        {
            "Sepal Length": sepal_length,
            "Sepal Width": sepal_width,
            "Petal Length": petal_length,
            "Petal Width": petal_width,
        },
        index=[0],
    )


def render_result(species: str, confidence: float) -> None:
    """Show the predicted species, its confidence and a photo.

    Args:
        species: Lower-case species name, for example ``"setosa"``.
        confidence: Probability of the predicted species between 0 and 1.
    """
    image_dir = get_project_root() / "assets" / "images"
    text_col, image_col = st.columns([3, 2], gap="large")
    with text_col:
        species_col, confidence_col = st.columns(2)
        species_col.metric("Predicted species", f"Iris {species}")
        confidence_col.metric("Confidence", f"{confidence:.0%}")
    with image_col:
        photo = image_dir / f"{species}.png"
        if photo.exists():
            st.image(str(photo), caption=f"Iris {species}", width=240)


def _style(fig: go.Figure, height: int) -> go.Figure:
    """Apply the shared chart layout: transparent, light grid, Inter text."""
    fig.update_layout(
        height=height,
        margin={"t": 8, "b": 8, "l": 8, "r": 48},
        font=CHART_FONT,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    fig.update_xaxes(gridcolor="rgba(128,128,128,0.18)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.18)", zeroline=False)
    return fig


def render_probabilities(names, proba) -> None:
    """Draw one horizontal bar per species showing the model's probability.

    Args:
        names: Species names in class order.
        proba: One row of class probabilities.
    """
    species = [str(name) for name in names]
    values = [float(p) * 100 for p in proba[0]]
    fig = go.Figure(
        go.Bar(
            x=values,
            y=species,
            orientation="h",
            marker_color=[SPECIES_COLOURS[s] for s in species],
            text=[f"{v:.0f}%" for v in values],
            textposition="outside",
            cliponaxis=False,
        )
    )
    _style(fig, 190)
    fig.update_xaxes(range=[0, 100], ticksuffix="%")
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, width="stretch")


def render_context(reference: pd.DataFrame, flower: pd.DataFrame) -> None:
    """Plot the user's flower among the reference flowers.

    Args:
        reference: The 150 reference flowers with a ``species`` column.
        flower: The user's one-row measurements.
    """
    fig = go.Figure()
    for species, colour in SPECIES_COLOURS.items():
        subset = reference[reference["species"] == species]
        fig.add_scatter(
            x=subset["petal_length"],
            y=subset["petal_width"],
            mode="markers",
            name=species,
            marker={"color": colour, "size": 8, "opacity": 0.85},
        )
    fig.add_scatter(
        x=flower["Petal Length"],
        y=flower["Petal Width"],
        mode="markers",
        name="your flower",
        marker={
            "color": USER_COLOUR,
            "size": 15,
            "symbol": "diamond",
            "line": {"width": 2, "color": "white"},
        },
    )
    _style(fig, 330)
    fig.update_layout(showlegend=True, legend={"orientation": "h", "y": 1.12})
    fig.update_xaxes(title="Petal length (cm)")
    fig.update_yaxes(title="Petal width (cm)")
    st.plotly_chart(fig, width="stretch")


def render_footer() -> None:
    """Show a one-line footer linking back to the Data Flakes site."""
    st.markdown(
        f"""
        <footer class="site-foot">
            Built by Elliott Fairhall. More projects and write-ups at
            <a href="{SITE_URL}/" target="_blank">data-flakes.dev</a>.
        </footer>
        """,
        unsafe_allow_html=True,
    )
