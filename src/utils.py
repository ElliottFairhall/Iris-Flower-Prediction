from pathlib import Path

import streamlit as st


def get_project_root() -> Path:
    """
    Returns the root directory of the project.
    Uses the current file location as a reference.
    """
    return Path(__file__).parent.parent


def load_css(file_path: Path):
    """
    Loads a CSS file and injects it into the Streamlit app.

    Args:
        file_path (Path): Path to the .css file.
    """
    with open(file_path, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
