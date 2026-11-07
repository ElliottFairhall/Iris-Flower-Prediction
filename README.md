# Iris species classifier

> Architecture and design rationale: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://elliottfairhall-iris-flower-prediction-main-e0ikd7.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Move four sliders to describe an iris and a random forest names the most likely species. Built with Streamlit,
scikit-learn and Plotly.

## What it does

- Live prediction as the sliders move, with the model's confidence
- Probability for each species
- A scatter plot placing your flower among the 150 reference flowers
- Cross-validated accuracy measured at start-up and quoted on the page

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Needs Python 3.11 or newer. The theme follows your system's light or dark setting and is defined in
`.streamlit/config.toml`.

## Test it

```bash
pip install -e ".[dev]"
pytest
```

The tests cover the analysis code, the page itself (run headlessly with Streamlit's `AppTest`) and the dependency pins:
`tests/test_pillow_security_*.py` fails if Pillow is capped below 13 or Python 3.10 is allowed, because older Pillow
releases carry known security advisories.

## Notes

The model is trained once and cached, with a fixed random seed so the same measurements always give the same answer.
`verify.py` is a quick command-line check of the model.

Part of [Data Flakes](https://www.data-flakes.dev/).
