import sys

import numpy as np
import pandas as pd

from src.model import IrisModel


def main():
    print("Starting verification...")
    try:
        # Initialize model
        model = IrisModel()
        print("Model initialized.")

        # Train model
        model.train()
        print("Model trained.")

        # Create dummy input
        data = {
            "Sepal Length": [5.4],
            "Sepal Width": [3.4],
            "Petal Length": [1.3],
            "Petal Width": [0.2],
        }
        input_df = pd.DataFrame(data)

        # Predict
        prediction = model.predict(input_df)
        print(f"Prediction: {prediction}")

        # Predict Proba
        proba = model.predict_proba(input_df)
        print(f"Probabilities: {proba}")

        # Validate types
        if not isinstance(prediction, np.ndarray):
            raise TypeError("Prediction should be numpy array")

        print("Verification SUCCESS.")
        sys.exit(0)
    except Exception as e:
        print(f"Verification FAILED: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
