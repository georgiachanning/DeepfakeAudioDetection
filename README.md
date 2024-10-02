# Deepfake Audio Detection

## Overview
This repository provides tools and models for detecting deepfake audio using machine learning techniques. The project focuses on enhancing detection robustness through explainability and feature engineering.

## Contents
- **custom_data.py**: Script to load and preprocess custom datasets.
- **features.py**: Feature extraction methods for audio analysis.
- **transformer.py**: Implementation of a transformer-based model for classification.
- **xgboost.py**: XGBoost model for baseline comparison.
- **Jupyter Notebooks**: Analysis, feature importance, and explainability.

## Usage
1. Preprocess the data using `custom_data.py`.
2. Extract features using `features.py`.
3. Train and evaluate models using `transformer.py` and `xgboost.py`.
4. Analyze results and model explainability using the provided notebooks.
5. Explainability tooling found in `.ipynb` files.

## Contributing
Contributions are welcome! Please submit issues or pull requests for improvements.

## License
[MIT License](LICENSE)