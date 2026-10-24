# House Price Prediction using Python

A beginner-friendly machine learning project that trains and compares Linear Regression, Decision Tree, and Random Forest models, evaluates them, saves the best pipeline, and provides a Streamlit web app for predictions.

## Features
- Dataset validation and basic cleaning
- Train/test split
- Model comparison using MAE, RMSE, and R²
- Saves the best model to `models/model.joblib`
- Interactive Streamlit prediction app
- Sample dataset included

> The included dataset is synthetic sample data for learning only. It is not real estate market data and predictions are not financial or valuation advice.

## 1. Requirements
- Python 3.10 or newer recommended
- Windows, macOS, or Linux

## 2. Install
Open a terminal in this project folder.

### Windows (PowerShell)
```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, use Command Prompt:
```bat
.venv\Scripts\activate.bat
```

### macOS/Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Train and evaluate
```bash
python src/train.py
```

This compares three models and saves the model with the lowest test-set MAE to `models/model.joblib`.

## 4. Run the web app
After training:
```bash
streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`.

## Dataset columns
- `area_sqft`: property area in square feet
- `bedrooms`: number of bedrooms
- `bathrooms`: number of bathrooms
- `age_years`: property age in years
- `price`: target price in rupees

`data/house_prices.csv` contains synthetic example data. Replace it with a legitimate dataset that uses the same columns to train on real data.

## Project structure
```text
price-prediction/
├── data/
│   └── house_prices.csv
├── models/
│   └── .gitkeep
├── src/
│   └── train.py
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Important notes
- MAE and RMSE are in the same units as the target (`price`).
- R² can be negative on poor predictions; it is not a percentage accuracy.
- For serious evaluation, use a larger representative dataset and cross-validation.
- Only load model files you trust. `joblib`/pickle-based files can execute code when loaded.
