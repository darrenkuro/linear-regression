<h1 align="center">Linear Regression</h1>

<p align="center">
    <img src="https://img.shields.io/badge/license-MIT-blue?style=flat-square&logo=opensourceinitiative&logoColor=white" alt="License"/>
    <img src="https://img.shields.io/badge/status-development-brightgreen?style=flat-square&logo=git&logoColor=white" alt="Status">
    <!-- <img src="https://img.shields.io/badge/score-125%2F100-3CB371?style=flat-square&logo=42&logoColor=white" alt="Score"/> -->
    <img src="https://img.shields.io/badge/date-Dec%2014th,%202025-ff6984?style=flat-square&logo=Cachet&logoColor=white" alt="Date"/>
</p>

> Univariate linear regression implemented manually from scratch: train, predict, and evaluate using gradient descent, complete with error analysis and visualization.

---

## 🚀 Overview

This project implements **linear regression** in Python to predict a target `y` from a single feature `x` using `ŷ = θ₀ + θ₁x`. It'll parse CSV data, normalize features, train with **gradient descent**, save/load parameters, and **visualize** the learning process. The aim is to understand the basic math and mechanics that laid the foundation of ML.

## 🧰 Tech Stack:
![Python](https://img.shields.io/badge/-Python_3-3776AB?style=flat-square&logo=python&logoColor=white)
![Matplotlib](https://img.shields.io/badge/-Matplotlib-11557C?style=flat-square&logo=plotly&logoColor=white)

## 📦 Features

- ✅ **CSV ingestion & normalization** (min–max or z-score)
- ✅ **Gradient descent trainer** with configurable `learning_rate`, `epochs`, and early stopping
- ✅ **Model I/O**: save/load `θ₀`, `θ₁` to a small JSON file
- ✅ **Prediction CLI** for single values or batches
- ✅ **Visualization**: regression line over time

---

## 🛠️ Configuration

### Prerequisites
- Python 3.10+
- `pip` (and optionally `venv`)
- Dataset CSV with two columns (e.g., `x,y`) and a header row
- `Matplotlib` required for visualization, but optional

### Installation & Usage
```bash
git clone https://github.com/darrenkuro/linear-regression.git && cd linear-regression

# Create venv (Recommended)
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 📫 Contact

Darren Kuro – [darrenkuro@icloud.com](mailto:darrenkuro@icloud.com)
GitHub: [@darrenkuro](https://github.com/darrenkuro)
