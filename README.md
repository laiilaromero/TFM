
# TFM – IFC + EPD Integration and Machine Learning for Embodied Carbon Estimation

This repository contains the code developed for a Master's Thesis focused on exploring an open and reproducible workflow that combines **IFC models**, **EPD data**, and **machine-learning techniques** to estimate the embodied carbon of concrete elements.

The project assesses the feasibility and limitations of combining **synthetic training data** with **real IFC-EPD data**, to support early-stage environmental assessments when information is incomplete.

---

## 🚀 Project Goals
- Extract geometric and material information directly from **IFC files** using `ifcopenshell`.
- Clean and structure EPD datasets for modules **A1–A3** (EN 15804).
- Merge IFC + EPD information to compute GWP values per element.
- Generate synthetic data to train ML models when real EPD data is insufficient.
- Train and evaluate three regression models:
  - Random Forest  
  - Linear Regression  
  - Neural Network (Keras)
- Apply trained models to real building data and analyze performance limitations.

This work does not aim to replace official LCA calculations, but rather to investigate whether predictive models can support early-stage decision-making under data uncertainty.

---

## 📂 Repository Structure

