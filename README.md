# UCI - Heart Disease Prediction Project

A machine learning project to predict the likelihood of heart disease in patients using structured clinical data. The project demonstrates the full pipeline from data preprocessing to model deployment with a simple Streamlit user interface.

---

## Project Objectives

- Explore and preprocess the heart disease dataset.  
- Perform feature analysis, dimensionality reduction, and feature selection.  
- Build and evaluate supervised and unsupervised machine learning models.  
- Fine-tune hyperparameters to improve model performance.  
- Deploy the final model using a simple Streamlit UI for interactive predictions.

---

## File Structure

```text
📁 Heart_Disease_Project/
├─ 📂 data/
│  └─ 🗃️ heart_disease.csv
├─ 📂 notebooks/
│  ├─ 📓 01_data_preprocessing.ipynb
│  ├─ 📓 02_pca_analysis.ipynb
│  ├─ 📓 03_feature_selection.ipynb
│  ├─ 📓 04_supervised_learning.ipynb
│  ├─ 📓 05_unsupervised_learning.ipynb
│  └─ 📓 06_hyperparameter_tuning.ipynb
├─ 📂 models/
│  └─ 🧠 final_model.pkl
├─ 📂 ui/
│  └─ 🎈 app.py
├─ 📂 results/
│  └─ 📊 evaluation_metrics.txt
└─ 📄 requirements.txt
