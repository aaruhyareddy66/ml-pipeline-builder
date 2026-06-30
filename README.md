# 🧩 Drag-n-Drop ML Pipeline Builder

An interactive machine learning pipeline builder that lets anyone — technical or not — load a dataset, configure preprocessing, choose a model, train it, and get AI-powered explanations of the results. No code required.

**🔗 Live App:** [ml-pipeline-builder-dzhawgnninyamxrfzj86eg.streamlit.app](https://ml-pipeline-builder-dzhawgnninyamxrfzj86eg.streamlit.app)
**🔗 GitHub:** [github.com/aaruhyareddy66/ml-pipeline-builder](https://github.com/aaruhyareddy66/ml-pipeline-builder)

---

## 📌 Why this project

Most ML tools assume you already know what you're doing. This project flips that — it gives anyone a visual, step-by-step interface to build and run a real classification pipeline, understand the results, and even make predictions on new data, without writing a single line of code.

---

## 🧠 How it works

1. **Load data** — pick from 3 built-in sample datasets (Iris, Wine, Breast Cancer) or upload any CSV file
2. **Drag to reorder** — rearrange the pipeline steps in any order using the drag-and-drop interface
3. **Preprocess** — choose a target column, handle missing values, scale numeric features; the app auto-detects column types (numeric, categorical, long text, ID-like) and handles each appropriately
4. **Choose model** — pick from 6 algorithms: Random Forest, SVM, Logistic Regression, Decision Tree, KNN, or Gradient Boosting
5. **Train & evaluate** — one click runs the full pipeline and shows accuracy, confusion matrix, classification report, and feature importance
6. **AI explanation** — Llama 3.3 (via Groq API) reads the results and explains them in plain English
7. **Predict on new data** — enter values for each feature and get an instant prediction with probability scores

---

## ✨ Features

- **Drag-and-drop pipeline ordering** using `streamlit-sortables`
- **Smart column detection** — automatically identifies numeric, categorical, long text, and ID-like columns
- **TF-IDF text support** — text-heavy CSVs (like resume datasets) are handled via TF-IDF vectorization
- **6 ML algorithms** to choose from
- **Feature importance chart** for tree-based models
- **Confusion matrix** with annotated heatmap
- **Classification report** with per-class precision, recall, and F1
- **AI explanation** powered by Llama 3.3 70B via Groq (fast, free)
- **Predict on new data** with probability distribution chart
- **Robust error handling** — friendly messages for wrong target columns, too-small datasets, ID columns, single-class targets, and more
- Works with **any CSV** — numeric, categorical, or raw text data

---

## 🛠️ Tech Stack

| Layer | Tech |
|---|---|
| UI & App | Streamlit |
| ML | scikit-learn (RandomForest, SVM, LogisticRegression, DecisionTree, KNN, GradientBoosting) |
| NLP | TF-IDF (sklearn), Groq API (Llama 3.3 70B) |
| Visualization | Plotly |
| Drag-and-drop | streamlit-sortables |
| Deployment | Streamlit Cloud |

---

## 📁 Project Structure
ml_pipeline_builder/
├── app.py                  # Main Streamlit app — full pipeline logic
├── backend/
│   └── main.py             # FastAPI backend (REST API version)
├── .streamlit/
│   └── config.toml         # Streamlit cloud deployment config
├── requirements.txt
└── README.md
---

## 🚀 Running locally

```bash
git clone https://github.com/aaruhyareddy66/ml-pipeline-builder.git
cd ml-pipeline-builder

python -m venv venv
venv\Scripts\activate        # Windows

pip install -r requirements.txt
```

Create a `.env` file:
Run the app:
```bash
streamlit run app.py
```

---

## 📊 Supported CSV types

| CSV type | Supported | How |
|---|---|---|
| Pure numeric | ✅ | Standard scaling + model |
| Mix of numeric + categorical | ✅ | Label encoding + scaling |
| Text-heavy (resumes, articles) | ✅ | TF-IDF vectorization |
| CSVs with ID columns | ✅ | Auto-detected and excluded |
| CSVs with missing values | ✅ | Drop rows or fill with mean |

---

## 🔭 Possible next steps

- Add regression support (currently classification only)
- Add hyperparameter tuning UI (GridSearch sliders)
- Allow model download after training
- Add cross-validation option
- Support multi-label classification

---

## 👤 Author

**Aaruhya Reddy**
GitHub: [@aaruhyareddy66](https://github.com/aaruhyareddy66)
Live App: [ml-pipeline-builder-dzhawgnninyamxrfzj86eg.streamlit.app](https://ml-pipeline-builder-dzhawgnninyamxrfzj86eg.streamlit.app)
