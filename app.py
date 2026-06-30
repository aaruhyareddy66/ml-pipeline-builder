import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.inspection import permutation_importance
import plotly.figure_factory as ff
import plotly.express as px
from scipy.sparse import hstack, csr_matrix
from streamlit_sortables import sort_items
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

st.set_page_config(page_title="ML Pipeline Builder", page_icon="🧩", layout="wide")

st.markdown("""
<style>
.block-title { color: #2E7D32; font-weight: bold; font-size: 1.2rem; margin-bottom: 14px; }
.arrow { text-align: center; font-size: 1.8rem; color: #4CAF50; margin: 4px 0; }
</style>
""", unsafe_allow_html=True)

st.title("🧩 Drag-n-Drop ML Pipeline Builder")
st.caption("Build a machine learning pipeline step by step — drag to reorder, predict on new data, AI-powered explanations")

SAMPLE_DATASETS = {
    "Iris (flower classification)": load_iris,
    "Wine (wine type classification)": load_wine,
    "Breast Cancer (diagnosis classification)": load_breast_cancer,
}

MODELS = {
    "Random Forest": RandomForestClassifier,
    "Support Vector Machine": SVC,
    "Logistic Regression": LogisticRegression,
    "Decision Tree": DecisionTreeClassifier,
    "K-Nearest Neighbors": KNeighborsClassifier,
    "Gradient Boosting": GradientBoostingClassifier,
}

if "df" not in st.session_state:
    st.session_state.df = None
if "default_target" not in st.session_state:
    st.session_state.default_target = None
if "pipeline_order" not in st.session_state:
    st.session_state.pipeline_order = ["📦 Load Dataset", "🧹 Preprocess", "🤖 Choose Model", "🚀 Train & Evaluate"]
if "trained_model" not in st.session_state:
    st.session_state.trained_model = None
if "feature_cols" not in st.session_state:
    st.session_state.feature_cols = None
if "label_encoder" not in st.session_state:
    st.session_state.label_encoder = None
if "last_results" not in st.session_state:
    st.session_state.last_results = None

def classify_column(series, max_classes=50):
    non_null = series.dropna()
    if len(non_null) == 0:
        return "empty"
    n_unique = non_null.nunique()
    avg_len = non_null.astype(str).str.len().mean()
    if pd.api.types.is_numeric_dtype(series):
        if n_unique <= max_classes:
            return "categorical_numeric"
        return "continuous"
    else:
        if avg_len > 50:
            return "long_text"
        if n_unique <= max_classes:
            return "categorical_text"
        return "high_cardinality_text"

def is_id_like(series, total_rows, threshold=0.9):
    return series.nunique() >= total_rows * threshold

def get_ai_explanation(results, model_name, dataset_name):
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        prompt = f"""You are an expert data scientist explaining ML results to a non-technical audience.

Model: {model_name}
Dataset: {dataset_name}
Accuracy: {results['accuracy']}%
Training samples: {results['train_size']}
Test samples: {results['test_size']}
Features used: {', '.join(results['features'][:10])}
Classification report summary: {results['report_summary']}

Write a clear, friendly 4-5 sentence explanation covering:
1. How well the model performed and what the accuracy means in simple terms
2. Which classes were predicted best and worst
3. What this means practically
4. One specific improvement suggestion

Keep it conversational, no bullet points, no markdown formatting."""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"AI explanation unavailable: {str(e)}"

# ---------- STEP 1: Load Data ----------
with st.container(border=True):
    st.markdown('<div class="block-title">📦 Step 1 — Load Dataset</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        source = st.radio("Choose data source", ["Sample Dataset", "Upload CSV"])
    with col2:
        if source == "Sample Dataset":
            chosen = st.selectbox("Pick a sample dataset", list(SAMPLE_DATASETS.keys()))
            if st.button("Load Dataset"):
                data = SAMPLE_DATASETS[chosen]()
                df = pd.DataFrame(data.data, columns=data.feature_names)
                df["target"] = data.target
                st.session_state.df = df
                st.session_state.default_target = "target"
                st.session_state.dataset_name = chosen
                st.success(f"Loaded {chosen} — {len(df)} rows")
        else:
            uploaded = st.file_uploader("Upload your CSV file", type=["csv"])
            if uploaded:
                try:
                    df = pd.read_csv(uploaded)
                    if len(df) < 10:
                        st.error("This file has fewer than 10 rows — too small to reliably train and test.")
                    elif len(df.columns) < 2:
                        st.error("This file needs at least 2 columns.")
                    else:
                        st.session_state.df = df
                        st.session_state.dataset_name = uploaded.name
                        candidates = [c for c in df.columns
                                      if classify_column(df[c]) in ("categorical_text", "categorical_numeric")
                                      and not is_id_like(df[c], len(df))]
                        st.session_state.default_target = candidates[-1] if candidates else df.columns[-1]
                        st.success(f"Loaded {uploaded.name} — {len(df)} rows, {len(df.columns)} columns")
                except Exception as e:
                    st.error(f"Couldn't read this file as a CSV: {str(e)}")

    if st.session_state.df is not None:
        st.dataframe(st.session_state.df.head(5), width="stretch")
        with st.expander("📊 Column overview"):
            overview = []
            for c in st.session_state.df.columns:
                kind = classify_column(st.session_state.df[c])
                if kind in ("categorical_numeric", "continuous") and is_id_like(st.session_state.df[c], len(st.session_state.df)):
                    kind = "id_like (will be excluded)"
                overview.append({"Column": c, "Detected type": kind, "Unique values": st.session_state.df[c].nunique()})
            st.dataframe(pd.DataFrame(overview), width="stretch")

if st.session_state.df is not None:
    st.markdown('<div class="arrow">⬇️</div>', unsafe_allow_html=True)

    # ---------- DRAG AND DROP ----------
    with st.container(border=True):
        st.markdown('<div class="block-title">🔀 Reorder Pipeline Steps (drag to rearrange)</div>', unsafe_allow_html=True)
        st.caption("Drag the cards below to customize your pipeline order.")
        ordered_steps = sort_items(st.session_state.pipeline_order, key="pipeline_order_widget")
        st.session_state.pipeline_order = ordered_steps

    st.markdown('<div class="arrow">⬇️</div>', unsafe_allow_html=True)

    # ---------- STEP 2: Preprocessing ----------
    with st.container(border=True):
        st.markdown('<div class="block-title">🧹 Step 2 — Preprocessing</div>', unsafe_allow_html=True)
        cols = list(st.session_state.df.columns)
        default_idx = cols.index(st.session_state.default_target) if st.session_state.default_target in cols else len(cols) - 1

        col1, col2, col3 = st.columns(3)
        with col1:
            target_col = st.selectbox("Target column (the category to predict)", cols, index=default_idx)
        with col2:
            missing_strategy = st.selectbox("Handle missing values", ["Drop rows", "Fill with mean (numeric only)", "None"])
        with col3:
            scale_data = st.checkbox("Scale numeric features", value=True)

        target_series = st.session_state.df[target_col]
        target_kind = classify_column(target_series)
        n_total = len(st.session_state.df)

        if is_id_like(target_series, n_total):
            st.error(f"❌ '{target_col}' looks like an ID column — pick a column with repeated categories.")
        elif target_kind == "continuous":
            st.warning(f"⚠️ '{target_col}' looks continuous ({target_series.nunique()} unique values) — may give poor results.")
        elif target_kind == "long_text":
            st.warning(f"⚠️ '{target_col}' looks like free text — pick a short category column.")
        elif target_series.nunique() < 2:
            st.error(f"❌ '{target_col}' only has 1 unique value — nothing to classify.")
        else:
            st.success(f"✅ '{target_col}' looks like a good classification target ({target_series.nunique()} classes)")

        text_cols = [c for c in st.session_state.df.columns if c != target_col and classify_column(st.session_state.df[c]) == "long_text"]
        if text_cols:
            st.info(f"📝 Detected long text column(s): {', '.join(text_cols)} — will be converted using TF-IDF.")
        use_text_features = st.checkbox("Include text columns as features (TF-IDF)", value=bool(text_cols), disabled=not text_cols)

    st.markdown('<div class="arrow">⬇️</div>', unsafe_allow_html=True)

    # ---------- STEP 3: Model ----------
    with st.container(border=True):
        st.markdown('<div class="block-title">🤖 Step 3 — Choose Model</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            model_choice = st.selectbox("Algorithm", list(MODELS.keys()))
        with col2:
            test_size = st.slider("Test set size", 0.1, 0.5, 0.2, 0.05)

    st.markdown('<div class="arrow">⬇️</div>', unsafe_allow_html=True)

    # ---------- STEP 4: Train ----------
    with st.container(border=True):
        st.markdown('<div class="block-title">🚀 Step 4 — Train & Evaluate</div>', unsafe_allow_html=True)
        order_str = " → ".join(st.session_state.pipeline_order)
        st.caption(f"Your pipeline order: {order_str}")

        run_clicked = st.button("▶️ Run Pipeline", type="primary", width="stretch")

        if run_clicked:
            status = st.empty()
            status.info("🔄 Pipeline started...")
            try:
                df = st.session_state.df.copy()
                n_total = len(df)

                if len(df) < 10:
                    status.error("Dataset has fewer than 10 rows.")
                    st.stop()

                if is_id_like(df[target_col], n_total):
                    status.error(f"'{target_col}' is an ID-like column.")
                    st.stop()

                if missing_strategy == "Drop rows":
                    df = df.dropna(subset=[target_col])
                    df = df.fillna("")
                elif missing_strategy == "Fill with mean (numeric only)":
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        imputer = SimpleImputer(strategy="mean")
                        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
                    df = df.fillna("")
                else:
                    df = df.fillna("")

                y_raw = df[target_col]
                if classify_column(y_raw) == "continuous":
                    status.error(f"'{target_col}' has too many unique numeric values.")
                    st.stop()

                le = LabelEncoder()
                y = y_raw.values if pd.api.types.is_numeric_dtype(y_raw) else le.fit_transform(y_raw.astype(str))
                st.session_state.label_encoder = le if not pd.api.types.is_numeric_dtype(y_raw) else None

                if len(np.unique(y)) < 2:
                    status.error(f"'{target_col}' only has 1 class.")
                    st.stop()

                feature_df = df.drop(columns=[target_col])
                numeric_feature_cols = [c for c in feature_df.columns
                                         if classify_column(feature_df[c]) == "categorical_numeric"
                                         and not is_id_like(feature_df[c], n_total)]
                continuous_feature_cols = [c for c in feature_df.columns
                                            if classify_column(feature_df[c]) == "continuous"
                                            and not is_id_like(feature_df[c], n_total)]
                short_cat_cols = [c for c in feature_df.columns if classify_column(feature_df[c]) == "categorical_text"]
                long_text_cols_feat = [c for c in feature_df.columns if classify_column(feature_df[c]) == "long_text"]
                id_dropped = [c for c in feature_df.columns
                               if classify_column(feature_df[c]) == "high_cardinality_text"
                               or is_id_like(feature_df[c], n_total)]

                all_numeric = numeric_feature_cols + continuous_feature_cols
                if id_dropped:
                    st.caption(f"ℹ️ Dropped ID-like columns: {', '.join(id_dropped)}")

                status.info(f"🔄 Building features...")
                parts = []
                numeric_col_names = []

                if all_numeric:
                    X_num = feature_df[all_numeric].apply(pd.to_numeric, errors="coerce").fillna(0)
                    if scale_data:
                        X_num = pd.DataFrame(StandardScaler().fit_transform(X_num), columns=all_numeric)
                    parts.append(csr_matrix(X_num.values))
                    numeric_col_names.extend(all_numeric)

                if short_cat_cols:
                    X_cat = feature_df[short_cat_cols].astype(str).apply(lambda col: LabelEncoder().fit_transform(col))
                    parts.append(csr_matrix(X_cat.values))
                    numeric_col_names.extend(short_cat_cols)

                if long_text_cols_feat and use_text_features:
                    status.info("🔄 Running TF-IDF...")
                    combined_text = feature_df[long_text_cols_feat].astype(str).agg(" ".join, axis=1)
                    tfidf = TfidfVectorizer(max_features=300, stop_words="english")
                    X_text = tfidf.fit_transform(combined_text)
                    parts.append(X_text)
                    numeric_col_names.extend([f"tfidf_{w}" for w in tfidf.get_feature_names_out()[:20]])

                if not parts:
                    status.error("No usable feature columns found.")
                    st.stop()

                X = hstack(parts).tocsr() if len(parts) > 1 else parts[0]
                st.session_state.feature_cols = numeric_col_names

                class_counts = np.bincount(y)
                min_class = class_counts.min()
                can_stratify = min_class >= 2 and X.shape[0] * test_size >= len(class_counts)
                stratify = y if can_stratify else None

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=stratify
                )

                status.info(f"🔄 Training {model_choice}...")
                model_class = MODELS[model_choice]
                if model_choice == "Support Vector Machine":
                    model = model_class(probability=True)
                elif model_choice == "Logistic Regression":
                    model = model_class(max_iter=1000)
                elif model_choice == "K-Nearest Neighbors":
                    model = model_class()
                else:
                    model = model_class(random_state=42)

                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                st.session_state.trained_model = model

                acc = accuracy_score(y_test, preds)
                cm = confusion_matrix(y_test, preds)
                report = classification_report(y_test, preds, output_dict=True, zero_division=0)

                report_summary = f"Classes: {list(np.unique(y))}. " + \
                    " ".join([f"Class {k}: F1={v['f1-score']:.2f}" for k, v in report.items() if k not in ['accuracy', 'macro avg', 'weighted avg']])

                st.session_state.last_results = {
                    "accuracy": round(acc * 100, 2),
                    "train_size": X_train.shape[0],
                    "test_size": X_test.shape[0],
                    "features": numeric_col_names,
                    "report_summary": report_summary,
                    "model_name": model_choice,
                    "y_test": y_test,
                    "preds": preds,
                    "cm": cm,
                    "report": report
                }

                status.empty()
                st.success(f"✅ Pipeline complete! Accuracy: **{acc*100:.2f}%**")

                colA, colB, colC = st.columns(3)
                colA.metric("Accuracy", f"{acc*100:.2f}%")
                colB.metric("Training samples", X_train.shape[0])
                colC.metric("Test samples", X_test.shape[0])

                # ---------- FEATURE IMPORTANCE ----------
                st.subheader("📊 Feature Importance")
                if hasattr(model, "feature_importances_") and len(numeric_col_names) == X_train.shape[1]:
                    importance_df = pd.DataFrame({
                        "Feature": numeric_col_names[:len(model.feature_importances_)],
                        "Importance": model.feature_importances_
                    }).sort_values("Importance", ascending=False).head(15)
                    fig_imp = px.bar(importance_df, x="Importance", y="Feature",
                                     orientation="h", color="Importance",
                                     color_continuous_scale="Greens",
                                     title="Top 15 Most Important Features")
                    fig_imp.update_layout(yaxis=dict(autorange="reversed"))
                    st.plotly_chart(fig_imp, width="stretch")
                else:
                    st.caption("ℹ️ Feature importance not available for this model type or sparse matrix.")

                # ---------- CONFUSION MATRIX ----------
                st.subheader("🔢 Confusion Matrix")
                fig_cm = ff.create_annotated_heatmap(
                    z=cm, colorscale="Greens",
                    x=[f"Pred {i}" for i in range(cm.shape[0])],
                    y=[f"Actual {i}" for i in range(cm.shape[0])]
                )
                st.plotly_chart(fig_cm, width="stretch")

                # ---------- CLASSIFICATION REPORT ----------
                st.subheader("📋 Classification Report")
                st.dataframe(pd.DataFrame(report).transpose(), width="stretch")

                # ---------- AI EXPLANATION ----------
                st.subheader("🤖 AI Explanation (Llama 3 via Groq)")
                with st.spinner("Generating AI explanation..."):
                    explanation = get_ai_explanation(
                        st.session_state.last_results,
                        model_choice,
                        st.session_state.get("dataset_name", "uploaded dataset")
                    )
                st.info(explanation)

            except Exception as e:
                status.error(f"❌ Something went wrong: {str(e)}")
                with st.expander("Technical details"):
                    import traceback
                    st.code(traceback.format_exc())
                st.info("💡 Try a sample dataset to confirm the pipeline works, then troubleshoot your CSV.")

    # ---------- STEP 5: PREDICT ON NEW DATA ----------
    if st.session_state.trained_model is not None and st.session_state.feature_cols is not None:
        st.markdown('<div class="arrow">⬇️</div>', unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown('<div class="block-title">🔮 Step 5 — Predict on New Data</div>', unsafe_allow_html=True)
            st.caption("Enter values for each feature below and click Predict to see what the model decides.")

            non_tfidf_cols = [c for c in st.session_state.feature_cols if not c.startswith("tfidf_")]

            if non_tfidf_cols:
                input_values = {}
                cols_per_row = 4
                col_chunks = [non_tfidf_cols[i:i+cols_per_row] for i in range(0, len(non_tfidf_cols), cols_per_row)]

                for chunk in col_chunks:
                    row_cols = st.columns(len(chunk))
                    for col, col_name in zip(row_cols, chunk):
                        with col:
                            input_values[col_name] = st.number_input(col_name, value=0.0, key=f"pred_{col_name}")

                if st.button("🔮 Predict", type="primary"):
                    try:
                        input_df = pd.DataFrame([input_values])
                        input_sparse = csr_matrix(input_df.values)
                        prediction = st.session_state.trained_model.predict(input_sparse)
                        pred_label = prediction[0]
                        if st.session_state.label_encoder is not None:
                            pred_label = st.session_state.label_encoder.inverse_transform([int(pred_label)])[0]

                        st.success(f"🎯 Predicted class: **{pred_label}**")

                        if hasattr(st.session_state.trained_model, "predict_proba"):
                            proba = st.session_state.trained_model.predict_proba(input_sparse)[0]
                            proba_df = pd.DataFrame({"Class": range(len(proba)), "Probability": proba})
                            fig_proba = px.bar(proba_df, x="Class", y="Probability",
                                               color="Probability", color_continuous_scale="Greens",
                                               title="Prediction Probability per Class")
                            st.plotly_chart(fig_proba, width="stretch")
                    except Exception as e:
                        st.error(f"Prediction failed: {str(e)}")
            else:
                st.caption("ℹ️ Prediction on new data is only available when numeric/categorical features are used (not pure text datasets).")