# classification_app.py
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

# ML & DL imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import set_random_seed, to_categorical

# ---------------------------
# PAGE/STYLES
# ---------------------------
st.set_page_config(page_title="Universal Classification Analyzer", page_icon="🧠", layout="wide")
st.markdown("""
<style>
.main > div:nth-child(1) {padding: 1rem 1rem 0 1rem;}
.stButton>button {background-color:#0f766e;color: white;}
.checkbox-horizontal .stCheckbox>div{display:flex;align-items:center;}
.card {
    background-color: #f0f2f6;
    border-radius: 10px;
    padding: 10px;
    margin-bottom: 10px;
}
.metric-title { font-weight: bold; font-size: 14px; color:#0f766e; }
</style>
""", unsafe_allow_html=True)

st.title("🧠 Universal Classification Analyzer")
st.write("Upload CSV dataset, run models, and see detailed metrics.")
st.markdown("---")

# ---------------------------
# File upload
# ---------------------------
uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"])
uploaded_df = None
if uploaded_file:
    try:
        uploaded_df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")

# ---------------------------
# Helper functions
# ---------------------------
def encode_object_columns(df):
    encoders = {}
    try:
        for c in df.select_dtypes(include=['object', 'category']).columns:
            le = LabelEncoder()
            df[c] = le.fit_transform(df[c].astype(str))
            encoders[c] = le
    except Exception as e:
        st.warning(f"Encoding warning: {str(e)}")
    return df, encoders

def build_dl_model(input_dim, output_dim):
    set_random_seed(42)
    if output_dim == 1:
        model = Sequential([
            Dense(128, input_dim=input_dim, activation='relu'),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    else:
        model = Sequential([
            Dense(128, input_dim=input_dim, activation='relu'),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(output_dim, activation='softmax')
        ])
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def evaluate_metrics(y_true, y_pred, y_prob=None):
    try:
        unique_vals = np.unique(y_true)
        average_type = 'binary' if len(unique_vals) == 2 else 'macro'
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average=average_type, zero_division=0)
        rec = recall_score(y_true, y_pred, average=average_type, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=average_type, zero_division=0)
        # Compute ROC-AUC only if probabilities exist
        if y_prob is not None:
            if len(unique_vals) == 2:
                auc = roc_auc_score(y_true, y_prob)
            else:
                try:
                    auc = roc_auc_score(y_true, y_prob, multi_class='ovr')
                except:
                    auc = np.nan
        else:
            auc = np.nan
        return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": auc}
    except Exception as e:
        st.warning(f"Metrics evaluation error: {str(e)}")
        return {"accuracy": np.nan, "precision": np.nan, "recall": np.nan, "f1": np.nan, "roc_auc": np.nan}

def run_sklearn_model(name, model_obj, X_tr, X_te, y_tr, y_te):
    try:
        model_obj.fit(X_tr, y_tr)
        y_pred = model_obj.predict(X_te)
        # Only use predict_proba if it exists
        if hasattr(model_obj, "predict_proba"):
            if len(np.unique(y_te)) == 2:
                y_prob = model_obj.predict_proba(X_te)[:, 1]
            else:
                y_prob = model_obj.predict_proba(X_te)
        else:
            y_prob = None
        met = evaluate_metrics(y_te, y_pred, y_prob)
        return (name, met, y_pred, y_prob)
    except Exception as e:
        return (name, {"error": str(e)}, None, None)

def execute_models(models_to_run, X_train, X_test, y_train, y_test, dl_epochs=10, dl_batch=16):
    results = []
    futures = []
    with ThreadPoolExecutor(max_workers=min(6, max(1, len(models_to_run)))) as exe:
        for name in models_to_run:
            if name != "ANN (Deep Learning)":
                futures.append(exe.submit(run_sklearn_model, name, models_map[name], X_train, X_test, y_train, y_test))
        for f in as_completed(futures):
            name, met, y_pred, y_prob = f.result()
            results.append((name, met, y_pred, y_prob))
    if "ANN (Deep Learning)" in models_to_run:
        try:
            output_dim = 1 if len(np.unique(y_train)) == 2 else len(np.unique(y_train))
            ann_model = build_dl_model(X_train.shape[1], output_dim)
            if output_dim == 1:
                ann_model.fit(X_train, y_train, epochs=dl_epochs, batch_size=dl_batch, verbose=0)
                y_prob = ann_model.predict(X_test).ravel()
                y_pred = (y_prob > 0.5).astype(int)
            else:
                y_train_cat = to_categorical(y_train)
                ann_model.fit(X_train, y_train_cat, epochs=dl_epochs, batch_size=dl_batch, verbose=0)
                y_prob = ann_model.predict(X_test)
                y_pred = np.argmax(y_prob, axis=1)
            met = evaluate_metrics(y_test, y_pred, y_prob)
            results.append(("ANN (Deep Learning)", met, y_pred, y_prob))
        except Exception as e:
            st.warning(f"ANN failed: {str(e)}")
    return results

# ---------------------------
# Main UI
# ---------------------------
if uploaded_df is not None:
    df = uploaded_df.copy()
    df = df.dropna()
    df, _ = encode_object_columns(df)

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10))
    st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

    target = st.selectbox("Select target column", df.columns, index=len(df.columns)-1)
    if target:
        X = df.drop(columns=[target])
        y = df[target]
        dist_str = ", ".join([f"{k}: {v:.2f}%" for k, v in y.value_counts(normalize=True).mul(100).items()])
        st.metric("Target distribution (%)", dist_str)

        # Quick EDA
        if st.checkbox("Show correlation heatmap"):
            fig, ax = plt.subplots(figsize=(13,13))
            sns.heatmap(df.corr(), cmap="coolwarm", ax=ax, annot=True)
            st.pyplot(fig)

        if st.checkbox("Show pairplot (sample 200 rows)"):
            sample = df.sample(min(200, df.shape[0]), random_state=42)
            fig = sns.pairplot(sample.select_dtypes(include=[np.number]).iloc[:, :6]).fig
            st.pyplot(fig)

        st.subheader("Modeling Options")
        test_size = st.slider("Test size", 0.1, 0.5, 0.2)
        random_state = st.number_input("Random seed", value=42, step=1)

        fast_mode = True
        scale_data = True
        dl_epochs = 10 if fast_mode else 30
        dl_batch = 16

        # Models map (all models)
        models_map = {
            "Logistic Regression": LogisticRegression(max_iter=1000, n_jobs=-1),
            "Decision Tree": DecisionTreeClassifier(random_state=random_state),
            "Extra Tree": ExtraTreeClassifier(random_state=random_state),
            "Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=random_state),
            "AdaBoost": AdaBoostClassifier(random_state=random_state),
            "Gradient Boosting": GradientBoostingClassifier(random_state=random_state),
            "Bagging": BaggingClassifier(n_jobs=-1, random_state=random_state),
            "SVM (RBF)": SVC(probability=True, kernel="rbf", random_state=random_state),
            "Linear SVM": LinearSVC(random_state=random_state, max_iter=5000),
            "KNN": KNeighborsClassifier(),
            "Naive Bayes (Gaussian)": GaussianNB(),
            "Naive Bayes (Bernoulli)": BernoulliNB(),
            "LDA": LinearDiscriminantAnalysis(),
            "QDA": QuadraticDiscriminantAnalysis(),
            "Ridge Classifier": RidgeClassifier(),
            "SGD Classifier": SGDClassifier(loss="log_loss", random_state=random_state),
            "ANN (Deep Learning)": "ANN"
        }

        # Only keep models that support both binary and multiclass
        allowed_models = [
            "Logistic Regression", "Decision Tree", "Extra Tree", "Random Forest",
            "AdaBoost", "Gradient Boosting", "Bagging", "SVM (RBF)", "KNN",
            "Naive Bayes (Gaussian)", "LDA", "QDA", "ANN (Deep Learning)"
        ]

        # ML/DL selection UI
        with st.expander("Select ML/DL models to run", expanded=True):
            selected_models = []
            cols = st.columns(3)
            for i, model_name in enumerate(models_map.keys()):
                if model_name not in allowed_models:
                    continue
                col = cols[i % 3]
                if col.checkbox(model_name, value=model_name in ["Logistic Regression", "Decision Tree", "Random Forest", "KNN", "SVM (RBF)", "Naive Bayes (Gaussian)", "LDA", "ANN (Deep Learning)"]):
                    selected_models.append(model_name)

        # Run button
        if "model_results" not in st.session_state:
            st.session_state.model_results = {}

        run_single = st.button("Run Selected Models")

        # Split & scale
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=int(random_state))
        if scale_data:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        # Run selected models
        if run_single:
            st.session_state.model_results = execute_models(selected_models, X_train, X_test, y_train, y_test, dl_epochs, dl_batch)
            results = st.session_state.model_results
            st.success("Selected models have been run!")

            st.subheader("📊 Run Selected Models Outcomes")
            for name, met, y_pred, y_prob in results:
                if "error" in met:
                    st.warning(f"{name} failed: {met['error']}")
                    continue
                st.markdown(f"**Model: {name}**")
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("Accuracy", f"{met['accuracy']:.3f}")
                col2.metric("Precision", f"{met['precision']:.3f}")
                col3.metric("Recall", f"{met['recall']:.3f}")
                col4.metric("F1 Score", f"{met['f1']:.3f}")
                col5.metric("ROC-AUC", f"{met['roc_auc']:.3f}")

                if y_pred is not None:
                    st.markdown("**Classification Report:**")
                    cr_dict = classification_report(y_test, y_pred, output_dict=True)
                    classes = [k for k in cr_dict.keys() if k not in ["accuracy", "macro avg", "weighted avg"]]
                    cr_rows = []
                    for cls in classes:
                        cr_rows.append({
                            "Class": cls,
                            "Precision": cr_dict[cls]['precision'],
                            "Recall": cr_dict[cls]['recall'],
                            "F1-Score": cr_dict[cls]['f1-score'],
                            "Support": int(cr_dict[cls]['support'])
                        })
                    cr_df = pd.DataFrame(cr_rows)
                    st.dataframe(cr_df.style.format({
                        "Precision": "{:.3f}",
                        "Recall": "{:.3f}",
                        "F1-Score": "{:.3f}"
                    }))

        # Comparison table at the end
        if st.session_state.model_results:
            results = st.session_state.model_results
            rows = []
            for name, met, _, _ in results:
                if "error" not in met:
                    rows.append([name, met["accuracy"], met["precision"], met["recall"], met["f1"], met["roc_auc"]])
            if rows:
                res_df = pd.DataFrame(rows, columns=["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"])
                res_df_sorted = res_df.sort_values("Accuracy", ascending=False).reset_index(drop=True)

                st.subheader("🏆 Model Comparison Table")
                st.dataframe(res_df_sorted.style.format({
                    "Accuracy": "{:.3f}",
                    "Precision": "{:.3f}",
                    "Recall": "{:.3f}",
                    "F1": "{:.3f}",
                    "ROC-AUC": "{:.3f}"
                }))

                csv_bytes = res_df_sorted.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download Comparison CSV", data=csv_bytes, file_name="comparison_table.csv", mime="text/csv")
