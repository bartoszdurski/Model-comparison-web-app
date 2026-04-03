import streamlit as st
import pandas as pd
import plotly.express as px
import os
import numpy as np
import time

# Model imports
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.preprocessing import LabelEncoder

# Tuning imports
from sklearn.model_selection import GridSearchCV, train_test_split

# Imputation imports
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Metric imports
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score, confusion_matrix

# --- HELPER: STYLING PLOTS ---
def style_plot(fig):
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='gray'),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

# --- HELPER: PARSE INPUT ---
def parse_list_input(input_str, type_func=int):
    """Zamienia ciąg '10, 20, 30' na listę [10, 20, 30]"""
    try:
        return [type_func(x.strip()) for x in input_str.split(',')]
    except:
        return []

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Model Comparison", layout="wide")

st.title("Model Comparison Tool")
st.markdown("Advanced comparison with Hyperparameter Tuning.")

# --- INIT SESSION STATE ---
if 'custom_models' not in st.session_state:
    st.session_state.custom_models = {}

# --- 1. DATA LOADING ---
st.sidebar.header("1. Data Selection")
current_dir = os.getcwd()
csv_files = [f for f in os.listdir(current_dir) if f.endswith('.csv')]

if not csv_files:
    st.error("No .csv files found!")
    st.stop()

selected_file = st.sidebar.selectbox("Select CSV file", csv_files)

@st.cache_data
def load_raw_data(filename):
    return pd.read_csv(filename)

raw_df = load_raw_data(selected_file)

if 'processed_df' not in st.session_state or st.session_state.get('current_file') != selected_file:
    st.session_state.processed_df = raw_df.copy()
    st.session_state.current_file = selected_file
    if 'model_results' in st.session_state: del st.session_state.model_results

# --- 2. PREPROCESSING ---
st.sidebar.header("2. Preprocessing")
cols_to_drop = st.sidebar.multiselect("Select columns to DROP", raw_df.columns)
missing_method = st.sidebar.selectbox("Imputation Method", ["None", "Simple Imputer (Mean)", "Simple Imputer (Median)", "Simple Imputer (Most Frequent)", "Simple Imputer (Constant - 0)"])
cols_to_impute = []
if missing_method != "None":
    cols_to_impute = st.sidebar.multiselect("Select columns to IMPUTE", [c for c in raw_df.columns if c not in cols_to_drop])

available_cols = [c for c in raw_df.columns if c not in cols_to_drop]
cols_to_ohe = st.sidebar.multiselect("Columns for ONE-HOT Encoding", available_cols)
remaining_for_le = [c for c in available_cols if c not in cols_to_ohe]
cols_to_le = st.sidebar.multiselect("Columns for LABEL Encoding", remaining_for_le)

if st.sidebar.button("Apply Preprocessing"):
    temp_df = raw_df.copy()
    if cols_to_drop: temp_df = temp_df.drop(columns=cols_to_drop)
    
    if missing_method != "None" and cols_to_impute:
        try:
            if missing_method == "Simple Imputer (Mean)": imp = SimpleImputer(strategy='mean')
            elif missing_method == "Simple Imputer (Median)": imp = SimpleImputer(strategy='median')
            elif missing_method == "Simple Imputer (Most Frequent)": imp = SimpleImputer(strategy='most_frequent')
            else: imp = SimpleImputer(strategy='constant', fill_value=0)
            temp_df[cols_to_impute] = imp.fit_transform(temp_df[cols_to_impute])
        except Exception as e: st.sidebar.error(f"Imputation Error: {e}")

    if cols_to_le:
        le = LabelEncoder()
        for col in cols_to_le: temp_df[col] = le.fit_transform(temp_df[col].astype(str))
        
    if cols_to_ohe:
        # POPRAWKA: drop_first=False zachowuje wszystkie kolumny (np. Sex_female ORAZ Sex_male)
        temp_df = pd.get_dummies(temp_df, columns=cols_to_ohe, drop_first=False)
        bool_cols = temp_df.select_dtypes(include=['bool']).columns
        temp_df[bool_cols] = temp_df[bool_cols].astype(int)
    
    st.session_state.processed_df = temp_df
    st.sidebar.success("Changes applied!")

df = st.session_state.processed_df

# --- DATA PREVIEW & STATISTICS ---
with st.expander("Data Preview & Statistics", expanded=True):
    st.subheader("Raw Data Preview")
    st.write(f"Shape: {df.shape}")
    st.dataframe(df.head())

    st.markdown("---")
    st.subheader("Statistics & Missing Values")
    
    col_stat1, col_stat2 = st.columns(2)

    with col_stat1:
        st.write("**Descriptive Statistics**")
        st.dataframe(df.describe())

    with col_stat2:
        st.write("**Missing Values Analysis**")
        info_df = pd.DataFrame({
            'Data Type': df.dtypes.astype(str),
            'Null Count': df.isnull().sum(),
            'Null (%)': (df.isnull().sum() / len(df) * 100),
            'Unique Values': df.nunique()
        })
        st.dataframe(
            info_df,
            column_config={
                "Null (%)": st.column_config.ProgressColumn(
                    "Null (%)", format="%.2f%%", min_value=0, max_value=100
                )
            },
            use_container_width=True
        )

# --- 3. CONFIGURATION & CUSTOM MODELS ---
st.sidebar.markdown("---")
st.sidebar.header("3. Problem & Target")

if len(df.columns) < 2: st.stop()
target_col = st.sidebar.selectbox("Select Target", df.columns)
problem_type = st.sidebar.radio("Problem Type", ["Classification", "Regression"])

# --- NEW SECTION: MODEL BUILDER ---
st.sidebar.markdown("---")
st.sidebar.header("4. Build Custom Model")

with st.sidebar.expander("Create New Configuration"):
    base_algos = ["Random Forest", "XGBoost", "Decision Tree", "Logistic/Linear Regression"]
    selected_base = st.selectbox("Base Algorithm", base_algos)
    custom_name = st.text_input("Name your model", f"My {selected_base}")
    tuning_mode = st.radio("Configuration Mode", ["Manual Parameters", "Grid Search (Auto-Tune)"])
    
    params = {}
    if selected_base == "Random Forest":
        if tuning_mode == "Manual Parameters":
            n_est = st.number_input("n_estimators", 10, 1000, 100)
            max_d = st.number_input("max_depth", 1, 100, 10)
            params = {"n_estimators": n_est, "max_depth": max_d}
        else:
            n_est_str = st.text_input("n_estimators list", "50, 100, 200")
            max_d_str = st.text_input("max_depth list", "5, 10, 20")
            params = {"n_estimators": parse_list_input(n_est_str), "max_depth": parse_list_input(max_d_str)}

    elif selected_base == "XGBoost":
        if tuning_mode == "Manual Parameters":
            lr = st.number_input("learning_rate", 0.001, 1.0, 0.1, format="%.3f")
            n_est = st.number_input("n_estimators", 10, 1000, 100)
            params = {"learning_rate": lr, "n_estimators": n_est}
        else:
            lr_str = st.text_input("learning_rate list", "0.01, 0.1, 0.3")
            n_est_str = st.text_input("n_estimators list", "50, 100, 200")
            params = {"learning_rate": parse_list_input(lr_str, float), "n_estimators": parse_list_input(n_est_str)}
            
    elif selected_base == "Decision Tree":
        if tuning_mode == "Manual Parameters":
            max_d = st.number_input("max_depth", 1, 50, 5)
            min_s = st.number_input("min_samples_split", 2, 20, 2)
            params = {"max_depth": max_d, "min_samples_split": min_s}
        else:
            max_d_str = st.text_input("max_depth list", "3, 5, 10, 20")
            params = {"max_depth": parse_list_input(max_d_str)}

    elif selected_base == "Logistic/Linear Regression":
        st.info("Simple regression usually has fewer parameters to tune here.")
    
    if st.button("Save Configuration"):
        if custom_name in st.session_state.custom_models:
            st.warning("Overwriting existing model with this name.")
        
        st.session_state.custom_models[custom_name] = {
            "type": selected_base,
            "mode": tuning_mode,
            "params": params,
            "problem_type": problem_type
        }
        st.success(f"Saved '{custom_name}'!")

# --- 5. SELECT MODELS TO RUN ---
st.sidebar.markdown("---")
st.sidebar.header("5. Select Models to Run")

defaults = []
if problem_type == "Classification":
    defaults = ["Logistic Regression (Default)", "Decision Tree (Default)", "Random Forest (Default)", "XGBoost (Default)", "CatBoost (Default)"]
else:
    defaults = ["Linear Regression (Default)", "Decision Tree (Default)", "Random Forest (Default)", "XGBoost (Default)", "CatBoost (Default)"]

custom_options = [name for name, cfg in st.session_state.custom_models.items() if cfg['problem_type'] == problem_type]
all_options = defaults + custom_options

selected_models_names = st.sidebar.multiselect("Choose models", all_options, default=defaults[:2])

# --- 6. TRAINING ENGINE ---
y = df[target_col]
X = df.drop(columns=[target_col]).select_dtypes(include=[np.number])

if st.sidebar.button("Run Analysis"):
    
    if not selected_models_names: st.error("Select models!"); st.stop()
    if X.isnull().values.any(): st.error("NaN values found! Fix in Preprocessing."); st.stop()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    results_metrics = []
    full_results = {}
    
    progress_bar = st.progress(0)
    step = 1.0 / len(selected_models_names)
    curr_prog = 0.0
    
    with st.spinner("Training & Tuning models..."):
        for name in selected_models_names:
            model = None
            is_grid = False
            
            if name in st.session_state.custom_models:
                config = st.session_state.custom_models[name]
                base_type = config['type']
                params = config['params']
                mode = config['mode']
                
                if problem_type == "Classification":
                    if base_type == "Random Forest": base = RandomForestClassifier()
                    elif base_type == "XGBoost": base = XGBClassifier(verbosity=0, use_label_encoder=False)
                    elif base_type == "Decision Tree": base = DecisionTreeClassifier()
                    else: base = LogisticRegression()
                else:
                    if base_type == "Random Forest": base = RandomForestRegressor()
                    elif base_type == "XGBoost": base = XGBRegressor(verbosity=0)
                    elif base_type == "Decision Tree": base = DecisionTreeRegressor()
                    else: base = LinearRegression()
                
                if mode == "Manual Parameters":
                    base.set_params(**params)
                    model = base
                else:
                    is_grid = True
                    model = GridSearchCV(base, params, cv=3, verbose=0, n_jobs=-1)
            else:
                clean_name = name.replace(" (Default)", "")
                if problem_type == "Classification":
                    if clean_name == "Logistic Regression": model = LogisticRegression(max_iter=1000)
                    elif clean_name == "Decision Tree": model = DecisionTreeClassifier()
                    elif clean_name == "Random Forest": model = RandomForestClassifier()
                    elif clean_name == "XGBoost": model = XGBClassifier(verbosity=0, use_label_encoder=False)
                    elif clean_name == "CatBoost": model = CatBoostClassifier(verbose=0)
                else:
                    if clean_name == "Linear Regression": model = LinearRegression()
                    elif clean_name == "Decision Tree": model = DecisionTreeRegressor()
                    elif clean_name == "Random Forest": model = RandomForestRegressor()
                    elif clean_name == "XGBoost": model = XGBRegressor(verbosity=0)
                    elif clean_name == "CatBoost": model = CatBoostRegressor(verbose=0)

            try:
                start_time = time.time()
                model.fit(X_train, y_train)
                end_time = time.time()
                duration = end_time - start_time
                
                final_model = model.best_estimator_ if is_grid else model
                best_params_txt = str(model.best_params_) if is_grid else "Default/Manual"

                y_pred = final_model.predict(X_test)
                
                if problem_type == "Classification":
                    acc = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred, average='weighted') if len(np.unique(y)) > 2 else f1_score(y_test, y_pred)
                    results_metrics.append({
                        "Model": name, "Accuracy": acc, "F1 Score": f1, "Time (s)": duration, "Params": best_params_txt
                    })
                    sort_by = "Accuracy"
                    metric_plot = "Accuracy"
                else:
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    results_metrics.append({
                        "Model": name, "MAE": mae, "R2 Score": r2, "Time (s)": duration, "Params": best_params_txt
                    })
                    sort_by = "R2 Score"
                    metric_plot = "R2 Score"
                
                full_results[name] = {"model": final_model, "y_pred": y_pred}

            except Exception as e:
                st.error(f"Error {name}: {str(e)}")

            curr_prog += step
            progress_bar.progress(min(curr_prog, 1.0))
            
    progress_bar.empty()
    
    st.session_state.model_results = {
        "metrics": results_metrics,
        "full": full_results,
        "X_test": X_test,
        "y_test": y_test,
        "sort_by": sort_by,
        "metric_plot": metric_plot,
        "problem_type": problem_type,
        "feat_names": X.columns
    }

# --- 7. RESULTS DISPLAY (COLLAPSIBLE) ---
if 'model_results' in st.session_state:
    data = st.session_state.model_results
    res_df = pd.DataFrame(data['metrics']).sort_values(by=data['sort_by'], ascending=False)
    
    # 1. LEADERBOARD
    with st.expander("Leaderboard", expanded=True):
        st.dataframe(res_df.style.background_gradient(subset=[data['sort_by']], cmap='Greens'), use_container_width=True)
    
    # 2. PERFORMANCE CHART
    with st.expander(f"Model Performance ({data['metric_plot']})", expanded=True):
        fig = px.bar(res_df, x="Model", y=data['metric_plot'], color="Model", text_auto='.3f')
        st.plotly_chart(style_plot(fig), use_container_width=True)

    # 3. TRAINING TIME CHART
    with st.expander("Training Time", expanded=False):
        fig = px.bar(res_df, x="Model", y="Time (s)", text_auto='.3f')
        st.plotly_chart(style_plot(fig), use_container_width=True)
        
    # 4. DEEP DIVE
    with st.expander("Deep Dive & Config Details", expanded=False):
        sel_model = st.selectbox("Select Model Details", res_df["Model"].tolist())
        if sel_model:
            params_used = res_df[res_df["Model"] == sel_model]["Params"].values[0]
            st.info(f"**Configuration / Best Params:** {params_used}")
            
            mod_obj = data['full'][sel_model]['model']
            y_p = data['full'][sel_model]['y_pred']
            
            d1, d2 = st.columns(2)
            with d1:
                st.write("**Feature Importance**")
                fi = None
                if hasattr(mod_obj, 'feature_importances_'): fi = mod_obj.feature_importances_
                elif hasattr(mod_obj, 'coef_'): fi = mod_obj.coef_[0]
                
                if fi is not None:
                    fi_df = pd.DataFrame({'Feature': data['feat_names'], 'Importance': fi}).sort_values('Importance')
                    fig_fi = px.bar(fi_df, x='Importance', y='Feature', orientation='h')
                    st.plotly_chart(style_plot(fig_fi), use_container_width=True)
                else: st.warning("Not available for this model.")
                
            with d2:
                if data['problem_type'] == "Classification":
                    st.write("**Confusion Matrix**")
                    cm = confusion_matrix(data['y_test'], y_p)
                    fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues')
                    st.plotly_chart(style_plot(fig_cm), use_container_width=True)
                else:
                    st.write("**Actual vs Predicted**")
                    res_d = pd.DataFrame({'Act': data['y_test'], 'Pred': y_p})
                    fig_res = px.scatter(res_d, x='Act', y='Pred', trendline='ols', trendline_color_override='red')
                    st.plotly_chart(style_plot(fig_res), use_container_width=True)