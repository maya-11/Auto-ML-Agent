import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ----------------- Page Configuration -----------------
st.set_page_config(
    page_title="AutoML Genius 🤖",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- CSS Styling -----------------
def local_css():
    st.markdown("""
    <style>
    .main-header {
        font-size: 3.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
        background: linear-gradient(45deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
        border-left: 4px solid #3498db;
    }
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
        border-top: 4px solid #3498db;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2c3e50;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #7f8c8d;
        text-transform: uppercase;
    }
    .stProgress > div > div > div > div {
        background-color: #3498db;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #2980b9;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
    }
    .model-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }
    .model-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.2);
    }
    </style>
    """, unsafe_allow_html=True)

local_css()

# ----------------- Local Analysis Functions -----------------
def generate_data_summary(df, target_column):
    """Generate a comprehensive data summary without API calls"""
    summary = []
    
    # Basic info
    summary.append("## 📊 Dataset Overview")
    summary.append(f"- **Shape**: {df.shape[0]} rows, {df.shape[1]} columns")
    summary.append(f"- **Target Variable**: {target_column}")
    summary.append("")
    
    # Data types
    summary.append("## 🔍 Data Types")
    for dtype in df.dtypes.unique():
        cols = df.select_dtypes(include=[dtype]).columns.tolist()
        summary.append(f"- **{dtype}**: {len(cols)} columns")
        if len(cols) > 0:
            summary.append(f"  - {', '.join(cols)}")
    summary.append("")
    
    # Missing values
    summary.append("## ❓ Missing Values")
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    if len(missing_cols) > 0:
        for col, count in missing_cols.items():
            percentage = (count / len(df)) * 100
            summary.append(f"- **{col}**: {count} missing values ({percentage:.2f}%)")
    else:
        summary.append("- No missing values found!")
    summary.append("")
    
    # Target analysis
    summary.append(f"## 🎯 Target Analysis: {target_column}")
    if df[target_column].dtype in ['object', 'category']:
        value_counts = df[target_column].value_counts()
        summary.append(f"- **Type**: Categorical")
        summary.append(f"- **Unique Values**: {len(value_counts)}")
        summary.append("- **Distribution**:")
        for value, count in value_counts.items():
            percentage = (count / len(df)) * 100
            summary.append(f"  - {value}: {count} ({percentage:.1f}%)")
    else:
        summary.append(f"- **Type**: Numerical")
        summary.append(f"- **Range**: {df[target_column].min():.2f} to {df[target_column].max():.2f}")
        summary.append(f"- **Mean**: {df[target_column].mean():.2f}")
        summary.append(f"- **Std Dev**: {df[target_column].std():.2f}")
    summary.append("")
    
    # Recommendations
    summary.append("## 💡 Preprocessing Recommendations")
    
    # Numeric features
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if target_column in numeric_features:
        numeric_features.remove(target_column)
    
    if numeric_features:
        summary.append("### For Numerical Features:")
        summary.append("- Impute missing values with mean/median")
        summary.append("- Scale features using StandardScaler")
        summary.append(f"- Columns: {', '.join(numeric_features)}")
    
    # Categorical features
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if target_column in categorical_features:
        categorical_features.remove(target_column)
    
    if categorical_features:
        summary.append("### For Categorical Features:")
        summary.append("- Impute missing values with mode")
        summary.append("- Encode using OneHotEncoder")
        summary.append(f"- Columns: {', '.join(categorical_features)}")
    
    # Target preprocessing
    summary.append(f"### For Target Variable ({target_column}):")
    if df[target_column].dtype in ['object', 'category']:
        summary.append("- Label encoding will be applied")
    else:
        summary.append("- No encoding needed (numerical)")
    
    return "\n".join(summary)

def generate_model_insights(leaderboard_df, best_model_name):
    """Generate model insights without API calls"""
    insights = []
    
    insights.append("## 🧠 Model Performance Insights")
    insights.append("")
    
    # Best model analysis
    best_accuracy = leaderboard_df[leaderboard_df['Model'] == best_model_name]['Accuracy'].values[0]
    insights.append(f"### 🏆 Best Model: {best_model_name}")
    insights.append(f"- **Accuracy**: {best_accuracy:.3f}")
    insights.append("")
    
    # Model comparison
    insights.append("### 📈 Model Comparison")
    for _, row in leaderboard_df.iterrows():
        insights.append(f"- **{row['Model']}**: Accuracy = {row['Accuracy']:.3f}, F1 = {row['F1-Score']:.3f}")
    insights.append("")
    
    # Recommendations
    insights.append("### 💡 Recommendations for Improvement")
    insights.append("- Try collecting more data if possible")
    insights.append("- Experiment with feature engineering")
    insights.append("- Consider hyperparameter tuning")
    insights.append("- Try ensemble methods or different algorithms")
    insights.append("- Address class imbalance if present")
    
    return "\n".join(insights)

# ----------------- Header Section -----------------
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown('<h1 class="main-header">AutoML Genius 🤖</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <p style='font-size: 1.2rem; color: #7f8c8d;'>
        Upload your dataset and let AI build the perfect machine learning model for you!
        </p>
    </div>
    """, unsafe_allow_html=True)

# ----------------- Sidebar -----------------
with st.sidebar:
    st.markdown("""
    <div style='text-align: center;'>
        <h2>⚙️ Settings</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.info("""
    **How to use:**
    1. Upload your CSV dataset
    2. Select the target column to predict
    3. Let AutoML Genius analyze your data
    4. Train models and compare performance
    5. Download the best model
    """)
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center;'>
        <h4>🔧 Advanced Options</h4>
    </div>
    """, unsafe_allow_html=True)
    
    test_size = st.slider("Test Set Size", 0.1, 0.5, 0.2, 0.05)
    random_state = st.number_input("Random State", 0, 100, 42)

# ----------------- Main Content -----------------
tab1, tab2, tab3 = st.tabs(["📊 Data Upload", "🤖 Model Training", "📈 Results"])

with tab1:
    st.markdown('<h2 class="sub-header">Upload Your Dataset</h2>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", help="Upload your dataset in CSV format")
    
    if uploaded_file is not None:
        with st.spinner("Loading your data..."):
            df = pd.read_csv(uploaded_file)
            time.sleep(0.5)  # Simulate loading
            
        st.markdown('<div class="success-box">✅ Dataset loaded successfully!</div>', unsafe_allow_html=True)
        
        # Dataset overview
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown('<div class="metric-card"><div class="metric-label">Rows</div><div class="metric-value">' + str(df.shape[0]) + '</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card"><div class="metric-label">Columns</div><div class="metric-value">' + str(df.shape[1]) + '</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-card"><div class="metric-label">Missing Values</div><div class="metric-value">' + str(df.isnull().sum().sum()) + '</div></div>', unsafe_allow_html=True)
        with col4:
            st.markdown('<div class="metric-card"><div class="metric-label">Size</div><div class="metric-value">' + f"{uploaded_file.size / 1024:.1f} KB" + '</div></div>', unsafe_allow_html=True)
        
        # Data preview
        st.markdown('<h3 class="sub-header">Data Preview</h3>', unsafe_allow_html=True)
        st.dataframe(df.head(10), use_container_width=True)
        
        # Target selection
        st.markdown('<h3 class="sub-header">Select Target Variable</h3>', unsafe_allow_html=True)
        target_column = st.selectbox("Choose the column you want to predict", df.columns.tolist(), 
                                   help="This will be the variable that models will try to predict")
        
        if target_column:
            st.markdown(f'<div class="info-box">🎯 Selected target: <strong>{target_column}</strong></div>', unsafe_allow_html=True)
            
            # Target distribution visualization
            st.markdown('<h3 class="sub-header">Target Distribution</h3>', unsafe_allow_html=True)
            
            if df[target_column].dtype in ['object', 'category']:
                target_counts = df[target_column].value_counts()
                fig = px.pie(values=target_counts.values, names=target_counts.index, 
                            title=f'Distribution of {target_column}')
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig = px.histogram(df, x=target_column, title=f'Distribution of {target_column}',
                                  nbins=20, color_discrete_sequence=['#3498db'])
                st.plotly_chart(fig, use_container_width=True)
            
            # Data Analysis
            st.markdown('<h3 class="sub-header">Data Analysis</h3>', unsafe_allow_html=True)
            
            if st.button("🧠 Analyze Data", key="analyze_btn"):
                with st.spinner("🤖 Analyzing your data..."):
                    summary_text = generate_data_summary(df, target_column)
                    st.markdown('<div class="card">' + summary_text.replace('\n', '<br>') + '</div>', unsafe_allow_html=True)

with tab2:
    if 'df' not in locals() or 'target_column' not in locals():
        st.warning("Please upload data and select a target variable in the 'Data Upload' tab first.")
    else:
        st.markdown('<h2 class="sub-header">Model Training</h2>', unsafe_allow_html=True)
        
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Convert target if categorical
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
            st.markdown('<div class="info-box">📊 Target variable encoded using LabelEncoder</div>', unsafe_allow_html=True)

        # Identify numeric and categorical columns
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

        # Show feature information
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Numeric Features</div><div class="metric-value">{len(numeric_features)}</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Categorical Features</div><div class="metric-value">{len(categorical_features)}</div></div>', unsafe_allow_html=True)

        # Preprocessing
        with st.expander("🔧 Preprocessing Details", expanded=True):
            st.write("**Numeric features:**", ", ".join(numeric_features) if numeric_features else "None")
            st.write("**Categorical features:**", ", ".join(categorical_features) if categorical_features else "None")
            
            numeric_transformer = Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ])
            
            categorical_transformer = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])

            preprocessor = ColumnTransformer([
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

            # Process the data
            with st.spinner("Preprocessing data..."):
                X_processed = preprocessor.fit_transform(X)
                
                # Get feature names after preprocessing
                feature_names = []
                if numeric_features:
                    feature_names.extend(numeric_features)
                if categorical_features:
                    ct = preprocessor.named_transformers_['cat']
                    ohe = ct.named_steps['onehot']
                    cat_feature_names = ohe.get_feature_names_out(categorical_features)
                    feature_names.extend(cat_feature_names)
                
                X_processed_df = pd.DataFrame(X_processed, columns=feature_names)

            st.markdown('<div class="success-box">✅ Dataset preprocessed successfully!</div>', unsafe_allow_html=True)
            
            if st.checkbox("Show processed data"):
                st.dataframe(X_processed_df.head())

        # Model selection
        st.markdown('<h3 class="sub-header">Select Models to Train</h3>', unsafe_allow_html=True)
        
        models_config = {
            "Logistic Regression": {"model": LogisticRegression(max_iter=1000, random_state=random_state), "color": "#3498db"},
            "Random Forest": {"model": RandomForestClassifier(random_state=random_state, n_estimators=100), "color": "#e74c3c"},
            "Gradient Boosting": {"model": GradientBoostingClassifier(random_state=random_state, n_estimators=100), "color": "#2ecc71"}
        }
        
        cols = st.columns(len(models_config))
        selected_models = {}
        
        for i, (name, config) in enumerate(models_config.items()):
            with cols[i]:
                if st.checkbox(name, value=True, key=f"model_{i}"):
                    selected_models[name] = config

        if selected_models and st.button("🚀 Train Selected Models", type="primary"):
            # Check if we can use stratification (need at least 2 samples per class)
            unique_classes, class_counts = np.unique(y, return_counts=True)
            can_stratify = all(count > 1 for count in class_counts) and len(unique_classes) > 1
            
            # Split the data
            if can_stratify:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_processed, y, test_size=test_size, random_state=random_state, stratify=y
                )
                st.markdown('<div class="info-box">📊 Using stratified sampling for train-test split</div>', unsafe_allow_html=True)
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_processed, y, test_size=test_size, random_state=random_state
                )
                st.markdown('<div class="warning-box">⚠️ Cannot use stratified sampling (some classes have only 1 sample). Using random split.</div>', unsafe_allow_html=True)
            
            leaderboard = []
            results = {}
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Create placeholders for metrics
            metric_placeholders = {}
            for i, name in enumerate(selected_models.keys()):
                metric_placeholders[name] = st.empty()
            
            for i, (name, config) in enumerate(selected_models.items()):
                status_text.text(f"Training {name}...")
                
                # Display model card
                with metric_placeholders[name]:
                    st.markdown(f"""
                    <div class="model-card" style="border-left: 4px solid {config['color']};">
                        <h3 style="color: {config['color']}; margin-top: 0;">{name}</h3>
                        <p>Training in progress...</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                try:
                    model = config["model"]
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    # Calculate metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    
                    # Only calculate precision/recall/f1 if we have multiple classes
                    if len(np.unique(y_test)) > 1:
                        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                    else:
                        precision = recall = f1 = 0.0
                        st.markdown('<div class="warning-box">⚠️ Only one class in test set - precision/recall/f1 cannot be calculated</div>', unsafe_allow_html=True)
                    
                    leaderboard.append({
                        "Model": name,
                        "Accuracy": round(accuracy, 3),
                        "Precision": round(precision, 3),
                        "Recall": round(recall, 3),
                        "F1-Score": round(f1, 3)
                    })
                    
                    results[name] = {
                        "model": model,
                        "accuracy": accuracy,
                        "predictions": y_pred,
                        "y_test": y_test
                    }
                    
                    # Update model card with results
                    with metric_placeholders[name]:
                        st.markdown(f"""
                        <div class="model-card" style="border-left: 4px solid {config['color']};">
                            <h3 style="color: {config['color']}; margin-top: 0;">{name}</h3>
                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                                <div style="background: #f8f9fa; padding: 10px; border-radius: 5px;">
                                    <div style="font-size: 0.8rem; color: #7f8c8d;">Accuracy</div>
                                    <div style="font-size: 1.5rem; font-weight: bold; color: {config['color']};">{accuracy:.3f}</div>
                                </div>
                                <div style="background: #f8f9fa; padding: 10px; border-radius: 5px;">
                                    <div style="font-size: 0.8rem; color: #7f8c8d;">F1-Score</div>
                                    <div style="font-size: 1.5rem; font-weight: bold; color: {config['color']};">{f1:.3f}</div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.warning(f"{name} training failed: {e}")
                
                progress_bar.progress((i + 1) / len(selected_models))
            
            status_text.text("Training complete!")
            st.balloons()
            
            # Store results in session state for the next tab
            st.session_state.leaderboard = leaderboard
            st.session_state.results = results
            st.session_state.preprocessor = preprocessor
            st.session_state.X = X
            st.session_state.y = y
            st.session_state.best_model_name = leaderboard[0]["Model"] if leaderboard else None

with tab3:
    if 'leaderboard' not in st.session_state:
        st.warning("Please train models in the 'Model Training' tab first.")
    else:
        st.markdown('<h2 class="sub-header">Results & Analysis</h2>', unsafe_allow_html=True)
        
        leaderboard = st.session_state.leaderboard
        results = st.session_state.results
        
        # Leaderboard
        st.markdown('<h3 class="sub-header">Model Leaderboard</h3>', unsafe_allow_html=True)
        leaderboard_df = pd.DataFrame(leaderboard).sort_values(by="Accuracy", ascending=False)
        
        # Create a visual comparison
        fig = go.Figure()
        for metric in ["Accuracy", "Precision", "Recall", "F1-Score"]:
            fig.add_trace(go.Bar(
                name=metric,
                x=leaderboard_df["Model"],
                y=leaderboard_df[metric],
                text=leaderboard_df[metric],
                textposition='auto',
            ))
        
        fig.update_layout(
            barmode='group',
            title="Model Performance Comparison",
            xaxis_title="Models",
            yaxis_title="Score",
            yaxis_range=[0, 1] if leaderboard_df["Accuracy"].max() <= 1 else None
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display leaderboard as table
        st.dataframe(leaderboard_df.style.format({
            "Accuracy": "{:.3f}",
            "Precision": "{:.3f}",
            "Recall": "{:.3f}",
            "F1-Score": "{:.3f}"
        }).highlight_max(subset=["Accuracy", "F1-Score"], color='lightgreen'), use_container_width=True)
        
        # Best model
        best_model_name = st.session_state.best_model_name
        best_model = results[best_model_name]["model"]
        
        st.markdown(f'<div class="success-box">🏆 Best model: <strong>{best_model_name}</strong> (Accuracy: {leaderboard_df.iloc[0]["Accuracy"]:.3f})</div>', unsafe_allow_html=True)
        
        # Confusion matrix for the best model
        st.markdown('<h3 class="sub-header">Confusion Matrix</h3>', unsafe_allow_html=True)
        y_test = results[best_model_name]["y_test"]
        y_pred = results[best_model_name]["predictions"]
        
        cm = confusion_matrix(y_test, y_pred)
        fig = px.imshow(cm, 
                       labels=dict(x="Predicted", y="Actual", color="Count"),
                       x=[f"Class {i}" for i in range(cm.shape[0])],
                       y=[f"Class {i}" for i in range(cm.shape[0])],
                       title=f"Confusion Matrix for {best_model_name}")
        st.plotly_chart(fig, use_container_width=True)
        
        # Download best model
        st.markdown('<h3 class="sub-header">Download Model</h3>', unsafe_allow_html=True)
        
        # Create and save the full pipeline
        preprocessor = st.session_state.preprocessor
        X = st.session_state.X
        y = st.session_state.y
        
        final_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', best_model)
        ])
        
        # Retrain on full data
        final_pipeline.fit(X, y)
        
        # Save the model
        joblib.dump(final_pipeline, "best_model_pipeline.pkl")
        
        # Provide download button
        with open("best_model_pipeline.pkl", "rb") as f:
            st.download_button(
                label="📥 Download Best Model",
                data=f,
                file_name="best_model_pipeline.pkl",
                mime="application/octet-stream",
                use_container_width=True
            )
        
        # Insights
        st.markdown('<h3 class="sub-header">Model Insights</h3>', unsafe_allow_html=True)
        
        if st.button("💡 Generate Model Insights"):
            with st.spinner("🤖 Generating insights..."):
                insights = generate_model_insights(leaderboard_df, best_model_name)
                st.markdown('<div class="card">' + insights.replace('\n', '<br>') + '</div>', unsafe_allow_html=True)

# ----------------- Footer -----------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d;'>
    <p>Built with ❤️ using Streamlit</p>
    <p>AutoML Genius 🤖 - Making Machine Learning Accessible to Everyone</p>
</div>
""", unsafe_allow_html=True)