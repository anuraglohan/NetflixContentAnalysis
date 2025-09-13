# 🎬 Netflix Content Evolution & Prediction Analysis

A comprehensive data science project analyzing Netflix's content evolution and building machine learning models to predict content types.

## 📋 Project Overview

This project performs:
- **Exploratory Data Analysis (EDA)** of Netflix content trends
- **Machine Learning Model Training** with 5+ algorithms
- **Interactive Streamlit Dashboard** for predictions and analytics

## 🚀 Features

### 📊 Analytics Dashboard
- Content growth visualization over years
- Genre distribution analysis
- Country-wise production insights
- Interactive charts and metrics

### 🤖 ML Predictor
- Predict Movie vs TV Show based on metadata
- Real-time predictions with confidence scores
- Feature importance analysis
- Model reasoning explanations

### 📈 Performance Analysis
- Model comparison across 5+ algorithms
- Accuracy, precision, and recall metrics
- Feature importance visualization
- Cross-validation results

## 🛠️ Technologies Used

- **Python**: Core programming language
- **Pandas & NumPy**: Data manipulation and analysis
- **Scikit-learn**: Machine learning algorithms
- **XGBoost**: Gradient boosting framework
- **Matplotlib & Seaborn**: Static visualizations
- **Plotly**: Interactive charts
- **Streamlit**: Web application framework

## 📁 Project Structure

\`\`\`
netflix-analysis/
├── scripts/
│   ├── netflix_analysis.py      # EDA and data preparation
│   └── model_training.py        # ML model training pipeline
├── streamlit_app.py             # Interactive web application
├── requirements.txt             # Python dependencies
└── README.md                   # Project documentation
\`\`\`

## 🏃‍♂️ How to Run

### Step 1: Install Dependencies
\`\`\`bash
pip install -r requirements.txt
\`\`\`

### Step 2: Run Data Analysis
\`\`\`bash
python scripts/netflix_analysis.py
\`\`\`

### Step 3: Train ML Models
\`\`\`bash
python scripts/model_training.py
\`\`\`

### Step 4: Launch Streamlit App
\`\`\`bash
streamlit run streamlit_app.py
\`\`\`

## 🤖 Machine Learning Models

The project trains and compares 7 different models:

1. **Logistic Regression** - Linear baseline model
2. **Ridge Regression** - L2 regularized linear model
3. **Lasso Regression** - L1 regularized linear model
4. **Decision Tree** - Tree-based classifier
5. **Random Forest** - Ensemble of decision trees
6. **Gradient Boosting** - Sequential boosting algorithm
7. **XGBoost** - Optimized gradient boosting

### 🎯 Prediction Task
- **Objective**: Predict whether content is a Movie (0) or TV Show (1)
- **Features**: Duration, Release Year, Description Length, Genre, Country, Rating
- **Expected Accuracy**: 80-87% (varies by model)

## 📊 Key Insights

- **Content Growth**: Exponential increase in Netflix content since 2015
- **Format Evolution**: Movies dominate but TV shows are growing rapidly
- **Global Expansion**: Production expanded beyond US to international markets
- **Genre Diversity**: Drama and Comedy lead, but increasing diversification

## 🎨 Streamlit App Features

### Dashboard Tabs:
1. **Analytics Dashboard**: Interactive visualizations and key metrics
2. **ML Predictor**: Real-time content type prediction
3. **Model Performance**: Detailed model comparison and analysis

### Interactive Elements:
- Sliders for numerical inputs
- Dropdown menus for categorical features
- Real-time prediction updates
- Confidence score visualization
- Model reasoning explanations

## 🔧 Customization

### Adding New Features:
1. Modify `prepare_data_for_ml()` in `netflix_analysis.py`
2. Update feature lists in `model_training.py`
3. Adjust input fields in `streamlit_app.py`

### Adding New Models:
1. Add model to `models` dictionary in `train_multiple_models()`
2. Handle any special preprocessing requirements
3. Update model comparison visualizations

## 📈 Performance Metrics

The best performing models typically achieve:
- **Accuracy**: 85-87%
- **Precision**: 86-88%
- **Recall**: 84-86%
- **F1-Score**: 85-87%

## 🎓 Learning Outcomes

This project demonstrates:
- **Data Science Pipeline**: From EDA to deployment
- **Model Comparison**: Systematic evaluation of multiple algorithms
- **Feature Engineering**: Creating meaningful predictors
- **Web Deployment**: Building interactive applications
- **Visualization**: Creating compelling data stories

## 🤝 Contributing

Feel free to contribute by:
- Adding new visualization types
- Implementing additional ML models
- Enhancing the Streamlit interface
- Improving model performance

## 📄 License

This project is open source and available under the MIT License.

---

**Built with ❤️ for data science learning and Netflix content analysis!**
