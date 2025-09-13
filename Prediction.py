import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from EDA import load_and_explore_data, prepare_data_for_ml

def train_multiple_models(X, y):
    """Train multiple ML models and compare performance"""
    print("\n🤖 Training Multiple Machine Learning Models...")
    print("=" * 60)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features for models that need it
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Ridge Classifier': Ridge(random_state=42),
        'Lasso Classifier': Lasso(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100),
        'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss')
    }
    
    results = {}
    trained_models = {}
    
    print("Training models...")
    for name, model in models.items():
        print(f"\n🔄 Training {name}...")
        
        try:
            if name in ['Ridge Classifier', 'Lasso Classifier']:
                # These are regression models, we'll adapt them for classification
                if name == 'Ridge Classifier':
                    model = LogisticRegression(penalty='l2', C=1.0, random_state=42, max_iter=1000)
                else:  # Lasso
                    model = LogisticRegression(penalty='l1', solver='liblinear', C=1.0, random_state=42, max_iter=1000)
            
            # Use scaled data for linear models
            if name in ['Logistic Regression', 'Ridge Classifier', 'Lasso Classifier']:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                # Cross-validation with scaled data
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                # Cross-validation with original data
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            results[name] = {
                'accuracy': accuracy,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'predictions': y_pred
            }
            
            trained_models[name] = model
            
            print(f"✅ {name}:")
            print(f"   Test Accuracy: {accuracy:.4f}")
            print(f"   CV Score: {cv_mean:.4f} (+/- {cv_std * 2:.4f})")
            
        except Exception as e:
            print(f"❌ Error training {name}: {str(e)}")
            continue
    
    return results, trained_models, X_test, y_test, scaler

def find_best_model(results):
    """Find the best performing model"""
    print("\n🏆 Model Performance Comparison:")
    print("=" * 50)
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Test_Accuracy': [results[model]['accuracy'] for model in results.keys()],
        'CV_Mean': [results[model]['cv_mean'] for model in results.keys()],
        'CV_Std': [results[model]['cv_std'] for model in results.keys()]
    })
    
    comparison_df = comparison_df.sort_values('Test_Accuracy', ascending=False)
    print(comparison_df.to_string(index=False))
    
    # Visualize results
    plt.figure(figsize=(12, 8))
    
    # Accuracy comparison
    plt.subplot(2, 1, 1)
    plt.bar(comparison_df['Model'], comparison_df['Test_Accuracy'], color='skyblue', alpha=0.7)
    plt.title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('Test Accuracy')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    # Add accuracy values on bars
    for i, v in enumerate(comparison_df['Test_Accuracy']):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    # Cross-validation scores with error bars
    plt.subplot(2, 1, 2)
    plt.errorbar(range(len(comparison_df)), comparison_df['CV_Mean'], 
                yerr=comparison_df['CV_Std'], fmt='o', capsize=5, capthick=2)
    plt.title('Cross-Validation Scores with Standard Deviation', fontsize=14, fontweight='bold')
    plt.ylabel('CV Accuracy')
    plt.xlabel('Models')
    plt.xticks(range(len(comparison_df)), comparison_df['Model'], rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Find best model
    best_model_name = comparison_df.iloc[0]['Model']
    best_accuracy = comparison_df.iloc[0]['Test_Accuracy']
    
    print(f"\n🥇 Best Model: {best_model_name}")
    print(f"🎯 Best Accuracy: {best_accuracy:.4f}")
    
    return best_model_name, comparison_df

def detailed_model_analysis(best_model_name, trained_models, results, X_test, y_test):
    """Perform detailed analysis of the best model"""
    print(f"\n🔬 Detailed Analysis of {best_model_name}:")
    print("=" * 50)
    
    best_model = trained_models[best_model_name]
    y_pred = results[best_model_name]['predictions']
    
    # Classification Report
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Movie', 'TV Show']))
    
    # Confusion Matrix
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Movie', 'TV Show'], 
                yticklabels=['Movie', 'TV Show'])
    plt.title(f'Confusion Matrix - {best_model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Feature Importance (if available)
    plt.subplot(1, 2, 2)
    if hasattr(best_model, 'feature_importances_'):
        feature_names = ['Duration', 'Release Year', 'Description Length', 
                        'Genre', 'Country', 'Rating']
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.bar(range(len(importances)), importances[indices])
        plt.title(f'Feature Importance - {best_model_name}')
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
        plt.ylabel('Importance')
    elif hasattr(best_model, 'coef_'):
        feature_names = ['Duration', 'Release Year', 'Description Length', 
                        'Genre', 'Country', 'Rating']
        coef = np.abs(best_model.coef_[0])
        indices = np.argsort(coef)[::-1]
        
        plt.bar(range(len(coef)), coef[indices])
        plt.title(f'Feature Coefficients - {best_model_name}')
        plt.xticks(range(len(coef)), [feature_names[i] for i in indices], rotation=45)
        plt.ylabel('|Coefficient|')
    else:
        plt.text(0.5, 0.5, 'Feature importance\nnot available\nfor this model', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Feature Analysis')
    
    plt.tight_layout()
    plt.show()
    
    return best_model

def save_best_model(best_model, best_model_name, scaler):
    """Save the best model and scaler for deployment"""
    print(f"\n💾 Saving {best_model_name} for deployment...")
    
    # Save model
    joblib.dump(best_model, 'best_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    
    # Save model info
    model_info = {
        'model_name': best_model_name,
        'features': ['duration_numeric', 'release_year', 'description_length', 
                    'genre_encoded', 'country_encoded', 'rating_encoded'],
        'target_mapping': {0: 'Movie', 1: 'TV Show'}
    }
    
    import json
    with open('model_info.json', 'w') as f:
        json.dump(model_info, f)
    
    print("✅ Model saved successfully!")
    print("Files created:")
    print("- best_model.pkl")
    print("- scaler.pkl") 
    print("- model_info.json")

def demonstrate_prediction(best_model, scaler, best_model_name):
    """Demonstrate model prediction with examples"""
    print(f"\n🎯 Model Prediction Demonstration:")
    print("=" * 40)
    
    # Example predictions
    examples = [
        {
            'duration_numeric': 120,  # 120 minutes
            'release_year': 2020,
            'description_length': 150,
            'genre_encoded': 0,  # Drama
            'country_encoded': 0,  # US
            'rating_encoded': 2   # R
        },
        {
            'duration_numeric': 300,  # 5 seasons * 60 min
            'release_year': 2018,
            'description_length': 200,
            'genre_encoded': 1,  # Comedy
            'country_encoded': 0,  # US
            'rating_encoded': 1   # TV-14
        }
    ]
    
    for i, example in enumerate(examples, 1):
        # Prepare input
        input_data = np.array([[
            example['duration_numeric'],
            example['release_year'],
            example['description_length'],
            example['genre_encoded'],
            example['country_encoded'],
            example['rating_encoded']
        ]])
        
        # Scale if needed
        if best_model_name in ['Logistic Regression', 'Ridge Classifier', 'Lasso Classifier']:
            input_data = scaler.transform(input_data)
        
        # Predict
        prediction = best_model.predict(input_data)[0]
        probability = best_model.predict_proba(input_data)[0] if hasattr(best_model, 'predict_proba') else None
        
        print(f"\nExample {i}:")
        print(f"Input: Duration={example['duration_numeric']}min, Year={example['release_year']}")
        print(f"Prediction: {'TV Show' if prediction == 1 else 'Movie'}")
        if probability is not None:
            print(f"Confidence: Movie={probability[0]:.2f}, TV Show={probability[1]:.2f}")

if __name__ == "__main__":
    # Load and prepare data
    print("🎬 Netflix ML Model Training Pipeline")
    print("=" * 50)
    
    df = load_and_explore_data()
    X, y, ml_df, le_genre, le_country, le_rating = prepare_data_for_ml(df)
    
    # Train models
    results, trained_models, X_test, y_test, scaler = train_multiple_models(X, y)
    
    # Find best model
    best_model_name, comparison_df = find_best_model(results)
    
    # Detailed analysis
    best_model = detailed_model_analysis(best_model_name, trained_models, results, X_test, y_test)
    
    # Save model
    save_best_model(best_model, best_model_name, scaler)
    
    # Demonstrate predictions
    demonstrate_prediction(best_model, scaler, best_model_name)
    
    print("\n🎉 Model training complete!")
    print("Ready to deploy with Streamlit app!")
