import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_explore_data():
    """Load Netflix dataset and perform initial exploration"""
    print("🎬 Netflix Content Evolution Analysis")
    print("=" * 50)
    
    # For this demo, we'll create a sample dataset similar to Netflix data
    # In real scenario, you'd load from: df = pd.read_csv('netflix_titles.csv')
    
    # Create sample Netflix-like data
    np.random.seed(42)
    n_samples = 8800
    
    # Generate sample data with proper probability distribution
    year_probs = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17]
    # Normalize probabilities to ensure they sum to 1.0
    year_probs = np.array(year_probs)
    year_probs = year_probs / year_probs.sum()
    
    years = np.random.choice(range(2008, 2024), n_samples, p=year_probs)
    
    types = np.random.choice(['Movie', 'TV Show'], n_samples, p=[0.7, 0.3])
    
    genres = ['Drama', 'Comedy', 'Action', 'Horror', 'Romance', 'Thriller', 'Documentary', 'Sci-Fi', 'Animation', 'Crime']
    genre_list = np.random.choice(genres, n_samples)
    
    countries = ['United States', 'India', 'United Kingdom', 'Canada', 'France', 'Japan', 'South Korea', 'Spain', 'Germany', 'Australia']
    country_list = np.random.choice(countries, n_samples)
    
    # Duration logic: Movies (80-180 mins), TV Shows (1-10 seasons)
    durations = []
    for t in types:
        if t == 'Movie':
            durations.append(f"{np.random.randint(80, 181)} min")
        else:
            durations.append(f"{np.random.randint(1, 11)} Season{'s' if np.random.randint(1, 11) > 1 else ''}")
    
    # Create DataFrame
    df = pd.DataFrame({
        'title': [f'Title_{i}' for i in range(n_samples)],
        'type': types,
        'release_year': years,
        'duration': durations,
        'listed_in': genre_list,
        'country': country_list,
        'description': [f'Description for title {i}' for i in range(n_samples)],
        'rating': np.random.choice(['TV-MA', 'TV-14', 'R', 'PG-13', 'TV-PG', 'PG', 'G'], n_samples)
    })
    
    print(f"📊 Dataset Overview:")
    print(f"Total titles: {len(df)}")
    print(f"Movies: {len(df[df['type'] == 'Movie'])}")
    print(f"TV Shows: {len(df[df['type'] == 'TV Show'])}")
    print(f"Year range: {df['release_year'].min()} - {df['release_year'].max()}")
    
    return df

def perform_eda(df):
    """Perform comprehensive Exploratory Data Analysis"""
    print("\n🔍 Performing Exploratory Data Analysis...")
    
    # 1. Content Growth Over Years
    plt.figure(figsize=(15, 10))
    
    # Movies vs TV Shows over time
    plt.subplot(2, 2, 1)
    yearly_content = df.groupby(['release_year', 'type']).size().unstack(fill_value=0)
    yearly_content.plot(kind='line', marker='o', linewidth=2)
    plt.title('Netflix Content Growth: Movies vs TV Shows', fontsize=14, fontweight='bold')
    plt.xlabel('Release Year')
    plt.ylabel('Number of Titles')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Genre Distribution
    plt.subplot(2, 2, 2)
    genre_counts = df['listed_in'].value_counts().head(10)
    colors = plt.cm.Set3(np.linspace(0, 1, len(genre_counts)))
    plt.pie(genre_counts.values, labels=genre_counts.index, autopct='%1.1f%%', colors=colors)
    plt.title('Top 10 Genres Distribution', fontsize=14, fontweight='bold')
    
    # 3. Country-wise Production
    plt.subplot(2, 2, 3)
    country_counts = df['country'].value_counts().head(10)
    plt.barh(country_counts.index, country_counts.values, color='skyblue')
    plt.title('Top 10 Countries by Content Production', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Titles')
    
    # 4. Content Type Distribution
    plt.subplot(2, 2, 4)
    type_counts = df['type'].value_counts()
    plt.bar(type_counts.index, type_counts.values, color=['#ff9999', '#66b3ff'])
    plt.title('Movies vs TV Shows Distribution', fontsize=14, fontweight='bold')
    plt.ylabel('Number of Titles')
    
    plt.tight_layout()
    plt.show()
    
    # Advanced Visualizations with Plotly
    print("\n📈 Creating Interactive Visualizations...")
    
    # Stacked Area Chart for Content Evolution
    yearly_content_cumsum = yearly_content.cumsum()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=yearly_content_cumsum.index, 
        y=yearly_content_cumsum['Movie'],
        fill='tonexty',
        mode='lines',
        name='Movies',
        line=dict(color='#ff6b6b')
    ))
    fig.add_trace(go.Scatter(
        x=yearly_content_cumsum.index, 
        y=yearly_content_cumsum['TV Show'],
        fill='tonexty',
        mode='lines',
        name='TV Shows',
        line=dict(color='#4ecdc4')
    ))
    
    fig.update_layout(
        title='Netflix Content Evolution: Cumulative Growth',
        xaxis_title='Year',
        yaxis_title='Cumulative Number of Titles',
        hovermode='x unified'
    )
    fig.show()
    
    # Heatmap of Content by Year and Genre
    plt.figure(figsize=(12, 8))
    year_genre = df.groupby(['release_year', 'listed_in']).size().unstack(fill_value=0)
    sns.heatmap(year_genre.T, cmap='YlOrRd', cbar_kws={'label': 'Number of Titles'})
    plt.title('Content Heatmap: Year vs Genre', fontsize=16, fontweight='bold')
    plt.xlabel('Release Year')
    plt.ylabel('Genre')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return df

def prepare_data_for_ml(df):
    """Prepare data for machine learning models"""
    print("\n🤖 Preparing Data for Machine Learning...")
    
    # Create a copy for ML processing
    ml_df = df.copy()
    
    # Feature Engineering
    # 1. Extract duration in minutes for movies, seasons for TV shows
    ml_df['duration_numeric'] = ml_df['duration'].apply(lambda x: 
        int(x.split()[0]) if 'min' in x else int(x.split()[0]) * 60  # Convert seasons to approximate minutes
    )
    
    # 2. Description length
    ml_df['description_length'] = ml_df['description'].str.len()
    
    # 3. Encode categorical variables
    from sklearn.preprocessing import LabelEncoder
    
    le_genre = LabelEncoder()
    ml_df['genre_encoded'] = le_genre.fit_transform(ml_df['listed_in'])
    
    le_country = LabelEncoder()
    ml_df['country_encoded'] = le_country.fit_transform(ml_df['country'])
    
    le_rating = LabelEncoder()
    ml_df['rating_encoded'] = le_rating.fit_transform(ml_df['rating'])
    
    # 4. Target variable (Movie=0, TV Show=1)
    ml_df['target'] = (ml_df['type'] == 'TV Show').astype(int)
    
    # Select features for modeling
    features = ['duration_numeric', 'release_year', 'description_length', 
                'genre_encoded', 'country_encoded', 'rating_encoded']
    
    X = ml_df[features]
    y = ml_df['target']
    
    print(f"✅ Features prepared: {features}")
    print(f"✅ Target distribution: {y.value_counts().to_dict()}")
    
    return X, y, ml_df, le_genre, le_country, le_rating

if __name__ == "__main__":
    # Execute the analysis
    df = load_and_explore_data()
    df = perform_eda(df)
    X, y, ml_df, le_genre, le_country, le_rating = prepare_data_for_ml(df)
    
    print("\n🎯 EDA Complete! Ready for Machine Learning...")
    print("Next step: Run model_training.py to train ML models")
