import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# Set Page configuration
st.set_page_config(
    page_title="Spotify Song Recommender",
    page_icon="🎵",
    layout="wide"
)

# Custom CSS to improve styling
st.markdown("""
<style>
    .main-title {
        font-size: 45px;
        font-weight: bold;
        color: #1DB954;
        margin-bottom: 5px;
    }
    .subtitle {
        font-size: 18px;
        color: #888888;
        margin-bottom: 25px;
    }
    .metric-card {
        background-color: #111111;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #222222;
    }
</style>
""", unsafe_allow_html=True)

# Title and Header
st.markdown('<div class="main-title">🎵 Spotify AI Song Recommender</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">An interactive, machine-learning-powered song discoverer built with Spotify audio features.</div>', unsafe_allow_html=True)

# Load dataset and cache it
@st.cache_data
def load_data():
    df = pd.read_csv("spotify_songs.csv")
    df = df.dropna(subset=["track_name", "track_artist", "track_album_name"])
    df_unique = df.drop_duplicates(subset=['track_name', 'track_artist']).reset_index(drop=True)
    return df_unique

try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Define audio features
audio_features = [
    'danceability', 'energy', 'loudness', 'speechiness', 
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
]

# Fit scaler and model
@st.cache_resource
def train_recommender(data):
    scaler = StandardScaler()
    feature_matrix = scaler.fit_transform(data[audio_features])
    
    model = NearestNeighbors(n_neighbors=21, metric='cosine', algorithm='brute')
    model.fit(feature_matrix)
    return model, feature_matrix, scaler

model, feature_matrix, scaler = train_recommender(df)

# Sidebar Navigation
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/1/19/Spotify_logo_without_text.svg", width=60)
st.sidebar.title("Navigation & Settings")

search_mode = st.sidebar.radio("Choose Recommendation Method:", ["Search by Song Name", "Design Your Own Song Vibes"])
num_recommendations = st.sidebar.slider("Number of Recommendations to Show", 5, 20, 10)

query_vector = None
target_features = {}

# Search by Song Name Mode
if search_mode == "Search by Song Name":
    st.sidebar.markdown("--- ")
    st.sidebar.markdown("### 🔍 Find Your Track")
    
    # Optional filtering by artist to make searching easier
    all_artists = sorted(df['track_artist'].unique())
    selected_artist = st.sidebar.selectbox("Filter by Artist (Optional)", ["All"] + list(all_artists))
    
    if selected_artist != "All":
        filtered_songs = sorted(df[df['track_artist'] == selected_artist]['track_name'].unique())
    else:
        filtered_songs = sorted(df['track_name'].unique())
        
    selected_song = st.sidebar.selectbox("Select a Song", filtered_songs)
    
    if selected_song:
        song_entry = df[df['track_name'] == selected_song].iloc[0]
        query_index = df[df['track_name'] == selected_song].index[0]
        query_vector = feature_matrix[query_index].reshape(1, -1)
        
        # Create interactive visual layout for Selected Track Details
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("Selected Track")
            st.success(f"**Title:** {song_entry['track_name']}\n\n**Artist:** {song_entry['track_artist']}\n\n**Genre:** {song_entry['playlist_genre'].title()} ({song_entry['playlist_subgenre']})")
            
        with col2:
            st.subheader("Acoustic Signature")
            # Visualize using horizontal progress indicators via st.progress
            for feat in ['danceability', 'energy', 'acousticness', 'valence']:
                val = float(song_entry[feat])
                st.write(f"**{feat.title()}**: `{val:.3f}`")
                st.progress(val)
                
        for f in audio_features:
            target_features[f] = float(song_entry[f])

# Design Custom Vibes Mode
else:
    st.sidebar.markdown("--- ")
    st.sidebar.markdown("### 🎛️ Tune Song Attributes")
    
    # Set target features using sliders
    target_features['danceability'] = st.sidebar.slider("Danceability (Beaty/Groovy)", 0.0, 1.0, 0.5, step=0.01)
    target_features['energy'] = st.sidebar.slider("Energy (Intensity/Speed)", 0.0, 1.0, 0.6, step=0.01)
    target_features['loudness'] = st.sidebar.slider("Loudness (dB)", -45.0, 2.0, -6.0, step=0.1)
    target_features['speechiness'] = st.sidebar.slider("Speechiness (Presence of Spoken Words)", 0.0, 1.0, 0.1, step=0.01)
    target_features['acousticness'] = st.sidebar.slider("Acousticness (Acoustic vs. Electronic)", 0.0, 1.0, 0.2, step=0.01)
    target_features['instrumentalness'] = st.sidebar.slider("Instrumentalness", 0.0, 1.0, 0.0, step=0.01)
    target_features['liveness'] = st.sidebar.slider("Liveness (Live Concert Feel)", 0.0, 1.0, 0.15, step=0.01)
    target_features['valence'] = st.sidebar.slider("Valence (Emotional Positiveness/Happiness)", 0.0, 1.0, 0.5, step=0.01)
    target_features['tempo'] = st.sidebar.slider("Tempo (Beats Per Minute)", 50.0, 220.0, 120.0, step=1.0)
    
    custom_features_df = pd.DataFrame([target_features])[audio_features]
    query_vector = scaler.transform(custom_features_df).reshape(1, -1)
    
    st.subheader("Your Custom Target Profile")
    
    # Create standard key metrics row
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("Danceability", f"{target_features['danceability']:.2%}")
    m_col2.metric("Energy", f"{target_features['energy']:.2%}")
    m_col3.metric("Valence (Mood)", f"{target_features['valence']:.2%}")
    m_col4.metric("Tempo (BPM)", f"{int(target_features['tempo'])}")

# Run ML Recommendations Engine
if query_vector is not None:
    st.markdown("--- ")
    st.subheader(f"🚀 Top {num_recommendations} Recommendations for You")
    
    n_neighbors_to_query = num_recommendations + 1 if search_mode == "Search by Song Name" else num_recommendations
    distances, indices = model.kneighbors(query_vector, n_neighbors=n_neighbors_to_query)
    
    if search_mode == "Search by Song Name":
        rec_indices = indices.flatten()[1:]
        rec_distances = distances.flatten()[1:]
    else:
        rec_indices = indices.flatten()[:num_recommendations]
        rec_distances = distances.flatten()[:num_recommendations]
        
    recommendations_df = df.iloc[rec_indices].copy()
    recommendations_df['Match Distance'] = rec_distances
    
    # Make recommendations interactive via tabs: List View vs Deep Feature Breakdown View
    tab1, tab2 = st.tabs(["📋 Recommendations List", "📊 Features Comparative View"])
    
    with tab1:
        display_df = recommendations_df[[
            'track_name', 'track_artist', 'playlist_genre', 'playlist_subgenre', 'Match Distance'
        ]].rename(columns={
            'track_name': 'Song Title',
            'track_artist': 'Artist',
            'playlist_genre': 'Genre',
            'playlist_subgenre': 'Subgenre',
            'Match Distance': 'Similarity Distance (Lower is closer)'
        }).reset_index(drop=True)
        
        st.dataframe(
            display_df.style.background_gradient(subset=['Similarity Distance (Lower is closer)'], cmap='viridis_r').format({
                'Similarity Distance (Lower is closer)': '{:.4f}'
            }),
            use_container_width=True
        )
        
    with tab2:
        st.markdown("**Compare your target profile with the averages of your recommended songs:**")
        rec_means = recommendations_df[audio_features].mean()
        
        comp_cols = st.columns(len(audio_features))
        for idx, feat in enumerate(audio_features):
            with comp_cols[idx]:
                target_val = target_features[feat]
                rec_val = rec_means[feat]
                diff = rec_val - target_val
                
                # If it's tempo or loudness, display absolute difference differently
                if feat in ['tempo', 'loudness']:
                    st.metric(label=feat.title(), value=f"{rec_val:.1f}", delta=f"{diff:.1f} vs Target")
                else:
                    st.metric(label=feat.title(), value=f"{rec_val:.1%}", delta=f"{diff:.1%} vs Target")