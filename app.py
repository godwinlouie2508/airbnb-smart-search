import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Airbnb Smart Search & Price Intelligence",
    page_icon="Airbnb-Emblem.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# AUTHENTIC AIRBNB STYLING
# Based on official Airbnb Design System (airbnb.design)
# Colors: Rausch (#FF385C), Babu (#00A699), Arches (#FC642D),
#         Hof (#484848), Foggy (#767676)
# Typography: Airbnb Cereal (fallback to Circular, system fonts)
# ============================================================================
st.markdown("""
    <style>
    /* Import fonts - Airbnb Cereal alternatives */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global Styles */
    * {
        font-family: 'Inter', 'Circular', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    }

    /* Main background */
    .main {
        background: #FFFFFF;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1440px;
    }

    /* Sidebar - Airbnb style */
    [data-testid="stSidebar"] {
        background: #F7F7F7;
        border-right: 1px solid #DDDDDD;
        padding-top: 2rem;
    }

    [data-testid="stSidebar"] .stMarkdown h2 {
        color: #222222;
        font-weight: 700;
        font-size: 1.5rem;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }

    [data-testid="stSidebar"] .stMarkdown p {
        color: #717171;
        font-size: 0.875rem;
        line-height: 1.5;
    }

    /* Title - Airbnb style */
    .airbnb-title {
        font-size: 3rem;
        font-weight: 800;
        color: #222222;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.03em;
        line-height: 1.1;
    }

    .airbnb-subtitle {
        font-size: 1.125rem;
        color: #717171;
        text-align: center;
        margin-bottom: 2.5rem;
        font-weight: 400;
        line-height: 1.5;
    }

    /* Search Box - Airbnb style with fully rounded corners */
    .stTextInput > div > div > input {
        border: 1px solid #DDDDDD !important;
        border-radius: 50px !important;
        padding: 16px 24px !important;
        font-size: 1rem !important;
        color: #222222 !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08) !important;
        transition: all 0.2s ease !important;
    }

    .stTextInput > div > div > input:focus {
        border-color: #222222 !important;
        box-shadow: 0 4px 16px rgba(0,0,0,0.12) !important;
        outline: none !important;
    }

    /* Target all wrapper divs for rounded corners */
    .stTextInput > div > div {
        border-radius: 50px !important;
        overflow: hidden !important;
    }

    .stTextInput > div {
        border-radius: 50px !important;
    }

    /* Remove any conflicting borders */
    .stTextInput input {
        border-radius: 50px !important;
    }

    .stTextInput > label {
        font-size: 0.875rem;
        font-weight: 600;
        color: #222222;
        margin-bottom: 8px;
    }

    /* Hide "Press Enter to apply" instruction */
    .stTextInput [data-testid="InputInstructions"] {
        display: none !important;
    }

    .stTextInput div[class*="InputInstructions"] {
        display: none !important;
    }

    /* Alternative selector for instruction text */
    .stTextInput > div > div > div[class*="instruction"] {
        display: none !important;
    }

    /* Match search input height with button */
    .stTextInput > div > div > input {
        height: 48px !important;
        padding: 12px 24px !important;
        box-sizing: border-box !important;
    }

    /* Remove extra spacing from input wrapper */
    .stTextInput > div {
        margin-bottom: 0 !important;
    }

    .stTextInput > div > div {
        min-height: 48px !important;
        max-height: 48px !important;
        height: 48px !important;
    }


    /* Radio buttons - Airbnb style */
    .stRadio > label {
        font-size: 0.875rem;
        font-weight: 600;
        color: #222222;
        margin-bottom: 8px;
    }

    .stRadio > div {
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
    }

    .stRadio > div > label {
        background: #FFFFFF;
        border: 1px solid #DDDDDD;
        border-radius: 24px;
        padding: 10px 20px;
        cursor: pointer;
        transition: all 0.2s ease;
        font-weight: 500;
        color: #222222;
    }

    .stRadio > div > label:hover {
        border-color: #222222;
        background: #F7F7F7;
    }

    .stRadio > div > label[data-baseweb="radio"] > div:first-child {
        display: none;
    }

    /* Use monospace font for price numbers in radio buttons for better alignment */
    .stRadio > div > label > div:last-child {
        font-variant-numeric: tabular-nums;
        letter-spacing: 0.02em;
    }

    /* Selectbox - Airbnb style */
    .stSelectbox > label {
        font-size: 0.875rem;
        font-weight: 600;
        color: #222222;
        margin-bottom: 8px;
    }

    .stSelectbox > div > div {
        border: 1px solid #DDDDDD;
        border-radius: 8px;
        padding: 4px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.08);
    }

    /* Ensure selectbox text is visible */
    .stSelectbox > div > div > div {
        color: #222222 !important;
    }

    .stSelectbox option {
        color: #222222 !important;
        font-size: 0.875rem !important;
    }

    /* Slider - Airbnb style */
    .stSlider > label {
        font-size: 0.875rem;
        font-weight: 600;
        color: #222222;
        margin-bottom: 8px;
    }

    .stSlider > div > div > div > div {
        background: #FF385C;
    }

    .stSlider > div > div > div {
        background: #DDDDDD;
    }

    /* Multiselect - Airbnb style */
    .stMultiSelect > label {
        font-size: 0.875rem;
        font-weight: 600;
        color: #222222;
        margin-bottom: 8px;
    }

    /* Button - Airbnb primary style */
    .stButton > button {
        background: linear-gradient(to right, #E61E4D 0%, #E31C5F 50%, #D70466 100%) !important;
        color: #FFFFFF !important;
        border-radius: 8px !important;
        padding: 12px 24px !important;
        height: 48px !important;
        min-height: 48px !important;
        max-height: 48px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        border: none !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.08), 0 4px 12px rgba(0,0,0,0.08) !important;
        transition: all 0.2s ease !important;
        width: 100% !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        white-space: nowrap !important;
        text-align: center !important;
        line-height: 1 !important;
        box-sizing: border-box !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.12), 0 8px 16px rgba(0,0,0,0.12) !important;
    }

    /* Ensure button container doesn't apply problematic flex */
    .stButton {
        display: block !important;
    }

    /* Cluster badges */
    .cluster-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    .cluster-budget {
        background: #E8F5F4;
        color: #00A699;
        border: 1px solid #00A699;
    }

    .cluster-standard {
        background: #FFE7E8;
        color: #FF385C;
        border: 1px solid #FF385C;
    }

    .cluster-luxury {
        background: #FFF0E8;
        color: #FC642D;
        border: 1px solid #FC642D;
    }

    /* Value badges */
    .value-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.813rem;
        font-weight: 600;
    }

    .value-great {
        background: #DEF7EC;
        color: #0E7A48;
        border: 1px solid #0E7A48;
    }

    .value-good {
        background: #FEF3C7;
        color: #92400E;
        border: 1px solid #D97706;
    }

    .value-overpriced {
        background: #FEE2E2;
        color: #991B1B;
        border: 1px solid #DC2626;
    }

    /* Listing card - Target containers with borders */
    div[data-testid="stVerticalBlock"]:has(> div[style*="border"]) {
        border-radius: 12px !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08) !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        cursor: pointer;
    }

    div[data-testid="stVerticalBlock"]:has(> div[style*="border"]):hover {
        box-shadow: 0 6px 20px rgba(0,0,0,0.15) !important;
        transform: translateY(-4px);
    }

    /* Style the actual bordered div inside container */
    div[data-testid="stVerticalBlock"] > div[style*="border"] {
        border: 1px solid #EBEBEB !important;
        border-radius: 12px !important;
        padding: 20px !important;
        background: #FFFFFF;
    }

    div[data-testid="stVerticalBlock"]:hover > div[style*="border"] {
        border-color: #222222 !important;
    }

    /* Result card - Legacy */
    .result-card {
        background: #FFFFFF;
        border: 1px solid #DDDDDD;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
        transition: all 0.2s ease;
        box-shadow: 0 1px 2px rgba(0,0,0,0.08);
    }

    .result-card:hover {
        box-shadow: 0 4px 16px rgba(0,0,0,0.12);
        transform: translateY(-2px);
        border-color: #222222;
    }

    .result-card-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 12px;
    }

    .listing-name {
        font-size: 1.125rem;
        font-weight: 600;
        color: #222222;
        margin: 0 0 8px 0;
        line-height: 1.3;
    }

    .listing-id {
        font-size: 0.75rem;
        color: #717171;
        font-weight: 500;
    }

    .property-details {
        display: flex;
        gap: 16px;
        margin: 12px 0;
        flex-wrap: wrap;
    }

    .detail-item {
        display: flex;
        align-items: center;
        gap: 6px;
        font-size: 0.875rem;
        color: #484848;
    }

    .detail-icon {
        font-size: 1rem;
    }

    .price-section {
        display: flex;
        align-items: center;
        gap: 16px;
        margin-top: 16px;
        padding-top: 16px;
        border-top: 1px solid #EBEBEB;
    }

    .actual-price {
        font-size: 1.5rem;
        font-weight: 700;
        color: #222222;
    }

    .predicted-price {
        font-size: 0.875rem;
        color: #717171;
    }

    .savings-badge {
        background: #DEF7EC;
        color: #0E7A48;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.813rem;
        font-weight: 600;
    }

    /* Stats panel */
    .stats-panel {
        background: linear-gradient(135deg, #667EEA 0%, #764BA2 100%);
        border-radius: 16px;
        padding: 24px;
        color: #FFFFFF;
        margin-bottom: 24px;
        box-shadow: 0 4px 16px rgba(102, 126, 234, 0.3);
    }

    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 20px;
        margin-top: 16px;
    }

    .stat-item {
        text-align: center;
    }

    .stat-value {
        font-size: 2rem;
        font-weight: 800;
        margin-bottom: 4px;
    }

    .stat-label {
        font-size: 0.875rem;
        opacity: 0.9;
        font-weight: 500;
    }

    /* Section header */
    .section-header {
        font-size: 1.75rem;
        font-weight: 700;
        color: #222222;
        margin: 32px 0 20px 0;
        letter-spacing: -0.02em;
    }

    /* Empty state - centered vertically */
    .empty-state {
        text-align: center;
        padding: 80px 20px 40px 20px;
        color: #717171;
        min-height: 30vh;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }

    .empty-state-icon {
        font-size: 4rem;
        margin-bottom: 16px;
    }

    .empty-state-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #222222;
        margin-bottom: 8px;
    }

    .empty-state-text {
        font-size: 1rem;
        color: #717171;
        margin-bottom: 20px;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Dataframe styling */
    .dataframe {
        font-size: 0.875rem;
        border: none !important;
    }

    .dataframe thead tr th {
        background: #F7F7F7 !important;
        color: #222222 !important;
        font-weight: 600 !important;
        border: none !important;
        padding: 12px 16px !important;
    }

    .dataframe tbody tr td {
        border: none !important;
        border-bottom: 1px solid #EBEBEB !important;
        padding: 12px 16px !important;
        color: #484848 !important;
    }

    .dataframe tbody tr:hover {
        background: #F7F7F7 !important;
    }

    /* Divider */
    hr {
        margin: 32px 0;
        border: none;
        border-top: 1px solid #EBEBEB;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA LOADING & CACHING
# ============================================================================

@st.cache_resource
def load_models():
    """Load ML models and build k-NN index"""
    try:
        # Load price prediction model
        with open('model.pkl', 'rb') as f:
            price_model = pickle.load(f)

        # Load model columns
        with open('model_columns.pkl', 'rb') as f:
            model_columns = pickle.load(f)['columns']

        # Load SentenceTransformer for query encoding
        text_model = SentenceTransformer('all-mpnet-base-v2')

        return price_model, model_columns, text_model
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        return None, None, None

@st.cache_data
def load_data():
    """Load listings data and embeddings"""
    try:
        # Load listings
        merged = pd.read_csv('listings_reviews_final.csv')
        merged['listing_id'] = merged['listing_id'].astype(int)

        # Load embeddings
        embed_df = pd.read_parquet('listing_embeddings.parquet')
        embed_df['listing_id'] = embed_df['listing_id'].astype(int)

        # Filter embeddings to match listings
        embed_df = embed_df[embed_df['listing_id'].isin(set(merged['listing_id']))]
        listing_ids = embed_df['listing_id'].values
        X_embeddings = embed_df.drop(columns=['listing_id']).values.astype('float32')

        # Assign clusters based on price quantiles
        merged_clean = merged.dropna(subset=['price'])
        merged_clean = merged_clean[merged_clean['price'] > 0]
        q33, q67 = merged_clean['price'].quantile(0.33), merged_clean['price'].quantile(0.67)

        def assign_cluster(price):
            if pd.isna(price) or price <= 0:
                return None
            return 'Budget' if price <= q33 else 'Standard' if price <= q67 else 'Luxury'

        merged['cluster'] = merged['price'].apply(assign_cluster)

        return merged, listing_ids, X_embeddings, q33, q67
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return None, None, None, None, None

@st.cache_resource
def build_knn_index(_X_embeddings):
    """Build k-NN index for semantic search"""
    nn_model = NearestNeighbors(metric='cosine')
    nn_model.fit(_X_embeddings)
    return nn_model

# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def classify_value(actual_price, predicted_price):
    """Classify value using notebook logic: Great Value (<-$20), Good Value (±$20), Overpriced (>+$20)"""
    diff = actual_price - predicted_price
    if diff < -20:
        return "Great Value"
    elif diff > 20:
        return "Overpriced"
    else:
        return "Good Value"

def predict_price_for_cluster(listing_data, cluster, price_model, model_columns):
    """Predict price for a listing in a specific cluster"""
    input_data = {
        'accommodates': listing_data['accommodates'],
        'bathrooms': listing_data['bathrooms'],
        'bedrooms': listing_data['bedrooms'],
        'review_scores_rating': listing_data.get('review_scores_rating', 4.5),
        'review_scores_accuracy': listing_data.get('review_scores_accuracy', 4.5),
        'segment_Budget': 1 if cluster == 'Budget' else 0,
        'segment_Luxury': 1 if cluster == 'Luxury' else 0,
        'segment_Standard': 1 if cluster == 'Standard' else 0
    }
    X_input = pd.DataFrame([input_data])[model_columns]
    return max(0, price_model.predict(X_input)[0])

def search_listings(query, cluster_filter, top_n, nn_model, text_model, merged, listing_ids, price_model, model_columns):
    """Perform semantic search with cluster filtering"""
    # Encode query
    query_embedding = text_model.encode([query], convert_to_numpy=True)

    # Search with extra buffer for filtering (increased to account for value filter)
    search_n = top_n * 10  # Always search for more to account for cluster and value filters
    distances, indices = nn_model.kneighbors(query_embedding, n_neighbors=min(search_n, len(listing_ids)))

    # Get matching listings
    results = merged[merged['listing_id'].isin(listing_ids[indices[0]])].copy()

    # Apply cluster filter
    if cluster_filter != "All":
        results = results[results['cluster'] == cluster_filter]

    # Clean data - don't limit yet, let value filter be applied first
    results = results.dropna(subset=['cluster', 'price', 'accommodates', 'bathrooms', 'bedrooms'])

    if len(results) == 0:
        return pd.DataFrame()

    # Add predictions and value ratings
    results['predicted_price'] = results.apply(
        lambda row: predict_price_for_cluster(row, row['cluster'], price_model, model_columns),
        axis=1
    )
    results['value_rating'] = results.apply(
        lambda row: classify_value(row['price'], row['predicted_price']),
        axis=1
    )
    results['savings'] = results['predicted_price'] - results['price']

    return results

# ============================================================================
# DISPLAY FUNCTIONS
# ============================================================================

def display_result_card(row, show_cluster=True):
    """Display a single result card with compact, space-efficient Airbnb styling - Option 6"""
    # Handle missing name
    name = row['name'] if pd.notna(row['name']) else f"Unnamed Listing #{int(row['listing_id'])}"
    if len(name) > 50:
        name = name[:50] + "..."

    # Cluster badge colors
    cluster_colors = {
        'Budget': ('#00A699', '🌿'),
        'Standard': ('#FF385C', '💎'),
        'Luxury': ('#FC642D', '✨')
    }

    # Value badge colors (no icon needed, fill color is enough)
    value_colors = {
        'Great Value': ('#D1FAE5', '#065F46'),
        'Good Value': ('#FEF3C7', '#92400E'),
        'Overpriced': ('#F3F4F6', '#6B7280')
    }

    cluster_color, cluster_icon = cluster_colors.get(row['cluster'], ('#484848', '📋'))
    value_bg, value_color = value_colors.get(row['value_rating'], ('#F7F7F7', '#484848'))

    # Sentiment
    sentiment_text = ""
    if 'mean_vader' in row and pd.notna(row['mean_vader']):
        sentiment_score = row['mean_vader']
        sentiment_emoji = "😊" if sentiment_score > 0.5 else "😐" if sentiment_score > -0.1 else "😞"
        emotion = row.get('dominant_emotion_review', '')
        if pd.notna(emotion):
            sentiment_text = f"{sentiment_emoji} {sentiment_score:.2f} · {emotion.capitalize()}"
        else:
            sentiment_text = f"{sentiment_emoji} {sentiment_score:.2f}"

    # Use st.container with border=True for proper card enclosure
    with st.container(border=True):
        # Savings tag at top right corner
        savings_tag = ""
        if row['savings'] > 0:
            savings_tag = f'''
                <div style="position: absolute; top: 12px; right: 12px; background: #A7F3D0; color: #047857;
                padding: 4px 10px; border-radius: 12px; font-size: 0.7rem; font-weight: 600;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);">💰 Save ${row["savings"]:.0f}</div>
            '''

        # Header - Name and cluster as tag (more compact) with relative positioning for savings tag
        st.markdown(f'<div style="position: relative;">{savings_tag}</div>', unsafe_allow_html=True)

        if show_cluster:
            st.markdown(f'<p style="margin: 0 0 4px 0; font-weight: 600; font-size: 0.95rem; color: #222;">{name}</p>', unsafe_allow_html=True)
            # Cluster as a styled tag
            st.markdown(
                f'<div style="display: inline-block; background: {cluster_color}15; color: {cluster_color}; '
                f'border: 1px solid {cluster_color}; padding: 3px 8px; border-radius: 10px; '
                f'font-size: 0.65rem; font-weight: 600; margin-bottom: 8px;">'
                f'{cluster_icon} {row["cluster"]}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(f'<p style="margin: 0 0 4px 0; font-weight: 600; font-size: 0.95rem; color: #222;">{name}</p>', unsafe_allow_html=True)

        # Property details in one compact row (smaller icons and spacing)
        st.markdown(
            f'<div style="display: flex; gap: 12px; margin: 8px 0; font-size: 0.85rem; color: #484848;">'
            f'<span>🛏️ {int(row["bedrooms"])}</span>'
            f'<span>🚿 {row["bathrooms"]:.1f}</span>'
            f'<span>👥 {int(row["accommodates"])}</span>'
            f'</div>',
            unsafe_allow_html=True
        )

        st.markdown('<hr style="margin: 8px 0; border: none; border-top: 1px solid #EBEBEB;">', unsafe_allow_html=True)

        # Side-by-side: Price on left, Value assessment on right (Option 6)
        price_col, divider_col, value_col = st.columns([2.5, 0.1, 2])

        with price_col:
            st.markdown(f'<div style="font-size: 1.75rem; font-weight: 700; color: #222; margin: 0;">${row["price"]:.0f}</div>', unsafe_allow_html=True)
            st.caption(f"per night")
            st.caption(f"Est. ${row['predicted_price']:.0f}")
            if sentiment_text:
                st.caption(sentiment_text)

        with divider_col:
            st.markdown(
                '<div style="border-left: 1px solid #EBEBEB; height: 100%; margin: 0 auto;"></div>',
                unsafe_allow_html=True
            )

        with value_col:
            # Value badge (no icon, just fill color)
            st.markdown(
                f'<div style="background: {value_bg}; color: {value_color}; '
                f'border: 1px solid {value_color}; padding: 6px 12px; '
                f'border-radius: 16px; font-size: 0.75rem; font-weight: 600; '
                f'text-align: center; margin-bottom: 8px;">{row["value_rating"]}</div>',
                unsafe_allow_html=True
            )
            # Book Now button (moved up since savings is now at top)
            book_button = st.button(
                "Book Now",
                key=f"book_{int(row['listing_id'])}",
                use_container_width=True
            )
            if book_button:
                st.toast(f"Booking demo for listing #{int(row['listing_id'])}", icon="🎉")

def display_stats_panel(results_df):
    """Display summary statistics panel"""
    if len(results_df) == 0:
        return

    total_results = len(results_df)
    avg_actual = results_df['price'].mean()
    avg_predicted = results_df['predicted_price'].mean()
    avg_savings = results_df['savings'].mean()
    great_value_count = len(results_df[results_df['value_rating'] == 'Great Value'])

    stats_html = f"""
    <div class="stats-panel">
        <h3 style="margin: 0 0 16px 0; font-size: 1.25rem; font-weight: 600;">📊 Search Summary</h3>
        <div class="stats-grid">
            <div class="stat-item">
                <div class="stat-value">{total_results}</div>
                <div class="stat-label">Results Found</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">${avg_actual:.0f}</div>
                <div class="stat-label">Avg Actual Price</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">${avg_predicted:.0f}</div>
                <div class="stat-label">Avg Predicted Price</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{great_value_count}</div>
                <div class="stat-label">Great Value Deals</div>
            </div>
        </div>
    </div>
    """
    st.markdown(stats_html, unsafe_allow_html=True)

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Load data first
    with st.spinner('🔄 Loading models and data...'):
        price_model, model_columns, text_model = load_models()
        merged, listing_ids, X_embeddings, q33, q67 = load_data()

        if price_model is None or merged is None:
            st.error("❌ Failed to load required data. Please ensure all files are present.")
            return

        nn_model = build_knn_index(X_embeddings)

    # Sidebar filters - Optimized for no scrolling
    with st.sidebar:
        st.markdown("## Search Filters")

        # Cluster filter
        st.markdown("**Market Segment**")
        cluster_options = ["All", "Budget", "Standard", "Luxury"]
        cluster_descriptions = {
            "All": "All ranges",
            "Budget": f"≤ ${q33:.0f}",
            "Standard": f"${q33:.0f} - ${q67:.0f}",
            "Luxury": f"> ${q67:.0f}"
        }

        cluster_filter = st.radio(
            "Select price range",
            cluster_options,
            format_func=lambda x: f"{x} ({cluster_descriptions[x]})",
            label_visibility="collapsed"
        )

        # Number of results
        st.markdown("**Results**")
        top_n = st.slider("", min_value=5, max_value=50, value=10, step=5, label_visibility="collapsed")

        # Sort by
        st.markdown("**Sort By**")
        sort_by = st.selectbox(
            "Sort by",
            ["Relevance", "Best Value", "Lowest Price", "Highest Price", "Most Savings"],
            label_visibility="collapsed"
        )

        # Value filter
        st.markdown("**Value Rating**")
        value_filter = st.multiselect(
            "Value rating",
            ["Great Value", "Good Value", "Overpriced"],
            default=["Great Value", "Good Value", "Overpriced"],
            label_visibility="collapsed"
        )

    # Initialize session state for query if not exists
    if 'query' not in st.session_state:
        st.session_state.query = ""

    # Header with logo on left and search bar on right
    st.markdown('<div style="padding-top: 24px;"></div>', unsafe_allow_html=True)
    col_logo, col_search = st.columns([1, 4], gap="medium", vertical_alignment="center")

    with col_logo:
        st.image("Airbnb-Logo.png", width=180)

    with col_search:
        col_input, col_button = st.columns([5, 1], gap="small", vertical_alignment="center")
        with col_input:
            query = st.text_input(
                "🔍 Search for listings",
                placeholder="e.g., cozy apartment near downtown with parking and great views...",
                label_visibility="collapsed",
                value=st.session_state.query,
                key="search_input"
            )
        with col_button:
            search_button = st.button("Search", type="primary", use_container_width=True)

    # Update session state with current query
    st.session_state.query = query

    # Perform search
    if query and (search_button or 'last_query' in st.session_state):
        # Store query in session state
        st.session_state.last_query = query

        with st.spinner('🔍 Searching listings...'):
            results = search_listings(
                query,
                cluster_filter,
                top_n,
                nn_model,
                text_model,
                merged,
                listing_ids,
                price_model,
                model_columns
            )

        # Apply value filter
        if len(results) > 0:
            results = results[results['value_rating'].isin(value_filter)]

        # Limit to top_n after all filters are applied
        if len(results) > 0:
            results = results.head(top_n)

        # Apply sorting
        if len(results) > 0:
            if sort_by == "Best Value":
                results = results.sort_values('savings', ascending=False)
            elif sort_by == "Lowest Price":
                results = results.sort_values('price', ascending=True)
            elif sort_by == "Highest Price":
                results = results.sort_values('price', ascending=False)
            elif sort_by == "Most Savings":
                results = results.sort_values('savings', ascending=False)

        # Display results
        if len(results) == 0:
            st.markdown("""
                <div class="empty-state">
                    <div class="empty-state-icon">🔍</div>
                    <div class="empty-state-title">No results found</div>
                    <div class="empty-state-text">Try adjusting your search query or filters</div>
                </div>
            """, unsafe_allow_html=True)
        else:
            # Section header
            cluster_icons = {
                "All": "",
                "Budget": "🌿",
                "Standard": "💎",
                "Luxury": "✨"
            }
            icon = cluster_icons.get(cluster_filter, "")
            icon_display = f"{icon} " if icon else ""

            st.markdown(f'<h2 class="section-header">{icon_display}{cluster_filter} Listings - {len(results)} Results</h2>', unsafe_allow_html=True)

            # Display result cards in grid layout - 2 columns with proper gap
            show_cluster = (cluster_filter == "All")

            results_list = list(results.iterrows())
            for i in range(0, len(results_list), 2):
                cols = st.columns([1, 1], gap="medium")
                with cols[0]:
                    display_result_card(results_list[i][1], show_cluster=show_cluster)
                if i + 1 < len(results_list):
                    with cols[1]:
                        display_result_card(results_list[i+1][1], show_cluster=show_cluster)

                # Add uniform vertical spacing between rows
                if i + 2 < len(results_list):
                    st.markdown('<div style="margin-bottom: 16px;"></div>', unsafe_allow_html=True)

            # Download option
            st.markdown("---")
            csv = results[['listing_id', 'name', 'cluster', 'bedrooms', 'bathrooms', 'accommodates',
                          'price', 'predicted_price', 'value_rating', 'savings']].to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name=f"airbnb_search_results_{cluster_filter.lower()}.csv",
                mime="text/csv",
                use_container_width=True
            )

    else:
        # Welcome state - centered container
        st.markdown('<div style="max-width: 1200px; margin: 0 auto;">', unsafe_allow_html=True)

        st.markdown("""
            <div class="empty-state">
                <div class="empty-state-icon">🔍</div>
                <div class="empty-state-title">Start Your Search</div>
                <div class="empty-state-text">Enter keywords above to find your perfect Airbnb listing</div>
            </div>
        """, unsafe_allow_html=True)

        # Show sample searches
        st.markdown('<h2 class="section-header" style="text-align: center;">💡 Try These Popular Searches</h2>', unsafe_allow_html=True)

        # Add spacing before buttons
        st.markdown('<div style="margin-top: 20px;"></div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🏙️ Downtown Apartment", use_container_width=True):
                st.session_state.query = "cozy apartment near downtown"
                st.rerun()

        with col2:
            if st.button("🏡 Family Home", use_container_width=True):
                st.session_state.query = "spacious family home with backyard"
                st.rerun()

        with col3:
            if st.button("🌆 Modern Studio", use_container_width=True):
                st.session_state.query = "modern studio with city views"
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

        # Compact spacing before footer
        st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)

    # Footer - compact design
    st.markdown('<hr style="margin: 10px 0; border: none; border-top: 1px solid #EBEBEB;">', unsafe_allow_html=True)
    st.markdown("""
        <div style="text-align: center; color: #717171; padding: 15px 20px; font-size: 0.813rem;">
            <p style="margin: 3px 0;">🤖 Powered by XGBoost ML + SentenceTransformers Semantic Search</p>
            <p style="opacity: 0.8; margin: 3px 0;">Design inspired by Airbnb's official design system</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
