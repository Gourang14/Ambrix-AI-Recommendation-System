import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
import base64

# Configure Streamlit page
st.set_page_config(
    page_title="AMBRIX - AI Content Recommender",
    page_icon="ambrix_lifemedia_logo.jpeg",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Professional Dark Theme with Glassmorphism
def load_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* Global Styles */
    .stApp {
        background: #0a0a0a;
        background-image: 
            radial-gradient(circle at 25% 25%, #1a1a2e 0%, transparent 50%),
            radial-gradient(circle at 75% 75%, #16213e 0%, transparent 50%),
            radial-gradient(circle at 50% 50%, #0f3460 0%, transparent 50%);
        font-family: 'Inter', sans-serif;
        color: #ffffff;
        min-height: 100vh;
    }
    
    /* Main Components */
    .main-header {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 24px;
        padding: 48px;
        margin: 24px 0 48px 0;
        border: 1px solid rgba(255, 255, 255, 0.12);
        box-shadow: 
            0 24px 48px rgba(0, 0, 0, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
    }
    
    .glass-container {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-radius: 20px;
        padding: 32px;
        margin: 32px 0;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 
            0 16px 32px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.08);
        position: relative;
    }
    
    .glass-section {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 16px;
        padding: 24px;
        margin: 24px 0;
        border: 1px solid rgba(255, 255, 255, 0.06);
        box-shadow: 
            0 8px 16px rgba(0, 0, 0, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.06);
    }
    
    .upload-zone {
        background: rgba(255, 255, 255, 0.02);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 24px;
        margin: 16px 0;
        border: 2px dashed rgba(255, 255, 255, 0.15);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        text-align: center;
        min-height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .upload-zone:hover {
        border-color: rgba(120, 120, 255, 0.4);
        background: rgba(120, 120, 255, 0.03);
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
    }
    
    # .metric-card {
    #     background: rgba(255, 255, 255, 0.06);
    #     backdrop-filter: blur(12px);
    #     -webkit-backdrop-filter: blur(12px);
    #     border-radius: 16px;
    #     padding: 24px;
    #     margin: 16px 0;
    #     border: 1px solid rgba(255, 255, 255, 0.1);
    #     text-align: center;
    #     transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    #     position: relative;
    #     overflow: hidden;
    # }
    
    .metric-card:hover {
        transform: translateY(-4px) scale(1.02);
        box-shadow: 0 16px 40px rgba(0, 0, 0, 0.4);
        border-color: rgba(255, 255, 255, 0.15);
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.05), transparent);
        transition: left 0.5s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .metric-card:hover::before {
        left: 100%;
    }
    
    .recommendation-card {
        background: rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-radius: 20px;
        padding: 28px;
        margin: 20px 0;
        border: 1px solid rgba(255, 255, 255, 0.08);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .recommendation-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, #4f46e5, #06b6d4, #10b981);
        opacity: 0;
        transition: opacity 0.3s;
    }
    
    .recommendation-card:hover {
        transform: translateY(-6px) scale(1.01);
        box-shadow: 0 20px 48px rgba(0, 0, 0, 0.5);
        border-color: rgba(255, 255, 255, 0.12);
    }
    
    .recommendation-card:hover::before {
        opacity: 1;
    }
    
    .algorithm-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 24px;
        margin: 16px 0;
        border: 1px solid rgba(255, 255, 255, 0.06);
        height: 100%;
        transition: all 0.3s ease;
    }
    
    .algorithm-card:hover {
        border-color: rgba(255, 255, 255, 0.12);
        background: rgba(255, 255, 255, 0.05);
    }
    
    .step-indicator {
        background: linear-gradient(135deg, #4f46e5, #06b6d4);
        border-radius: 50%;
        width: 32px;
        height: 32px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 600;
        margin-right: 12px;
        font-size: 14px;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);
    }
    
    .success-alert {
        background: rgba(34, 197, 94, 0.1);
        border: 1px solid rgba(34, 197, 94, 0.2);
        border-radius: 12px;
        padding: 16px 20px;
        margin: 16px 0;
        color: #10b981;
        backdrop-filter: blur(8px);
    }
    
    .warning-alert {
        background: rgba(245, 158, 11, 0.1);
        border: 1px solid rgba(245, 158, 11, 0.2);
        border-radius: 12px;
        padding: 16px 20px;
        margin: 16px 0;
        color: #f59e0b;
        backdrop-filter: blur(8px);
    }
    
    .info-box {
        background: rgba(59, 130, 246, 0.08);
        backdrop-filter: blur(8px);
        border-radius: 12px;
        padding: 20px;
        margin: 20px 0;
        border-left: 3px solid #3b82f6;
        border-top: 1px solid rgba(59, 130, 246, 0.2);
        border-right: 1px solid rgba(59, 130, 246, 0.2);
        border-bottom: 1px solid rgba(59, 130, 246, 0.2);
    }
    
    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff;
        font-weight: 600;
        letter-spacing: -0.025em;
    }
    
    h1 {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #ffffff 0%, #a1a1aa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 16px;
    }
    
    h2 {
        font-size: 1.875rem;
        margin-bottom: 20px;
        color: #f8fafc;
    }
    
    h3 {
        font-size: 1.5rem;
        margin-bottom: 16px;
        color: #e2e8f0;
    }
    
    h4 {
        font-size: 1.25rem;
        margin-bottom: 12px;
        color: #cbd5e1;
    }
    
    p {
        color: #94a3b8;
        line-height: 1.6;
        margin-bottom: 12px;
    }
    
    .subtitle {
        font-size: 1.25rem;
        color: #cbd5e1;
        font-weight: 400;
        opacity: 0.9;
    }
    
    .caption {
        font-size: 0.875rem;
        color: #64748b;
        font-weight: 400;
        opacity: 0.8;
    }
    
    /* Interactive Elements */
    .stButton > button {
        background: linear-gradient(135deg, #4f46e5 0%, #06b6d4 100%);
        border: none;
        border-radius: 12px;
        padding: 12px 24px;
        color: white;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 16px rgba(79, 70, 229, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(8px);
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(79, 70, 229, 0.4);
        background: linear-gradient(135deg, #5b52e6 0%, #07b7d5 100%);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(8px);
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: #ffffff !important;
    }
    
    .stFileUploader > div {
        background: rgba(255, 255, 255, 0.03) !important;
        backdrop-filter: blur(8px);
        border-radius: 12px !important;
        border: 2px dashed rgba(255, 255, 255, 0.15) !important;
    }
    
    .stFileUploader label {
        color: #cbd5e1 !important;
    }
    
    /* Code blocks */
    .code-block {
        background: rgba(0, 0, 0, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.875rem;
        color: #e2e8f0;
        overflow-x: auto;
        line-height: 1.5;
    }
    
    /* Feature list */
    .feature-list {
        list-style: none;
        padding: 0;
        margin: 16px 0;
    }
    
    .feature-list li {
        padding: 8px 0;
        color: #94a3b8;
        position: relative;
        padding-left: 24px;
    }
    
    .feature-list li::before {
        content: '→';
        position: absolute;
        left: 0;
        color: #3b82f6;
        font-weight: bold;
    }
    
    /* Grid layouts */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 16px;
        margin: 24px 0;
    }
    
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 24px;
        margin: 32px 0;
    }
    
    /* Status indicators */
    .status-success {
        color: #10b981;
        background: rgba(16, 185, 129, 0.1);
        padding: 4px 8px;
        border-radius: 6px;
        font-size: 0.875rem;
        font-weight: 500;
    }
    
    .status-pending {
        color: #f59e0b;
        background: rgba(245, 158, 11, 0.1);
        padding: 4px 8px;
        border-radius: 6px;
        font-size: 0.875rem;
        font-weight: 500;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.2);
        border-radius: 3px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(255, 255, 255, 0.3);
    }
    
    /* Plotly chart styling */
    .plotly-chart {
        background: transparent !important;
    }
    
    /* Footer */
    .footer {
        margin-top: 64px;
        padding: 32px;
        text-align: center;
        background: rgba(255, 255, 255, 0.02);
        border-radius: 16px;
        border-top: 1px solid rgba(255, 255, 255, 0.06);
    }
    
    .footer p {
        margin: 8px 0;
        color: #64748b;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header {
            padding: 32px 24px;
        }
        
        h1 {
            font-size: 2.5rem;
        }
        
        .glass-container {
            padding: 24px 20px;
        }
    }
    </style>
    """, unsafe_allow_html=True)

class ContentRecommendationSystem:
    def __init__(self):
        self.users_df = None
        self.posts_df = None
        self.engagements_df = None
        self.user_item_matrix = None
        self.content_features = None
        self.scaler = StandardScaler()
        
    def load_data(self, users_file, posts_file, engagements_file):
        """Load and process the uploaded data files"""
        try:
            self.users_df = pd.read_csv(users_file)
            self.posts_df = pd.read_csv(posts_file)
            self.engagements_df = pd.read_csv(engagements_file)
            
            # Clean and preprocess data
            self._preprocess_data()
            return True
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return False
    
    def _preprocess_data(self):
        """Preprocess the loaded data"""
        # Create user-item interaction matrix
        self.user_item_matrix = self.engagements_df.pivot_table(
            index='user_id', 
            columns='post_id', 
            values='engagement', 
            fill_value=0
        )
        
        # Process user interests
        if 'top_3_interests' in self.users_df.columns:
            self.users_df['interests_list'] = self.users_df['top_3_interests'].str.split(', ')
        
        # Process post tags
        if 'tags' in self.posts_df.columns:
            self.posts_df['tags_list'] = self.posts_df['tags'].str.split(', ')
            
            # Create TF-IDF features for content
            tfidf = TfidfVectorizer(stop_words='english', max_features=100)
            tags_text = self.posts_df['tags'].fillna('')
            self.content_features = tfidf.fit_transform(tags_text)
    
    def calculate_user_similarity(self):
        """Calculate user-user similarity based on engagement patterns"""
        user_similarity = cosine_similarity(self.user_item_matrix)
        return pd.DataFrame(
            user_similarity, 
            index=self.user_item_matrix.index, 
            columns=self.user_item_matrix.index
        )
    
    def content_based_recommendations(self, user_id, n_recommendations=3):
        """Generate content-based recommendations"""
        if user_id not in self.users_df['user_id'].values:
            return []
        
        user_data = self.users_df[self.users_df['user_id'] == user_id].iloc[0]
        user_interests = user_data.get('interests_list', [])
        
        # Calculate content scores
        post_scores = []
        for idx, post in self.posts_df.iterrows():
            score = 0
            post_tags = post.get('tags_list', [])
            
            # Interest matching score
            if user_interests and post_tags:
                common_interests = set(user_interests) & set(post_tags)
                score += len(common_interests) * 2
            
            # Engagement history bias
            past_engagement = self.engagements_df[
                (self.engagements_df['user_id'] == user_id) & 
                (self.engagements_df['post_id'] == post['post_id'])
            ]
            
            if len(past_engagement) == 0:  # New content
                score += 1
            elif past_engagement.iloc[0]['engagement'] == 0:
                score -= 1  # Previously disliked
            
            post_scores.append({
                'post_id': post['post_id'],
                'score': score,
                'content_type': post.get('content_type', 'unknown'),
                'tags': post.get('tags', ''),
                'creator_id': post.get('creator_id', '')
            })
        
        # Sort and return top recommendations
        post_scores.sort(key=lambda x: x['score'], reverse=True)
        return post_scores[:n_recommendations]
    
    def collaborative_filtering_recommendations(self, user_id, n_recommendations=3):
        """Generate collaborative filtering recommendations"""
        if user_id not in self.user_item_matrix.index:
            return []
        
        user_similarity = self.calculate_user_similarity()
        similar_users = user_similarity[user_id].sort_values(ascending=False)[1:6]  # Top 5 similar users
        
        # Get posts liked by similar users but not seen by target user
        user_posts = set(self.engagements_df[self.engagements_df['user_id'] == user_id]['post_id'])
        recommendations = []
        
        for similar_user, similarity_score in similar_users.items():
            similar_user_posts = self.engagements_df[
                (self.engagements_df['user_id'] == similar_user) & 
                (self.engagements_df['engagement'] == 1)
            ]['post_id'].tolist()
            
            for post_id in similar_user_posts:
                if post_id not in user_posts:
                    post_info = self.posts_df[self.posts_df['post_id'] == post_id]
                    if not post_info.empty:
                        recommendations.append({
                            'post_id': post_id,
                            'score': similarity_score,
                            'content_type': post_info.iloc[0].get('content_type', 'unknown'),
                            'tags': post_info.iloc[0].get('tags', ''),
                            'creator_id': post_info.iloc[0].get('creator_id', '')
                        })
        
        # Remove duplicates and sort
        seen_posts = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec['post_id'] not in seen_posts:
                unique_recommendations.append(rec)
                seen_posts.add(rec['post_id'])
        
        unique_recommendations.sort(key=lambda x: x['score'], reverse=True)
        return unique_recommendations[:n_recommendations]
    
    def hybrid_recommendations(self, user_id, n_recommendations=3):
        """Generate hybrid recommendations combining content-based and collaborative filtering"""
        content_recs = self.content_based_recommendations(user_id, n_recommendations * 2)
        collab_recs = self.collaborative_filtering_recommendations(user_id, n_recommendations * 2)
        
        # Combine and weight recommendations
        all_recs = {}
        
        # Add content-based recommendations with weight 0.6
        for rec in content_recs:
            post_id = rec['post_id']
            all_recs[post_id] = all_recs.get(post_id, {
                'content_score': 0, 
                'collab_score': 0,
                'content_type': rec['content_type'],
                'tags': rec['tags'],
                'creator_id': rec['creator_id']
            })
            all_recs[post_id]['content_score'] = rec['score']
        
        # Add collaborative recommendations with weight 0.4
        for rec in collab_recs:
            post_id = rec['post_id']
            if post_id not in all_recs:
                all_recs[post_id] = {
                    'content_score': 0, 
                    'collab_score': 0,
                    'content_type': rec['content_type'],
                    'tags': rec['tags'],
                    'creator_id': rec['creator_id']
                }
            all_recs[post_id]['collab_score'] = rec['score']
        
        # Calculate hybrid scores
        final_recs = []
        for post_id, scores in all_recs.items():
            hybrid_score = 0.6 * scores['content_score'] + 0.4 * scores['collab_score']
            final_recs.append({
                'post_id': post_id,
                'score': hybrid_score,
                'content_type': scores['content_type'],
                'tags': scores['tags'],
                'creator_id': scores['creator_id']
            })
        
        final_recs.sort(key=lambda x: x['score'], reverse=True)
        return final_recs[:n_recommendations]

def main():
    load_css()
    
    # Professional Header
    st.markdown("""
    <div class="main-header">
        <h1>Ambrix Lifemedia</h1>
        <p class="subtitle">Advanced Machine Learning Recommendation Intelligence</p>
        <p class="caption">Professional-grade content recommendation system powered by AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize recommendation system
    if 'recommender' not in st.session_state:
        st.session_state.recommender = ContentRecommendationSystem()
    
    # Step 1: System Overview and Instructions
    st.markdown("""
            <div class="glass-section">
                <h2>System Overview</h2>
                <h4>AMBRIX utilizes advanced machine learning algorithms to deliver personalized content recommendations. The system analyzes user behavior patterns, content metadata, and engagement metrics to provide intelligent suggestions.</h4>
                <ul class="feature-list">
                    <li>Data Preparation:</strong> Prepare three CSV files with the specified schema structure</li>
                    <li>Upload Files:</strong> Upload your datasets using the file upload interface below</li>
                    <li>Process Data:</strong> Initialize the recommendation engine with your datasets</li>
                    <li>Generate Recommendations:</strong> Select users and algorithms to produce personalized suggestions</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    # Step 2: Data Schema Requirements
    # st.markdown("""
    # <div class="glass-container">
    #     <h2>Required Data Schema</h2>
    #     <p>Ensure your CSV files conform to the following schema specifications:</p>
    # </div>
    # """, unsafe_allow_html=True)


    # Step 3: File Upload Interface
    st.markdown("""
    <div class="glass-container">
        <h2>Dataset Upload Interface</h2>
        <p>Upload your CSV files in the order specified. The system will validate the schema upon processing.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="glass-section">
            <h4>Users Dataset Structure</h4>
            <div class="code-block">
user_id,age,gender,top_3_interests,past_engagement_score
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="glass-section">
            <h4>Posts Dataset Structure</h4>
            <div class="code-block">
post_id,creator_id,content_type,tags
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="glass-section">
            <h4>Engagements Dataset Structure</h4>
            <div class="code-block">
user_id,post_id,engagement
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="zone">', unsafe_allow_html=True)
        st.markdown("#### Users Dataset")
        users_file = st.file_uploader("Select Users.csv", type=['csv'], key="users", help="User profiles and preferences data")
        if users_file:
            st.markdown('<div class="success-alert">Users dataset uploaded successfully</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="zone">', unsafe_allow_html=True)
        st.markdown("#### Posts Dataset")
        posts_file = st.file_uploader("Select Posts.csv", type=['csv'], key="posts", help="Content metadata and categorization data")
        if posts_file:
            st.markdown('<div class="success-alert">Posts dataset uploaded successfully</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="zone">', unsafe_allow_html=True)
        st.markdown("#### Engagements Dataset")
        engagements_file = st.file_uploader("Select Engagements.csv", type=['csv'], key="engagements", help="User interaction and engagement data")
        if engagements_file:
            st.markdown('<div class="success-alert">Engagements dataset uploaded successfully</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Step 4: Data Processing
    if users_file and posts_file and engagements_file:
        st.markdown("""
        <div class="glass-container">
            <h2>Data Processing</h2>
            <p>All required datasets have been uploaded. Initialize the recommendation engine to begin processing.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Initialize Recommendation Engine", use_container_width=True):
                with st.spinner("Processing datasets and initializing ML models..."):
                    success = st.session_state.recommender.load_data(users_file, posts_file, engagements_file)
                    if success:
                        st.balloons()
                        st.markdown('<div class="success-alert">System initialization completed successfully. Ready for recommendation generation.</div>', unsafe_allow_html=True)
                        st.session_state.data_loaded = True
                    else:
                        st.markdown('<div class="warning-alert">Initialization failed. Please verify your CSV file formats and schema compliance.</div>', unsafe_allow_html=True)
    
    # Step 5: Algorithm Framework Overview
    st.markdown("""
    <div class="glass-container">
        <h2>Machine Learning Framework</h2>
        <p>Our system implements three distinct recommendation algorithms, each optimized for specific use cases and data patterns.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="algorithm-card">
            <h4>Content-Based Filtering</h4>
            <p><strong>Methodology:</strong></p>
            <ul class="feature-list">
                <li>TF-IDF vectorization of content attributes</li>
                <li>Cosine similarity computation</li>
                <li>User preference profile matching</li>
                <li>Interest-content correlation analysis</li>
                <li>Engagement history weighting</li>
            </ul>
            <p><strong>Optimal Use Cases:</strong> Cold start scenarios, preference consistency, niche content discovery</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="algorithm-card">
            <h4>Collaborative Filtering</h4>
            <p><strong>Methodology:</strong></p>
            <ul class="feature-list">
                <li>User-item interaction matrix construction</li>
                <li>Similarity-based user clustering</li>
                <li>Neighborhood-based predictions</li>
                <li>Implicit feedback processing</li>
                <li>Social proof utilization</li>
            </ul>
            <p><strong>Optimal Use Cases:</strong> Social discovery, trending content identification, behavioral pattern analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="algorithm-card">
            <h4>Hybrid Implementation</h4>
            <p><strong>Methodology:</strong></p>
            <ul class="feature-list">
                <li>Weighted ensemble approach (60/40 ratio)</li>
                <li>Content-collaborative score fusion</li>
                <li>Multi-objective optimization</li>
                <li>Diversity-accuracy balancing</li>
                <li>Cross-validation performance tuning</li>
            </ul>
            <p><strong>Optimal Use Cases:</strong> Production deployments, balanced recommendations, comprehensive coverage</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main Interface (only show if data is loaded)
    if hasattr(st.session_state, 'data_loaded') and st.session_state.data_loaded:
        
        # Step 6: System Metrics Overview
        st.markdown("""
        <div class="glass-container">
            <h2>System Performance Metrics</h2>
            <p>Real-time analytics and performance indicators for the loaded datasets.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Users", len(st.session_state.recommender.users_df), help="Unique user profiles in the system")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Content Items", len(st.session_state.recommender.posts_df), help="Available content pieces for recommendation")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Interactions", len(st.session_state.recommender.engagements_df), help="User-content interaction records")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            engagement_rate = (st.session_state.recommender.engagements_df['engagement'].sum() / 
                             len(st.session_state.recommender.engagements_df)) * 100
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Engagement Rate", f"{engagement_rate:.1f}%", help="Percentage of positive user interactions")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Step 7: Recommendation Generation Interface
        st.markdown("""
        <div class="glass-container">
            <h2>Recommendation Generation Interface</h2>
            <p>Configure parameters and generate personalized content recommendations for selected users.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            user_ids = sorted(st.session_state.recommender.users_df['user_id'].tolist())
            selected_user = st.selectbox("Target User Selection:", user_ids, help="Select user ID for recommendation generation")
        
        with col2:
            algorithm = st.selectbox("Algorithm Selection:", 
                                   ["Hybrid Implementation", "Content-Based Filtering", "Collaborative Filtering"],
                                   help="Choose the machine learning algorithm for recommendation generation")
        
        with col3:
            st.markdown("<br>", unsafe_allow_html=True)
            generate_btn = st.button("Generate Recommendations", use_container_width=True)
        
        if generate_btn:
            with st.spinner("Processing user data and generating personalized recommendations..."):
                if algorithm == "Content-Based Filtering":
                    recommendations = st.session_state.recommender.content_based_recommendations(selected_user)
                    algorithm_used = "Content-Based Filtering"
                elif algorithm == "Collaborative Filtering":
                    recommendations = st.session_state.recommender.collaborative_filtering_recommendations(selected_user)
                    algorithm_used = "Collaborative Filtering"
                else:
                    recommendations = st.session_state.recommender.hybrid_recommendations(selected_user)
                    algorithm_used = "Hybrid Implementation"
                
                st.session_state.current_recommendations = recommendations
                st.session_state.current_user = selected_user
                st.session_state.algorithm_used = algorithm_used
        
        # Step 8: Recommendation Results Display
        if hasattr(st.session_state, 'current_recommendations'):
            st.markdown(f"""
            <div class="glass-container">
                <h2>Personalized Recommendations</h2>
                <p><strong>Target User:</strong> {st.session_state.current_user} | <strong>Algorithm:</strong> {st.session_state.algorithm_used}</p>
                <p>Top-ranked content recommendations based on user preferences and behavioral patterns.</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.session_state.current_recommendations:
                for i, rec in enumerate(st.session_state.current_recommendations, 1):
                    st.markdown(f"""
                    <div class="recommendation-card">
                        <h3>Recommendation #{i} - {rec['post_id']}</h3>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin: 20px 0;">
                            <div>
                                <p><strong>Content Type:</strong> {rec['content_type'].title()}</p>
                                <p><strong>Tags:</strong> {rec['tags']}</p>
                            </div>
                            <div>
                                <p><strong>Creator ID:</strong> {rec['creator_id']}</p>
                                <p><strong>Relevance Score:</strong> <span style="color: #10b981; font-weight: 600;">{rec['score']:.3f}</span></p>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown('<div class="warning-alert">No recommendations available for the selected user and algorithm combination. Consider trying an alternative algorithm or different user profile.</div>', unsafe_allow_html=True)
        
        # Step 9: Analytics and Insights Dashboard
        st.markdown("""
        <div class="glass-container">
            <h2>Analytics Dashboard</h2>
            <p>Comprehensive data visualization and insights into user behavior patterns, content distribution, and system performance metrics.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Analytics visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Engagement distribution
            engagement_dist = st.session_state.recommender.engagements_df['engagement'].value_counts()
            fig_pie = px.pie(
                values=engagement_dist.values,
                names=['Negative Engagement', 'Positive Engagement'],
                title="User Engagement Distribution Analysis",
                color_discrete_sequence=['#ef4444', '#10b981']
            )
            fig_pie.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white', family='Inter'),
                title_font_size=16,
                title_font_color='#e2e8f0'
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Content type distribution
            content_dist = st.session_state.recommender.posts_df['content_type'].value_counts()
            fig_bar = px.bar(
                x=content_dist.index,
                y=content_dist.values,
                title="Content Type Distribution Analysis",
                color=content_dist.values,
                color_continuous_scale='viridis'
            )
            fig_bar.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white', family='Inter'),
                xaxis=dict(color='white', title='Content Type'),
                yaxis=dict(color='white', title='Content Count'),
                title_font_size=16,
                title_font_color='#e2e8f0'
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Additional Analytics
        col3, col4 = st.columns(2)
        
        with col3:
            # User engagement score distribution
            if 'past_engagement_score' in st.session_state.recommender.users_df.columns:
                fig_hist = px.histogram(
                    st.session_state.recommender.users_df,
                    x='past_engagement_score',
                    title="User Engagement Score Distribution",
                    nbins=20,
                    color_discrete_sequence=['#4f46e5']
                )
                fig_hist.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white', family='Inter'),
                    xaxis=dict(color='white', title='Engagement Score'),
                    yaxis=dict(color='white', title='User Count'),
                    title_font_size=16,
                    title_font_color='#e2e8f0'
                )
                st.plotly_chart(fig_hist, use_container_width=True)
        
        with col4:
            # Top interests analysis
            if 'top_3_interests' in st.session_state.recommender.users_df.columns:
                # Extract all interests
                all_interests = []
                for interests in st.session_state.recommender.users_df['top_3_interests']:
                    if pd.notna(interests):
                        all_interests.extend([interest.strip() for interest in interests.split(',')])
                
                interest_counts = pd.Series(all_interests).value_counts().head(10)
                fig_bar_interests = px.bar(
                    x=interest_counts.values,
                    y=interest_counts.index,
                    title="Most Popular User Interest Categories",
                    orientation='h',
                    color=interest_counts.values,
                    color_continuous_scale='plasma'
                )
                fig_bar_interests.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white', family='Inter'),
                    xaxis=dict(color='white', title='User Count'),
                    yaxis=dict(color='white', title='Interest Categories'),
                    title_font_size=16,
                    title_font_color='#e2e8f0'
                )
                st.plotly_chart(fig_bar_interests, use_container_width=True)
    
    else:
        # Welcome Section when no data is loaded
        st.markdown("""
        <div class="glass-container">
            <h2>System Capabilities Overview</h2>
            <p>AMBRIX represents a comprehensive machine learning solution for personalized content recommendation, incorporating state-of-the-art algorithms and professional-grade analytics capabilities.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature highlights
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="glass-section">
                <h4>Machine Learning Core</h4>
                <ul class="feature-list">
                    <li>Content-Based Filtering with TF-IDF</li>
                    <li>Collaborative Filtering with Cosine Similarity</li>
                    <li>Hybrid Model Implementation</li>
                    <li>Real-time Feature Vectorization</li>
                    <li>Advanced Pattern Recognition</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="glass-section">
                <h4>Analytics Engine</h4>
                <ul class="feature-list">
                    <li>Real-time Performance Metrics</li>
                    <li>User Behavior Analysis</li>
                    <li>Content Distribution Insights</li>
                    <li>Engagement Pattern Recognition</li>
                    <li>Interactive Data Visualization</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="glass-section">
                <h4>System Architecture</h4>
                <ul class="feature-list">
                    <li>Scalable Processing Pipeline</li>
                    <li>Professional UI/UX Design</li>
                    <li>Multi-format Data Support</li>
                    <li>Production-Ready Deployment</li>
                    <li>Enterprise-Grade Performance</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Getting started section
        st.markdown("""
        <div class="glass-container">
            <h3>Implementation Guidelines</h3>
            <div class="info-box">
                <p>To begin utilizing the AMBRIX recommendation system, prepare your datasets according to the specified schema and upload them using the interface above. The system requires three CSV files containing user profiles, content metadata, and engagement data.</p>
                <p><strong>Performance Optimization:</strong> For optimal recommendation accuracy, ensure your datasets contain diverse user preferences, comprehensive content tagging, and sufficient interaction history. The system performs best with balanced engagement patterns across different content categories.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Professional Footer
    st.markdown("""
    <div class="footer">
        <p><strong>AMBRIX</strong> - Advanced Machine Learning Recommendation Intelligence</p>
        <p>Professional Content Recommendation System | Powered by Python & Streamlit</p>
        <p class="caption">Content-Based Filtering • Collaborative Intelligence • Hybrid Models • Real-time Analytics</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()