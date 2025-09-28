# AMBRIX - AI-Powered Lifestyle Content Recommendation System

I developed a **hybrid recommendation system** that combines **content similarity** (matching user interests to post tags) with **collaborative filtering** (what similar users interacted with). I implemented cold start handling using popularity ranking and exposed everything in an interactive Streamlit UI so stakeholders can test recommendations per user.

## Access the live app here:  
https://ambrixai.streamlit.app/

---

## Features
- **Content-Based Filtering** (TF-IDF + Cosine Similarity)
- **Collaborative Filtering** (User-User Similarity)
- **Hybrid Model** (Balanced personalization + discovery)
- **Analytics Dashboard** (Engagement insights, content distribution, user trends)
- **Real-time Recommendations** with interactive UI
- Built with **Streamlit & Python**

---

## Expected Data Format

You need **three CSV files**:

### Eg: Users.csv
| user_id | age | gender | top_3_interests       | past_engagement_score |
|---------|-----|--------|-----------------------|-----------------------|
| U1      | 24  | F      | sports, art, gaming  | 0.61                  |
| U2      | 32  | F      | travel, food, fashion | 0.93                  |

### Eg: Posts.csv
| post_id | creator_id | content_type | tags              |
|---------|------------|--------------|------------------|
| P1      | U44        | video        | sports, food     |
| P2      | U26        | video        | music, travel    |

### Eg: Engagements.csv
| user_id | post_id | engagement |
|---------|---------|------------|
| U1      | P52     | 1          |
| U1      | P44     | 0          |
| U1      | P1      | 1          |

---

## How to Run Locally

1. Clone the repo:
   ```bash
   git clone https://github.com/Gourang14/Ambrix-AI-Recommender-System.git
   cd Ambrix-AI-Recommender-System
2. Terminal:
   ```bash
   python -m venv venv
   source venv/bin/activate   # (Linux/Mac)
   venv\Scripts\activate      # (Windows)

   pip install -r requirements.txt

   streamlit run streamlit_app.py

---

## Tech Stack

1. Frontend/UI → Streamlit

2. ML Models → Scikit-learn (TF-IDF, Cosine Similarity)

3. Data Processing → Pandas, NumPy

4. Visualization → Matplotlib, Altair

5. Deployment → Streamlit Cloud

---

## Example Output

1. Upload CSVs
2. Select User + Algorithm
3. Get Top 3 Personalized Recommendations
4. View Analytics Dashboard

---
[![Live Demo](https://img.shields.io/badge/Streamlit-View%20Live-blue)](https://ambrixai.streamlit.app/)
