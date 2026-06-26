import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="🛒 Shopper Spectrum", page_icon="🛒", layout="wide")

@st.cache_resource
def load_models():
    with open('models/kmeans_model.pkl','rb') as f: kmeans = pickle.load(f)
    with open('models/scaler.pkl','rb') as f: scaler = pickle.load(f)
    with open('models/cluster_labels.pkl','rb') as f: cluster_labels = pickle.load(f)
    with open('models/product_similarity_small.pkl','rb') as f: product_sim_df = pickle.load(f)
    with open('models/product_list.pkl','rb') as f: product_list = pickle.load(f)
    return kmeans, scaler, cluster_labels, product_sim_df, product_list

kmeans, scaler, cluster_labels, product_sim_df, product_list = load_models()

# Sidebar
st.sidebar.title("🛒 Shopper Spectrum")
st.sidebar.caption("E-Commerce Analytics")
st.sidebar.markdown("---")
module = st.sidebar.radio("Navigate:", ["🏠 Home", "🎯 Product Recommendations", "👥 Customer Segmentation"])

# ── HOME ──────────────────────────────────────────────────────
if module == "🏠 Home":
    st.title("🛒 Shopper Spectrum")
    st.subheader("Customer Segmentation & Product Recommendation in E-Commerce")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", "4,338")
    col2.metric("Total Products", "500+")
    col3.metric("Silhouette Score", "0.6162")
    col4.metric("Segments", "4")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.info("""
        ### 🎯 Product Recommendations
        Enter a **product name** → get **5 similar product recommendations**
        using **Item-based Collaborative Filtering** (Cosine Similarity).
        """)
    with col2:
        st.success("""
        ### 👥 Customer Segmentation
        Enter **Recency, Frequency, Monetary** values → predict **customer segment**
        using **KMeans Clustering** (K=4) on RFM features.
        """)

    st.markdown("---")
    st.markdown("### 📊 Customer Segments")
    seg_data = {
        "Segment": ["🌟 High-Value", "✅ Regular", "🔔 Occasional", "⚠️ At-Risk"],
        "Recency": ["Low (recent)", "Medium", "High", "Very High"],
        "Frequency": ["Very High", "High", "Low", "Low"],
        "Monetary": ["Very High", "Medium", "Low", "Low"],
        "Strategy": ["Reward & retain", "Upsell", "Re-engage", "Win-back campaigns"]
    }
    st.dataframe(pd.DataFrame(seg_data), use_container_width=True)

# ── PRODUCT RECOMMENDATIONS ───────────────────────────────────
elif module == "🎯 Product Recommendations":
    st.title("🎯 Product Recommendation System")
    st.caption("Item-based Collaborative Filtering · Cosine Similarity · Top 500 Products")
    st.markdown("---")

    col1, col2 = st.columns([3,1])
    with col1:
        selected = st.selectbox("Select a Product:", [""] + product_list)
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        top_n = st.selectbox("# Recommendations:", [5, 3, 10], index=0)

    manual = st.text_input("Or type a product name (partial search supported):")

    if st.button("🔍 Get Recommendations", type="primary", use_container_width=True):
        query = manual.strip().upper() if manual.strip() else selected.strip().upper()

        if not query:
            st.warning("Please select or enter a product name.")
        else:
            if query not in product_sim_df.index:
                matches = [p for p in product_sim_df.index if query in p]
                if not matches:
                    st.error(f"❌ '{query}' not found. Try a different name.")
                    st.stop()
                query = matches[0]
                st.info(f"Showing results for: **{query}**")

            similar = product_sim_df[query].sort_values(ascending=False)
            similar = similar[similar.index != query].head(top_n)

            st.success(f"✅ Top {top_n} products similar to: **{query}**")
            st.markdown("---")

            for rank, (prod, score) in enumerate(zip(similar.index, similar.values), 1):
                c1, c2, c3 = st.columns([1, 6, 2])
                c1.markdown(f"### {rank}")
                c2.markdown(f"**{prod.title()}**")
                c3.metric("Score", f"{score:.3f}")
                st.markdown("---")

# ── CUSTOMER SEGMENTATION ────────────────────────────────────
elif module == "👥 Customer Segmentation":
    st.title("👥 Customer Segment Predictor")
    st.caption("RFM Analysis · KMeans Clustering (K=4)")
    st.markdown("---")

    st.markdown("### Enter Customer RFM Values")
    col1, col2, col3 = st.columns(3)
    with col1:
        recency = st.number_input("📅 Recency (days since last purchase)", 0, 1000, 30)
        st.caption("Lower = more recent = better")
    with col2:
        frequency = st.number_input("🔁 Frequency (number of orders)", 1, 500, 5)
        st.caption("Higher = more loyal")
    with col3:
        monetary = st.number_input("💰 Monetary (total spend £)", 0.0, 500000.0, 500.0, 50.0)
        st.caption("Higher = more valuable")

    if st.button("🔮 Predict Segment", type="primary", use_container_width=True):
        rfm_input = np.array([[recency, frequency, monetary]])
        rfm_scaled = scaler.transform(rfm_input)
        cluster = kmeans.predict(rfm_scaled)[0]
        segment = cluster_labels.get(cluster, "Unknown")

        st.markdown("---")

        info = {
            'High-Value': {
                'emoji': '🌟', 'color': 'success',
                'desc': "Top performer — recent, frequent, high spending.",
                'action': "💡 Reward with VIP loyalty programs, early access & exclusive offers."
            },
            'Regular': {
                'emoji': '✅', 'color': 'info',
                'desc': "Steady buyer — consistent but not premium.",
                'action': "💡 Upsell with personalized recommendations & bundle deals."
            },
            'Occasional': {
                'emoji': '🔔', 'color': 'warning',
                'desc': "Infrequent buyer — buys rarely.",
                'action': "💡 Re-engage with seasonal promotions & reminder emails."
            },
            'At-Risk': {
                'emoji': '⚠️', 'color': 'error',
                'desc': "Haven't purchased in a long time.",
                'action': "💡 Launch immediate win-back campaign with special discount."
            }
        }

        d = info.get(segment, info['Occasional'])
        fn = getattr(st, d['color'])
        fn(f"### {d['emoji']} Segment: **{segment}**\n{d['desc']}\n\n{d['action']}")

        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        c1.metric("📅 Recency", f"{recency} days")
        c2.metric("🔁 Frequency", f"{frequency} orders")
        c3.metric("💰 Monetary", f"£{monetary:,.2f}")

        st.markdown("### 📊 Your Profile vs Segment Averages")
        seg_avg = {
            'High-Value': {'Recency': 7,   'Frequency': 82, 'Monetary': 127187},
            'Regular':    {'Recency': 15,  'Frequency': 22, 'Monetary': 12690},
            'Occasional': {'Recency': 43,  'Frequency': 3,  'Monetary': 1353},
            'At-Risk':    {'Recency': 248, 'Frequency': 1,  'Monetary': 478},
        }
        avg = seg_avg.get(segment, seg_avg['Occasional'])
        compare_df = pd.DataFrame({
            'Metric':        ['Recency (days)', 'Frequency', 'Monetary (£)'],
            'Your Value':    [recency, frequency, monetary],
            'Segment Avg':   [avg['Recency'], avg['Frequency'], avg['Monetary']]
        })
        st.dataframe(compare_df, use_container_width=True)
