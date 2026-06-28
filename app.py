import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image
import os

st.set_page_config(page_title="🛒 Shopper Spectrum", page_icon="🛒", layout="wide")

# ── Dark Theme CSS ────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background-color: #0e1117; }
[data-testid="stSidebar"] { background-color: #1a1a2e; }
[data-testid="metric-container"] {
    background: #1a1a2e; border: 1px solid #333;
    border-radius: 10px; padding: 12px;
}
h1,h2,h3,h4 { color: #ffffff !important; }
p, label { color: #cccccc !important; }
.stRadio label { color: #cccccc !important; }
div[data-testid="stMetricValue"] { color: #00d4ff !important; font-size: 1.8rem !important; }
div[data-testid="stMetricLabel"] { color: #aaaaaa !important; }
.insight-box {
    background: #1a1a2e; border-left: 4px solid #00d4ff;
    padding: 12px 16px; border-radius: 6px; margin: 8px 0;
    color: #ccc;
}
</style>
""", unsafe_allow_html=True)

# ── Load Models ───────────────────────────────────────────────
@st.cache_resource
def load_all():
    with open('models/kmeans_model.pkl','rb') as f: kmeans = pickle.load(f)
    with open('models/scaler.pkl','rb') as f: scaler = pickle.load(f)
    with open('models/cluster_labels.pkl','rb') as f: cluster_labels = pickle.load(f)
    with open('models/product_similarity_small.pkl','rb') as f: psim = pickle.load(f)
    with open('models/product_list.pkl','rb') as f: plist = pickle.load(f)
    with open('models/summary.pkl','rb') as f: summary = pickle.load(f)
    return kmeans, scaler, cluster_labels, psim, plist, summary

kmeans, scaler, cluster_labels, psim, plist, summary = load_all()

def show_plot(path, caption=None):
    if os.path.exists(path):
        img = Image.open(path)
        st.image(img, use_container_width=True, caption=caption)
    else:
        st.warning(f"Plot not found: {path}")

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.markdown("## 🛒 Shopper Spectrum")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", [
    "Executive Dashboard",
    "Sales Analytics",
    "Country Analysis",
    "RFM Analysis",
    "Elbow Method",
    "Customer Segmentation",
    "Similarity Matrix",
    "Product Recommendation",
    "Customer Prediction",
    "Business Insights"
])

# ══════════════════════════════════════════════════════════════
# 1. EXECUTIVE DASHBOARD
# ══════════════════════════════════════════════════════════════
if page == "Executive Dashboard":
    st.title("📊 Executive Dashboard")
    st.markdown("*High-level overview of business performance*")
    st.markdown("---")

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("👥 Customers",    f"{summary['total_customers']:,}")
    c2.metric("📦 Products",     f"{summary['total_products']:,}")
    c3.metric("💰 Total Revenue",f"£{summary['total_revenue']/1000:.0f}K")
    c4.metric("🧾 Transactions", f"{summary['total_transactions']:,}")
    c5.metric("🌍 Countries",    f"{summary['total_countries']}")

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.metric("🏆 Top Country",    summary['top_country'])
    c2.metric("⭐ Avg Order Value", f"£{summary['avg_order_value']:.2f}")
    c3.metric("🎯 Silhouette Score",f"{summary['silhouette_score']:.4f}")

    st.markdown("---")
    st.markdown("### 📈 Monthly Revenue Trend")
    show_plot("plots/monthly_revenue.png")

    st.markdown("### 🏆 Top Products")
    show_plot("plots/top_products.png")

# ══════════════════════════════════════════════════════════════
# 2. SALES ANALYTICS
# ══════════════════════════════════════════════════════════════
elif page == "Sales Analytics":
    st.title("📈 Sales Analytics")
    st.markdown("*Revenue trends and product performance*")
    st.markdown("---")

    st.markdown("### 📅 Monthly Revenue Trend")
    show_plot("plots/monthly_revenue.png")

    st.markdown("---")
    st.markdown("### 🏆 Top 10 Best-Selling Products")
    show_plot("plots/top_products.png")

    st.markdown("---")
    st.markdown("### 📊 Key Sales Metrics")
    monthly = summary['monthly_revenue']
    months = list(monthly.keys())
    revenues = list(monthly.values())

    c1, c2, c3 = st.columns(3)
    c1.metric("Peak Revenue Month", months[revenues.index(max(revenues))], f"£{max(revenues):.0f}K")
    c2.metric("Lowest Revenue Month", months[revenues.index(min(revenues))], f"£{min(revenues):.0f}K")
    c3.metric("Avg Monthly Revenue", f"£{sum(revenues)/len(revenues):.0f}K")

# ══════════════════════════════════════════════════════════════
# 3. COUNTRY ANALYSIS
# ══════════════════════════════════════════════════════════════
elif page == "Country Analysis":
    st.title("🌍 Country Analysis")
    st.markdown("*Geographic distribution of sales and customers*")
    st.markdown("---")
    show_plot("plots/country_analysis.png")

    st.markdown("---")
    st.markdown(f"""
    <div class="insight-box">💡 <b>Top Country:</b> {summary['top_country']} generates the highest revenue.<br>
    🌐 <b>Global Reach:</b> {summary['total_countries']} countries served worldwide.</div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# 4. RFM ANALYSIS
# ══════════════════════════════════════════════════════════════
elif page == "RFM Analysis":
    st.title("📐 RFM Analysis")
    st.markdown("*Recency · Frequency · Monetary — the foundation of customer segmentation*")
    st.markdown("---")

    c1,c2,c3 = st.columns(3)
    c1.markdown("""
    **📅 Recency**
    Days since last purchase.
    Lower = more recent = better customer.
    """)
    c2.markdown("""
    **🔁 Frequency**
    Number of orders placed.
    Higher = more loyal customer.
    """)
    c3.markdown("""
    **💰 Monetary**
    Total amount spent.
    Higher = more valuable customer.
    """)

    st.markdown("---")
    st.markdown("### 📊 RFM Distribution Charts")
    show_plot("plots/rfm_distributions.png")

    st.markdown("---")
    st.markdown("### 🔥 Cluster Profile Heatmap")
    show_plot("plots/cluster_heatmap.png")

    st.markdown("---")
    st.markdown("### 📋 Segment RFM Averages")
    cp = summary['cluster_profile']
    profile_df = pd.DataFrame({
        seg: {
            'Recency (days)': cp['Recency'].get(seg, '-'),
            'Frequency (orders)': cp['Frequency'].get(seg, '-'),
            'Monetary (£)': f"£{cp['Monetary'].get(seg, 0):,.0f}"
        } for seg in ['High-Value','Regular','Occasional','At-Risk'] if seg in cp['Recency']
    }).T
    st.dataframe(profile_df, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# 5. ELBOW METHOD
# ══════════════════════════════════════════════════════════════
elif page == "Elbow Method":
    st.title("📉 Elbow Method")
    st.markdown("*Finding the optimal number of clusters (K)*")
    st.markdown("---")

    st.markdown("""
    The **Elbow Method** helps us decide how many customer groups (K) to create.
    We look for the point where inertia stops decreasing sharply — like an **elbow** in the curve.
    The **Silhouette Score** confirms the best K by measuring how well-separated the clusters are.
    """)

    st.markdown("---")
    show_plot("plots/elbow_curve.png")

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.metric("Optimal K",         "4")
    c2.metric("Inertia at K=4",    f"{summary['inertia']:.0f}")
    c3.metric("Silhouette Score",  f"{summary['silhouette_score']:.4f}")

    st.markdown("""
    <div class="insight-box">
    ✅ <b>K=4 is optimal</b> — beyond K=4 the inertia reduction becomes minimal (diminishing returns).<br>
    ✅ <b>Silhouette Score 0.61</b> indicates well-separated, meaningful clusters.
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# 6. CUSTOMER SEGMENTATION
# ══════════════════════════════════════════════════════════════
elif page == "Customer Segmentation":
    st.title("👥 Customer Segmentation")
    st.markdown("*KMeans Clustering on RFM features — K=4*")
    st.markdown("---")

    sc = summary['seg_counts']
    sr = summary['seg_revenue']
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("🌟 High-Value",  sc.get('High-Value',0),  f"£{sr.get('High-Value',0)/1000:.0f}K revenue")
    c2.metric("✅ Regular",     sc.get('Regular',0),     f"£{sr.get('Regular',0)/1000:.0f}K revenue")
    c3.metric("🔔 Occasional",  sc.get('Occasional',0),  f"£{sr.get('Occasional',0)/1000:.0f}K revenue")
    c4.metric("⚠️ At-Risk",     sc.get('At-Risk',0),     f"£{sr.get('At-Risk',0)/1000:.0f}K revenue")

    st.markdown("---")
    st.markdown("### 🔵 PCA Cluster Visualization")
    show_plot("plots/pca_clusters.png")

    st.markdown("---")
    st.markdown("### 📊 Segment Distribution & Revenue")
    show_plot("plots/segment_analysis.png")

    st.markdown("---")
    st.markdown("### 🔥 Cluster Profile Heatmap")
    show_plot("plots/cluster_heatmap.png")

    st.markdown("---")
    st.markdown("### 💡 Segment Descriptions")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **🌟 High-Value**
        Recent buyers, purchase often, spend the most.
        Strategy: VIP rewards, exclusive offers, loyalty programs.

        **✅ Regular**
        Consistent buyers with moderate spend.
        Strategy: Upsell, bundle deals, personalized offers.
        """)
    with col2:
        st.markdown("""
        **🔔 Occasional**
        Buy rarely, low spend.
        Strategy: Re-engagement emails, seasonal promotions.

        **⚠️ At-Risk**
        Long time since last purchase.
        Strategy: Win-back campaigns, special discounts.
        """)

# ══════════════════════════════════════════════════════════════
# 7. SIMILARITY MATRIX
# ══════════════════════════════════════════════════════════════
elif page == "Similarity Matrix":
    st.title("🔗 Product Similarity Matrix")
    st.markdown("*Cosine similarity between top 15 products*")
    st.markdown("---")

    st.markdown("""
    This heatmap shows how similar products are to each other based on
    **co-purchase patterns**. A score close to **1.0** means customers
    who buy one product also tend to buy the other.
    """)

    show_plot("plots/similarity_matrix.png")

    st.markdown("""
    <div class="insight-box">
    🔵 <b>Dark blue</b> = very similar products (frequently bought together)<br>
    ⚪ <b>Light</b> = less similar products<br>
    This matrix powers the product recommendation engine.
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# 8. PRODUCT RECOMMENDATION
# ══════════════════════════════════════════════════════════════
elif page == "Product Recommendation":
    st.title("🎯 Product Recommendation System")
    st.caption("Item-based Collaborative Filtering · Cosine Similarity")
    st.markdown("---")

    c1, c2 = st.columns([3,1])
    with c1:
        selected = st.selectbox("Select a Product:", [""] + plist)
    with c2:
        st.markdown("<br>", unsafe_allow_html=True)
        top_n = st.selectbox("# Recommendations:", [5, 3, 10])

    manual = st.text_input("Or type a product name (partial search supported):")

    if st.button("🔍 Get Recommendations", type="primary", use_container_width=True):
        query = manual.strip().upper() if manual.strip() else selected.strip().upper()
        if not query:
            st.warning("Please select or enter a product name.")
        else:
            if query not in psim.index:
                matches = [p for p in psim.index if query in p]
                if not matches:
                    st.error(f"❌ '{query}' not found. Try partial name.")
                    st.stop()
                query = matches[0]
                st.info(f"Using closest match: **{query}**")

            similar = psim[query].sort_values(ascending=False)
            similar = similar[similar.index != query].head(top_n)

            st.success(f"✅ Top {top_n} products similar to: **{query.title()}**")
            st.markdown("---")
            for rank, (prod, score) in enumerate(zip(similar.index, similar.values), 1):
                c1,c2,c3 = st.columns([1,6,2])
                c1.markdown(f"### {rank}")
                c2.markdown(f"**{prod.title()}**")
                c3.metric("Similarity", f"{score:.3f}")
                st.markdown("---")

# ══════════════════════════════════════════════════════════════
# 9. CUSTOMER PREDICTION
# ══════════════════════════════════════════════════════════════
elif page == "Customer Prediction":
    st.title("🔮 Customer Segment Prediction")
    st.caption("Enter RFM values to predict which segment a customer belongs to")
    st.markdown("---")

    c1,c2,c3 = st.columns(3)
    with c1:
        recency  = st.number_input("📅 Recency (days since last purchase)", 0, 1000, 30)
        st.caption("Lower = more recent = better")
    with c2:
        frequency = st.number_input("🔁 Frequency (number of orders)", 1, 500, 5)
        st.caption("Higher = more loyal")
    with c3:
        monetary = st.number_input("💰 Monetary (total spend £)", 0.0, 500000.0, 500.0, 50.0)
        st.caption("Higher = more valuable")

    if st.button("🔮 Predict Segment", type="primary", use_container_width=True):
        inp = np.array([[recency, frequency, monetary]])
        inp_scaled = scaler.transform(inp)
        cluster = kmeans.predict(inp_scaled)[0]
        segment = cluster_labels.get(cluster, "Unknown")

        st.markdown("---")
        info = {
            'High-Value': ('🌟','success',"Top performer — recent, frequent, high spending.","Reward with VIP loyalty programs, early access & exclusive offers."),
            'Regular':    ('✅','info',   "Steady buyer — consistent but not premium.",       "Upsell with personalized recommendations & bundle deals."),
            'Occasional': ('🔔','warning',"Infrequent buyer — buys rarely.",                  "Re-engage with seasonal promotions & reminder emails."),
            'At-Risk':    ('⚠️','error',  "Haven't purchased in a long time.",                "Launch immediate win-back campaign with special discount."),
        }
        emoji, color, desc, action = info.get(segment, info['Occasional'])
        getattr(st, color)(f"### {emoji} Segment: **{segment}**\n{desc}\n\n💡 {action}")

        st.markdown("---")
        c1,c2,c3 = st.columns(3)
        c1.metric("📅 Recency",   f"{recency} days")
        c2.metric("🔁 Frequency", f"{frequency} orders")
        c3.metric("💰 Monetary",  f"£{monetary:,.2f}")

        st.markdown("### 📊 Your Profile vs Segment Averages")
        seg_avg = {
            'High-Value': (7,   82, 127187),
            'Regular':    (15,  22, 12690),
            'Occasional': (43,  3,  1353),
            'At-Risk':    (248, 1,  478),
        }
        r_avg, f_avg, m_avg = seg_avg.get(segment, (43,3,1353))
        st.dataframe(pd.DataFrame({
            'Metric':       ['Recency (days)', 'Frequency', 'Monetary (£)'],
            'Your Value':   [recency,  frequency,  f"£{monetary:,.0f}"],
            'Segment Avg':  [r_avg,    f_avg,      f"£{m_avg:,}"],
        }), use_container_width=True)

# ══════════════════════════════════════════════════════════════
# 10. BUSINESS INSIGHTS
# ══════════════════════════════════════════════════════════════
elif page == "Business Insights":
    st.title("💡 Business Insights")
    st.markdown("*Key findings and actionable recommendations*")
    st.markdown("---")

    sc = summary['seg_counts']
    sr = summary['seg_revenue']
    total_rev = summary['total_revenue']

    st.markdown("### 🔑 Key Findings")
    col1, col2 = st.columns(2)
    with col1:
        hv_pct = sc.get('High-Value',0)/summary['total_customers']*100
        hv_rev_pct = sr.get('High-Value',0)/total_rev*100
        ar_pct = sc.get('At-Risk',0)/summary['total_customers']*100

        st.markdown(f"""
        <div class="insight-box">
        🌟 <b>High-Value customers</b> are only <b>{hv_pct:.1f}%</b> of total
        customers but contribute <b>{hv_rev_pct:.1f}%</b> of revenue.
        </div>
        <div class="insight-box">
        ⚠️ <b>{ar_pct:.1f}%</b> of customers are At-Risk —
        targeted win-back campaigns can recover significant revenue.
        </div>
        <div class="insight-box">
        🌍 <b>{summary['top_country']}</b> is the top revenue-generating country.
        </div>
        """, unsafe_allow_html=True)
    with col2:
        occ_pct = sc.get('Occasional',0)/summary['total_customers']*100
        st.markdown(f"""
        <div class="insight-box">
        🔔 <b>{occ_pct:.1f}%</b> customers are Occasional buyers —
        a huge opportunity for conversion with the right campaigns.
        </div>
        <div class="insight-box">
        ⭐ Average order value is <b>£{summary['avg_order_value']:.2f}</b> —
        bundle offers can increase this significantly.
        </div>
        <div class="insight-box">
        🎯 Clustering Silhouette Score: <b>{summary['silhouette_score']:.4f}</b> —
        indicates well-separated, meaningful customer groups.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📋 Recommended Actions by Segment")
    actions = pd.DataFrame({
        'Segment':   ['🌟 High-Value','✅ Regular','🔔 Occasional','⚠️ At-Risk'],
        'Size':      [sc.get('High-Value',0), sc.get('Regular',0), sc.get('Occasional',0), sc.get('At-Risk',0)],
        'Revenue':   [f"£{sr.get('High-Value',0)/1000:.0f}K", f"£{sr.get('Regular',0)/1000:.0f}K",
                      f"£{sr.get('Occasional',0)/1000:.0f}K", f"£{sr.get('At-Risk',0)/1000:.0f}K"],
        'Action':    ['VIP rewards + exclusive access','Bundle deals + upsell','Seasonal promos + reminders','Win-back emails + discounts'],
        'Priority':  ['🔴 High','🟡 Medium','🟢 Normal','🔴 Urgent'],
    })
    st.dataframe(actions, use_container_width=True, hide_index=True)
