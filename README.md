🛒 SHOPPER SPECTRUM — PROJECT FILES
=====================================

FILES:
------
app.py                  → Streamlit Web App (main file)
requirements.txt        → Python dependencies
models/
  kmeans_model.pkl      → Trained KMeans (K=4)
  scaler.pkl            → StandardScaler for RFM
  cluster_labels.pkl    → Cluster → Segment name mapping
  product_similarity.pkl→ Product cosine similarity matrix (3866x3866)
  product_list.pkl      → List of all product names
plots/
  eda_countries.png     → Country analysis charts
  eda_top_products.png  → Top products bar chart
  eda_monthly_trend.png → Monthly revenue trend
  elbow_curve.png       → Elbow + Silhouette score plots
  customer_clusters.png → PCA cluster scatter plot
  rfm_distributions.png → RFM distribution histograms

RESULTS:
--------
✅ Dataset: 541,909 rows → 392,692 after cleaning
✅ Customers: 4,338
✅ Products: 3,866
✅ Silhouette Score: 0.6162 (good clustering!)
✅ Segments: High-Value (13), Regular (204), Occasional (3054), At-Risk (1067)

HOW TO RUN LOCALLY:
-------------------
1. Put all files in one folder
2. pip install -r requirements.txt
3. streamlit run app.py
4. Open: http://localhost:8501

HOW TO DEPLOY ON STREAMLIT CLOUD (FREE):
------------------------------------------
1. Upload all files to a GitHub repo
2. Go to https://share.streamlit.io
3. Connect your GitHub repo
4. Set main file: app.py
5. Deploy! You'll get a public URL like:
   https://your-app-name.streamlit.app
