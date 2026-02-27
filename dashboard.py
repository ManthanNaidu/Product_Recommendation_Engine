import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------

st.set_page_config(
    page_title="Customer Intelligence Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------------------------------------
# STYLING
# -------------------------------------------------

st.markdown("""
<style>
.main {
    background-color: #F8F9FB;
}
.metric-container {
    background: white;
    padding: 20px;
    border-radius: 16px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.04);
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# TITLE
# -------------------------------------------------

st.markdown("## 🛒 Customer Intelligence Platform")
st.markdown(
    "<span style='color:#6B7280'>Segmentation • Personalization • Insights</span>",
    unsafe_allow_html=True
)
st.markdown("---")

# -------------------------------------------------
# LOAD DATA (PATH SAFE)
# -------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

@st.cache_data
def load_data():
    customers = pd.read_csv(BASE_DIR / "data/processed/mall_customers_segmented.csv")
    transactions = pd.read_csv(BASE_DIR / "data/ecommerce_transactions.csv")
    return customers, transactions

customers, transactions = load_data()

# -------------------------------------------------
# BUILD USER ITEM MATRIX
# -------------------------------------------------

user_item = transactions.pivot_table(
    index="user_id",
    columns="product_id",
    values="quantity",
    aggfunc="sum",
    fill_value=0
)

similarity = cosine_similarity(user_item)

similarity_df = pd.DataFrame(
    similarity,
    index=user_item.index,
    columns=user_item.index
)

# -------------------------------------------------
# SAFE PRODUCTS TABLE
# -------------------------------------------------

if "price_x" in transactions.columns:
    products = transactions[["product_id", "category", "price_x"]].drop_duplicates()
    products = products.rename(columns={"price_x": "price"})
elif "price" in transactions.columns:
    products = transactions[["product_id", "category", "price"]].drop_duplicates()
else:
    products = transactions[["product_id", "category"]].drop_duplicates()
    products["price"] = "N/A"

# -------------------------------------------------
# MODEL PERFORMANCE METRICS
# -------------------------------------------------

# Silhouette Score
try:
    X = customers[["Annual Income ($)", "Spending Score (1-100)"]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    silhouette = silhouette_score(X_scaled, customers["cluster"])
    silhouette = round(silhouette, 3)

except:
    silhouette = "N/A"

# Precision@5
def precision_at_k(user_item_matrix, similarity_df, k=5):

    precisions = []

    for user in user_item_matrix.index[:100]:

        actual = set(
            user_item_matrix.loc[user][
                user_item_matrix.loc[user] > 0
            ].index
        )

        similar_users = similarity_df[user].sort_values(ascending=False)[1:11]
        purchases = user_item_matrix.loc[similar_users.index]

        scores = np.dot(similar_users.values, purchases)
        scores = pd.Series(scores, index=user_item_matrix.columns)

        recommended = set(scores.sort_values(ascending=False).head(k).index)

        if len(recommended) == 0:
            continue

        precision = len(actual & recommended) / len(recommended)
        precisions.append(precision)

    return round(sum(precisions) / len(precisions), 3)

try:
    precision5 = precision_at_k(user_item, similarity_df)
except:
    precision5 = "N/A"

# -------------------------------------------------
# KPI SECTION
# -------------------------------------------------

c1, c2, c3, c4, c5, c6 = st.columns(6)

with c1:
    st.markdown(f"""
    <div class="metric-container">
    <h3>{customers.shape[0]}</h3>
    <p>Total Customers</p>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-container">
    <h3>{transactions.shape[0]}</h3>
    <p>Total Transactions</p>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-container">
    <h3>{round(customers["Spending Score (1-100)"].mean(),2)}</h3>
    <p>Avg Spending Score</p>
    </div>
    """, unsafe_allow_html=True)

with c4:
    premium = customers[
        customers["segment"] == "High Income - High Spending"
    ].shape[0]

    st.markdown(f"""
    <div class="metric-container">
    <h3>{premium}</h3>
    <p>Premium Customers</p>
    </div>
    """, unsafe_allow_html=True)

with c5:
    st.markdown(f"""
    <div class="metric-container">
    <h3>{silhouette}</h3>
    <p>Clustering Score</p>
    </div>
    """, unsafe_allow_html=True)

with c6:
    st.markdown(f"""
    <div class="metric-container">
    <h3>{precision5}</h3>
    <p>Recommendation Precision@5</p>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------------------------
# SEGMENTATION VISUALIZATION
# -------------------------------------------------

fig = px.scatter(
    customers,
    x="Annual Income ($)",
    y="Spending Score (1-100)",
    color="segment",
    hover_data=["Age"],
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------
# SEGMENT DISTRIBUTION
# -------------------------------------------------

segment_counts = customers["segment"].value_counts().reset_index()
segment_counts.columns = ["segment", "count"]

fig2 = px.bar(
    segment_counts,
    x="segment",
    y="count",
    color="segment",
    template="plotly_white"
)

st.plotly_chart(fig2, use_container_width=True)

# -------------------------------------------------
# RECOMMENDATION ENGINE WITH CONFIDENCE
# -------------------------------------------------

def recommend_products(user_id, n=5):

    if user_id not in similarity_df.index:
        return None

    similar_users = similarity_df[user_id].sort_values(ascending=False)[1:11]

    purchases = user_item.loc[similar_users.index]

    scores = np.dot(similar_users.values, purchases)

    scores = pd.Series(scores, index=user_item.columns)

    bought = user_item.loc[user_id]

    scores = scores[bought == 0]

    recs = scores.sort_values(ascending=False).head(n)

    result = products[
        products["product_id"].isin(recs.index)
    ].copy()

    result["confidence_score"] = result["product_id"].map(recs)

    result["confidence_score"] = (
        result["confidence_score"] /
        result["confidence_score"].max()
    ).round(3)

    return result.sort_values("confidence_score", ascending=False)

# -------------------------------------------------
# USER RECOMMENDATION PANEL
# -------------------------------------------------

st.markdown("---")
st.markdown("## 🎯 Personalized Recommendations")

user_id = st.number_input(
    "Customer ID",
    min_value=int(customers["CustomerID"].min()),
    max_value=int(customers["CustomerID"].max()),
    step=1
)

# Show customer segment
if user_id in customers["CustomerID"].values:

    segment = customers[
        customers["CustomerID"] == user_id
    ]["segment"].values[0]

    st.info(f"Customer Segment: {segment}")

if st.button("Generate Recommendations"):

    recs = recommend_products(user_id)

    if recs is not None and not recs.empty:

        st.success("Based on similar customer behavior")

        st.dataframe(
            recs,
            use_container_width=True
        )

    else:

        st.warning("No recommendations available")

# -------------------------------------------------
# END
# -------------------------------------------------
