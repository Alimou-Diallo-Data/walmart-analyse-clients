import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

# =========================================================
# Deep Learning optionnel
# =========================================================
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Input
    from tensorflow.keras.callbacks import EarlyStopping
    DL_AVAILABLE = True
except Exception:
    DL_AVAILABLE = False

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="Executive AI Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATA_FILE = "Walmart_customer_purchases.csv"

# =========================================================
# CSS PREMIUM
# =========================================================
st.markdown("""
<style>
    html, body, [class*="css"] {
        font-family: "Inter", sans-serif;
    }

    .stApp {
        background: linear-gradient(90deg, #03060d 0%, #050a14 100%);
        color: #f3f4f6;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #262733 0%, #2b2c39 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }

    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 1550px;
    }

    .top-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 0.2rem;
    }

    .top-subtitle {
        color: #9ca3af;
        font-size: 1rem;
        margin-bottom: 1rem;
    }

    .divider-line {
        height: 1px;
        background: rgba(255,255,255,0.10);
        margin-top: 0.5rem;
        margin-bottom: 1rem;
    }

    [data-testid="stMetric"] {
        background: #f9fafb;
        border-radius: 18px;
        padding: 16px;
        box-shadow: 0 4px 18px rgba(0,0,0,0.18);
        border: 1px solid rgba(0,0,0,0.03);
    }

    [data-testid="stMetricLabel"] {
        color: #6b7280;
        font-weight: 700;
    }

    [data-testid="stMetricValue"] {
        color: #111827 !important;
        font-weight: 800;
    }

    .section-title {
        color: #f3f4f6;
        font-size: 1.05rem;
        font-weight: 800;
        margin-bottom: 0.45rem;
    }

    .section-subtitle {
        color: #9ca3af;
        font-size: 0.92rem;
        margin-bottom: 0.9rem;
    }

    .sidebar-title {
        font-size: 1.1rem;
        font-weight: 800;
        color: white;
        margin-bottom: 0.8rem;
    }

    .sidebar-label {
        color: #ffffff;
        font-weight: 700;
        font-size: 0.95rem;
        margin-bottom: 0.5rem;
    }

    .mini-divider {
        height: 1px;
        background: rgba(255,255,255,0.15);
        margin: 16px 0 16px 0;
    }

    .insight-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 16px;
        padding: 1rem;
        color: #e5e7eb;
        min-height: 120px;
    }

    .insight-title {
        font-weight: 800;
        margin-bottom: 0.35rem;
        color: #ffffff;
    }

    .insight-text {
        color: #cbd5e1;
        font-size: 0.93rem;
    }

    .stTabs [data-baseweb="tab"] {
        color: #e5e7eb;
        font-weight: 700;
        font-size: 15px;
    }

    .stTabs [aria-selected="true"] {
        color: white !important;
    }

    div[data-baseweb="select"] > div {
        background-color: #0b1220 !important;
        border: 1px solid rgba(255,255,255,0.10) !important;
        border-radius: 12px !important;
        color: white !important;
    }

    .stMultiSelect label, .stSlider label, .stSelectbox label, .stNumberInput label {
        color: #f3f4f6 !important;
        font-weight: 700 !important;
    }

    .stButton > button, .stDownloadButton > button, div[data-testid="stFormSubmitButton"] > button {
        width: 100%;
        border-radius: 12px;
        height: 44px;
        font-weight: 700;
    }

    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# HELPERS STYLE
# =========================================================
def dark_layout(fig, height=360):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e5e7eb"),
        margin=dict(l=10, r=10, t=45, b=20),
        height=height,
        legend=dict(
            orientation="h",
            y=1.08,
            x=0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e5e7eb")
        )
    )
    return fig

def loyal_mask(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["yes", "oui", "1", "true"])

def safe_qcut(series: pd.Series, q: int, labels):
    s = series.rank(method="first")
    unique_count = s.nunique()
    bins = min(q, unique_count)
    if bins < 2:
        return pd.Series([labels[-1]] * len(series), index=series.index)
    return pd.qcut(s, q=bins, labels=labels[:bins], duplicates="drop")

# =========================================================
# LOAD DATA
# =========================================================
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Le fichier '{path}' est introuvable. Mets-le dans le même dossier que app.py."
        )

    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    for col in ["Age", "Purchase_Amount", "Rating"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")

    if "Age" in df.columns:
        df["Age_Group"] = pd.cut(
            df["Age"],
            bins=[0, 25, 35, 50, 100],
            labels=["18-25", "26-35", "36-50", "51+"],
            include_lowest=True
        )

    return df

def ensure_product_category(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    if "Product_Category" not in data.columns and "Purchase_Amount" in data.columns:
        data["Product_Category"] = pd.cut(
            data["Purchase_Amount"],
            bins=[0, 50, 100, 200, 500, np.inf],
            labels=["Budget", "Standard", "Plus", "Premium", "Elite"],
            include_lowest=True
        )
    return data

# =========================================================
# FEATURE ENGINEERING
# =========================================================
@st.cache_data(show_spinner=False)
def enrich_data(df: pd.DataFrame) -> pd.DataFrame:
    data = ensure_product_category(df).copy()

    if "Purchase_Amount" in data.columns and "Rating" in data.columns:
        max_purchase = max(float(data["Purchase_Amount"].max(skipna=True) or 0), 1.0)
        loyal = loyal_mask(data["Repeat_Customer"]).astype(int) if "Repeat_Customer" in data.columns else 0

        data["Customer_Score"] = (
            0.45 * (data["Purchase_Amount"].fillna(0) / max_purchase) +
            0.30 * (data["Rating"].fillna(0) / 5.0) +
            0.25 * loyal
        ) * 100

        rating_factor = data["Rating"].fillna(3) / 5.0
        data["Estimated_LTV"] = (
            data["Purchase_Amount"].fillna(0) *
            (1.2 + 1.8 * loyal + 0.8 * rating_factor)
        )

    else:
        data["Customer_Score"] = 0.0
        data["Estimated_LTV"] = 0.0

    def next_best_action(row):
        score = row.get("Customer_Score", 0)
        rating = row.get("Rating", 0)
        amount = row.get("Purchase_Amount", 0)
        repeat = str(row.get("Repeat_Customer", "")).lower() in ["yes", "oui", "1", "true"]

        if repeat and score >= 75:
            return "🎁 Offre VIP"
        if repeat and amount >= 150:
            return "⬆️ Upsell Premium"
        if rating >= 4 and not repeat:
            return "📩 Relance personnalisée"
        if rating < 3:
            return "🛠️ Action service client"
        if amount < 50 and not repeat:
            return "🏷️ Coupon réactivation"
        return "🛒 Campagne cross-sell"

    data["Next_Best_Action"] = data.apply(next_best_action, axis=1)

    return data

# =========================================================
# RFM
# =========================================================
@st.cache_data(show_spinner=False)
def compute_rfm(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["Customer_ID"] = np.arange(len(data))
    data["InvoiceDate"] = pd.date_range(end=pd.Timestamp.today(), periods=len(data), freq="D")

    rfm = data.groupby("Customer_ID", as_index=False).agg(
        Recency=("InvoiceDate", lambda x: (pd.Timestamp.today() - x.max()).days),
        Frequency=("Purchase_Amount", "count"),
        Monetary=("Purchase_Amount", "sum")
    )

    r_labels = [5, 4, 3, 2, 1]
    fm_labels = [1, 2, 3, 4, 5]

    rfm["R_score"] = safe_qcut(-rfm["Recency"], 5, r_labels)
    rfm["F_score"] = safe_qcut(rfm["Frequency"], 5, fm_labels)
    rfm["M_score"] = safe_qcut(rfm["Monetary"], 5, fm_labels)

    rfm["R_score"] = rfm["R_score"].astype(int)
    rfm["F_score"] = rfm["F_score"].astype(int)
    rfm["M_score"] = rfm["M_score"].astype(int)
    rfm["RFM_Score"] = rfm["R_score"] + rfm["F_score"] + rfm["M_score"]

    def assign_segment(score):
        if score >= 13:
            return "🔥 Champions"
        elif score >= 10:
            return "💎 Fidèles"
        elif score >= 7:
            return "⚡ Potentiel"
        return "🧊 À relancer"

    rfm["Segment"] = rfm["RFM_Score"].apply(assign_segment)
    return rfm

# =========================================================
# CLUSTERING
# =========================================================
@st.cache_data(show_spinner=False)
def compute_clustering(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["Age", "Purchase_Amount", "Rating"]
    data = df[cols].dropna().copy()

    if len(data) < 10:
        data["Cluster"] = 0
        return data

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)

    n_clusters = 4 if len(data) >= 20 else 3
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    data["Cluster"] = kmeans.fit_predict(X_scaled).astype(str)
    return data

# =========================================================
# MODELS
# =========================================================
@st.cache_resource(show_spinner=False)
def train_loyalty_model(df: pd.DataFrame):
    required = ["Age", "Purchase_Amount", "Rating", "Repeat_Customer"]
    if not all(c in df.columns for c in required):
        return None, None

    data = df[required].dropna().copy()
    le = LabelEncoder()
    data["Target"] = le.fit_transform(data["Repeat_Customer"].astype(str))

    X = data[["Age", "Purchase_Amount", "Rating"]]
    y = data["Target"]

    model = RandomForestClassifier(
        n_estimators=80,
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)
    return model, le

@st.cache_resource(show_spinner=False)
def train_churn_model(df: pd.DataFrame):
    required = ["Age", "Purchase_Amount", "Rating", "Repeat_Customer"]
    if not all(c in df.columns for c in required):
        return None

    data = df[required].dropna().copy()
    data["Churn"] = (~loyal_mask(data["Repeat_Customer"])).astype(int)

    X = data[["Age", "Purchase_Amount", "Rating"]]
    y = data["Churn"]

    model = RandomForestClassifier(
        n_estimators=80,
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)
    return model

@st.cache_resource(show_spinner=False)
def train_deep_learning_model(df: pd.DataFrame):
    if not DL_AVAILABLE:
        return None, None

    required = ["Age", "Purchase_Amount", "Rating", "Repeat_Customer"]
    if not all(c in df.columns for c in required):
        return None, None

    data = df[required].dropna().copy()
    data["Target"] = loyal_mask(data["Repeat_Customer"]).astype(int)

    X = data[["Age", "Purchase_Amount", "Rating"]].astype(float)
    y = data["Target"].astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(32, activation="relu"),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid")
    ])

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=25,
        batch_size=32,
        verbose=0,
        callbacks=[early_stop]
    )

    return model, scaler

# =========================================================
# CHARTS
# =========================================================
def create_value_distribution_chart(df: pd.DataFrame) -> go.Figure:
    values = df["Purchase_Amount"].dropna().astype(float)

    hist_counts, hist_bins = np.histogram(values, bins=28)
    x_vals = 0.5 * (hist_bins[1:] + hist_bins[:-1])
    y_vals = hist_counts

    purchase_mean = values.mean()
    purchase_median = values.median()

    fig = go.Figure()

    fig.add_bar(
        x=x_vals,
        y=y_vals,
        name="Distribution",
        marker=dict(color="#1e88ff", line=dict(color="#4fb3ff", width=1)),
        opacity=0.82,
        hovertemplate="<b>Montant:</b> %{x:.1f}<br><b>Transactions:</b> %{y}<extra></extra>"
    )

    fig.add_scatter(
        x=x_vals,
        y=y_vals,
        mode="lines",
        name="Tendance",
        line=dict(color="#00e5ff", width=3, shape="spline")
    )

    fig.add_vline(
        x=purchase_mean,
        line_width=2,
        line_dash="dash",
        line_color="#00e5ff",
        annotation_text=f"Moyenne: {purchase_mean:.1f}",
        annotation_position="top left"
    )

    fig.add_vline(
        x=purchase_median,
        line_width=2,
        line_dash="dot",
        line_color="#ffd166",
        annotation_text=f"Médiane: {purchase_median:.1f}",
        annotation_position="top right"
    )

    dark_layout(fig, height=420)
    fig.update_layout(
        xaxis_title="Montant d'achat",
        yaxis_title="Nombre de transactions",
        bargap=0.10
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)", zeroline=False)
    return fig

# =========================================================
# LOAD
# =========================================================
try:
    df_raw = load_data(DATA_FILE)
    df_enriched = enrich_data(df_raw)
    loyalty_model, loyalty_encoder = train_loyalty_model(df_raw)
    churn_model = train_churn_model(df_raw)
    dl_model, dl_scaler = train_deep_learning_model(df_raw)
except Exception as e:
    st.error(f"Erreur au chargement : {e}")
    st.stop()

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("<div class='sidebar-title'>Executive Suite</div>", unsafe_allow_html=True)

    page = st.radio(
        "Menu",
        ["📊 Dashboard Premium", "🧠 Simulateur IA"],
        index=0
    )

    st.markdown("<div class='mini-divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-label'>Filtres</div>", unsafe_allow_html=True)

    filtered_df = df_enriched.copy()

    if "Product_Category" in filtered_df.columns:
        categories = sorted(filtered_df["Product_Category"].dropna().astype(str).unique().tolist())
        selected_categories = st.multiselect(
            "Catégories / Valeur",
            options=categories,
            default=categories
        )
        if selected_categories:
            filtered_df = filtered_df[
                filtered_df["Product_Category"].astype(str).isin(selected_categories)
            ]

    if "Age_Group" in filtered_df.columns:
        age_groups = sorted(filtered_df["Age_Group"].dropna().astype(str).unique().tolist())
        selected_age = st.multiselect(
            "Segments d'âge",
            options=age_groups,
            default=age_groups
        )
        if selected_age:
            filtered_df = filtered_df[
                filtered_df["Age_Group"].astype(str).isin(selected_age)
            ]

    if "Purchase_Amount" in filtered_df.columns and not filtered_df["Purchase_Amount"].dropna().empty:
        min_amt = float(filtered_df["Purchase_Amount"].min())
        max_amt = float(filtered_df["Purchase_Amount"].max())
        amount_range = st.slider(
            "Montant d'achat",
            min_value=min_amt,
            max_value=max_amt,
            value=(min_amt, max_amt)
        )
        filtered_df = filtered_df[
            filtered_df["Purchase_Amount"].between(amount_range[0], amount_range[1])
        ]

    if "Rating" in filtered_df.columns and not filtered_df["Rating"].dropna().empty:
        min_rating = float(filtered_df["Rating"].min())
        max_rating = float(filtered_df["Rating"].max())
        rating_range = st.slider(
            "Note client",
            min_value=min_rating,
            max_value=max_rating,
            value=(min_rating, max_rating)
        )
        filtered_df = filtered_df[
            filtered_df["Rating"].between(rating_range[0], rating_range[1])
        ]

# =========================================================
# DASHBOARD
# =========================================================
if page == "📊 Dashboard Premium":
    df = filtered_df.copy()

    st.markdown("<div class='top-title'>📊 Executive Analytics Dashboard</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='top-subtitle'>Vue premium des ventes, segments, recommandations IA et valeur client.</div>",
        unsafe_allow_html=True
    )

    if df.empty:
        st.warning("Aucune donnée ne correspond aux filtres sélectionnés.")
        st.stop()

    total_sales = float(df["Purchase_Amount"].sum()) if "Purchase_Amount" in df.columns else 0
    total_customers = int(len(df))
    avg_score = float(df["Customer_Score"].mean()) if "Customer_Score" in df.columns else 0
    avg_ltv = float(df["Estimated_LTV"].mean()) if "Estimated_LTV" in df.columns else 0
    loyalty_rate = float(loyal_mask(df["Repeat_Customer"]).mean() * 100) if "Repeat_Customer" in df.columns else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Revenue Total", f"${total_sales:,.0f}")
    c2.metric("Clients", f"{total_customers:,}")
    c3.metric("Score Moyen", f"{avg_score:.1f}")
    c4.metric("LTV Moyen", f"${avg_ltv:,.0f}")
    c5.metric("Fidélité", f"{loyalty_rate:.1f}%")

    st.markdown("<div class='divider-line'></div>", unsafe_allow_html=True)

    left, right = st.columns([2.1, 1.1])

    with left:
        st.markdown("<div class='section-title'>📈 Analyse de la Valeur</div>", unsafe_allow_html=True)
        st.markdown("<div class='section-subtitle'>Répartition premium des montants d'achat avec tendance.</div>", unsafe_allow_html=True)
        if "Purchase_Amount" in df.columns and not df["Purchase_Amount"].dropna().empty:
            st.plotly_chart(
                create_value_distribution_chart(df),
                width="stretch",
                config={"displayModeBar": False}
            )

    with right:
        st.markdown("<div class='section-title'>💰 Segments de Valeur Client</div>", unsafe_allow_html=True)
        st.markdown("<div class='section-subtitle'>Répartition des clients par niveau de dépense.</div>", unsafe_allow_html=True)

        if "Product_Category" in df.columns:
            market_share = (
                df["Product_Category"]
                .astype(str)
                .value_counts()
                .rename_axis("Category")
                .reset_index(name="Count")
            )
            fig_pie = px.pie(
                market_share,
                names="Category",
                values="Count",
                hole=0.60
            )
            dark_layout(fig_pie, height=420)
            fig_pie.update_traces(
                textinfo="percent+label",
                marker=dict(line=dict(color="rgba(0,0,0,0)", width=0))
            )
            st.plotly_chart(fig_pie, width="stretch", config={"displayModeBar": False})

    st.markdown("<div class='divider-line'></div>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📌 Vue Exécutive",
        "👥 Segments",
        "🎯 Actions IA",
        "📈 Churn & LTV",
        "🤖 ML Avancé",
        "📋 Données"
    ])

    with tab1:
        i1, i2, i3 = st.columns(3)

        with i1:
            st.markdown(f"""
            <div class="insight-card">
                <div class="insight-title">Performance commerciale</div>
                <div class="insight-text">
                    Le revenu cumulé atteint <b>${total_sales:,.0f}</b> avec une base de <b>{total_customers:,}</b> clients analysés.
                </div>
            </div>
            """, unsafe_allow_html=True)

        with i2:
            best_action = df["Next_Best_Action"].value_counts().idxmax() if "Next_Best_Action" in df.columns else "N/A"
            st.markdown(f"""
            <div class="insight-card">
                <div class="insight-title">Action dominante</div>
                <div class="insight-text">
                    L’action la plus recommandée actuellement est <b>{best_action}</b>, ce qui donne une lecture marketing immédiate.
                </div>
            </div>
            """, unsafe_allow_html=True)

        with i3:
            st.markdown(f"""
            <div class="insight-card">
                <div class="insight-title">Valeur client</div>
                <div class="insight-text">
                    Le <b>LTV moyen</b> estimé est de <b>${avg_ltv:,.0f}</b>, avec un <b>score moyen</b> de <b>{avg_score:.1f}</b>.
                </div>
            </div>
            """, unsafe_allow_html=True)

        g1, g2 = st.columns(2)

        with g1:
            if "Age_Group" in df.columns and "Purchase_Amount" in df.columns:
                spend_by_age = (
                    df.groupby("Age_Group", observed=False)["Purchase_Amount"]
                    .mean()
                    .reset_index()
                )
                fig_bar = px.bar(
                    spend_by_age,
                    x="Age_Group",
                    y="Purchase_Amount",
                    text_auto=".2f",
                    title="Panier moyen par segment d'âge"
                )
                dark_layout(fig_bar, height=360)
                fig_bar.update_traces(marker_color="#0d8bff")
                st.plotly_chart(fig_bar, width="stretch", config={"displayModeBar": False})

        with g2:
            if "Rating" in df.columns:
                fig_rating = px.histogram(
                    df,
                    x="Rating",
                    nbins=15,
                    title="Distribution des notes clients"
                )
                dark_layout(fig_rating, height=360)
                fig_rating.update_traces(marker_color="#1f6feb")
                st.plotly_chart(fig_rating, width="stretch", config={"displayModeBar": False})

    with tab2:
        g1, g2 = st.columns(2)

        with g1:
            if "Age_Group" in df.columns:
                age_dist = (
                    df.groupby("Age_Group", observed=False)
                    .size()
                    .reset_index(name="Count")
                )
                fig_age = px.bar(
                    age_dist,
                    x="Age_Group",
                    y="Count",
                    text_auto=True,
                    title="Volume clients par segment"
                )
                dark_layout(fig_age, height=360)
                fig_age.update_traces(marker_color="#118dff")
                st.plotly_chart(fig_age, width="stretch", config={"displayModeBar": False})

        with g2:
            if "Repeat_Customer" in df.columns and "Age_Group" in df.columns:
                loyalty_by_age = (
                    df.groupby("Age_Group", observed=False)["Repeat_Customer"]
                    .apply(lambda x: loyal_mask(x).mean() * 100)
                    .reset_index(name="Loyalty_Rate")
                )
                fig_loyal = px.bar(
                    loyalty_by_age,
                    x="Age_Group",
                    y="Loyalty_Rate",
                    text_auto=".1f",
                    title="Taux de fidélité par segment"
                )
                dark_layout(fig_loyal, height=360)
                fig_loyal.update_traces(marker_color="#0d8bff")
                fig_loyal.update_yaxes(title="Taux (%)")
                st.plotly_chart(fig_loyal, width="stretch", config={"displayModeBar": False})

        rfm = compute_rfm(df)
        c_rfm1, c_rfm2 = st.columns(2)

        with c_rfm1:
            rfm_seg = (
                rfm["Segment"]
                .value_counts()
                .rename_axis("Segment")
                .reset_index(name="Count")
            )
            fig_rfm = px.pie(rfm_seg, names="Segment", values="Count", hole=0.55, title="Segmentation RFM")
            dark_layout(fig_rfm, height=360)
            st.plotly_chart(fig_rfm, width="stretch", config={"displayModeBar": False})

        with c_rfm2:
            clusters_df = compute_clustering(df)
            if not clusters_df.empty and "Cluster" in clusters_df.columns:
                fig_cluster = px.scatter(
                    clusters_df,
                    x="Purchase_Amount",
                    y="Rating",
                    color="Cluster",
                    title="Clustering ML des clients"
                )
                dark_layout(fig_cluster, height=360)
                st.plotly_chart(fig_cluster, width="stretch", config={"displayModeBar": False})

    with tab3:
        actions_df = (
            df["Next_Best_Action"]
            .value_counts()
            .rename_axis("Action")
            .reset_index(name="Count")
        )

        fig_actions = px.bar(
            actions_df,
            x="Action",
            y="Count",
            text_auto=True,
            title="Répartition des actions recommandées"
        )
        dark_layout(fig_actions, height=380)
        fig_actions.update_traces(marker_color="#0d8bff")
        st.plotly_chart(fig_actions, width="stretch", config={"displayModeBar": False})

        preview_cols = [
            "Age", "Purchase_Amount", "Rating", "Repeat_Customer",
            "Customer_Score", "Estimated_LTV", "Next_Best_Action"
        ]
        preview_cols = [c for c in preview_cols if c in df.columns]
        st.dataframe(df[preview_cols].head(20), width="stretch")

    with tab4:
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("<div class='section-title'>📈 Risque de churn</div>", unsafe_allow_html=True)
            if churn_model is not None:
                churn_input = df[["Age", "Purchase_Amount", "Rating"]].dropna().copy()
                if not churn_input.empty:
                    churn_probs = churn_model.predict_proba(churn_input)[:, 1] * 100
                    churn_df = churn_input.copy()
                    churn_df["Churn_Risk"] = churn_probs

                    fig_churn = px.histogram(
                        churn_df,
                        x="Churn_Risk",
                        nbins=20,
                        title="Distribution du risque de churn"
                    )
                    dark_layout(fig_churn, height=360)
                    fig_churn.update_traces(marker_color="#ef4444")
                    st.plotly_chart(fig_churn, width="stretch", config={"displayModeBar": False})

        with c2:
            st.markdown("<div class='section-title'>💰 Lifetime Value</div>", unsafe_allow_html=True)
            if "Estimated_LTV" in df.columns:
                fig_ltv = px.histogram(
                    df,
                    x="Estimated_LTV",
                    nbins=20,
                    title="Distribution du LTV estimé"
                )
                dark_layout(fig_ltv, height=360)
                fig_ltv.update_traces(marker_color="#22c55e")
                st.plotly_chart(fig_ltv, width="stretch", config={"displayModeBar": False})

        top_risk = None
        if churn_model is not None:
            churn_input = df[["Age", "Purchase_Amount", "Rating"]].dropna().copy()
            if not churn_input.empty:
                churn_probs = churn_model.predict_proba(churn_input)[:, 1] * 100
                risk_table = df.loc[churn_input.index].copy()
                risk_table["Churn_Risk"] = churn_probs
                top_risk = risk_table.sort_values("Churn_Risk", ascending=False).head(10)

        top_value = df.sort_values("Estimated_LTV", ascending=False).head(10) if "Estimated_LTV" in df.columns else None

        a, b = st.columns(2)
        with a:
            if top_risk is not None:
                cols = [c for c in ["Age", "Purchase_Amount", "Rating", "Churn_Risk", "Next_Best_Action"] if c in top_risk.columns]
                st.markdown("**Top clients à risque**")
                st.dataframe(top_risk[cols], width="stretch")
        with b:
            if top_value is not None:
                cols = [c for c in ["Age", "Purchase_Amount", "Rating", "Estimated_LTV", "Next_Best_Action"] if c in top_value.columns]
                st.markdown("**Top clients à forte valeur**")
                st.dataframe(top_value[cols], width="stretch")

    with tab5:
        st.markdown("<div class='section-title'>🤖 Deep Learning</div>", unsafe_allow_html=True)

        if DL_AVAILABLE and dl_model is not None and dl_scaler is not None:
            dl_input = df[["Age", "Purchase_Amount", "Rating"]].dropna().astype(float)
            if not dl_input.empty:
                dl_scaled = dl_scaler.transform(dl_input)
                dl_pred = dl_model.predict(dl_scaled, verbose=0).flatten() * 100

                dl_results = dl_input.copy()
                dl_results["DL_Fidelity_Prob"] = dl_pred

                fig_dl = px.histogram(
                    dl_results,
                    x="DL_Fidelity_Prob",
                    nbins=20,
                    title="Probabilité de fidélité — Deep Learning"
                )
                dark_layout(fig_dl, height=360)
                fig_dl.update_traces(marker_color="#8b5cf6")
                st.plotly_chart(fig_dl, width="stretch", config={"displayModeBar": False})
        else:
            st.warning("TensorFlow n'est pas installé. Active-le avec : pip install tensorflow")

    with tab6:
        display_cols = [
            "Age", "Age_Group", "Purchase_Amount", "Rating", "Repeat_Customer",
            "Product_Category", "Customer_Score", "Estimated_LTV", "Next_Best_Action"
        ]
        display_cols = [c for c in display_cols if c in df.columns]

        st.dataframe(df[display_cols].head(100), width="stretch")

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Télécharger les données enrichies",
            data=csv,
            file_name="executive_dashboard_enriched.csv",
            mime="text/csv"
        )

# =========================================================
# SIMULATEUR IA
# =========================================================
elif page == "🧠 Simulateur IA":
    st.markdown("<div class='top-title'>🧠 IA Prediction Studio</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='top-subtitle'>Simulation de fidélité, churn et score client en temps réel.</div>",
        unsafe_allow_html=True
    )

    left, right = st.columns([1, 1.1])

    with left:
        with st.form("prediction_form"):
            age_input = st.number_input("Âge du client", min_value=18, max_value=100, value=30)
            amount_input = st.number_input("Montant d'achat", min_value=1.0, max_value=10000.0, value=150.0)
            rating_input = st.slider("Note du client", min_value=1.0, max_value=5.0, value=4.0, step=0.1)
            submitted = st.form_submit_button("Lancer l'analyse")

    with right:
        if submitted:
            new_data = pd.DataFrame([{
                "Age": age_input,
                "Purchase_Amount": amount_input,
                "Rating": rating_input
            }])

            loyal_prob = None
            churn_prob = None

            if loyalty_model is not None:
                proba = loyalty_model.predict_proba(new_data)[0]
                positive_index = 1 if len(loyalty_encoder.classes_) > 1 else 0
                loyal_prob = float(proba[positive_index]) * 100

            if churn_model is not None:
                churn_prob = float(churn_model.predict_proba(new_data)[0][1]) * 100

            temp_df = new_data.copy()
            temp_df["Repeat_Customer"] = "Yes" if (loyal_prob or 0) >= 50 else "No"
            temp_df = enrich_data(temp_df)

            score = float(temp_df["Customer_Score"].iloc[0])
            ltv = float(temp_df["Estimated_LTV"].iloc[0])
            action = str(temp_df["Next_Best_Action"].iloc[0])

            k1, k2, k3 = st.columns(3)
            k1.metric("Score client", f"{score:.1f}")
            k2.metric("LTV estimé", f"${ltv:,.0f}")
            k3.metric("Action", action)

            if loyal_prob is not None:
                if loyal_prob >= 50:
                    st.success(f"Client fidèle prédit — confiance estimée : {loyal_prob:.1f}%")
                else:
                    st.error(f"Client occasionnel prédit — fidélité estimée : {loyal_prob:.1f}%")

            g1, g2 = st.columns(2)

            with g1:
                if loyal_prob is not None:
                    gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=loyal_prob,
                        title={'text': "Score de Fidélité"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "#0d8bff"},
                            'bgcolor': "rgba(255,255,255,0.03)",
                            'borderwidth': 0,
                        }
                    ))
                    dark_layout(gauge, height=320)
                    st.plotly_chart(gauge, width="stretch", config={"displayModeBar": False})

            with g2:
                if churn_prob is not None:
                    churn_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=churn_prob,
                        title={'text': "Risque de Churn"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "#ef4444"},
                            'bgcolor': "rgba(255,255,255,0.03)",
                            'borderwidth': 0,
                        }
                    ))
                    dark_layout(churn_gauge, height=320)
                    st.plotly_chart(churn_gauge, width="stretch", config={"displayModeBar": False})

            if loyalty_model is not None:
                importance_df = pd.DataFrame({
                    "Variable": ["Age", "Purchase_Amount", "Rating"],
                    "Importance": loyalty_model.feature_importances_
                }).sort_values("Importance", ascending=False)

                fig_imp = px.bar(
                    importance_df,
                    x="Variable",
                    y="Importance",
                    text_auto=".2f",
                    title="Variables explicatives"
                )
                dark_layout(fig_imp, height=330)
                fig_imp.update_traces(marker_color="#118dff")
                st.plotly_chart(fig_imp, width="stretch", config={"displayModeBar": False})
        else:
            st.info("Remplis le formulaire pour afficher l'analyse.")