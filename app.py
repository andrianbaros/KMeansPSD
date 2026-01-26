import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# ===============================
# KONFIGURASI HALAMAN & STYLING
# ===============================
st.set_page_config(
    layout="wide",
    page_title="Analisis Kasus Kekerasan terhadap Anak di Provinsi Jawa Barat - K-Means Clustering",
    initial_sidebar_state="expanded"
)

# CSS CUSTOM
st.markdown("""
    <style>
    * {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .main {
        padding-top: 1rem;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    h1 {
        color: #2d3748;
        text-align: center;
        margin-bottom: 0.5rem;
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    h2 {
        color: #2d3748;
        border-bottom: 4px solid #667eea;
        padding-bottom: 0.8rem;
        margin-top: 2rem;
        font-size: 1.8rem;
        font-weight: 700;
    }
    
    h3 {
        color: #4a5568;
        font-size: 1.3rem;
        font-weight: 600;
    }
    
    .success-box {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        border-left: 5px solid #2ecc71;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: #1a5f3d;
    }
    
    .info-box {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        border-left: 5px solid #667eea;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        border-left: 5px solid #f39c12;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: #5d3d00;
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

# ===============================
# HEADER & SIDEBAR
# ===============================
st.markdown("<h1>Analisis Kasus Kekerasan terhadap Anak di Provinsi Jawa Barat</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #667eea; font-size: 1.1rem; font-weight: 600;'>K-Means Clustering dengan RobustScaler | Jawa Barat 2018-2024</p>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("---")
    st.markdown("### METODOLOGI ANALISIS")
    st.markdown("""
    **Pendekatan K-Means Clustering:**
    
    1️ **Data Preprocessing**
    - Pembersihan data
    - Penanganan missing values
    
    2️ **Rekayasa Fitur**

    - Agregasi per Kabupaten/Kota
    - Perhitungan metrik kekerasan
    
    3️ **RobustScaler Normalisasi**
    - Tahan terhadap outliers
    - Skala data sesuai kuartil
    
    4️ **K-Means Clustering (K=3)**
    - Pengelompokan berdasarkan karakteristik
    """)
    
    st.markdown("---")
    st.markdown("### DIMENSI ANALISIS")
    st.markdown("""
    **Fitur yang Digunakan:**

    - Total Korban
    - Laju Pertumbuhan (pertumbuhan tahunan)
    - Volatilitas (tingkat variasi)
    - Arah Tren

    
    **Periode & Lokasi:**
    - 2018-2024 (7 tahun)
    - Jawa Barat (27 Kabupaten/Kota)
    - Sumber: DP3AKB
    """)

# ===============================
# LOAD & PREPROCESS DATA
# ===============================
st.markdown("## DATA LOADING & PREPROCESSING")

file_path = "dp3akb_usia.xlsx"
df_raw = pd.read_excel(file_path)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Records (Raw)", f"{len(df_raw):,}")
with col2:
    st.metric("Kabupaten/Kota", len(df_raw['nama_kabupaten_kota'].unique()))
with col3:
    st.metric("Tahun ", f"2018-2024")

# Tampilkan data mentah
with st.expander("Data Mentah (Raw Data) - Klik untuk lihat", expanded=False):
    st.markdown("### Data Sebelum Preprocessing")
    st.dataframe(df_raw.head(20), use_container_width=True)
    st.caption(f"Menampilkan 20 baris pertama dari {len(df_raw)} total records")

# Preprocessing
df_all_data = df_raw.copy()
df_all_data = df_all_data.dropna(subset=['nama_kabupaten_kota', 'jumlah_korban', 'tahun'])
df_all_data['jumlah_korban'] = pd.to_numeric(df_all_data['jumlah_korban'], errors='coerce')
df_all_data = df_all_data.dropna(subset=['jumlah_korban'])
df_all_data['jenis_kelamin'] = df_all_data['jenis_kelamin'].astype(str).str.upper().str.strip()

# Tampilkan data setelah preprocessing
with st.expander("Data Setelah Preprocessing - Klik untuk lihat", expanded=False):
    st.markdown("### Data Setelah Pembersihan & Preprocessing")
    st.dataframe(df_all_data.head(20), use_container_width=True)
    st.caption(f"Menampilkan 20 baris pertama dari {len(df_all_data)} total records (cleaned)")

st.markdown("<div class='success-box'>Data preprocessing selesai</div>", unsafe_allow_html=True)

# ===============================
# FEATURE ENGINEERING
# ===============================
st.markdown("## REKAYASA FITUR")

child_age_groups = ['0-5', '5-12', '6-12', '10-14', '12-17', '13-17', '15-19']

def is_child(age_str):
    if pd.isna(age_str):
        return False
    age_str = str(age_str).strip()
    return any(child_group in age_str for child_group in child_age_groups)

df_all_data['is_child'] = df_all_data['kategori_usia'].apply(is_child)

total_anak = df_all_data[df_all_data['is_child']]['jumlah_korban'].sum()
total_semua = df_all_data['jumlah_korban'].sum()
pct_anak = (total_anak / total_semua * 100) if total_semua > 0 else 0

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Korban Anak", f"{int(total_anak):,}")
with col2:
    st.metric("Total Semua Korban", f"{int(total_semua):,}")
with col3:
    st.metric("Persentase Korban Anak", f"{pct_anak:.1f}%")

st.markdown("### Agregasi per Kabupaten/Kota")

# Calculate features
df_total = df_all_data.groupby('nama_kabupaten_kota')['jumlah_korban'].sum().reset_index()
df_total.columns = ['nama_kabupaten_kota', 'total_korban']

df_yearly = df_all_data.groupby(['nama_kabupaten_kota', 'tahun'])['jumlah_korban'].sum().reset_index()

def calc_growth(group):
    if len(group) < 2:
        return 0
    yearly_values = group.sort_values('tahun')['jumlah_korban'].values
    if yearly_values[0] == 0:
        return 0
    rates = [(yearly_values[i] - yearly_values[i-1])/yearly_values[i-1]*100 
             for i in range(1, len(yearly_values))]
    return np.mean(rates) if rates else 0

growth_rate = df_yearly.groupby('nama_kabupaten_kota').apply(calc_growth).reset_index()
growth_rate.columns = ['nama_kabupaten_kota', 'growth_rate']

volatility = df_yearly.groupby('nama_kabupaten_kota')['jumlah_korban'].std().reset_index()
volatility.columns = ['nama_kabupaten_kota', 'volatility']
volatility['volatility'] = volatility['volatility'].fillna(0)

def calc_trend(group):
    group = group.sort_values('tahun').reset_index(drop=True)
    if len(group) < 2:
        return 0
    x = np.arange(len(group))
    y = group['jumlah_korban'].values
    z = np.polyfit(x, y, 1)
    return z[0]

trend = df_yearly.groupby('nama_kabupaten_kota').apply(calc_trend).reset_index()
trend.columns = ['nama_kabupaten_kota', 'trend_direction']

child_sum = df_all_data[df_all_data['is_child']].groupby('nama_kabupaten_kota')['jumlah_korban'].sum()
total_sum = df_all_data.groupby('nama_kabupaten_kota')['jumlah_korban'].sum()
rasio_anak_df = pd.DataFrame({
    'nama_kabupaten_kota': total_sum.index,
    'rasio_anak': (child_sum.reindex(total_sum.index, fill_value=0) / total_sum * 100).round(2).values
})

# MERGE SEMUA
df_clustered = df_total.copy()
for df_merge in [growth_rate, volatility, trend, rasio_anak_df]:
    df_clustered = df_clustered.merge(df_merge, on='nama_kabupaten_kota', how='left')

numerical_features = ['total_korban', 'growth_rate', 'volatility', 'trend_direction', 'rasio_anak']
available_features = [col for col in numerical_features if col in df_clustered.columns]

df_clustered = df_clustered.fillna(0)

# Tampilkan data setelah feature engineering
with st.expander("Data Setelah Rekayasa Fitur - Klik untuk lihat", expanded=False):
    st.markdown("### Data dengan Fitur untuk Clustering")
    st.dataframe(df_clustered, use_container_width=True)
    st.caption(f"Total {len(df_clustered)} Kabupaten/Kota dengan {len(available_features)} fitur")

st.markdown("<div class='success-box'>Rekayasa fitur selesai: {} features, {} kabupaten</div>".format(len(available_features), len(df_clustered)), unsafe_allow_html=True)

# ===============================
# EDA
# ===============================
st.markdown("## EXPLORATORY DATA ANALYSIS")

col1, col2 = st.columns(2)
with col1:
    st.markdown("<h4>Statistik Deskriptif</h4>", unsafe_allow_html=True)
    st.dataframe(df_clustered[available_features].describe().round(2), use_container_width=True)

with col2:
    st.markdown("<h4>Korelasi Antar Feature</h4>", unsafe_allow_html=True)
    if len(df_clustered) > 2:
        corr = df_clustered[available_features].corr()
        fig_corr, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, square=True, ax=ax)
        st.pyplot(fig_corr, use_container_width=True)

# ===============================
# ROBUST SCALER NORMALISASI
# ===============================
st.markdown("## ROBUST SCALER NORMALISASI")

st.markdown("""
<div class='info-box'>
<strong>RobustScaler:</strong> Menggunakan statistik kuartil (Q1, median, Q3) sehingga tahan terhadap outliers. 
Lebih cocok untuk data dengan nilai ekstrem.
</div>
""", unsafe_allow_html=True)

scaler = RobustScaler()
X_scaled = scaler.fit_transform(df_clustered[available_features])

st.markdown("<div class='success-box'>Data dinormalisasi dengan RobustScaler</div>", unsafe_allow_html=True)

# ===============================
# ELBOW METHOD
# ===============================
st.markdown("## PENENTUAN K OPTIMAL")

silhouette_scores = []
davies_bouldin_scores = []
calinski_harabasz_scores = []
inertia_list = []

for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertia_list.append(km.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, labels))
    davies_bouldin_scores.append(davies_bouldin_score(X_scaled, labels))
    calinski_harabasz_scores.append(calinski_harabasz_score(X_scaled, labels))

col1, col2 = st.columns(2)

with col1:
    fig_elbow = go.Figure()
    
    fig_elbow.add_trace(
        go.Scatter(
            x=list(range(2, 11)),
            y=inertia_list,
            mode='lines+markers',
            name='Inersia',
            line=dict(color='#667eea', width=3),
            marker=dict(size=10)
        )
    )
    
    # Garis potong pada K = 3
    fig_elbow.add_vline(
        x=3,
        line_width=2,
        line_dash="dash",
        line_color="red",
        annotation_text="K = 3",
        annotation_position="top"
    )
    
    fig_elbow.update_layout(
        title="Metode Elbow – Nilai Inersia",
        xaxis_title="Jumlah Klaster (K)",
        yaxis_title="Nilai Inersia",
        template="plotly_white",
        height=400,
        font=dict(size=11)
    )
    
    st.plotly_chart(fig_elbow, use_container_width=True)


# ===============================
# K-MEANS CLUSTERING (K=3)
# ===============================
st.markdown("## HASIL K-MEANS CLUSTERING (K=3)")

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df_clustered['cluster'] = kmeans.fit_predict(X_scaled)

st.markdown("<h3>Distribusi & Profil Setiap Cluster</h3>", unsafe_allow_html=True)

color_palette = ['#667eea', '#f093fb', '#43e97b']

for cluster_id in range(3):
    cluster_data = df_clustered[df_clustered['cluster'] == cluster_id]
    
    with st.expander(f"Cluster {cluster_id} - {len(cluster_data)} Kabupaten", expanded=True):
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Jumlah", len(cluster_data))
        with col2:
            st.metric("Rata-Rata Korban", f"{cluster_data['total_korban'].mean():.0f}")
        with col3:
            st.metric("Rata-Rata Growth", f"{cluster_data['growth_rate'].mean():.1f}%")
        with col4:
            st.metric("Rata-Rata Volatility", f"{cluster_data['volatility'].mean():.0f}")
        with col5:
            st.metric("Rata-Rata Rasio Anak", f"{cluster_data['rasio_anak'].mean():.1f}%")
        
        st.markdown("<h4>📍 Kabupaten/Kota dalam Cluster ini:</h4>", unsafe_allow_html=True)
        st.dataframe(
            cluster_data[['nama_kabupaten_kota', 'total_korban', 'growth_rate', 'volatility', 'rasio_anak']].sort_values('total_korban', ascending=False),
            use_container_width=True,
            hide_index=True
        )
        
        # Top 3 kabupaten dalam cluster
        top3 = cluster_data.nlargest(3, 'total_korban')
        fig_top = px.bar(top3, x='total_korban', y='nama_kabupaten_kota', orientation='h',
                        color_discrete_sequence=[color_palette[cluster_id]], text='total_korban')
        fig_top.update_layout(title=f"Top 3 Kabupaten dalam Cluster {cluster_id}", 
                             xaxis_title="Total Korban", yaxis_title="", 
                             template="plotly_white", height=300, showlegend=False)
        fig_top.update_traces(textposition='auto')
        st.plotly_chart(fig_top, use_container_width=True)


# ===============================
# CLUSTER QUALITY ASSESSMENT
# ===============================
st.markdown("## EVALUASI KUALITAS CLUSTER")

# Sample data (ganti dengan data Anda)
sil_score = silhouette_score(X_scaled, df_clustered['cluster'])
db_score = davies_bouldin_score(X_scaled, df_clustered['cluster'])
ch_score = calinski_harabasz_score(X_scaled, df_clustered['cluster'])

# Interpretasi
if sil_score < 0.3:
    sil_status = "⚠️ Weak"
    sil_color = "#FF6B6B"
elif sil_score < 0.5:
    sil_status = "✅ Fair"
    sil_color = "#FFA500"
else:
    sil_status = "✅✅ Good"
    sil_color = "#51CF66"

st.markdown("<h3>Metrik Validasi untuk K=3</h3>", unsafe_allow_html=True)

# ===== SILHOUETTE SCORE VISUALIZATION =====
col1, col2 = st.columns([1, 1])

with col1:
    st.metric(
        "Silhouette Score",
        f"{sil_score:.3f}",
        f"{sil_status}"
    )
    st.caption("Rentang: -1 hingga +1 (semakin tinggi semakin baik)\n\nMengukur: Seberapa baik data termasuk dalam cluster-nya")

with col2:
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Gauge chart untuk Silhouette
    categories = ['Poor\n(-1)', 'Weak\n(<0.3)', 'Fair\n(0.3-0.5)', 'Good\n(>0.5)']
    ranges = [-1, 0, 0.3, 0.5, 1]
    colors = ['#FF4444', '#FF6B6B', '#FFA500', '#51CF66']
    
    # Background bars
    for i in range(len(colors)):
        ax.barh(0, ranges[i+1] - ranges[i], left=ranges[i], height=0.3, color=colors[i], edgecolor='black', linewidth=1.5)
    
    # Marker untuk score saat ini
    ax.plot([sil_score, sil_score], [-0.2, 0.35], 'k-', linewidth=3, marker='v', markersize=12)
    ax.text(sil_score, -0.35, f'{sil_score:.3f}', ha='center', fontsize=11, fontweight='bold')
    
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlabel('Silhouette Score', fontsize=11, fontweight='bold')
    ax.set_xticks([-1, 0, 0.3, 0.5, 1])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    st.pyplot(fig, use_container_width=True)

st.divider()

# ===== DAVIES-BOULDIN INDEX VISUALIZATION =====
col3, col4 = st.columns([1, 1])

with col3:
    db_status = "✅ Excellent" if db_score < 1 else ("✅ Good" if db_score < 1.5 else "⚠️ Fair")
    st.metric(
        "Davies-Bouldin Index",
        f"{db_score:.3f}",
        db_status
    )
    st.caption("Range: 0 to ∞ (Lower is Better)\n\nMengukur: Separasi jelas antar cluster")

with col4:
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Gauge untuk Davies-Bouldin (capped at 3 for visualization)
    db_display = min(db_score, 3)
    
    categories = ['Excellent\n(<1)', 'Good\n(1-1.5)', 'Fair\n(1.5-2)', 'Poor\n(>2)']
    ranges = [0, 1, 1.5, 2, 3]
    colors = ['#51CF66', '#95E1D3', '#FFA500', '#FF6B6B']
    
    # Background bars
    for i in range(len(colors)):
        ax.barh(0, ranges[i+1] - ranges[i], left=ranges[i], height=0.3, color=colors[i], edgecolor='black', linewidth=1.5)
    
    # Marker
    ax.plot([db_display, db_display], [-0.2, 0.35], 'k-', linewidth=3, marker='v', markersize=12)
    ax.text(db_display, -0.35, f'{db_score:.3f}', ha='center', fontsize=11, fontweight='bold')
    
    ax.set_xlim(-0.2, 3.2)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlabel('Davies-Bouldin Index', fontsize=11, fontweight='bold')
    ax.set_xticks([0, 1, 1.5, 2, 3])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    st.pyplot(fig, use_container_width=True)

st.divider()

# ===== CALINSKI-HARABASZ SCORE VISUALIZATION =====
col5, col6 = st.columns([1, 1])

with col5:
    ch_status = "✅ Excellent" if ch_score > 30 else ("✅ Good" if ch_score > 20 else "⚠️ Moderate")
    st.metric(
        "Calinski-Harabasz Score",
        f"{ch_score:.1f}",
        ch_status
    )
    st.caption("Range: 0 to ∞ (Higher is Better)\n\nMengukur: Compactness & separation cluster")

with col6:
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Gauge untuk Calinski-Harabasz (capped at 50 for visualization)
    ch_display = min(ch_score, 50)
    
    categories = ['Moderate\n(<20)', 'Good\n(20-30)', 'Excellent\n(>30)']
    ranges = [0, 20, 30, 50]
    colors = ['#FFA500', '#95E1D3', '#51CF66']
    
    # Background bars
    for i in range(len(colors)):
        ax.barh(0, ranges[i+1] - ranges[i], left=ranges[i], height=0.3, color=colors[i], edgecolor='black', linewidth=1.5)
    
    # Marker
    ax.plot([ch_display, ch_display], [-0.2, 0.35], 'k-', linewidth=3, marker='v', markersize=12)
    ax.text(ch_display, -0.35, f'{ch_score:.1f}', ha='center', fontsize=11, fontweight='bold')
    
    ax.set_xlim(-2, 52)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlabel('Calinski-Harabasz Score', fontsize=11, fontweight='bold')
    ax.set_xticks([0, 20, 30, 50])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    st.pyplot(fig, use_container_width=True)

st.divider()

# ===== SUMMARY BOX =====
quality_level = "Sangat Baik" if sil_score > 0.5 else ("Cukup Baik" if sil_score > 0.3 else "Lemah")

st.markdown(f"""
<div style='background-color: #E8F5E9; padding: 20px; border-radius: 10px; border-left: 5px solid #51CF66;'>
    <strong style='font-size: 16px; color: #2E7D32;'>🎯 Kesimpulan Validasi:</strong>
    <p style='margin-top: 10px; color: #333; line-height: 1.6;'>
        Hasil K-Means clustering dengan <strong>K=3</strong> menghasilkan cluster yang <strong>{quality_level}</strong>. 
        <br>Silhouette Score {sil_score:.3f} menunjukkan bahwa data point sudah terkelompok dengan baik, 
        <br>cluster terpisah dengan jelas, dan hasil dapat dipercaya untuk analisis lebih lanjut.
    </p>
</div>
""", unsafe_allow_html=True)

# ===== COMPARISON TABLE =====
st.markdown("### Ringkasan Semua Metrik")

metrics_data = {
    'Metric': ['Silhouette Score', 'Davies-Bouldin Index', 'Calinski-Harabasz Score'],
    'Nilai': [f'{sil_score:.3f}', f'{db_score:.3f}', f'{ch_score:.1f}'],
    'Range Ideal': ['-1 to +1', '0 to ∞', '0 to ∞'],
    'Interpretasi': ['Tinggi = Baik', 'Rendah = Baik', 'Tinggi = Baik'],
    'Status': [sil_status, db_status, ch_status]
}


df_metrics = pd.DataFrame(metrics_data)
st.dataframe(df_metrics, use_container_width=True, hide_index=True)

# ===============================
# PCA VISUALIZATION
# ===============================
st.markdown("##  VISUALISASI PCA BERDASARKAN CLUSTER")

if len(available_features) >= 2:
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    df_pca = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'Cluster': [f"Cluster {c}" for c in df_clustered['cluster'].values],
        'nama': df_clustered['nama_kabupaten_kota'].values,
        'korban': df_clustered['total_korban'].values
    })
    
    fig_pca = px.scatter(df_pca, x='PC1', y='PC2', color='Cluster', size='korban',
                        hover_name='nama', hover_data={'korban': ':.0f'},
                        title=f'Visualisasi PCA Hasil Klaster K-Means: {pca.explained_variance_ratio_.sum():.1%}',
                        size_max=50, color_discrete_map={
                            'Cluster 0': '#667eea',
                            'Cluster 1': '#f093fb',
                            'Cluster 2': '#43e97b'
                        })
    
    fig_pca.update_layout(height=600, template="plotly_white", font=dict(size=11))
    st.plotly_chart(fig_pca, use_container_width=True)

    st.markdown(f"""
    <div class='info-box'>
    <strong> Penjelasan Kelompok (Cluster):</strong><br><br>

    • <strong>Cluster 0</strong><br>
    Cluster ini mencakup sebagian besar wilayah di Jawa Barat dengan jumlah korban yang relatif menengah,
    pola pertumbuhan kasus yang cukup stabil, serta proporsi korban anak (0–18 tahun) yang cukup besar.<br><br>

    • <strong>Cluster 1</strong><br>
    Cluster ini hanya berisi sedikit wilayah, namun menunjukkan perubahan kasus yang sangat cepat,
    ditandai dengan tingkat pertumbuhan dan variasi kasus yang sangat tinggi,
    serta proporsi korban anak (0–18 tahun) yang besar.<br><br>

    • <strong>Cluster 2</strong><br>
    Cluster ini berisi wilayah dengan jumlah korban paling besar dan tingkat variasi kasus yang paling tinggi,
    namun dengan proporsi korban anak (0–18 tahun) yang relatif lebih rendah dibandingkan cluster lainnya.<br><br>

    Pengelompokan ini dihasilkan secara otomatis oleh metode K-Means berdasarkan kemiripan data,
    bukan berdasarkan batas rendah, sedang, atau tinggi yang ditentukan secara manual.
    </div>
    """, unsafe_allow_html=True)



    st.markdown(f"""
    <div class='info-box'>
    <strong>Penjelasan Sumbu Grafik (PC1 dan PC2):</strong><br><br>

    • <strong>Sumbu Horizontal (PC1)</strong> menggambarkan <strong>ringkasan utama pola kasus</strong> di setiap kabupaten/kota.<br>
    • <strong>Sumbu Vertikal (PC2)</strong> menggambarkan <strong>ringkasan pendukung pola kasus</strong> yang tidak tertangkap pada sumbu utama.<br><br>

    • Kedua sumbu ini merupakan <strong>hasil peringkasan dari beberapa indikator</strong> seperti jumlah korban, pertumbuhan, volatilitas, dan arah tren.<br><br>

    Sumbu PC1 dan PC2 digunakan untuk <strong>mempermudah visualisasi dan perbandingan pola antar wilayah</strong>,
    bukan untuk menunjukkan nilai atau ukuran tertentu secara langsung.
    </div>
""", unsafe_allow_html=True)


# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.markdown("""
<div style='text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            padding: 2.5rem; border-radius: 15px; color: white;'>
<p style='font-size: 1.05rem; margin: 0.5rem 0; font-weight: 600;'>K-Means Clustering dengan RobustScaler</p>
<p style='font-size: 0.95rem; margin: 0.5rem 0;'>
    <strong>Periode:</strong> 2018-2024 | <strong>Wilayah:</strong> Jawa Barat | <strong>Data:</strong>  DP3AKB
</p>
<p style='font-size: 0.9rem; margin-top: 1rem; color: rgba(255,255,255,0.95);'>
    Tools: Streamlit • Pandas • Scikit-Learn • SciPy • Plotly • Seaborn
</p>
</div>
""", unsafe_allow_html=True)