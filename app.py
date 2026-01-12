import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px

# ===============================
# KONFIGURASI HALAMAN
# ===============================
st.set_page_config(layout="wide")
st.title("📊 Analisis Kekerasan Menggunakan K-Means Clustering")

# ===============================
# LOAD DATA
# ===============================
df_all_data = pd.read_excel("dp3akb_usia.xlsx")

st.subheader("📄 Data Awal (Sebelum Preprocessing)")
st.dataframe(df_all_data)

# ===============================
# PREPROCESSING DATA
# ===============================
st.subheader("🧹 Preprocessing Data")

st.write("🔍 Missing Values (Sebelum):")
st.write(df_all_data.isnull().sum())

df_all_data = df_all_data.dropna(
    subset=['nama_kabupaten_kota', 'jumlah_korban', 'tahun']
)

df_all_data['jumlah_korban'] = pd.to_numeric(
    df_all_data['jumlah_korban'],
    errors='coerce'
)
df_all_data = df_all_data.dropna(subset=['jumlah_korban'])

df_all_data['kategori_usia_bersih'] = (
    df_all_data['kategori_usia']
    .astype(str)
    .str.upper()
    .str.replace('TAHUN', '', regex=False)
    .str.strip()
)

st.write("🔍 Missing Values (Sesudah):")
st.write(df_all_data.isnull().sum())

st.success("✅ Preprocessing selesai")

# ===============================
# DATA SETELAH PREPROCESSING
# ===============================
st.subheader("📄 Data Setelah Preprocessing")
st.dataframe(df_all_data)

# ===============================
# AGREGASI PER KABUPATEN/KOTA
# ===============================
df_clustered = df_all_data.groupby('nama_kabupaten_kota').agg(
    jumlah_korban=('jumlah_korban', 'sum')
).reset_index()

# LOG TRANSFORM SETELAH AGREGASI
df_clustered['jumlah_korban_log'] = np.log1p(df_clustered['jumlah_korban'])

st.subheader("📍 Agregasi Jumlah Korban per Kabupaten/Kota")
st.dataframe(df_clustered)

# ===============================
# DISTRIBUSI DATA
# ===============================
st.subheader("📊 Distribusi Jumlah Korban")

fig_dist1, ax1 = plt.subplots(figsize=(8,4))
ax1.hist(df_clustered['jumlah_korban'], bins=20)
ax1.set_title("Distribusi Jumlah Korban (Asli)")
ax1.set_xlabel("Jumlah Korban")
ax1.set_ylabel("Frekuensi")
ax1.grid(True)
st.pyplot(fig_dist1)

fig_dist2, ax2 = plt.subplots(figsize=(8,4))
ax2.hist(df_clustered['jumlah_korban_log'], bins=20)
ax2.set_title("Distribusi Jumlah Korban (Log Transform)")
ax2.set_xlabel("Log(Jumlah Korban + 1)")
ax2.set_ylabel("Frekuensi")
ax2.grid(True)
st.pyplot(fig_dist2)

# ===============================
# ELBOW METHOD
# ===============================
st.subheader("📐 Elbow Method")

X_elbow = df_clustered[['jumlah_korban_log']]
X_elbow_scaled = StandardScaler().fit_transform(X_elbow)

inertia = []
K = range(1, 11)

for k in K:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_elbow_scaled)
    inertia.append(km.inertia_)

fig_elbow, ax_elbow = plt.subplots(figsize=(8,5))
ax_elbow.plot(K, inertia, marker='o')
ax_elbow.set_xlabel("Jumlah Cluster (k)")
ax_elbow.set_ylabel("Inertia")
ax_elbow.set_title("Elbow Method")
ax_elbow.grid(True)
st.pyplot(fig_elbow)

# ===============================

# ===============================
# K-MEANS CLUSTERING (K = 3)
# ===============================
X = df_clustered[['jumlah_korban_log']]
X_scaled = StandardScaler().fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42)
df_clustered['cluster'] = kmeans.fit_predict(X_scaled)

# ===============================
# LABEL CLUSTER OTOMATIS (3 LEVEL)
# ===============================
cluster_order = (
    df_clustered
    .groupby('cluster')['jumlah_korban']
    .mean()
    .sort_values()
    .index
)

label_names = [
    'Rendah',
    'Sedang',
    'Tinggi'
]

label_map = {
    cluster_order[i]: label_names[i]
    for i in range(len(cluster_order))
}

df_clustered['label_cluster'] = df_clustered['cluster'].map(label_map)

st.subheader("🧠 Hasil K-Means Clustering")
st.dataframe(df_clustered)

# ===============================
# RATA-RATA KORBAN PER CLUSTER
# ===============================
st.subheader("📊 Rata-rata Jumlah Korban per Cluster")

cluster_mean = (
    df_clustered
    .groupby('label_cluster')['jumlah_korban']
    .mean()
    .reindex(label_names)
)

fig_mean, ax_mean = plt.subplots(figsize=(8,5))
cluster_mean.plot(kind='bar', ax=ax_mean)
ax_mean.set_xlabel("Cluster")
ax_mean.set_ylabel("Rata-rata Jumlah Korban")
ax_mean.set_title("Rata-rata Jumlah Korban per Tingkat Kerawanan")
ax_mean.grid(axis='y')
st.pyplot(fig_mean)

# ===============================
# JUMLAH KAB/KOTA PER CLUSTER
# ===============================
st.subheader("📊 Jumlah Kabupaten/Kota per Cluster")

cluster_count = (
    df_clustered['label_cluster']
    .value_counts()
    .reindex(label_names)
)

fig_count, ax_count = plt.subplots(figsize=(8,5))
cluster_count.plot(kind='bar', ax=ax_count)
ax_count.set_xlabel("Cluster")
ax_count.set_ylabel("Jumlah Kabupaten/Kota")
ax_count.set_title("Distribusi Kabupaten/Kota per Cluster")
ax_count.grid(axis='y')
st.pyplot(fig_count)

# ===============================
# TOP 10
# ===============================
df_top = df_clustered.sort_values(
    by='jumlah_korban',
    ascending=False
).head(10)

st.subheader("🔟 Top 10 Kabupaten/Kota dengan Korban Tertinggi")
st.dataframe(df_top)

# ===============================
# SCATTER PLOT INTERAKTIF
# ===============================
st.subheader("📌 Scatter Plot Clustering (Hover)")

fig4 = px.scatter(
    df_clustered,
    x=df_clustered.index,
    y='jumlah_korban',
    color='label_cluster',
    size='jumlah_korban',
    hover_name='nama_kabupaten_kota',
    hover_data={
        'jumlah_korban': True,
        'label_cluster': True
    },
    title='Scatter Plot Clustering Kabupaten/Kota'
)

fig4.update_layout(
    xaxis_title="Kabupaten/Kota (Index)",
    yaxis_title="Jumlah Korban",
    xaxis=dict(showticklabels=False),
    template="plotly_white"
)

st.plotly_chart(fig4, use_container_width=True)

# ===============================
# RINGKASAN AKHIR
# ===============================
st.subheader("📄 Ringkasan Statistik Tiap Cluster")

summary_cluster = df_clustered.groupby('label_cluster').agg(
    Jumlah_Kabupaten=('nama_kabupaten_kota', 'count'),
    Total_Korban=('jumlah_korban', 'sum'),
    Rata_rata_Korban=('jumlah_korban', 'mean'),
    Min_Korban=('jumlah_korban', 'min'),
    Max_Korban=('jumlah_korban', 'max')
).reindex(label_names)

st.dataframe(summary_cluster)
