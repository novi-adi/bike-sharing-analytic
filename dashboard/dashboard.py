import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# Fungsi untuk memuat data
def load_data():
    main_df = pd.read_csv('main_data.csv')
    day_df = pd.read_csv('../data/day.csv')  # Sesuaikan dengan path file Anda
    hour_df = pd.read_csv('../data/hour.csv')  # Sesuaikan dengan path file Anda
    day_df['dteday'] = pd.to_datetime(day_df['dteday'])
    hour_df['dteday'] = pd.to_datetime(hour_df['dteday'])
    return day_df, hour_df

day_df, hour_df = load_data()

# Judul Dashboard
st.title("Bike Sharing Usage Analysis")
with st.sidebar:
    # Menambahkan logo bike sharing
    st.image("https://images.unsplash.com/photo-1455641374154-422f32e234cd?q=80&w=1932&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D")
    # Filter Rentang Tanggal
    st.header("Select Date Range")
    start_date = st.date_input("Start date", day_df['dteday'].min())
    end_date = st.date_input("End date", day_df['dteday'].max())
    filtered_day_df = day_df[(day_df['dteday'] >= pd.to_datetime(start_date)) & (day_df['dteday'] <= pd.to_datetime(end_date))]

# Visualisasi 1: Tren dan Musiman
# Mengelompokkan data berdasarkan bulan dan tahun untuk melihat tren
st.header("1. Monthly Bike Usage Trend Over Years (Filtered by Date)")
monthly_trend = filtered_day_df.groupby(['yr', 'mnth']).agg({'cnt': np.mean}).reset_index()
fig, ax = plt.subplots()
sns.lineplot(data=monthly_trend, x='mnth', y='cnt', hue='yr', marker="o", ax=ax)
ax.set(xlabel='Month', ylabel='Average Daily Bike Count',
       title='Monthly Average Bike Usage Trend Over Years')
st.pyplot(fig)

# Visualisasi 2: Pengaruh Kondisi Cuaca
# Matriks korelasi antara kondisi cuaca dan penggunaan sepeda
st.header("2. Correlation between Weather Conditions and Bike Usage (Filtered by Date)")
weather_features = ['temp', 'atemp', 'hum', 'windspeed', 'cnt']
correlation_matrix = filtered_day_df[weather_features].corr()
fig, ax = plt.subplots()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
ax.set(title='Correlation between Weather Conditions and Bike Usage')
st.pyplot(fig)

# Visualisasi 3: Perilaku Pengguna dalam Keseharian
st.header("3. Hourly Bike Usage Trend for Weekdays vs Weekends")
hour_df['dteday'] = pd.to_datetime(hour_df['dteday'])

# Menambahkan kolom untuk menandai akhir pekan
hour_df['is_weekend'] = hour_df['weekday'].apply(lambda x: 1 if x in [0, 6] else 0)

# Mengelompokkan data berdasarkan jam dalam sehari dan tipe hari (hari kerja vs akhir pekan)
hourly_trend = hour_df.groupby(['hr', 'is_weekend']).agg({'cnt': 'mean'}).reset_index()

# Plotting tren penggunaan sepeda berdasarkan jam dan tipe hari
fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(data=hourly_trend, x='hr', y='cnt', hue='is_weekend', marker="o", ax=ax)
ax.set_title('Hourly Bike Usage Trend for Weekdays vs Weekends')
ax.set_xlabel('Hour of the Day')
ax.set_ylabel('Average Hourly Bike Count')
ax.set_xticks(np.arange(0, 24))
ax.legend(title='Weekend', labels=['Weekday', 'Weekend'])
st.pyplot(fig)

# Clustering Analysis
st.header("4. Clustering Analysis of Bike Rental Days")

# Pilih jumlah cluster
number_of_clusters = st.slider("Select Number of Clusters", 2, 10, 4)

# Pilih kolom untuk clustering
features = ['cnt', 'temp', 'hum', 'windspeed', 'season', 'weathersit']
scaler = StandardScaler()
scaled_features = scaler.fit_transform(day_df[features])

# Aplikasikan K-Means Clustering
kmeans = KMeans(n_clusters=number_of_clusters, random_state=0)
clusters = kmeans.fit_predict(scaled_features)
day_df['cluster'] = clusters

# Visualisasikan hasil clustering
fig, ax = plt.subplots()
sns.scatterplot(x=day_df['temp'], y=day_df['cnt'], hue=day_df['cluster'], palette='viridis', ax=ax)
ax.set_xlabel('Temperature')
ax.set_ylabel('Count of Bike Rentals')
ax.set_title('Clusters of Bike Rental Days')
st.pyplot(fig)
