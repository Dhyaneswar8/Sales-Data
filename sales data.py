# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import zscore

# %%
df = pd.read_csv("online_sales_dataset.csv")
df.head(10)

# %%
df.isnull().sum()

# %%
df['CustomerID'] = df['CustomerID'].fillna('Unknown')

# %%
df['ShippingCost'] = df['ShippingCost'].fillna(df['ShippingCost'].median())

# %%
df['ShippingCost'] = df['ShippingCost'].fillna(df['ShippingCost'].median())

# %%
df = df.drop_duplicates(subset=['InvoiceNo', 'StockCode'], keep='first')

# %%
df['PaymentMethod'] = df['PaymentMethod'].replace({'paypall': 'paypal'})
df['ReturnStatus'] = df['ReturnStatus'].str.title()

# %%
df['Revenue'] = df['Quantity'] * df['UnitPrice']

# %%
df['Net Revenue'] = df['Revenue'] * (1 - df['Discount'])

# %%
df = df[df['Quantity'] > 0]
df = df[df['UnitPrice'] > 0]

# %%
print(df[['InvoiceNo', 'StockCode', 'Quantity', 'UnitPrice', 'Revenue', 'Net Revenue', 'PaymentMethod', 'ReturnStatus']])

# %% [markdown]
# Exploratory Data Analysis (EDA)

# %%
# Top products by Net Revenue
top_products_revenue = df.groupby(['Description'])['Net Revenue'].sum().sort_values(ascending=False).head(10)
print("Top 10 Products by Net Revenue:\n", top_products_revenue)

# Top products by Quantity sold
top_products_quantity = df.groupby(['Description'])['Quantity'].sum().sort_values(ascending=False).head(10)
print("Top 10 Products by Quantity Sold:\n", top_products_quantity)

# Plotting top products by revenue
plt.figure(figsize=(10,6))
sns.barplot(x=top_products_revenue.values, y=top_products_revenue.index, palette='viridis')
plt.title('Top 10 Products by Net Revenue')
plt.xlabel('Net Revenue')
plt.ylabel('Product')
plt.show()


# %%
# Top countries by Net Revenue
top_countries_revenue = df.groupby('Country')['Net Revenue'].sum().sort_values(ascending=False).head(10)
print("Top 10 Countries by Net Revenue:\n", top_countries_revenue)

# Plotting
plt.figure(figsize=(10,6))
sns.barplot(x=top_countries_revenue.values, y=top_countries_revenue.index, palette='rocket')
plt.title('Top 10 Countries by Net Revenue')
plt.xlabel('Net Revenue')
plt.ylabel('Country')
plt.show()


# %%
# Sales Channel Revenue
sales_channel_revenue = df.groupby('SalesChannel')['Net Revenue'].sum().sort_values(ascending=False)
print("Revenue by Sales Channel:\n", sales_channel_revenue)

# Plotting
plt.figure(figsize=(6,4))
sns.barplot(x=sales_channel_revenue.index, y=sales_channel_revenue.values, palette='mako')
plt.title('Revenue by Sales Channel')
plt.xlabel('Sales Channel')
plt.ylabel('Net Revenue')
plt.show()


# %% [markdown]
# Detect Seasonal Trends and Anomalies

# %%
#1. Convert InvoiceDate to datetime
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['InvoiceMonth'] = df['InvoiceDate'].dt.to_period('M')


# %%
#2. Monthly Sales Trend
monthly_revenue = df.groupby('InvoiceMonth')['Net Revenue'].sum()

# Plot the monthly revenue
plt.figure(figsize=(12,6))
monthly_revenue.plot(marker='o')
plt.title('Monthly Net Revenue Trend')
plt.xlabel('Month')
plt.ylabel('Net Revenue')
plt.grid(True)
plt.show()


# %%
#3. Seasonal Decomposition
# Convert InvoiceMonth to timestamp for seasonal decomposition
monthly_revenue_ts = monthly_revenue.to_timestamp()

# Decomposition
decomposition = seasonal_decompose(monthly_revenue_ts, model='additive', period=12)

# Plot
decomposition.plot()
plt.show()


# %% [markdown]
# Segment Customers Based on Purchasing Behavior

# %%
#1. Create Customer-Level Metrics
#We'll use RFM (Recency, Frequency, Monetary) analysis.
import datetime as dt

# Assuming the latest date in the dataset as the reference date
snapshot_date = df['InvoiceDate'].max() + dt.timedelta(days=1)

# RFM Table
rfm_df = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'nunique',
    'Net Revenue': 'sum'
})

rfm_df.rename(columns={
    'InvoiceDate': 'Recency',
    'InvoiceNo': 'Frequency',
    'Net Revenue': 'Monetary'
}, inplace=True)

print(rfm_df.head())


# %%
#2. Standardize RFM for Clustering
# Standardize
rfm_scaled = rfm_df[['Recency', 'Frequency', 'Monetary']].apply(zscore)

# Check
print(rfm_scaled.head())


# %%
#3. KMeans Clustering on Customers
# Build KMeans model (let's try 4 segments for now)
kmeans = KMeans(n_clusters=4, random_state=42)
rfm_df['Cluster'] = kmeans.fit_predict(rfm_scaled)

# Analyze clusters
cluster_summary = rfm_df.groupby('Cluster').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': ['mean', 'count']
})

print(cluster_summary)


# %%
#4. Plot Clusters
# 2D Plot (Recency vs Frequency)
plt.figure(figsize=(8,6))
sns.scatterplot(data=rfm_df, x='Recency', y='Frequency', hue='Cluster', palette='Set1', s=100)
plt.title('Customer Segments (Recency vs Frequency)')
plt.xlabel('Recency (days)')
plt.ylabel('Frequency (transactions)')
plt.legend(title='Cluster')
plt.show()


# %% [markdown]
# 3. Visualization:

# %%
#1. Sales Trend Analysis Chart
# Plot the monthly Net Revenue trend
plt.figure(figsize=(12,6))
monthly_revenue.plot(marker='o', color='teal')
plt.title('Monthly Net Revenue Trend', fontsize=16)
plt.xlabel('Month')
plt.ylabel('Net Revenue')
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# %%
#2. Customer Segmentation Visual
plt.figure(figsize=(10,6))
sns.scatterplot(
    data=rfm_df, 
    x='Recency', 
    y='Frequency', 
    hue='Cluster', 
    palette='Set2', 
    s=100,
    edgecolor='black'
)
plt.title('Customer Segments (Recency vs Frequency)', fontsize=16)
plt.xlabel('Recency (Days Since Last Purchase)')
plt.ylabel('Frequency (Number of Transactions)')
plt.legend(title='Cluster')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()


# %%
#3. Top Products by Net Revenue Chart
plt.figure(figsize=(10,6))
sns.barplot(
    x=top_products_revenue.values, 
    y=top_products_revenue.index, 
    palette='crest'
)
plt.title('Top 10 Products by Net Revenue', fontsize=16)
plt.xlabel('Net Revenue')
plt.ylabel('Product')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()


# %% [markdown]
# 4. Insights and Recommendations:

# %% [markdown]
# 5. Sales Forecasting (Next Month's Prediction)

# %%
#Prepare Time Series Data
# Convert InvoiceDate to datetime (if not already)
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Set InvoiceDate as index
df.set_index('InvoiceDate', inplace=True)

# Resample to get monthly Net Revenue
monthly_sales = df['Net Revenue'].resample('M').sum()

# Plot to visualize
plt.figure(figsize=(12, 6))
plt.plot(monthly_sales, marker='o')
plt.title('Monthly Net Revenue Over Time')
plt.xlabel('Date')
plt.ylabel('Net Revenue')
plt.grid(True)
plt.show()


# %%
#Decompose the Time Series
decompose_result = seasonal_decompose(monthly_sales, model='additive')
decompose_result.plot()
plt.show()


# %%
#Build & Fit an ARIMA Model
from statsmodels.tsa.arima.model import ARIMA

# Build ARIMA model (p,d,q) parameters can be tuned later
model = ARIMA(monthly_sales, order=(1, 1, 1))
model_fit = model.fit()

# Forecast next month's sales
forecast = model_fit.forecast(steps=1)
print("Predicted Next Month's Net Revenue: ${:.2f}".format(forecast[0]))


# %%
#Plot Forecast vs Actual
# Forecast next 3 months
forecast_steps = 3
future_forecast = model_fit.forecast(steps=forecast_steps)

# Plot
plt.figure(figsize=(12, 6))
plt.plot(monthly_sales, label='Actual')
plt.plot(pd.date_range(start=monthly_sales.index[-1], periods=forecast_steps + 1, freq='M')[1:], 
         future_forecast, marker='o', label='Forecast', color='red')
plt.title('Actual vs Forecasted Net Revenue')
plt.xlabel('Date')
plt.ylabel('Net Revenue')
plt.legend()
plt.grid(True)
plt.show()



