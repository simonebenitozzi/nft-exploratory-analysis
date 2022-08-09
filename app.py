import streamlit as st

from utils.visualization import color_palette_mapping
from utils.numerical import nan_average
import os
import warnings
import sqlite3

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import seaborn as sns

import numpy as np
import nfts.dataset

import pandas as pd
import math

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, OPTICS
from sklearn.metrics import silhouette_samples, silhouette_score

from yellowbrick.cluster import SilhouetteVisualizer
from kneed import KneeLocator
import pyclustertend

### --- Streamlit Configuration --- ###

st.set_page_config(page_title="nfts Clutering", page_icon="img/NFT.png")

nfts_merged_df = pd.read_csv("data/nfts_merged.csv")

### --- Preprocessing: Imputation of NaN Values --- ###

# MCM (Most Common Value) Imputation for mints_values, transfers_values, market_values
nfts_merged_df.mints_avg_transaction_value.fillna(nan_average(nfts_merged_df, "mints_avg_transaction_value"), inplace=True)
nfts_merged_df.transfers_avg_transaction_value.fillna(nan_average(nfts_merged_df, "transfers_avg_transaction_value"), inplace=True)
nfts_merged_df.avg_market_value.fillna(nan_average(nfts_merged_df, "avg_market_value"), inplace=True)

# 0 for the other values (since they represents counts)
nfts_merged_df.mints_timestamp_range.fillna(0, inplace=True)
nfts_merged_df.transfers_count.fillna(0, inplace=True)
nfts_merged_df.num_owners.fillna(0, inplace=True)


### --- Principal Component Analysis --- ###
st.header("Principal Component Analysis")

# Selecting significative features
features = ["mints_avg_transaction_value", "mints_timestamp_range", "transfers_avg_transaction_value",
             "transfers_count", "num_owners", "avg_market_value"]

# Separating out the features
x_pca = nfts_merged_df.loc[:, features]

# Standardizing the features
x_pca = StandardScaler().fit_transform(x_pca)
standardized_df = pd.DataFrame(x_pca, columns = features)

## -- pca with all components (6) -- ##
pca = PCA()

principal_components = pca.fit_transform(x_pca)
cumulative_sum_variance = np.cumsum(pca.explained_variance_ratio_)

st.write(f"The next chart shows the cumulative Variance explained by each of the 6 Principal Components")
st.write(f"The 4 Principal Components alone explain {round(sum(pca.explained_variance_ratio_[0:4])*10000)/100}% of the Variance: it makes sense then, that the Clustering algorithms will be executed on the 4 principal components")

fig = plt.figure(figsize=(10,5))
plt.plot(range(1, len(cumulative_sum_variance)+1), cumulative_sum_variance)
plt.xlabel("Number of Components")
_ = plt.ylabel("Explained Variance (%)")
_ = plt.title("Variance Accumulation")

st.pyplot(fig)

## -- 2 Pricipal Components -- ##
st.subheader("2 Principal Components")

pca_2 = PCA(n_components=2)
pca_2.fit_transform(x_pca)
x_pca_2 = pca_2.transform(x_pca)

fig = plt.figure(figsize=(10,7))
sns.scatterplot(x=x_pca_2[:,0], y=x_pca_2[:,1], s=50)
_ = plt.title(f"2D Scatterplot: {round(cumulative_sum_variance[1]*10000)/100}% of variance captured")
_ = plt.xlabel("First Principal Component")
_ = plt.ylabel("Second Principal Component")

st.pyplot(fig)

## -- 3 Pricipal Components -- ##
st.subheader("3 Principal Components")

pca_3 = PCA(n_components=3)
pca_3.fit_transform(x_pca)
x_pca_3 = pca_3.transform(x_pca)

fig = plt.figure(figsize=(15,10))
ax = plt.axes(projection="3d")
sctt = ax.scatter3D(x_pca_3[:,0], x_pca_3[:,1], x_pca_3[:,2], s=50, alpha=0.6)
_ = plt.title(f"3D Scatterplot: {round(cumulative_sum_variance[2]*10000)/100}% of variance captured")
_ = ax.set_xlabel("First Principal Component")
_ = ax.set_ylabel("Second Principal Component")
_ = ax.set_zlabel("Third Principal Component")

st.pyplot(fig)

## -- 4 Pricipal Components -- ##
st.subheader("4 Principal Components")

pca_4 = PCA(n_components=4)
principal_components = pca_4.fit_transform(x_pca)
pca4_df = pd.DataFrame(principal_components)

x_pca_4 = pca_4.transform(x_pca) # variable for scatterplot visualization

fig = plt.figure(figsize=(15,10))
plt.plot(x_pca_4)
_ = plt.title(f"Transformed Data by PCA: {round(cumulative_sum_variance[3]*10000)/100}% variance")
_ = plt.xlabel("Observation")
_ = plt.ylabel("Transformed Data")

st.pyplot(fig)

### --- Clustering Tendency --- ###

# TODO: VAT matrix needed: see if it can be executed in some more time

### --- K-Means --- ###
st.header("K-Means Clustering")

## -- Elbow Method -- ##

sum_squared_dist = []
K = range(1,50)
for k in K:
    km = KMeans(n_clusters=k, random_state=0)
    km = km.fit(x_pca_4)
    sum_squared_dist.append(km.inertia_)
kn = KneeLocator(K, sum_squared_dist, curve='convex', direction='decreasing')

fig = plt.figure(figsize=(10,7))
_ = plt.plot(K,sum_squared_dist, 'bx-')
_ = plt.title("Clustering Quality - Elbow Point")
_ = plt.xlabel('number of clusters k')
_ = plt.ylabel('Sum of squared distances')
_ = plt.vlines(kn.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dashed', label=f"Elbow point: {kn.knee}")
_ = plt.legend()

st.pyplot(fig)

## -- Silhouette Score -- ##

silhouette_list = []
K = range(2,15)
for k in K:
    km = KMeans(n_clusters=k, random_state=0)
    km_labels = km.fit_predict(x_pca_4)
    silhouette_avg = silhouette_score(x_pca_4, km_labels)
    silhouette_list.append(silhouette_avg)

fig = plt.figure(figsize=(10,7))
_ = plt.plot(K,silhouette_list, 'bx-')
_ = plt.title("Clustering Quality - Silhouette Score")
_ = plt.xlabel('Number of clusters k')
_ = plt.ylabel('Silhouette Score')
_ = plt.xticks(K)

st.pyplot(fig)

## -- K Evaluation -- ##

fig, ax = plt.subplots(2, 2, figsize=(30,15))
k = 2
for i in [2, 3, 4, 5]:
    # Create KMeans instance for different number of clusters
    km = KMeans(n_clusters=i, random_state=0)
    q, mod = divmod(k, 2)

    # Create SilhouetteVisualizer instance with KMeans instance and Fit the visualizer
    visualizer = SilhouetteVisualizer(km, colors='yellowbrick', ax=ax[q-1][mod])
    visualizer.fit(x_pca_4)

    ax[q-1][mod].set_title(f"Silhouette Score for k={i}")
    ax[q-1][mod].set_xlabel("Silhouette Score")
    ax[q-1][mod].set_ylabel("Istances")

    k+=1

st.pyplot(fig)

## -- K-Means with K=3 --##
st.subheader("K-Means with K=3")

km = KMeans(n_clusters=3, random_state=0).fit(x_pca_4)
fig = plt.figure(figsize=(10,7))
sns.scatterplot(x=x_pca_2[:,0], y=x_pca_2[:,1], s=50, hue=km.labels_, palette=sns.color_palette(None, 3))
_ = plt.title(f"K-Means Clustering plotted on 2 Principal Components (K=3)")
_ = plt.xlabel("First Principal Component")
_ = plt.ylabel("Second Principal Component")

st.pyplot(fig)

fig = plt.figure(figsize=(15,10))
ax = plt.axes(projection="3d")
sctt = ax.scatter3D(x_pca_3[:,0], x_pca_3[:,1], x_pca_3[:,2], s=50, alpha=0.6, c=color_palette_mapping(km.labels_))
_ = plt.title(f"K-Means Clustering plotted on 3 Principal Components (K=3)")
_ = ax.set_xlabel("First Principal Component")
_ = ax.set_ylabel("Second Principal Component")
_ = ax.set_zlabel("Third Principal Component")

st.pyplot(fig)

### --- DBSCAN Clustering --- ###
st.header("DBSCAN Clustering")

dbscan = DBSCAN().fit(x_pca_4)
n_clusters = len(np.unique(dbscan.labels_))
fig = plt.figure(figsize=(10,7))
sns.scatterplot(x=x_pca_2[:,0], y=x_pca_2[:,1], s=50, hue=dbscan.labels_, palette=sns.color_palette(None, n_clusters))
_ = plt.title(f"DBSCAN Clustering plotted on 2 Principal Components")
_ = plt.xlabel("First Principal Component")
_ = plt.ylabel("Second Principal Component")

st.pyplot(fig)

fig = plt.figure(figsize=(15,10))
ax = plt.axes(projection="3d")
sctt = ax.scatter3D(x_pca_3[:,0], x_pca_3[:,1], x_pca_3[:,2], s=50, alpha=0.6, c=color_palette_mapping(dbscan.labels_+1))
_ = plt.title(f"DBSCAN Clustering plotted on 3 Principal Components")
_ = ax.set_xlabel("First Principal Component")
_ = ax.set_ylabel("Second Principal Component")
_ = ax.set_zlabel("Third Principal Component")

st.pyplot(fig)

### --- OPTICS Clustering --- ###
st.header("OPTICS Clustering")

optics = OPTICS().fit(x_pca_4)
n_clusters = len(np.unique(optics.labels_))
fig = plt.figure(figsize=(10,7))
sns.scatterplot(x=x_pca_2[:,0], y=x_pca_2[:,1], s=50, hue=optics.labels_, palette=sns.color_palette(None, n_clusters)).legend_.remove()
_ = plt.title(f"OPTICS Clustering plotted on 2 Principal Components")
_ = plt.xlabel("First Principal Component")
_ = plt.ylabel("Second Principal Component")

st.pyplot(fig)

fig = plt.figure(figsize=(15,10))
ax = plt.axes(projection="3d")
sctt = ax.scatter3D(x_pca_3[:,0], x_pca_3[:,1], x_pca_3[:,2], s=50, alpha=0.6, c=color_palette_mapping(optics.labels_))
_ = plt.title(f"OPTICS Clustering plotted on 3 Principal Components")
_ = ax.set_xlabel("First Principal Component")
_ = ax.set_ylabel("Second Principal Component")
_ = ax.set_zlabel("Third Principal Component")

st.pyplot(fig)