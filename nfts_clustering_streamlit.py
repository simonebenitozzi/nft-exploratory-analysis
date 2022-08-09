import streamlit as st

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

nfts_merged_df = pd.read_csv("data/nfts_merged.csv")

# --- Principal COmponent Analysis ---