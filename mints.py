#%%

import os
import sqlite3

import matplotlib.pyplot as plt
import numpy as np
import nfts.dataset

#%% dataset opening

dirname = os.path.dirname(__file__)
DATASET_PATH = os.path.join(dirname, 'nfts.sqlite')
ds = nfts.dataset.FromSQLite(DATASET_PATH)

# load current owners dataframe
mints_df = ds.load_dataframe("mints")
print(mints_df.head())

##% Mints per Contract



#%% NFTs minted per address



#%% Minting period



#%% Do NFTs charge for minting?



##% Conclusions



