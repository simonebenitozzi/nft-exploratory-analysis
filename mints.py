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

#%% Mints per Contract

mint_stats_df = mints_df.groupby('nft_address', as_index=False).size().rename(columns={'size':'num_nfts'})
mint_stats_df.hist('num_nfts', bins=200, log=True)

mint_stats_df.quantile(q = np.arange(0.1, 1.1, 0.1))

analysis_sample_df = mint_stats_df[mint_stats_df['num_nfts']>=100]
analysis_sample_df.head()

analysis_sample_df.count()

#%% NFTs minted per address



#%% Minting period



#%% Do NFTs charge for minting?



##% Conclusions



