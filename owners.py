#%%
import os
import sqlite3
import matplotlib.pyplot as plt
import nfts.dataset
import pandas as pd

#%% dataset opening

# =============================================================================
# dirname = os.path.dirname(__file__)
# DATASET_PATH = os.path.join(dirname, 'nfts.sqlite\\nfts.sqlite')
# =============================================================================
DATASET_PATH = "C:\\Users\\simon\\.spyder-py3\\nfts.sqlite\\nfts.sqlite"
ds = nfts.dataset.FromSQLite(DATASET_PATH)

# dataset description
nfts.dataset.explain()

# load current owners dataframe
current_owners_df = ds.load_dataframe("current_owners")

#%% Who owns NFTs?
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 20)
print(current_owners_df.head())

top_owners_df = current_owners_df.groupby(['owner'], as_index=False).size().rename(columns={"size":"num_tokens"})
top_owners_df.sort_values("num_tokens", inplace=True, ascending=False)

print(top_owners_df.head(20))

#%% NFT ownership histogram
plt.xlabel('Numbers of tokens owned - n')
plt.ylabel('Numbers of addresses owning n tokens (log scale)')
_, _, _ = plt.hist(top_owners_df['num_tokens'], bins=100, log=True)

#%% Low scale owners

scale_cutoff = 1500
low_scale_owners = [num_tokens for num_tokens in top_owners_df['num_tokens'] 
                    if num_tokens<=scale_cutoff]

plt.xlabel(f'Numbers of tokens owned - n < {scale_cutoff}')
plt.ylabel('Numbers of addresses owning n tokens')
_, _, _ = plt.hist(low_scale_owners, bins=int(scale_cutoff/50), log=False)

#%%

plt.xlabel(f'Numbers of tokens owned - n < {scale_cutoff}')
plt.ylabel('Numbers of addresses owning n tokens (log scale)')
_, _, _ = plt.hist(low_scale_owners, bins=int(scale_cutoff/50), log=True)

#%%

# how much does the 1% (5, 10, 15, . . .) own?