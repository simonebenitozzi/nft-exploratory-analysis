import numpy as np
import seaborn as sns

# average value for a DataFrame column, ignoring NaN's
def nan_average(df, column):
    not_nan_df = df[np.isnan(df[column]) == False]
    avg = sum(not_nan_df[column]) / len(not_nan_df)
    return avg

# given a list of labels, assigns a color to each istance, based on its label
def color_palette_mapping(labels):
    n_clusters = len(np.unique(labels))
    palette = sns.color_palette(None, n_clusters)
    
    colors = []
    for l in labels:
        colors.append(palette[l])

    return colors