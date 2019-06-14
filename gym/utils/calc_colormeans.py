import pandas as pd
import numpy as np

'''
Script that runs on panos directly to obtain the mean and standard deviation of the color channels in 
the dataset.
'''

region = "saint-urbain"
path = "/home/rogerg/Documents/autonomous_pedestrian_project/navi/"
data_df = pd.read_hdf(path + "hyrule-gym/data/" + region + "/processed/data.hdf5", key='df', mode='r')

r_means = []
g_means = []
b_means = []
df_indices = data_df.index.values.tolist()
for i in df_indices:
    img = data_df.loc[i]['thumbnail'] / 250.0
    r_means.append(np.mean(img[:, :, 0]))
    g_means.append(np.mean(img[:, :, 1]))
    b_means.append(np.mean(img[:, :, 2]))

# Note: I am assuming that this is the right order. TO be confirmed.
print("Red mean:", np.mean(r_means), "StD:", np.std(r_means))
print("Green mean:", np.mean(g_means), "StD:", np.std(g_means))
print("Blue mean:", np.mean(b_means), "StD:", np.std(b_means))
