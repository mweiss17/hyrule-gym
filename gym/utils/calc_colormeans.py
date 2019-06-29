import numpy as np
import gzip
import pickle
import os
import matplotlib.pyplot as plt
'''
Script that runs on panos directly to obtain the mean and standard deviation of the color channels in 
the dataset.
'''

region = "saint-urbain"
path = "/home/rogerg/Documents/autonomous_pedestrian_project/navi/hyrule-gym/data/data/mini-corl/processed/images.pkl.gz"
#path = os.path.expanduser(
#    "~/dev/hyrule-gym/data/data/mini-corl/processed/images.pkl.gz")
# data_df = pd.read_hdf(path + "hyrule-gym/data/" + region + "/processed/data.hdf5", key='df', mode='r')

f = gzip.GzipFile(path, "r")
images_df = np.array(list(pickle.load(f).values())).astype(np.uint8)

# just an example to check that the data is okay
print(images_df.shape)
plt.imshow(images_df[1])
plt.show()

images = images_df.astype(np.float) / 255

# we don't need to iterate if we use numpy arrays
print(
    f"Red mean: {images[:,:,:,0].mean()} StD: {images[:,:,:,0].std()} \n"
    f"Green mean: {images[:,:,:,1].mean()} StD: {images[:,:,:,1].std()} \n"
    f"Blue mean: {images[:,:,:,2].mean()} StD: {images[:,:,:,2].std()} ")
