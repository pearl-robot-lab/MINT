import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.dpi'] = 500
matplotlib.rcParams['font.size'] = 30
matplotlib.use('Qt5Agg')

import numpy as np

# read the folders in the directory
import os

files=os.listdir(".")
plt.figure(figsize=(25,10))
# iterate over the folders
for file in files:
    if file.endswith(".npy"):
        # read the file
        data=np.load(file)
        x=np.arange(0,data.shape[1])
        y=np.sum(data[0,:,:,0],axis=-1)
        # plot the data
        plt.plot(x,y, label=file[:-4], linewidth=5)
plt.grid()
plt.legend(bbox_to_anchor=(0.65, 0.5), loc='center', borderaxespad=0)
plt.box(False)
plt.xlim([-30,len(x)+30])
plt.xlabel('Timestep')
plt.ylabel('Number of keypoints')
# save the plot
plt.savefig('plot.svg',bbox_inches='tight', transparent=True)
# close the plot
plt.close()