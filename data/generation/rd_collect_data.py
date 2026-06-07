# %%
# %%
import scipy.io as scio
import numpy as np
import os
import h5py
import numpy as np
# %%
# Load data
filebase='./'

files =['RD_gauss_cov'+str(i) for i in range(1,17)]

# files=['RD_gauss_cov1','RD_gauss_cov2','RD_gauss_cov3','RD_gauss_cov4',
#        'RD_gauss_cov5','RD_gauss_cov6','RD_gauss_cov7', 'RD_gauss_cov8',]
u0s=np.empty((0,255))
Ks=np.empty((0,255))
solutions=np.empty((0,101,255))
for file in files:
    data=scio.loadmat(filebase+file+'.mat')
    u0s=np.append(u0s,data['u0s'],axis=0)
    solutions=np.append(solutions,data['solutions'],axis=0) 
    Ks=np.append(Ks,data['Ks'],axis=0)

x_grid=data['x_grid']
t_grid=data['t_grid']
# %%
# training_data = {'x_grid':x_grid,'t_grid':t_grid,'ICs': ICs, 'solutions': solutions}
# femFile = os.path.join(filebase, 'diffusion_gauss_cov40k.mat')
# scio.savemat(femFile, training_data)
# %%
training_data = {'x_grid':x_grid,'t_grid':t_grid,'u0s': u0s,'Ks':Ks, 'solutions': solutions}
femFile = os.path.join(filebase, 'RD_gauss_cov80k.h5')
with h5py.File(femFile , 'w') as hf:
    for key, value in training_data.items():
        hf.create_dataset(key, data=value)
print("Data saved to: ", femFile)
# %%
# hf=h5py.File(femFile, 'r')
# x_grid=hf['x_grid'][:]
# with h5py.File(femFile, 'r') as hf:
#     # List all groups (datasets) in the file
#     print("Keys: ", list(hf.keys()))
    
#     # Access a specific dataset
#     data = hf['dataset_name'][:]k
# %%
