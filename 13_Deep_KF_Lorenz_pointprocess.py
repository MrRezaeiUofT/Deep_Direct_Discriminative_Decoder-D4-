import torch
from utils import *
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA,KernelPCA
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
#############################################################################
import numpy as np
import matplotlib.pyplot as plt
from Deep_KF import *
np.random.seed(1234)
from Lorenz import lorenz_sample_generator
np.random.seed(1234)
number_of_observations = 50

min_sigma = 0.1
latent_dim = 3
len_history_design =20
max_firing_rate = 20
max_hist_dependency_Langevin_data =len_history_design//2
pca = KernelPCA(n_components=latent_dim, kernel='rbf')

lorenz_config = {'initial_val' : (0., 1., 1.05),
                    'dt': 0.01,
                    'traj_size':500,
                    's':10,
                    'r':18,
                    'b':2.667,
                 'noise_enable':True,
                 'noise_std':4,
                 'scale_range': [-1,1]
                 }
Lorenz_simulator= lorenz_sample_generator(lorenz_config)
xs, ys, zs = Lorenz_simulator.sample_trajectory()
f, axes = plt.subplots(3, 1, sharex=True, sharey=False)
axes[0].plot(xs.squeeze())
axes[1].plot(ys.squeeze())
axes[2].plot(zs.squeeze())

Spikes = Lorenz_simulator.generate_spikes(xs,ys,zs,number_of_observations,min_sigma,max_hist_dependency_Langevin_data,max_firing_rate)
Spikes = np.delete( Spikes,np.where(Spikes.sum(axis=0)<10)[0] , axis=1)

# Spikes = gaussian_kernel_smoother(Spikes,2,10)

InputDim = Spikes.shape[1]
plt.figure()
plt.imshow(Spikes.T)
X=np.concatenate([xs,ys,zs],axis=-1)
##################################################################################
Spikes =(Spikes -np.mean(Spikes,axis=0))/np.std(Spikes,axis=0)
X =(X -np.mean(X,axis=0))/np.std(X,axis=0)
[Spikes_train, Spikes_test,
 X_train, X_test] = train_test_split( Spikes, X, test_size=0.2, shuffle=False)



Spikes_train = torch.from_numpy(np.swapaxes(np.expand_dims(Spikes_train,axis=0),0,2)).float()
Spikes_test = torch.from_numpy(np.swapaxes(np.expand_dims(Spikes_test,axis=0),0,2)).float()
X_train = torch.from_numpy(X_train).float()
X_test = torch.from_numpy(X_test).float()



sample_size= Spikes.shape[1]
T = X_train.shape[0]
z_dim=X_train.shape[1]
use_cuda = False
dkf = DKF(use_cuda=use_cuda,
          annealing_factor=0.1, input_dim=1,
          z_dim=z_dim, rnn_dim=100,
          emission_dim=sample_size, transition_dim=z_dim)

svi = SVI(dkf.model, dkf.guide, optim=pyro.optim.Adam({"lr":0.001}), loss=Trace_ELBO(num_particles=1))

from scipy.stats import pearsonr

torch.manual_seed(10)
num_epochs = 20
losses = []
val_losses = []
pyro.clear_param_store()
for epoch in range(num_epochs):

    loss = 0
    loss += svi.step(Spikes_train) / Spikes_train.size(1)
    losses.append(loss)
    val_loss = 0
    val_loss += svi.evaluate_loss(Spikes_train) / Spikes_train.size(1)
    print("Epoch: {0}, Loss: {1:.3f}, Val Loss: {2:.3f}, sigma: {3:.3f}".format(epoch + 1, loss, val_loss,
                                                                                dkf.sigma.item()))
    val_losses.append(val_loss)


all_samples = get_result_dkf(dkf, 1, sample_size, Spikes_train, T, 'joint')
fig, (ax1, ax2, ax3) = plt.subplots(3)
ax1.plot(X_train[:, 0].squeeze(), 'k')
ax2.plot(X_train[:, 1].squeeze(),'k')
ax3.plot(X_train[:, 2].squeeze(),'k')

for ii_sample in range(len(all_samples)):
    z_0_cc = np.abs(pearsonr(X_train[:, 0].squeeze(), all_samples[ii_sample][:, 0].squeeze())[0])
    z_0_MSE = mean_squared_error(X_train[:, 0].squeeze(), all_samples[ii_sample][:, 0].squeeze())

    z_1_cc = np.abs(pearsonr(X_train[:, 1].squeeze(), all_samples[ii_sample][:, 1].squeeze())[0])
    z_1_MSE = mean_squared_error(X_train[:, 1].squeeze(), all_samples[ii_sample][:, 1].squeeze())

    z_2_cc = np.abs(pearsonr(X_train[:, 2].squeeze(), all_samples[ii_sample][:, 2].squeeze())[0])
    z_2_MSE = mean_squared_error(X_train[:, 2].squeeze(), all_samples[ii_sample][:, 2].squeeze())
    print("CC: {0:.3f}, MSE: {1:.3f}".format((z_0_cc+z_1_cc+z_1_cc)/3, (z_1_MSE+z_0_MSE+z_2_MSE)/3))
    ax1.plot(all_samples[ii_sample][:, 0].squeeze())
    ax2.plot(all_samples[ii_sample][:, 1].squeeze())
    ax3.plot(all_samples[ii_sample][:, 2].squeeze())

plt.figure()
result_df = pd.DataFrame(X_train.numpy().squeeze(), columns=['x', 'y', 'z'])
result_df[['x_hat', 'y_hat', 'z_hat']] = all_samples[ii_sample].squeeze()
cor = result_df.corr()
sns.heatmap(np.abs(cor), annot=True, cmap=plt.cm.Reds)
plt.title('Deep-KF_corr_result')
plt.show()

### infere lorenz parameters from observarion
#
# from scipy.optimize import minimize
# observ = X
# def lorenz_obsr( pars):
#     s=pars[0]
#     r=pars[1]
#     b=pars[2]
#
#
#     x=observ[:-1,0]
#     y = observ[:-1, 1]
#     z = observ[:-1, 2]
#
#     x_dot = np.diff(observ[:, 0])
#     y_dot =  np.diff(observ[:, 1])
#     z_dot =  np.diff(observ[:, 2])
#     loss= np.nansum((x_dot- s*(x-y))**2)
#     loss += np.nansum((y_dot - (x * (r - z)-y)) ** 2)
#     loss += np.nansum((z_dot - (x *y  - b*z)) ** 2)
#     return loss
#
#
#
# def find_lorenz_params():
#
#     res = minimize(lorenz_obsr, [1,1,1],
#                    options={'gtol': 1e-6, 'disp': True})
#     return res
#
# result_l=find_lorenz_params()
# sigma=result_l.x[0]
# rho=result_l.x[1]
# beta=result_l.x[2]