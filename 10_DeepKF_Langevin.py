import torch
from utils import *

from sklearn.decomposition import PCA,KernelPCA
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
#############################################################################
import numpy as np
import matplotlib.pyplot as plt
from Langevin_utils import *
from Deep_KF import *
np.random.seed(1234)
number_of_observations = 100


latent_dim = 2
len_history_design =20
pca = KernelPCA(n_components=latent_dim, kernel='rbf')

#################
my_k = 1
my_max_time = 10
initial_position = .1
initial_velocity = .5
scale_size = 4
my_gamma=10
my_kBT=0.5
my_dt=0.01
min_sigma = 0.01
max_firing_rate = 15
max_hist_dependency_Langevin_data =len_history_design//2
times, positions, velocities, total_energies = baoab(harmonic_oscillator_energy_force, \
                                                                            my_max_time, my_dt, my_gamma, my_kBT, \
                                                                            initial_position, initial_velocity,\
                                                                            k=my_k)
positions = np.array(positions).reshape([-1,1])
positions -= positions.mean()
positions = positions*scale_size

velocities = np.array(velocities).reshape([-1,1])
velocities -= velocities.mean()
velocities = velocities*scale_size

Spikes = generate_spikes(positions,velocities,number_of_observations,min_sigma,max_hist_dependency_Langevin_data,max_firing_rate)

Spikes = np.delete( Spikes,np.where(Spikes.sum(axis=0)<10)[0] , axis=1)


plt.figure()
plt.plot(times,positions,marker='.',label='position',linestyle='-')
plt.plot(times,velocities,marker='',label='position',linestyle='-')
plt.xlabel('time')
plt.show()
plt.figure()
plt.imshow(Spikes.T)
Spikes = gaussian_kernel_smoother(Spikes,2,10)

InputDim = Spikes.shape[1]

X=np.concatenate([positions,velocities],axis=-1)
##################################################################################
Spikes =(Spikes -np.mean(Spikes,axis=0))
''' smooth spikes'''
from scipy import signal
# for ii in range(Spikes.shape[1]):
#     win = signal.windows.hann(10)
#     Spikes[:,ii] = signal.convolve(Spikes[:,ii], win, mode='same') / sum(win)

X =(X -np.mean(X,axis=0))/np.std(X,axis=0)
[ Spikes_train, Spikes_test,
 X_train, X_test] = train_test_split(Spikes, X, test_size=0.2, shuffle=False)



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
          z_dim=z_dim, rnn_dim=10,
          emission_dim=sample_size, transition_dim=z_dim)

svi = SVI(dkf.model, dkf.guide, optim=pyro.optim.Adam({"lr":0.001}), loss=Trace_ELBO(num_particles=1))

from scipy.stats import pearsonr

torch.manual_seed(10)
num_epochs = 50
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


all_samples = get_result_dkf(dkf, 10, sample_size, Spikes_train, T, 'joint')
fig, (ax1, ax2) = plt.subplots(2)
ax1.plot(X_train[:, 0].squeeze())
ax2.plot(X_train[:, 1].squeeze())
for ii_sample in range(len(all_samples)):
    z_0_cc = np.abs(pearsonr(X_train[:, 0].squeeze(), all_samples[ii_sample][:, 0].squeeze())[0])
    z_0_MSE = mean_squared_error(X_train[:, 0].squeeze(), all_samples[ii_sample][:, 0].squeeze())

    z_1_cc = np.abs(pearsonr(X_train[:, 1].squeeze(), all_samples[ii_sample][:, 1].squeeze())[0])
    z_1_MSE = mean_squared_error(X_train[:, 1].squeeze(), all_samples[ii_sample][:, 1].squeeze())


    print("CC: {0:.3f}, MSE: {1:.3f}".format((z_0_cc+z_1_cc)/2, (z_1_MSE+z_0_MSE)/2))
    ax1.plot(all_samples[ii_sample][:, 0].squeeze())
    ax2.plot(all_samples[ii_sample][:, 1].squeeze())

import pandas as pd
import seaborn as sns

plt.figure()
result_df = pd.DataFrame(X_train.numpy().squeeze(), columns=['x', 'y'])
result_df[['x_hat', 'y_hat']] = all_samples[ii_sample].squeeze()
cor = result_df.corr()
sns.heatmap(np.abs(cor), annot=True, cmap=plt.cm.Reds)
plt.title('Deep-KF_corr_result')
plt.show()