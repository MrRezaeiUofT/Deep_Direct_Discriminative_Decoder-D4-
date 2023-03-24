import torch

from D4_model import D4RegMNV
import numpy as np
from utils import *
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA,KernelPCA
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
#############################################################################
import numpy as np
import matplotlib.pyplot as plt
from Langevin_utils import *

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
X =(X -np.mean(X,axis=0))/np.std(X,axis=0)
XDsign = calDesignMatrix_V2(Spikes,len_history_design+1).squeeze()
[XDsign_train, XDsign_test, Spikes_train, Spikes_test,
 X_train, X_test] = train_test_split( XDsign,Spikes, X, test_size=0.2, shuffle=False)

principalComponents_XDsign = pca.fit_transform(Spikes_train)
config_D4={
    'supervised':False,
    'state_process_type': 'random_walk',# Lorenz, random_walk
    'data_dim':InputDim,
    'history_length':len_history_design,
    'latent_dim':latent_dim,
    'pp_hidden_dim':20,
    'pp_nlayers':3,
    'MCMC_size':10,
    'pp_dropout_rate':.5,
    'learing_rate':1e-3,
    'batch_size': 400,
    'epochs':400,
    'visualization_step':10,
    'EM_itr':100,
    'kl_lambda':.0,
    'kl_lambda_prior':0.0,
    'pca_comp':torch.transpose(torch.tensor(principalComponents_XDsign, dtype=torch.double),0,1)
}
if config_D4['supervised']:
  ELBO, posterior,posterior_smooth, corr_score_tr, mae_score_tr = D4.variational_I(XDsign_train,X_train)
else:
  ELBO, posterior,posterior_smooth, corr_score_tr, mae_score_tr = D4.variational_em(XDsign_train,X_train)



# ax = plt.figure().add_subplot(projection='3d')
# ax.plot(get_normalized(X_train[:,0].squeeze(),config_D4),'k', lw=0.5)
# ax.plot(get_normalized(posterior[0][:,0].detach().numpy().squeeze(),config_D4),'r', lw=0.5)


''' Test result show'''
posterior_test,posterior_smooth_test, cc_te, mae_te = D4.get_posterior(XDsign_test,X_test)

# ax = plt.figure().add_subplot(projection='3d')
# ax.plot(get_normalized(X_test[:,0].squeeze(),config_D4),'k', lw=0.5)
# ax.plot(get_normalized(posterior_test[0][:,0].detach().numpy().squeeze(),config_D4),'r', lw=0.5)


f, axes = plt.subplots(3, 1, sharex=True, sharey=False)
axes[0].plot((np.array(ELBO)), label='Q')
axes[0].set_title('Q')
axes[1].plot((np.array(cc_tr)), label='CC-Tr')
axes[1].set_title('CC-Tr')
axes[2].plot((np.array(mae_tr)) ,label='MAE-Tr')
axes[2].set_title('MAE-Tr')


