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

from Lorenz import lorenz_sample_generator
np.random.seed(1234)
number_of_observations = 200

min_sigma = 0.1
latent_dim = 3
len_history_design =20
max_firing_rate = 20
max_hist_dependency_Langevin_data =len_history_design//2
pca = KernelPCA(n_components=latent_dim, kernel='rbf')

lorenz_config = {'initial_val' : (0., 1., 1.05),
                    'dt': 0.01,
                    'traj_size':5000,
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
XDsign = calDesignMatrix_V2(Spikes,len_history_design+1).squeeze()
[XDsign_train, XDsign_test, Spikes_train, Spikes_test,
 X_train, X_test] = train_test_split( XDsign,Spikes, X, test_size=0.2, shuffle=False)

principalComponents_XDsign = pca.fit_transform(Spikes_train)
config_D4={
    'supervised':True,
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
    'epochs':100,
    'visualization_step':10,
    'EM_itr':10,
    'kl_lambda':.1,
    'kl_lambda_prior':0.0,
    'pca_comp':torch.transpose(torch.tensor(principalComponents_XDsign, dtype=torch.double),0,1)
}
D4= D4RegMNV(config_D4)
if config_D4['supervised']:
  ELBO, posterior,posterior_smooth, corr_score_tr, mae_score_tr = D4.variational_I(XDsign_train,X_train)
else:
  ELBO, posterior,posterior_smooth, corr_score_tr, mae_score_tr = D4.variational_em(XDsign_train,X_train)



ax = plt.figure().add_subplot(projection='3d')
ax.plot(get_normalized(X_train[:,0].squeeze(),config_D4),
        get_normalized(X_train[:,1].squeeze(),config_D4),
        get_normalized(X_train[:,2].squeeze(),config_D4),'k', lw=0.5)
ax.plot(get_normalized(posterior[0][:,0].detach().numpy().squeeze(),config_D4),
        get_normalized(posterior[0][:,1].detach().numpy().squeeze(),config_D4),
        get_normalized(posterior[0][:,2].detach().numpy().squeeze(),config_D4),'r', lw=0.5)
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
ax.set_title("Lorenz Attractor")

''' Test result show'''
posterior_test,posterior_smooth_test,corr_score_te, mae_score_te = D4.get_posterior(XDsign_test,X_test)

ax = plt.figure().add_subplot(projection='3d')
ax.plot(get_normalized(X_test[:,0].squeeze(),config_D4),
        get_normalized(X_test[:,1].squeeze(),config_D4),
        get_normalized(X_test[:,2].squeeze(),config_D4),'k', lw=0.5)
ax.plot(get_normalized(posterior_test[0][:,0].detach().numpy().squeeze(),config_D4),
        get_normalized(posterior_test[0][:,1].detach().numpy().squeeze(),config_D4),
        get_normalized(posterior_test[0][:,2].detach().numpy().squeeze(),config_D4),'r', lw=0.5)
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
ax.set_title("Lorenz Attractor")



f, axes = plt.subplots(3, 1, sharex=True, sharey=False)
axes[0].plot((np.array(ELBO)), label='Q')
axes[0].set_title('Q')
axes[1].plot((np.array(corr_score_tr)), label='CC-Tr')
axes[1].set_title('CC-Tr')
axes[2].plot((np.array(mae_score_tr)) ,label='MAE-Tr')
axes[2].set_title('MAE-Tr')
