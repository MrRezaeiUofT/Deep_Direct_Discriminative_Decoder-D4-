from D4_model import D4RegMNV
import numpy as np
from Data_Generator_V2 import Data_Generator_V2 ,cov_gen
import torch
from utils import *
from sklearn.decomposition import PCA, KernelPCA


import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import torch
obser_dim = 20
LFP_ch=obser_dim//2
latent_dim = 2
len_history_design=5
pca = KernelPCA(n_components=latent_dim, kernel='rbf')
config_synthetic_Data = {  ## set parameters fpr  x state
        'alphaX': .9,  # state transition Coff
        'sigmaX': .1,  # noise variance
        'initial_X': 0,  # state initialization
        'lim_X': [-5, 5],  # bounds for possible values for the state x
        'h_K':5,

        ## parameters for the LFP signals generation
        'number_LFPs': LFP_ch,
        'LFPs_sigma': np.linspace(.7,1.4, num=LFP_ch),
        'LFPs_init': np.zeros([LFP_ch, 1]).squeeze(),
        'LFPs_alpha': np.ones( (LFP_ch,)),
        ## simulation time
        'dt': 0.001,
        'simulation_duration': .2,
}

Synth_gen = Data_Generator_V2(config_synthetic_Data,rnd=np.random.randint(1430))
X1 = Synth_gen.generate_state_fun()
Y1 = Synth_gen.generate_LFPs_obs_fun()

config_synthetic_Data['sigmaX'] = .4
Synth_gen = Data_Generator_V2(config_synthetic_Data,rnd=np.random.randint(1230))
X2 = Synth_gen.generate_state_fun()
Y2 = Synth_gen.generate_LFPs_obs_fun()

X=np.concatenate([X1,X2],axis=-1)
Y=np.concatenate([Y1,Y2],axis=-1)

Y =(Y -np.mean(Y,axis=0))/np.std(Y,axis=0)
X =(X -np.mean(X,axis=0))/np.std(X,axis=0)

XDsign = calDesignMatrix_V2(Y,len_history_design+1).squeeze()
principalComponents_XDsign = pca.fit_transform(Y)

config_D4={
    'supervised':True,
    'state_process_type': 'random_walk',  # Lorenz, random_walk
    'data_dim':obser_dim,
    'history_length':len_history_design,
    'latent_dim':latent_dim,
    'pp_hidden_dim':10,
    'pp_nlayers':2,
    'MCMC_size':20,
    'pp_dropout_rate':.5,
    'learing_rate':0.001,
    'batch_size': 200,
    'epochs':3,
    'visualization_step':8,
    'EM_itr':100,
    'kl_lambda':.1,
    'kl_lambda_prior': .1,
    'pca_comp':torch.transpose(torch.tensor(principalComponents_XDsign, dtype=torch.double),0,1)
}

D4= D4RegMNV(config_D4)
ELBO, posterior,posterior_smooth, _, _ = D4.variational_I(XDsign,X)

plt.figure()
plt.plot(ELBO)
