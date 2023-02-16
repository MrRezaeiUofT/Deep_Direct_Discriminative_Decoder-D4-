
from Data_Generator_V2 import Data_Generator_V2
from utils import *
obser_dim = 20
LFP_ch=obser_dim//2
latent_dim = 2
config_synthetic_Data = {  ## set parameters fpr  x state
        'alphaX': .9,  # state transition Coff
        'sigmaX': .1,  # noise variance
        'initial_X': 0,  # state initialization
        'lim_X': [-5, 5],  # bounds for possible values for the state x
        'h_K':0,
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
