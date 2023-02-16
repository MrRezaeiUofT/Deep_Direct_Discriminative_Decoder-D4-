import numpy as np
'''Synthetic Dataset '''


'''
generate a 1-D random walk state and Gaussian and Bernoulli observations based on the state values in each timepoint
non-diagonal covariance marix 
'''
class Data_Generator_V2:
    def __init__(self, config, rnd):
        np.random.seed(rnd)
        self.number_sample =int(config['simulation_duration'] / config['dt']) # number of data point
        self.dt = config['dt'] # time resolution
        self.alphaX = config['alphaX']
        self.sigmaX = config['sigmaX']
        self.initial_X = config['initial_X']
        self.lim_X=config['lim_X']
        self.hk = config['h_K']
        self.X= np.zeros([self.number_sample, 1])
        self.X[0] = self.initial_X


        self.number_LFPs = config['number_LFPs']
        self.LFPs_sigm = config['LFPs_sigma']
        self.LFPs_alpha = config['LFPs_alpha']
        self.LFPs_init = config['LFPs_init']
        self.LFPs = np.zeros([self.number_sample, self.number_LFPs])


    ''' models p(x_k|x_(k-1) ~ N (\alpha x_(k-1), \sigma) where x_k is a 2D state'''
    def generate_state_fun(self):
        for i in range(1, self.number_sample-1):
            temp=self.alphaX * self.X[i] + np.random.normal(0, self.sigmaX) # 1D random walk
            if (temp > self.lim_X[0]) and (temp < self.lim_X[1]): # keep the state values bounded
                self.X[i+1] = temp
            else:
                self.X[i+1] = self.X[i]
            self.X=self.X-self.X.mean()
        return self.X



    def generate_LFPs_obs_fun(self):
        ''' generate multi-variate Gaussian observations from the x-state '''
        for j in range(self.number_LFPs):
            self.LFPs[:, j]= np.random.normal(self.LFPs_alpha[j] * (self.X[:]),self.LFPs_sigm[j]).squeeze()
        return self.LFPs


def cov_gen(min_c,max_c, number_LFP_chs ,scale=10):
    sigma_ii=np.linspace(min_c,max_c, number_LFP_chs)
    cov=np.zeros((number_LFP_chs,number_LFP_chs))
    for i in range(number_LFP_chs):
        for j in range(i):

            if i==j:
                cov[i,i]=sigma_ii[i]
            else:
                temp=np.random.uniform(0,1,1)
                cov[i,j]=temp/scale
                cov[j, i] = temp / scale


