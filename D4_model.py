# PyTorch
import torch
import numpy as np
from torch.distributions import Categorical, MultivariateNormal, Normal, \
    LowRankMultivariateNormal, kl_divergence
import torch.nn as nn
from torch.autograd import Variable
from utils import *
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
device = torch.device('cpu')
dtype = torch.float32
# torch.autograd.set_detect_anomaly(True)
from scipy.stats import pearsonr
# Helper function to convert between numpy arrays and tensors
to_t = lambda array: torch.tensor(array, device=device, dtype=dtype)
from_t = lambda tensor: tensor.to("cpu").detach().numpy()

def diagonal_matrix_create(diag_elements):
    output=torch.zeros((diag_elements.shape[0],diag_elements.shape[1],diag_elements.shape[1]))
    for ii in range( diag_elements.shape[0]):
        output[ii] = torch.diag(torch.squeeze(diag_elements[ii]))
    return output




class IndependentLinearRegression(nn.Module):
    def __init__(self, weights, bias, diag_covariance):
        super().__init__()
        """
        Parameters
        ----------
        weights: N x D tensor of regression weights
        bias: N tensor of regression bias
        diag_covariance: N tensor of non-negative variances
        """
        self.data_dim, self.covariate_dim = weights.shape[-2:]
        assert bias.shape[-1] == self.data_dim
        assert diag_covariance.shape[-1] == self.data_dim
        self.weights = weights
        self.bias = bias
        self.diag_covariance = diag_covariance
        self.softplus = nn.Softplus()

    def forward(self, covariates):

        predictions = torch.matmul( covariates, self.weights)
        predictions += self.bias
        lkhd = Normal(predictions,  self.softplus(self.diag_covariance) + 1e-5)
        return predictions, lkhd

    def log_prob(self, data, covariates):
        """
        Compute the log probability of the data given the covariates using the
        model parameters. Note that this function's signature is slightly
        different from what you implemented in Lab 7.

        Parameters
        ----------
        data: a tensor with lagging dimension $N$, the dimension of the data.
        covariates: a tensor with lagging dimension $D$, the covariate dimension

        Returns
        -------
        lp: a tensor of log likelihoods for each data point and covariate pair.
        """
        predictions = torch.matmul( covariates, self.weights)
        predictions += self.bias
        lkhd = Normal(predictions,   self.softplus(self.diag_covariance) + 1e-5)
        return lkhd.log_prob(data).sum(axis=-1)


class DependentLinearRegression(nn.Module):
    def __init__(self, weights, bias, Ch_covariance):
        super().__init__()
        """
        Parameters
        ----------
        weights: N x D tensor of regression weights
        bias: N tensor of regression bias
        diag_covariance: N tensor of non-negative variances
        """
        self.data_dim, self.covariate_dim = weights.shape[-2:]
        assert bias.shape[-1] == self.data_dim
        assert Ch_covariance.shape[-1] == self.data_dim
        self.weights = weights
        self.bias = bias
        self.Ch_covariance = Ch_covariance
        self.softplus = nn.Softplus()

    def forward(self, covariates):

        predictions = torch.matmul( covariates,  torch.matmul(self.weights,torch.transpose(self.weights,1,0)))
        predictions += self.bias
        lkhd = MultivariateNormal(predictions, torch.matmul(self.Ch_covariance,torch.transpose(self.Ch_covariance,1,0)) +1e-5*torch.eye(self.data_dim))
        return predictions, lkhd

    def log_prob(self, data, covariates):
        """
        Compute the log probability of the data given the covariates using the
        model parameters. Note that this function's signature is slightly
        different from what you implemented in Lab 7.

        Parameters
        ----------
        data: a tensor with lagging dimension $N$, the dimension of the data.
        covariates: a tensor with lagging dimension $D$, the covariate dimension

        Returns
        -------
        lp: a tensor of log likelihoods for each data point and covariate pair.
        """
        predictions = torch.matmul( covariates, torch.matmul(self.weights,torch.transpose(self.weights,1,0)))
        predictions += self.bias
        lkhd = MultivariateNormal(predictions,  torch.matmul(self.Ch_covariance,torch.transpose(self.Ch_covariance,1,0))+1e-5*torch.eye(self.data_dim))
        return lkhd.log_prob(data).sum(axis=-1)

    def get_weights(self,X,T):
        ws = torch.matmul(self.weights,torch.transpose(self.weights,1,0))
        ws_T = ws.clone().repeat(T, 1,1)
        return ws_T

class Lorenz_state_process_MVN(nn.Module):
    def __init__(self, weights, bias,  Ch_covariance):
        super().__init__()
        """
        Parameters
        ----------
        weights: N x D tensor of regression weights
        bias: N tensor of regression bias
        diag_covariance: N tensor of non-negative variances
        """

        # self.data_dim, self.covariate_dim = weights.shape[-2:]
        # assert bias.shape[-1] == self.data_dim
        # assert Ch_covariance.shape[-1] == self.data_dim
        self.weights = weights
        self.bias = bias
        self.Ch_covariance = Ch_covariance
        self.softplus = nn.Softplus()
    def update_weights(self,X):
        if len(X.shape)>1:
            T, M = X.shape[-2:]
        else:
            T = X.shape[-1]
        ws=torch.tensor([[-self.weights[0].clone(),self.weights[0].clone(),0],
                         [ self.weights[1].clone(), -1, 0],
                         [0,0, -self.weights[2].clone()]])
        ws_T=ws.clone().repeat(T, 1, 1)
        # ws_T = torch.matmul(self.weights,torch.transpose(self.weights,0,1)).clone().repeat(T, 1, 1)
        if len(X.shape)>1:
            ws_T[:, 1, -1] = 1 * X.clone()[:, 0]
            ws_T[:, 2, 0] = -X.clone()[:, 1]
        else:
            ws_T[:, 1, -1] = 1 * X.clone()[ 0]
            ws_T[:, 2, 0] = -X.clone()[ 1]

        return ws_T
    def forward(self, covariates):

        ws=self.update_weights(covariates)

        predictions = torch.matmul( covariates, ws )+self.bias

        lkhd = MultivariateNormal(predictions, torch.matmul(self.Ch_covariance,torch.transpose(self.Ch_covariance,1,0)))
        return predictions, lkhd

    def log_prob(self, data, covariates):
        """
        Compute the log probability of the data given the covariates using the
        model parameters. Note that this function's signature is slightly
        different from what you implemented in Lab 7.

        Parameters
        ----------
        data: a tensor with lagging dimension $N$, the dimension of the data.
        covariates: a tensor with lagging dimension $D$, the covariate dimension

        Returns
        -------
        lp: a tensor of log likelihoods for each data point and covariate pair.
        """
        ws = self.update_weights(covariates)

        predictions = torch.matmul(covariates, ws)+self.bias

        lkhd = MultivariateNormal(predictions,  torch.matmul(self.Ch_covariance,torch.transpose(self.Ch_covariance,1,0)))
        return lkhd.log_prob(data).sum(axis=-1)

    def get_weights(self,X,T):

        return self.update_weights(X)

class PredictionProcess_DNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.Input_size = config['data_dim']
        self.output_size = config['latent_dim']
        self.hidden_dim = config['pp_hidden_dim']
        self.n_layers = config['pp_nlayers']
        self.dropout_prob = config['pp_dropout_rate']
        self.history_length = config['history_length']


        self.flatten = nn.Flatten()
        self.softplus=nn.Softplus()
        self.q_mu = nn.Sequential(
            nn.Linear( (1+self.history_length) *self.Input_size,self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Linear(self.hidden_dim, self.output_size)
        )
        self.q_log_var = nn.Sequential(
            nn.Linear( (1+self.history_length)  *self.Input_size, self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Linear(self.hidden_dim, self.output_size)
        )

    def reparameterize(self, log_var):
        # std can not be negative, thats why we use log variance
        sigma = torch.exp(0.5 * log_var) + 1e-5

        return sigma

    def forward(self, x):

        X_F = self.flatten(x)
        mu = self.q_mu(X_F)
        log_var = self.q_log_var(X_F)
        # log_var =( torch.reshape(log_var, (-1,self.output_size,self.output_size)))
        var= diagonal_matrix_create(self.softplus(log_var))+1e-5*torch.eye(self.output_size)
        # var = torch.matmul(self.reparameterize(log_var),torch.transpose(self.reparameterize(log_var),2,1))

        return  mu, var , mu, var

class PredictionProcess_RNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.Input_size = config['data_dim']
        self.output_size = config['latent_dim']
        self.hidden_dim = config['pp_hidden_dim']
        self.n_layers = config['pp_nlayers']
        self.dropout_prob = config['pp_dropout_rate']
        self.history_length = config['history_length']

        self.rnn =nn.RNN(
            self.Input_size, self.hidden_dim, self.n_layers, batch_first=True, dropout=self.dropout_prob
        )
        # self.fc2 = nn.Linear(self.hidden_dim, self.output_size)
        # self.fc1 = nn.Linear(self.Input_size, self.hidden_dim)
        # self.fc3 = nn.Linear(self.Input_size, self.output_size)
        self.flatten = nn.Flatten()
        self.softplus=nn.Softplus()
        self.q_mu = nn.Sequential(
            nn.Linear(( self.history_length)*self.hidden_dim+self.Input_size,self.hidden_dim),
            # nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Linear(self.hidden_dim, self.output_size)
        )
        self.q_log_var = nn.Sequential(
            nn.Linear((self.history_length)*self.hidden_dim+self.Input_size, self.hidden_dim),
            # nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Linear(self.hidden_dim, self.output_size)
        )

        self.qh_mu = nn.Sequential(
            nn.Linear((self.history_length) * self.hidden_dim, self.hidden_dim),
            # nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Linear(self.hidden_dim, self.output_size)
        )
        self.qh_log_var = nn.Sequential(
            nn.Linear(( self.history_length) * self.hidden_dim, self.hidden_dim),
            # nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Linear(self.hidden_dim, self.output_size)
        )

    def reparameterize(self, log_var):
        # std can not be negative, thats why we use log variance
        sigma = torch.exp(0.5 * log_var) + 1e-5

        return sigma

    def forward(self, x):
        h0 = Variable(torch.randn(self.n_layers, x.shape[0], self.hidden_dim))
        # One time step
        # out_all, hn = self.rnn(x, h0)
        # out_all = torch.reshape(out_all, (x.shape[0],-1))

        # x_embed = torch.cat((out, torch.squeeze(hn)), -1)
        # mu = self.q_mu(out_all)
        # log_var = self.q_log_var(out_all)

        out_h, hnh = self.rnn(x[:,:-1,:], h0)
        out_h = torch.reshape(out_h, (x.shape[0], -1))
        # x_embed = torch.cat((out, torch.squeeze(hn)), -1)


        x_embed = torch.cat((out_h, torch.squeeze(x[:,-1,:])), -1)
        mu = self.q_mu(x_embed)
        log_var = self.q_log_var(x_embed)
        var = diagonal_matrix_create(self.softplus(log_var)) + 1e-5 * torch.eye(self.output_size)

        return  mu, var, mu, var


class D4RegMNV(object):
    ''' gaussian approximation of the D4'''
    def __init__(self, config):
        self.config = config
        self.data_dim = config['data_dim']
        self.latent_dim = config['latent_dim']
        self.MCMC_size = config['MCMC_size']
        self.learing_rate = config['learing_rate']
        self.batch_size = config['batch_size']
        self.epochs = config['epochs']
        self.EM_itr = config['EM_itr']
        self.visualization_step = config['visualization_step']
        self.kl_lambda = config['kl_lambda']
        self.kl_lambda_prior = config['kl_lambda_prior']
        self.softplus = nn.Softplus()
        self.supervised = config['supervised']
        self.pca_comp = config['pca_comp']
        self.state_process_type = config['state_process_type']
        self.state_process_cov_init = ((torch.cov(self.pca_comp.clone())))/(self.latent_dim**2)
        # print(self.state_process_cov_init)
    def create_state_process(self):
        if self.supervised:
            Transition_matrix = torch.eye(self.latent_dim)
            CH_cov = torch.cholesky(.1*torch.eye(self.latent_dim))
        else:
            temp =(torch.real( torch.linalg.eig(torch.cov(self.pca_comp)).eigenvectors))
            # temp = torch.cholesky(torch.cov(self.pca_comp))
            Transition_matrix = temp.float()
            CH_cov = torch.cholesky(torch.eye(self.latent_dim) * np.random.uniform(0.1,1,self.latent_dim))
        if self.state_process_type == 'Lorenz':
            self.p_x_x_1 = Lorenz_state_process_MVN(
             torch.tensor(1*torch.ones(self.latent_dim),requires_grad=True),
            torch.tensor(torch.zeros(self.latent_dim), requires_grad=True),
           torch.tensor(CH_cov,requires_grad=True))
        elif self.state_process_type == 'random_walk':
            self.p_x_x_1 = DependentLinearRegression(
             torch.tensor(Transition_matrix,requires_grad=True),
            torch.tensor(torch.zeros(self.latent_dim),requires_grad=True),
           torch.tensor(CH_cov,requires_grad=True))

        self.mux0 = torch.tensor(torch.zeros(self.latent_dim),requires_grad=True,dtype=torch.float32)
        self.sigma0 = torch.tensor(torch.ones(self.latent_dim),requires_grad=True,dtype=torch.float32)
        self.px0= Normal(self.mux0,self.softplus( self.sigma0.clone() ) )

    def create_prediction_process(self):

        # Initialize the observation model p(x| y,h)
        self.p_x_y_h = PredictionProcess_RNN(self.config)

    def compute_posterior(self,prediction_process_out,prediction_h):
        # observations
        ''' observation process'''

        mu_o, sigma_o = prediction_process_out
        mu_oh, sigma_oh = prediction_h
        T = mu_o.shape[0]  # length of sequence

        ''' state process'''
        mu_0 = self.mux0
        sigma_0 =torch.diag( (self.px0.scale))

        W_x = self.p_x_x_1.get_weights(mu_o.clone(),T)
        b_x = self.p_x_x_1.bias
        sigma_xx =  torch.matmul(self.p_x_x_1.Ch_covariance ,torch.transpose(self.p_x_x_1.Ch_covariance ,1,0))+ 1e-5


        ''' posterior estimators'''
        vk = torch.zeros(T, self.latent_dim)
        Mk = torch.zeros(T, self.latent_dim, self.latent_dim)
        sigmak = torch.zeros(T, self.latent_dim, self.latent_dim)  # sigma of posterior
        muk = torch.zeros(T, self.latent_dim)  # mean of posterior

        for ii in range(T):

            if ii == 0:
                vk[ii] = mu_0.clone()
                Mk[ii] = sigma_0.clone()

                sigmak[ii] = torch.inverse(torch.inverse(Mk[ii].clone()) +
                                            torch.inverse((sigma_o[ii])))

                mk1vk = torch.matmul(torch.inverse(Mk[ii].clone()), vk[ii].clone())[:, None]
                qk1fk = torch.matmul(
                    torch.inverse((sigma_o[ii])), mu_o[ii].clone()
                )[:,None]
                muk[ii] = torch.squeeze(torch.matmul(sigmak[ii].clone(), mk1vk + qk1fk))

            else:
                vk[ii] = torch.matmul((W_x[ii]), muk[ii - 1].clone()) + b_x

                Mk[ii] = torch.matmul(torch.matmul(W_x[ii], sigmak[ii - 1].clone()), torch.transpose(W_x[ii],1, 0)) + sigma_xx.clone()
                #
                sigmak[ii] = torch.inverse(
                    torch.inverse(Mk[ii].clone())
                    # - torch.inverse((sigma_oh[ii ]))
                    + torch.inverse((sigma_o[ii])))#+1e-5*torch.eye(self.latent_dim)

                mk1vk = torch.matmul(torch.inverse(Mk[ii].clone()), vk[ii].clone())[:, None]
                qk1fk = torch.matmul(
                    torch.inverse((sigma_o[ii])), mu_o[ii].clone())[:,None]

                # qk1fk_h = torch.matmul(
                #     torch.inverse((sigma_oh[ii])), mu_oh[ii].clone())[:, None]


                muk[ii] = torch.squeeze(torch.matmul(sigmak[ii].clone(), mk1vk
                                                     + qk1fk
                                                     # -qk1fk_h
                                                     ))

        return [muk, sigmak]

    def compute_smoother(self, posterior ):
        [muk,sigmak]=posterior
        T = muk.shape[0]  # length of sequence
        mu_0 = self.mux0
        sigma_0 = torch.diag((self.px0.scale))
        sigmak_smooth = torch.zeros(T, self.latent_dim, self.latent_dim)  # sigma of posterior
        muk_smooth = torch.zeros(T, self.latent_dim)  # mean of posterior

        W_x =   self.p_x_x_1.get_weights(muk.clone(),T)

        sigmak_smooth[-1] = sigmak[-1].clone()
        muk_smooth[-1] = muk[-1].clone()
        for qq in range(T - 2, -1, -1):

            J = torch.linalg.solve(sigmak[qq+1].clone(),torch.matmul(sigmak[qq].clone(),torch.transpose(W_x[qq],1,0) ))

            sigmak_smooth[qq] = sigmak[qq].clone() +\
                                    torch.matmul(J.clone(),torch.matmul(sigmak_smooth[qq+1].clone()-sigmak[qq+1].clone(), torch.transpose(J.clone(),1,0)))

            muk_smooth[qq] = muk[qq].clone() + torch.matmul( J.clone(), muk_smooth[qq+1].clone()-
                                                         torch.matmul(W_x[qq],muk[qq].clone()))

        J = torch.linalg.solve(sigmak[0].clone(), torch.matmul(sigma_0.clone(), torch.transpose(W_x[qq], 1, 0)))

        self.sigma_h_0 = sigma_0.clone() + \
                            torch.matmul(J.clone(), torch.matmul(sigmak_smooth[0].clone() - sigmak[0].clone(),
                                                                 torch.transpose(J.clone(), 1, 0)))
        self.mu_h_0 = mu_0.clone() + torch.matmul( J.clone(), muk_smooth[0].clone()-
                                                         torch.matmul(W_x[qq],mu_0.clone()))



        return [muk_smooth,sigmak_smooth]

    def elbo_state_process(self, posterior_sample):
        elbo=0
        if self.supervised:
            elbo_x0 = 0#self.px0.log_prob(posterior_sample[0]).mean().mean()
            # Pxx
            elbo_xx = self.p_x_x_1.log_prob(posterior_sample[1:], posterior_sample[:1]).mean().mean()

            elbo += (elbo_xx) + elbo_x0

        else:
            post_0 = MultivariateNormal(self.mu_h_0, self.sigma_h_0).sample((self.MCMC_size,)).T
            for ii in range(self.MCMC_size):
                #Px0
                elbo_x0 = self.px0.log_prob(post_0[:,ii]).mean().mean()
                #Pxx
                elbo_xx = self.p_x_x_1.log_prob(posterior_sample[1:,:,ii], posterior_sample[:1,:,ii]).mean().mean()
                # kl_prior = kl_divergence(MultivariateNormal(self.posterior[0].to(torch.float32).detach(),
                #                                             self.posterior[1].to(torch.float32).detach()),
                #                          MultivariateNormal(torch.zeros(self.latent_dim).to(torch.float32),
                #                                             self.state_process_cov_init.to(torch.float32).detach())).mean().mean()
                elbo+=((elbo_xx)/posterior_sample.shape[-1]
                       + elbo_x0/posterior_sample.shape[-1]
                       # - self.kl_lambda_prior * kl_prior / posterior_sample.shape[-1]
                )


        return elbo
    def elbo_prediction_process(self,posterior_sample,prediction_process_out,prediction_h):
        elbo=0
        mu_o, sigma_o = prediction_process_out
        mu_oh, sigma_oh = prediction_h

        lkhd = MultivariateNormal(mu_o, sigma_o)
        lkhd_h = MultivariateNormal(mu_oh, sigma_oh)
        if self.supervised:
            elbo_xyh = lkhd.log_prob(posterior_sample).mean().mean()
            elbo_xh =  lkhd_h.log_prob(posterior_sample).mean().mean()

            elbo += (elbo_xyh)-self.kl_lambda * elbo_xh

        else:

            for ii in range(self.MCMC_size):
                elbo_xyh =lkhd.log_prob(posterior_sample[:,:,ii]).mean().mean()
                # elbo_xh = lkhd_h.log_prob(posterior_sample[:,:,ii]).mean().mean()
                elbo_xh=kl_divergence(MultivariateNormal(self.posterior_smooth[0].detach(), self.posterior_smooth[1].detach()),lkhd_h).mean().mean()
                elbo+=(elbo_xyh)/posterior_sample.shape[-1] +self.kl_lambda * elbo_xh/posterior_sample.shape[-1]


        return elbo

    def MCMC_sampler(self,posterior):
        [mu,sigma] = posterior

        T=mu.shape[0]
        L = mu.shape[1]
        Samples = torch.zeros((T, L, self.MCMC_size))
        eps =torch.randn_like(Samples[0])
        for ii in range(T):
            # CL = torch.linalg.cholesky(sigma[ii].clone())
            # Samples[ii, :, :] = torch.transpose(mu[ii].clone() +
            #                                             torch.transpose(torch.matmul(sigma[ii].clone(), eps), 0, 1), 0,1)

            Samples[ii, :, :] = MultivariateNormal(mu[ii],sigma[ii]).sample((self.MCMC_size,)).T
        return Samples

    def get_score_posterior(self,posterior,posterior_smooth, X, Loss, itr):
        X =X.detach().numpy()
        X = (X - X.mean(axis=0)) / X.std(axis=0)
        [muk, sigmak] = posterior
        sigmak = get_diagonal(sigmak.clone())
        mean_posterior = muk.clone().detach().numpy()
        mean_posterior = (mean_posterior-mean_posterior.mean(axis=0))/mean_posterior.std(axis=0)
        sigma_posterior = sigmak.clone().detach().numpy()

        ####
        [muk_s, sigmak_s] = posterior_smooth
        sigmak_s = get_diagonal(sigmak_s.clone())
        mean_posterior_s = muk_s.clone().detach().numpy()
        mean_posterior_s = (mean_posterior_s - mean_posterior_s.mean(axis=0)) / mean_posterior_s.std(axis=0)
        sigma_posterior_s= sigmak_s.clone().detach().numpy()


        #### get scores
        corr = np.zeros((self.latent_dim, self.latent_dim))
        MAE = np.zeros((self.latent_dim, self.latent_dim))
        for i in range(self.latent_dim):
            for j in range(self.latent_dim):
                corr[i, j] = pearsonr(X[:, i], posterior[0].detach().numpy()[:, j])[0]
                if corr[i, j]<0:
                    MAE[i, j] =  mean_absolute_error(X[:, i], -mean_posterior[:, j])
                else:
                    MAE[i, j] =  mean_absolute_error(X[:, i], mean_posterior[:, j])
        corr_score = np.abs(corr).max(axis=1).max()
        mae_score = np.abs(MAE).max(axis=1).min()
        ### visualize
        if itr % self.visualization_step == 0:
            print('corr-score =%f, mae-score=%f\n'%(corr_score,mae_score))
            xs_id = np.arange(0, mean_posterior.shape[0])
            f, axes = plt.subplots(self.latent_dim, 2, sharex=True, sharey=False)
            for ii in range(self.latent_dim):
                axes[ii, 0].fill_between(xs_id,
                                         (mean_posterior[:, ii] - 2 * sigma_posterior[:, ii]).squeeze(),
                                         (mean_posterior[:, ii] + 2 * sigma_posterior[:, ii]).squeeze(),
                                         color='r', label='marginal neural 95%', alpha=.5)

                axes[ii, 0].plot(xs_id, mean_posterior[:, ii], 'r')
                axes[ii, 0].plot(xs_id, X[:, ii], 'k')

            for ii in range(self.latent_dim):
                axes[ii, 1].fill_between(xs_id,
                                         (mean_posterior_s[:, ii] - 2 * sigma_posterior_s[:, ii]).squeeze(),
                                         (mean_posterior_s[:, ii] + 2 * sigma_posterior_s[:, ii]).squeeze(),
                                         color='b', label='marginal neural 95%', alpha=.5)

                axes[ii, 1].plot(xs_id, mean_posterior_s[:, ii], 'b')
                axes[ii, 1].plot(xs_id, X[:, ii], 'k')
            print('EM-itr-%d with Total_loss=%f\n' % (itr, Loss))

        return corr_score, mae_score

    def variational_em(self,data,X ):
        """
        Fit the model parameters via variational EM.
        """
        y = to_t(data)
        X = to_t(X)
        self.create_state_process()
        self.create_prediction_process()

        # Run CAVI
        avg_elbos = []
        posterior = None
        optim_state = torch.optim.Adam([self.p_x_x_1.weights,
                                        self.p_x_x_1.bias,self.p_x_x_1.Ch_covariance, self.mux0, self.sigma0],
                                       lr=self.learing_rate, weight_decay=1e-6)
        optim_obser = torch.optim.Adam(self.p_x_y_h.parameters(), lr=self.learing_rate, weight_decay=1e-6)
        corr_score = []
        mae_score = []
        for itr in range(self.EM_itr):

            # Variational E step with CAVI
            optim_obser.zero_grad()
            optim_state.zero_grad()
            predictions = self.p_x_y_h(y.clone())
            mu,sigma, mu_h, sigma_h = predictions
            prediction_process_out = [mu.clone(),sigma.clone()]
            # prediction_h = [mu_h.clone(), sigma_h.clone()]
            prediction_h = self.get_prediction_h(prediction_process_out)
            self.posterior = self.compute_posterior(prediction_process_out,prediction_h )
            self.posterior_smooth = self.compute_smoother(self.posterior)
            posterior_sample = self.MCMC_sampler(self.posterior_smooth)
            loss_prediction = -1 * self.elbo_prediction_process(posterior_sample, prediction_process_out,prediction_h)
            loss_state = -1 * self.elbo_state_process( posterior_sample)
            total_loss = (loss_state + loss_prediction)
            total_loss.backward(retain_graph=True)
            optim_state.step()
            optim_obser.step()

            elbo = - total_loss
            avg_elbos.append(elbo.detach().numpy())

            temp_corr_score, temp_mae_score = self.get_score_posterior(self.posterior,self.posterior_smooth, X,total_loss,itr)
            corr_score.append(temp_corr_score)
            mae_score.append(temp_mae_score)
        return  avg_elbos, self.posterior, self.posterior_smooth, corr_score, mae_score

    def variational_I(self, data, X):
            """
            Fit the model parameters via variational EM.
            """
            y = to_t(data)
            x = to_t(X)

            # Run CAVI
            self.create_state_process()
            avg_elbos = []
            posterior = None
            optim_state = torch.optim.Adam([self.p_x_x_1.weights,
                                            self.p_x_x_1.bias, self.p_x_x_1.Ch_covariance, self.sigma0, self.mux0],
                                           lr=self.learing_rate, weight_decay=1e-6)


            for itr in range(self.EM_itr):
                optim_state.zero_grad()
                loss_state = -1 * self.elbo_state_process(x)
                loss_state.backward(retain_graph=True)
                optim_state.step()
                elbo = - loss_state
                avg_elbos.append(elbo.detach().numpy())

            corr_score=[]
            mae_score=[]
            self.create_prediction_process()
            optim_obser = torch.optim.Adam(self.p_x_y_h.parameters(), lr=self.learing_rate, weight_decay=1e-6)
            for ii in range(self.epochs):
                    optim_obser.zero_grad()
                    predictions = self.p_x_y_h(y.clone())
                    mu, sigma, mu_h, sigma_h = predictions
                    prediction_process_out = [mu.clone(), sigma.clone()]
                    # prediction_h = [mu_h.clone(), sigma_h.clone()]
                    prediction_h = self.get_prediction_h(prediction_process_out)
                    loss_prediction = -1 * self.elbo_prediction_process(x, prediction_process_out, prediction_h)
                    loss_prediction.backward(retain_graph=True)
                    optim_obser.step()


            prediction_h = self.get_prediction_h(prediction_process_out)
            self.posterior = self.compute_posterior(prediction_process_out, prediction_h)
            self.posterior_smooth = self.compute_smoother(self.posterior)
            temp_corr_score, temp_mae_score = self.get_score_posterior(self.posterior, self.posterior_smooth, x, loss_prediction, 0)
            corr_score.append(temp_corr_score)
            mae_score.append(temp_mae_score)
            return avg_elbos, self.posterior,self.posterior , corr_score, mae_score

    def get_prediction_h(self, prediction_process_out):
        mu_o, sigma_o = prediction_process_out
        T = mu_o.shape[0]

        sigma_xx = torch.matmul(self.p_x_x_1.Ch_covariance, torch.transpose(self.p_x_x_1.Ch_covariance, 1, 0)).clone() + 1e-5
        W_x = self.p_x_x_1.get_weights(mu_o.clone(), T)
        b_x = self.p_x_x_1.bias

        mu_0 = self.mux0
        sigma_0 =(self.px0.scale)

        mu_o_h=torch.zeros_like(mu_o)
        sigma_o_h = torch.zeros((T,self.latent_dim,self.latent_dim))
        sigma_o_h[0] = torch.diag(sigma_0)
        mu_o_h[0]=mu_0
        for ii in range(1,T):
            sigma_o_h[ii]= (  (torch.inverse(torch.inverse(sigma_xx) +
                                            torch.inverse((sigma_o[ii-1])))))

            mu_o_h[ii] =  torch.matmul(mu_o[ii-1],W_x[ii])+b_x
        return [mu_o_h, sigma_o_h]

    def variational_em_v2(self, data, X):
        """
        Fit the model parameters via variational EM.
        """
        y = to_t(data)
        x = to_t(X)
        self.create_prediction_process()
        self.create_state_process()

        avg_elbos = []
        optim_state = torch.optim.Adam([self.p_x_x_1.weights,
                                        self.p_x_x_1.bias, self.p_x_x_1.Ch_covariance, self.mux0, self.sigma0],
                                       lr=self.learing_rate, weight_decay=1e-6)

        optim_obser = torch.optim.Adam(self.p_x_y_h.parameters(), lr=self.learing_rate, weight_decay=1e-6)
        corr_score=[]
        mae_score =[]
        for itr in range(self.EM_itr):

            optim_obser.zero_grad()
            predictions = self.p_x_y_h(y)
            mu, sigma, mu_h, sigma_h = predictions
            # prediction_h = [mu_h.clone(), sigma_h.clone()]
            prediction_process_out = [mu.clone(), sigma.clone()]
            prediction_h = self.get_prediction_h(prediction_process_out)
            self.posterior = self.compute_posterior(prediction_process_out, prediction_h)
            self.posterior_smooth = self.compute_smoother(self.posterior)
            posterior_sample = self.MCMC_sampler(self.posterior_smooth)
            loss_prediction = -1 * self.elbo_prediction_process(posterior_sample, prediction_process_out,prediction_h)
            loss_prediction.backward(retain_graph=True)
            optim_obser.step()
            # if itr >0:
            #     elbo = - (loss_state + loss_prediction)
            #     avg_elbos.append(elbo.detach().numpy())

            predictions = self.p_x_y_h(y)
            mu, sigma, mu_h, sigma_h = predictions
            # prediction_h = [mu_h.clone(), sigma_h.clone()]
            prediction_process_out = [mu.clone(), sigma.clone()]
            prediction_h = self.get_prediction_h(prediction_process_out)
            self.posterior = self.compute_posterior(prediction_process_out,prediction_h)
            self.posterior_smooth = self.compute_smoother(self.posterior)
            posterior_sample = self.MCMC_sampler(self.posterior_smooth)
            optim_state.zero_grad()
            loss_state = -1 * self.elbo_state_process(posterior_sample)
            loss_state.backward(retain_graph=True)
            optim_state.step()



            elbo = - (loss_state + loss_prediction)
            avg_elbos.append(elbo.detach().numpy())

            temp_corr_score, temp_mae_score = self.get_score_posterior(self.posterior, self.posterior_smooth, x, loss_state, itr)
            corr_score.append(temp_corr_score)
            mae_score.append(temp_mae_score)

        return avg_elbos, self.posterior, self.posterior_smooth, corr_score, mae_score

    def get_posterior(self, data,X):

        y = to_t(data)
        x = to_t(X)
        predictions = self.p_x_y_h(y)
        mu, sigma, mu_h, sigma_h = predictions
        prediction_process_out = [mu.clone(), sigma.clone()]
        # prediction_h = [mu_h.clone(), sigma_h.clone()]
        prediction_h = self.get_prediction_h(prediction_process_out)
        posterior = self.compute_posterior(prediction_process_out, prediction_h)

        posterior_smooth = self.compute_smoother(posterior)

        corr_score, mae_score = self.get_score_posterior(posterior, posterior_smooth, x, 0, 0)

        return posterior, posterior_smooth, corr_score, mae_score


