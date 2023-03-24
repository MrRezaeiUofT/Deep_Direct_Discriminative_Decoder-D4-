# Deep_Direct_Discriminative_Decoder-D4

The state-space models (SSMs) are widely utilized in the analysis of time-series data. SSMs rely on an explicit definition of the state and observation processes. Characterizing these processes is not always easy and becomes a modeling challenge when the dimension of observed data grows or the observed data distribution deviates from the normal distribution. Here, we propose a new formulation of SSM for high-dimensional observation processes. We call this solution the deep direct discriminative process (D4). The D4 brings deep neural networks' expressiveness and scalability to the SSM formulation letting us build a novel solution that efficiently estimates the underlying state processes through high-dimensional observation signal.%For the D4, we define the Bayesian filter solution and develop a training algorithm that finds the model-free parameters. We demonstrate the D4 solutions in simulated and real data such as Lorenz attractors, Langevin dynamics, random walk dynamics, and rat hippocampus spiking neural data and show that the D4's performance precedes traditional SSMs and RNNs. The D4 can be applied to a broader class of time-series data where the connection between high-dimensional observation and the underlying latent process is hard to characterize.

To facilitate the deployment of experiments, use this [Colab notebook](Run_D4_inColab.ipynb).

See the details in [ArXiv, 2022]( https://arxiv.org/pdf/2205.10947.pdf). Cite this paper [DOI](https://doi.org/10.1162/neco_a_01491).

[An example of the Lorenz attractor modeling.](Lorenz_attractor.png)
