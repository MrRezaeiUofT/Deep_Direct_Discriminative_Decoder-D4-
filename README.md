Here's an improved and polished version of your GitHub `README.md` file in Markdown format:

---

# Deep Direct Discriminative Decoder (D4)

State-Space Models (SSMs) are widely used for analyzing time-series data by modeling the underlying dynamics through explicit definitions of state and observation processes. However, in high-dimensional settings or when the observed data distribution deviates from Gaussianity, accurately characterizing these processes becomes increasingly difficult.

To address these limitations, we introduce the **Deep Direct Discriminative Decoder (D4)** — a novel formulation of SSMs designed for high-dimensional observations. D4 leverages the expressiveness and scalability of deep neural networks to learn efficient mappings from high-dimensional observations to latent state representations. Unlike traditional SSMs and RNNs, D4 directly estimates the latent dynamics without requiring explicit generative models for the observation process.

We define a Bayesian filtering solution for D4 and propose a training algorithm to learn model-free parameters from data. The effectiveness of D4 is demonstrated on both synthetic and real-world datasets, including:

- **Lorenz attractors**
- **Langevin dynamics**
- **Random walk processes**
- **Rat hippocampus spiking neural data**

Experimental results show that D4 consistently outperforms classical SSMs and RNN-based approaches in recovering the latent states from high-dimensional signals. Its flexibility makes it applicable to a wide range of time-series applications where the observation-to-latent mapping is complex and unknown.

---

## 📄 Paper

For more details, see our paper on [ArXiv (2022)](https://arxiv.org/pdf/2205.10947.pdf)  
Cite the work using the following [DOI](https://doi.org/10.1162/neco_a_01491):

```
@article{rezaei2023deep,
  title={Deep Direct Discriminative Decoder for High-Dimensional State-Space Models},
  author={Rezaei, Mohammad R. and Lankarany, Milad},
  journal={Neural Computation},
  volume={35},
  number={8},
  pages={1622--1658},
  year={2023},
  publisher={MIT Press}
}
```


