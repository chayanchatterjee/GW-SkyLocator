# CBC-SkyNet
This repository contains code for the paper: "Rapid localization of gravitational wave sources from compact binary coalescences using deep learning" by Chayan Chatterjee, Linqing Wen and Damon Beveridge (under preparation). 

We introduce CBC-SkyNet (Compact Binary Coalescence - Sky Localization Neural Network), the first deep learing model to predict Right Ascension (RA) and Declination (Dec) posterior distributions from all kinds of CBC sources - binary black holes, binary neutron stars and neutron star - black hole binary, in milli-second latency, using gravitational wave (GW) data. We use a [Normalizing Flow](https://arxiv.org/abs/1505.05770) model, in partcular, a [Masked Autoregressive Flow](https://arxiv.org/abs/1705.07057) (MAF), trained on the complex signal-to-noise ratio (SNR) time series data, obtained from matched filtering of GW strains with optimal Numerical Relativity templates. 
Normalizing Flow models learn a mapping between a simple base distribution (like a Gaussian) to a more complex distribution. In our case, we train a MAF to map samples drawn from a multivariate Gaussian (our base distribution) to the more complex posterior distribution of the RA and Dec parameters. The MAF uses the [MADE](https://arxiv.org/abs/1502.03509) architecture to learn the transformation parameters that defines the mapping between the two parameter spaces. Since the RA and Dec posteriors are conditional probability distributions, we condition our MAF network on features extracted from real and imaginary parts of the SNR time series, using two identical 1-D [ResNet-34](https://arxiv.org/abs/1512.03385) networks. The diagram of the network architecture is shown [below](Picture3.png):




