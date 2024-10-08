# Welcome to probaforms

[![PyPI version](https://badge.fury.io/py/probaforms.svg)](https://badge.fury.io/py/probaforms)
[![Tests](https://github.com/HSE-LAMBDA/probaforms/actions/workflows/tests.yml/badge.svg)](https://github.com/HSE-LAMBDA/probaforms/actions/workflows/tests.yml)
[![Docs](https://github.com/HSE-LAMBDA/probaforms/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/HSE-LAMBDA/probaforms/actions/workflows/pages/pages-build-deployment)
[![Downloads](https://static.pepy.tech/badge/probaforms)](https://pepy.tech/project/probaforms)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`Probaforms` is a python library of conditional Generative Adversarial Networks, Normalizing Flows, Variational Autoencoders and other generative models for tabular data. All models have a sklearn-like interface to enable rapid use in a variety of science and engineering applications.

## Implemented conditional models

[//]: # (Use Vancouver reference style below)

|     **Model**     |     **Type**     | **Paper**                                                                                                                                                                                             | 
|:-----------------:|:----------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ConditionalNormal |       MDN        | Bishop CM. [Mixture density networks](https://publications.aston.ac.uk/id/eprint/373/1/NCRG_94_004.pdf). 1994.                                                                                                                                                        |
|       CVAE        |       VAE        | Kingma DP, Welling M. [Auto-encoding variational bayes](https://openreview.net/forum?id=33X9fd2-9FyZd). [arXiv:1312.6114](https://arxiv.org/abs/1312.6114). ICLR 2014.                                |
|  ConditionalWGAN  |       GAN        | Arjovsky M, Chintala S, Bottou L. [Wasserstein generative adversarial networks](https://proceedings.mlr.press/v70/arjovsky17a.html). [arXiv:1701.07875](https://arxiv.org/abs/1701.07875). ICML 2017. |
|      RealNVP      | Normalizing Flow | Dinh L, Sohl-Dickstein J, Bengio S. [Density estimation using real nvp](https://openreview.net/forum?id=HkpbnH9lx).  [arXiv:1605.08803](https://arxiv.org/abs/1605.08803). ICLR 2017.                 |


## Installation
```
pip install probaforms
```
or
```python
git clone https://github.com/hse-cs/probaforms
cd probaforms
pip install -e .
```

or

```
poetry install
```

## Basic usage
(See more examples in the [documentation](https://hse-cs.github.io/probaforms).)

The following code snippet generates a noisy synthetic data, fits a conditional generative model, sample new objects, and displays the results.

```python
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from probaforms.models import RealNVP

# generate sample X with conditions C
X, y = make_moons(n_samples=1000, noise=0.1)
C = y.reshape(-1, 1)

# fit nomalizing flow model
model = RealNVP(lr=0.01, n_epochs=100)
model.fit(X, C)

# sample new objects
X_gen = model.sample(C)

# display the results
plt.scatter(X_gen[y==0, 0], X_gen[y==0, 1])
plt.scatter(X_gen[y==1, 0], X_gen[y==1, 1])
plt.show()
```

## Support

- Home: [https://github.com/hse-cs/probaforms](https://github.com/hse-cs/probaforms)
- Documentation: [https://hse-cs.github.io/probaforms](https://hse-cs.github.io/probaforms)
- For any usage questions, suggestions and bugs use the [issue page](https://github.com/hse-cs/probaforms/issues), please.

## Thanks to all our contributors

<a href="https://github.com/HSE-LAMBDA/probaforms/graphs/contributors">
  <img src="https://contributors-img.web.app/image?repo=HSE-LAMBDA/probaforms" />
</a>
