# Welcome to probaforms

[![PyPI version](https://badge.fury.io/py/probaforms.svg)](https://badge.fury.io/py/probaforms)
[![Tests](https://github.com/HSE-LAMBDA/probaforms/actions/workflows/tests.yml/badge.svg)](https://github.com/HSE-LAMBDA/probaforms/actions/workflows/tests.yml)
[![Docs](https://github.com/HSE-LAMBDA/probaforms/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/HSE-LAMBDA/probaforms/actions/workflows/pages/pages-build-deployment)
[![Downloads](https://static.pepy.tech/badge/probaforms)](https://pepy.tech/project/probaforms)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`Probaforms` is a python library of conditional Generative Adversarial Networks, Normalizing Flows, Variational Autoencoders and other generative models for tabular data. All models have a sklearn-like interface to enable rapid use in a variety of science and engineering applications.

## Implemented conditional models
- Variational Autoencoder (CVAE)
- Wasserstein GAN (WGAN)
- Real NVP

## Installation
```
pip install probaforms
```
or
```python
git clone https://github.com/HSE-LAMBDA/probaforms.git
cd probaforms
pip install -e .
```

or

```
poetry install
```

## Basic usage
(See more examples in the [documentation](https://hse-lambda.github.io/probaforms).)

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

- Home: [https://github.com/HSE-LAMBDA/probaforms](https://github.com/HSE-LAMBDA/probaforms)
- Documentation: [https://hse-lambda.github.io/probaforms](https://hse-lambda.github.io/probaforms)
- For any usage questions, suggestions and bugs use the [issue page](https://github.com/HSE-LAMBDA/probaforms/issues), please.

## Thanks to all our contributors

<a href="https://github.com/HSE-LAMBDA/probaforms/graphs/contributors">
  <img src="https://contributors-img.web.app/image?repo=HSE-LAMBDA/probaforms" />
</a>
