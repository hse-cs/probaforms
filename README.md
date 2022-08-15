# Welcome to probaforms

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`Probaforms` is a python library of conditional Generative Adversarial Networks, Normalizing Flows, Variational Autoencoders and other generative models for tabular data. All models have a sklearn-like interface to enable rapid use in a variety of science and engineering applications.

## Installation

```python
git clone https://github.com/HSE-LAMBDA/roerich.git
cd roerich
pip install -e .
```

or

```
poetry install
```

## Basic usage

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

## Thanks to all our contributors

<a href="https://github.com/HSE-LAMBDA/probaforms/graphs/contributors">
  <img src="https://contributors-img.web.app/image?repo=HSE-LAMBDA/probaforms" />
</a>
