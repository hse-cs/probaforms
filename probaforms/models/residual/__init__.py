'''
"Residual Flows for Invertible Generative Modeling" (arxiv.org/abs/1906.02735)
Realization of (un-)conditional Residual Flow for tabular data
Code based on github.com/tatsy/normalizing-flows-pytorch
Conditioning idea: "Learning Likelihoods with Conditional Normalizing Flows" arxiv.org/abs/1912.00042)
'''

from .model import ResidualConditional
from .model import ResidualUnconditional
from .model import ResidualFlowModel
