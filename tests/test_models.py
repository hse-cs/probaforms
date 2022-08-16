import numpy as np
import pytest

from probaforms.models.interfaces import GenModel

def subclasses(cls):
    return set(cls.__subclasses__()).union(s for c in cls.__subclasses__() for s in subclasses(c))


@pytest.mark.parametrize("model", subclasses(GenModel))
def test_with_conditions(model):
    n = 100
    X = np.random.normal(size=(n, 5))
    C = np.random.normal(size=(n, 3))
    gen = model()
    gen.fit(X, C)
    X_gen = gen.sample(C)
    assert X_gen.shape == X.shape


@pytest.mark.parametrize("model", subclasses(GenModel))
def test_without_conditions(model):
    n = 100
    X = np.random.normal(size=(n, 5))
    gen = model()
    gen.fit(X, C=None)
    X_gen = gen.sample(C=n)
    assert X_gen.shape == X.shape
