import numpy as np
from aicumene_mssa import MSSA

def test_fit_shapes():
    fs = 100.0
    t = np.arange(400)/fs
    x = np.vstack([np.sin(2*np.pi*8*t), np.sin(2*np.pi*8*t+0.3)])
    m = MSSA(window_length=40).fit(x)
    assert m.hankel_.shape[0] == 80
    assert m.signal_subspace(2).shape == (80,2)

def test_full_reconstruction():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(2,100))
    m = MSSA(window_length=20).fit(x)
    rec = m.reconstruct(range(len(m.singular_values_)))
    assert np.allclose(rec, x, atol=1e-8)
