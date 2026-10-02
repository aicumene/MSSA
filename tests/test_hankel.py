import numpy as np
from aicumene_mssa import block_hankel, diagonal_average_block_hankel

def test_block_hankel_order():
    x = np.arange(10, dtype=float)[None, :]
    H = block_hankel(x, window_length=3)
    assert H.shape == (3, 8)
    assert np.allclose(H[:,0], [2,1,0])
    assert np.allclose(H[:,-1], [9,8,7])

def test_hankel_roundtrip():
    x = np.vstack([np.arange(20,dtype=float), np.arange(20,dtype=float)**2])
    H = block_hankel(x, window_length=5)
    rec = diagonal_average_block_hankel(H, 2, 20, 5)
    assert np.allclose(rec, x)
