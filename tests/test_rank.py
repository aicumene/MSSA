import numpy as np
from aicumene_mssa import block_hankel
from aicumene_mssa.rank import estimate_rank_bai_yin

def test_rank_positive_for_clean_oscillator():
    fs = 100.0
    t = np.arange(400)/fs
    x = np.sin(2*np.pi*10*t)[None,:]
    H = block_hankel(x, window_length=40)
    est = estimate_rank_bai_yin(H, finite_sample_correction=0.0)
    assert est.rank >= 2
