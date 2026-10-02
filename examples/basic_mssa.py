import numpy as np
from aicumene_mssa import MSSA
from aicumene_mssa.rank import estimate_rank_bai_yin

fs = 100.0
t = np.arange(1000)/fs
x = np.vstack([
    np.sin(2*np.pi*8*t) + 0.5*np.sin(2*np.pi*17*t),
    0.8*np.sin(2*np.pi*8*t+0.2) + 0.4*np.sin(2*np.pi*17*t-0.3),
])

model = MSSA(window_length=60).fit(x)
rank = estimate_rank_bai_yin(model.hankel_, finite_sample_correction=0.0)
print("Estimated rank:", rank.rank)
print("Top singular values:", model.singular_values_[:10])
print("Reconstruction:", model.reconstruct(range(rank.rank)).shape)
