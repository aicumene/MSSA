"""AICumene Multichannel Singular Spectrum Analysis."""

from .hankel import block_hankel, diagonal_average_block_hankel
from .mssa import MSSA, MSSAResult
from .rank import RankEstimate, estimate_rank_bai_yin
from .diagnostics import participation_ratio, entropy_rank, stable_rank

__all__ = [
    "block_hankel",
    "diagonal_average_block_hankel",
    "MSSA",
    "MSSAResult",
    "RankEstimate",
    "estimate_rank_bai_yin",
    "participation_ratio",
    "entropy_rank",
    "stable_rank",
]

__version__ = "0.1.0"
