from typing import TYPE_CHECKING, List
from pydantic import BaseModel

class MFRecommenderConfig(BaseModel):
    global_sigma: float = 0.01
    local_sigma: float = 0.01
    n_factors: int = 20
    n_items: int = 1241
    n_users: int = 1128
    seed: int = 42
    do_kron: bool = True


