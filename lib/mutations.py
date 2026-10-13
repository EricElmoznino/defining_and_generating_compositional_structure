import numpy as np
from pymoo.core.mutation import Mutation


class IntegerRandomResetMutation(Mutation):
    def __init__(self, prob=None, **kwargs):
        # prob_var = per-gene probability; prob (per-individual) stays 1.0
        super().__init__(prob=1.0, prob_var=prob, **kwargs)

    def _do(self, problem, X, random_state=None, **kwargs):
        Xp = np.copy(X)
        prob_var = self.get_prob_var(problem, size=X.shape)
        mask = random_state.random(X.shape) < prob_var

        # Sample new ints within bounds
        new_vals = random_state.integers(problem.xl, problem.xu + 1, size=X.shape)

        Xp[mask] = new_vals[mask]
        return Xp
