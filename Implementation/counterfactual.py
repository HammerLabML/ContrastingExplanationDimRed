from abc import ABC, abstractmethod


class CounterfactualExplanation():
    def __init__(self, **kwds):
        super().__init__(**kwds)
    
    def compute_diverse_explanations(self, x_orig, y_cf, n_explanations=3):
        X_cf = []
        
        # First counterfactual can be computed using the standard approach
        X_cf.append(self.compute_explanation(x_orig, y_cf))

        # Compute more & diverse counterfactuals
        for _ in range(n_explanations - 1):
            x_cf = self._compute_diverse_counterfactual(x_orig, y_cf, X_cf)
            if x_cf is None:
                break

            X_cf.append(x_cf)

        return X_cf

    @abstractmethod
    def _compute_diverse_counterfactual(self, x_orig, y_cf, X_cf):
        raise NotImplementedError()

    @abstractmethod
    def compute_explanation(self, x_orig, y_cf):
        raise NotImplementedError()
