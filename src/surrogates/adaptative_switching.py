from src.surrogates.base_methods import get_base_predictor
import utils

import numpy as np


def get_surrogate_predictor(name, inputs, targets):
    """Wrapper to get a clean instance of the model"""
    if name != "as" or name != "adaptive switching":
        predictor = get_base_predictor(name, inputs, targets)
        return predictor
    predictor = AdaptiveSwitching()
    predictor.fit(inputs, targets)
    return predictor


class AdaptiveSwitching:
    """ensemble surrogate model"""

    def __init__(self, n_fold=10, model_pool=["rbf", "gp", "carts", "mlp"]):
        self.name = "adaptive switching"
        self.model_pool = model_pool
        self.n_fold = n_fold

        self.model = None

    def fit(self, inputs, targets):
        """Selects a bets predictor with cross-validation."""
        # Sanity check for the predict
        test_msg = "# of training samples have to be > # of dimensions"
        assert len(inputs) > len(inputs[0]), test_msg

        # Select the best predictor
        _best_model = self._n_fold_validation(inputs, targets, n=self.n_fold)
        self.model = get_base_predictor(_best_model, inputs, targets)

    def _n_fold_validation(self, train_data, train_target, n=10):
        n_samples = len(train_data)
        perm = np.random.permutation(n_samples)

        kendall_tau = np.full((n, len(self.model_pool)), np.nan)

        for i, tst_split in enumerate(np.array_split(perm, n)):
            trn_split = np.setdiff1d(perm, tst_split, assume_unique=True)

            # loop over all considered surrogate model in pool
            for j, model in enumerate(self.model_pool):
                acc_predictor = get_base_predictor(
                    model,
                    train_data[trn_split],
                    train_target[trn_split],
                )
                rmse, rho, tau = utils.get_correlation(
                    acc_predictor.predict(train_data[tst_split]),
                    train_target[tst_split],
                )
                kendall_tau[i, j] = tau

        for j, model in enumerate(self.model_pool):
            print("model = {}, tau = {}".format(model, np.mean(kendall_tau, axis=0)[j]))

        # Index to select
        tau_metric = np.mean(kendall_tau, axis=0) - np.std(kendall_tau, axis=0)
        best_model = self.model_pool[int(np.argmax(tau_metric))]
        print(f"winner model = {best_model}, tau = {tau_metric}")
        return best_model

    def predict(self, test_data):
        return self.model.predict(test_data)
