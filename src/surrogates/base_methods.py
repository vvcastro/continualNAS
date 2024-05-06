from surrogates.base_models._carts import CART
from surrogates.base_models._rbf import RBF
from surrogates.base_models._mlp import MLP
from surrogates.base_models._gp import GP


def get_base_predictor(model, inputs, targets):
    test_msg = "# of training samples have to be > # of dimensions"
    assert len(inputs) > len(inputs[0]), test_msg

    if model == "rbf":
        acc_predictor = RBF()
        acc_predictor.fit(inputs, targets)

    elif model == "carts":
        acc_predictor = CART(n_tree=5000)
        acc_predictor.fit(inputs, targets)

    elif model == "gp":
        acc_predictor = GP()
        acc_predictor.fit(inputs, targets)

    elif model == "mlp":
        acc_predictor = MLP(n_feature=inputs.shape[1])
        acc_predictor.fit(x=inputs, y=targets)

    else:
        raise NotImplementedError

    return acc_predictor
