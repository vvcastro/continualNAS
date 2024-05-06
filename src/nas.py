import numpy as np
import subprocess
import shutil
import json
import os


from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.model.problem import Problem
from pymoo.optimize import minimize
from pymoo.factory import get_performance_indicator
from pymoo.algorithms.so_genetic_algorithm import GA
from pymoo.factory import get_algorithm, get_crossover, get_mutation

from surrogates.adaptative_switching import get_surrogate_predictor
from evaluator.evaluator import OFAEvaluator, get_net_info
from search_space.ofa import OFASearchSpace
import utils

# from utils import prepare_eval_folder, MySampling, BinaryCrossover, MyMutation
# from utils import get_correlation


def get_nadir_points(archive):
    """
    Computes the Nadir point (the bounded Pareto front) for all the
    architectures in the problem.
    """
    n_metrics = len(archive[0][1])
    nadir_point = [np.max([x[1][i] for x in archive]) for i in range(n_metrics)]
    return np.array(nadir_point)


class ContinualNAS:
    def __init__(self):
        self.search_space = OFASearchSpace(family="mobilenetv3")
        self.surrogate_predictor = "as"
        self.initial_samples = 100
        pass

    def sample_initial_set(self, n_iter):
        """
        Sample the initial architectures for the archive set.
        - Note: When `n_iter < 0` the problem transform to a random search
        problem
        """
        if n_iter < 1:
            return self.search_space.sample(n_samples=self.initial_samples)
        return self.search_space.initialise(n_samples=self.initial_samples)

    def search(self, n_iter):
        sampled_archs = self.sample_initial_set(n_iter)
        eval_metrics = self._evaluate(sampled_archs)

        # Here we initialise the archive set with all the metrics
        # from model evaluation
        online_archive = [info for info in zip(sampled_archs, eval_metrics)]

        # Compute the Nadir point for the hypervolume (non-dominated points )
        nadir_point = get_nadir_points(online_archive)

        for it in range(1, n_iter + 1):
            acc_predictor, a_top1_err_pred = self._fit_acc_predictor(online_archive)
            candidates, c_top1_err_pred = self._next(
                online_archive, acc_predictor, n_iter
            )

            c_top1_err, complexity = self._evaluate(candidates, it=it)

            # check for accuracy predictor's performance
            rmse, rho, tau = utils.get_correlation(
                np.vstack((a_top1_err_pred, c_top1_err_pred)),
                np.array([x[1] for x in online_archive] + c_top1_err),
            )

            # add to archive
            # Algo 1 line 15 / Fig. 3(e) in the paper
            for member in zip(candidates, c_top1_err, complexity):
                online_archive.append(member)

            # calculate hypervolume
            hv = self._calc_hv(
                nadir_point,
                np.column_stack(
                    [[x[1][i] for x in online_archive] for i in range(len(nadir_point))]
                ),
            )

            # print iteration-wise statistics
            print("Iter {}: hv = {:.2f}".format(it, hv))
            print(
                "fitting {}: RMSE = {:.4f}, Spearman's Rho = {:.4f}, Kendall’s Tau = {:.4f}".format(
                    self.predictor, rmse, rho, tau
                )
            )

    def _evaluate(self, archs: list, it):
        gen_dir = os.path.join(self.save_path, "iter_{}".format(it))

        subprocess.call("sh {}/run_bash.sh".format(gen_dir), shell=True)

        top1_err, complexity = [], []

        for i in range(len(archs)):
            try:
                stats = json.load(open(os.path.join(gen_dir, "net_{}.stats".format(i))))
            except FileNotFoundError:
                # just in case the subprocess evaluation failed
                stats = {"top1": 0, self.sec_obj: 1000}
                # makes the solution artificially bad so it won't survive
                # store this architecture to a separate in case we want to revisit after the search
                os.makedirs(os.path.join(self.save_path, "failed"), exist_ok=True)
                shutil.copy(
                    os.path.join(gen_dir, "net_{}.subnet".format(i)),
                    os.path.join(
                        self.save_path, "failed", "it_{}_net_{}".format(it, i)
                    ),
                )

            top1_err.append(100 - stats["top1"])
            complexity.append(stats[self.sec_obj])

        return top1_err, complexity

    def _fit_surrogate_predictor(self, archive, metric_idx=0):
        """
        Fits a surrogate model to predict the desired metric from the
        archive dataset
        @params
        - `archive`: list [ (architecture, [...metrics])]
        - `metric_idx`: refers to the metric to optimise, relative ordering from metrics.

        The default metric to train is `0=accuracy`.
        """
        inputs = np.array([self.search_space.encode(x[0]) for x in archive])
        targets = np.array([x[1][metric_idx] for x in archive])
        _predictor = get_surrogate_predictor(inputs, targets)
        return _predictor, _predictor.predict(inputs)

    def _next(self, archive, predictor, K):
        """
        Searches for the next K-candidates for high-fidelity evaluation of the
        lower level optimisation problem.

        > Correspond to lines [10...11] in the reference paper.
        """

        encoded_archive = [self.search_space.encode(x[0]) for x in archive]
        archive_metrics = np.column_stack(
            ([[x[1][i] for x in archive] for i in range(len(archive[0][1]))])
        )

        # 1. Get non-dominated architectures from archive
        front = NonDominatedSorting().do(
            archive_metrics,
            only_non_dominated_front=True,
        )
        encoded_front = np.array(encoded_archive)[front]

        # initialize the candidate finding optimization problem
        problem = AuxiliarySingleLevelProblem(
            self.search_space,
            predictor,
            self.sec_obj,
            {"n_classes": self.n_classes, "model_path": self.supernet_path},
        )

        # initiate a multi-objective solver to optimize the problem
        method = get_algorithm(
            "nsga2",
            pop_size=40,
            sampling=encoded_front,
            crossover=get_crossover("int_two_point", prob=0.9),
            mutation=get_mutation("int_pm", eta=1.0),
            eliminate_duplicates=True,
        )

        # kick-off the search
        res = minimize(
            problem,
            method,
            termination=("n_gen", 20),
            save_history=True,
            verbose=True,
        )

        # check for duplicates
        not_duplicate = np.logical_not(
            [
                (x in [x[0] for x in archive])
                for x in [self.search_space.decode(x) for x in res.pop.get("X")]
            ]
        )

        # the following lines corresponding to Algo 1 line 11 / Fig. 3(c)-(d) in the paper
        # form a subset selection problem to short list K from pop_size
        indices = self._subset_selection(
            res.pop[not_duplicate],
            archive_metrics[front, 1],
            K,
        )
        pop = res.pop[not_duplicate][indices]

        candidates = []
        for x in pop.get("X"):
            candidates.append(self.search_space.decode(x))

        # decode integer bit-string to config and also return predicted top1_err
        return candidates, predictor.predict(pop.get("X"))


class AuxiliarySingleLevelProblem(Problem):
    """The optimization problem for finding the next N candidate architectures"""

    def __init__(self, search_space, predictor, sec_obj="flops", supernet=None):
        super().__init__(n_var=46, n_obj=2, n_constr=0, type_var=np.int)

        self.ss = search_space
        self.predictor = predictor
        self.xl = np.zeros(self.n_var)
        self.xu = 2 * np.ones(self.n_var)
        self.xu[-1] = int(len(self.ss.resolution) - 1)
        self.sec_obj = sec_obj
        self.lut = {"cpu": "data/i7-8700K_lut.yaml"}

        # supernet engine for measuring complexity
        self.engine = OFAEvaluator(
            n_classes=supernet["n_classes"], model_path=supernet["model_path"]
        )

    def _evaluate(self, x, out, *args, **kwargs):
        f = np.full((x.shape[0], self.n_obj), np.nan)

        top1_err = self.predictor.predict(x)[:, 0]  # predicted top1 error

        for i, (_x, err) in enumerate(zip(x, top1_err)):
            config = self.ss.decode(_x)
            subnet, _ = self.engine.sample(
                {"ks": config["ks"], "e": config["e"], "d": config["d"]}
            )
            info = get_net_info(
                subnet,
                (3, config["r"], config["r"]),
                measure_latency=self.sec_obj,
                print_info=False,
                clean=True,
                lut=self.lut,
            )
            f[i, 0] = err
            f[i, 1] = info[self.sec_obj]

        out["F"] = f
