import os
import time
import argparse
import numpy as np
from pathlib import Path
from types import SimpleNamespace
from gdcm.data.load_data import FeaturesData
from gdcm.data.preprocess import preprocess_features
from gdcm.common.utils import load_a_dict, save_a_dict, print_the_evaluated_results

np.set_printoptions(suppress=True, precision=3, linewidth=140)


def args_parser(arguments):

    _pp = arguments.pp.lower()
    _run = arguments.run
    _data_name = arguments.data_name.lower()
    _algorithm_name = arguments.algorithm_name.lower()
    _n_clusters = arguments.n_clusters
    _n_repeats = arguments.n_repeats
    _n_init = arguments.n_init

    return _pp, _run, _data_name, _algorithm_name, _n_clusters, _n_repeats, _n_init


configs = {
    "results_path": Path("/home/soroosh/Programmes/GDCM/Results"),
    "figures_path": Path("/home/soroosh/Programmes/GDCM/Figures"),
    "params_path": Path("/home/soroosh/Programmes/GDCM/Params"),
    "data_path": Path("/home/soroosh/Programmes/GDCM/Datasets"),
}

configs = SimpleNamespace(**configs)

if not configs.results_path.exists():
    configs.results_path.mkdir()

if not configs.figures_path.exists():
    configs.figures_path.mkdir()

if not configs.params_path.exists():
    configs.params_path.mkdir()

if __name__ == "__main__":

    # all the string inputs will be converted to lower case.
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_name", type=str, default="IRIS",
        help="Dataset's name, e.g., IRIS, or Lawyers, or dd_fix_demo."
    )

    parser.add_argument(
        "--algorithm_name", type=str, default="km_clu",
        help="None case sensitive first letter abbreviated name of an estimator proceeds "
             "  with _clu e.g., K-Means clustering := km_clu."
             "Note: First letter of the methods' name should be used for abbreviation."
    )

    parser.add_argument(
        "--run", type=int, default=0,
        help="Run the model or load the saved"
             " weights and reproduce the results."
    )

    parser.add_argument(
        "--pp", type=str, default="mm",
        help="Data preprocessing method:"
             " MinMax/Z-Scoring/etc."
    )

    parser.add_argument(
        "--n_clusters", type=int, default=5,
        help="Number of clusters/classes/discrete target values."
    )

    parser.add_argument(
        "--n_repeats", type=int, default=10,
        help="Number of repeats of a data set or of a specific distribution"
    )

    parser.add_argument(
        "--n_init", type=int, default=10,
        help="Number of repeats with different seed initialization to select the best results on a data set."
    )

    args = parser.parse_args()

    pp, run, data_name, algorithm_name, n_clusters, n_repeats, n_init = args_parser(arguments=args)

    print(
        "configuration: \n",
        "  estimator:", algorithm_name, "\n",
        "  data_name:", data_name, "\n",
        "  pre-processing:", pp, "\n",
        "  run:", run, "\n",
    )

    # Adding some details for the sake of clarity in storing and visualization
    configs.run = run
    specifier = " -alg: " + algorithm_name + \
                " -data: " + data_name + \
                " -n_init: " + str(n_init)

    configs.specifier = specifier
    configs.data_name = data_name
    configs.n_repeats = n_repeats

    # to add the repeat numbers to the data_name variable for synthetic data
    if "n=" in data_name or "k=" in data_name or "v=" in data_name:
        synthetic_data = True
    else:
        synthetic_data = False

    if run == 1:
        results = {}
        for repeat in range(1, configs.n_repeats+1):

            repeat = str(repeat)
            results[repeat] = {}

            if algorithm_name.split("_")[-1].lower() == "clu":
                print(
                    "clustering features_only data: applying competitors on " + data_name + " repeat=" + repeat, "\n"
                )

                from gdcm.algorithms.clustering_methods_competitors import ClusteringEstimators

                if synthetic_data is True:
                    dire = "F/synthetic"
                    dn = data_name + "_" + repeat

                else:
                    dire = "F"
                    dn = data_name

                data_path = os.path.join(configs.data_path, dire)
                fd = FeaturesData(name=dn, path=data_path)

                x, xn, y_true = fd.get_dataset()
                results[repeat]['y_true'] = y_true

                x = preprocess_features(x=x, pp=pp)
                if xn.shape[0] != 0:
                    xn = preprocess_features(x=xn, pp=pp)
                n_clusters = len(np.unique(y_true))

                # instantiate and fit
                start = time.process_time()
                cu = ClusteringEstimators(
                    algorithm_name=algorithm_name,
                    n_clusters=n_clusters,
                    n_init=n_init
                )

                cu.instantiate_estimator_with_parameters()
                y_pred = cu.fit_estimator(x=x, y=y_true)
                end = time.process_time()

                # save results and logs
                results[repeat]['y_pred'] = y_pred
                results[repeat]['time'] = end-start
                results[repeat]['inertia'] = cu.inertia
                results[repeat]['data_scatter'] = cu.data_scatter

            else:
                assert False, "Ill-defined algorithm name!"

        # save results dict and configs
        save_a_dict(
            a_dict=results, name=configs.specifier, save_path=configs.results_path
        )

        save_a_dict(
            a_dict=configs, name=configs.specifier, save_path=configs.params_path
        )

        print_the_evaluated_results(results=results)

    elif run != 1:

        # load results dict and configs
        results = load_a_dict(
            name=configs.specifier, save_path=configs.results_path
        )

        configs = load_a_dict(
            name=configs.specifier, save_path=configs.params_path
        )

        print("configs \n", configs.specifier, "\n")

        print_the_evaluated_results(results=results)


