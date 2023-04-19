import jax
import numpy as np
import jax.numpy as jnp
from copy import deepcopy

from jax.config import config
config.update("jax_enable_x64", True)

from sklearn.metrics import adjusted_rand_score,\
    normalized_mutual_info_score


class GDCMf:

    """Computes various Gradient Descent Clustering Methods for Features (GDCMf).

    Parameters
    ----------

    n_clusters : int, default=10
        The number of clusters to form as well as the number of
        centroids to generate.

    p : float, default=2
        P value in Minkowski distance (metric), otherwise it is ignored.

    init : {'k-means++', 'random', 'user'}, string, default='random'
        Method for initialization:
        'k-means++' : selects initial cluster centers for k-mean
        clustering in a smart way to speed up convergence.
        'random': choose `n_clusters` observations (rows) at random from data
        for the initial centroids.
        'user': user-defined centroid indices for initial centroids.

    n_init : int, default=10
        Number of time the RGDC algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of inertia.

    max_iter : int, default=10
        Maximum number of iterations of a GDC algorithm for a single run.

    tau : float, default=1e-4
        Tolerance threshold of the data scatter (or the L1-norm of the gradient vectors).
        Since in our experiments, we could not find any meaningful pattern or
         stopping value for this parameter,
        thus we excluded from the methods' description in our paper.
        However, for consistency issues we preserve it in our implementation
        and later version we will remove it.
        The default value is set to a very small number to cancel the impact of this parameter.

    mu_1 : float, default=9e-1,
          Exponential decay rate for the first moment estimates in Adam or decay rate for Nestrov GDC

    mu_2 : float, default=999e-3,
          Exponential decay rate for the second moment estimates (i.e. squared gradients) in Adam.

    step_size : float, default=1e-3
        Gradient Descent step size, also known as learning rate to update the cluster centers.

    centroids_idx :  array of shape (n_clusters, 1), default = None
              A 1-D array containing user-defined seed indices to initiate centroids

    verbose : int, default=0
        Verbosity mode.

    update_rule = str, default="agdc",
        one of the three following  cases are supported:
            1)  vgdc: applies vanilla gdc algorithm (VGDC);
            2) ngdc : applies gdc with Nestrov momentum algorithm (NGDC);
            3) agdc : applies gdc with Adam algorithm (Adam GDC).

    batch_size = int, default=1,
        the size of the batch to apply the update rule.
        Fixing the batch size equal to one should be considered one of the main contribution of the proposed methods.

    range_len : int, default=None
        An Integer to form a two-element list, window, where the first and second elements, in respect,
         are the iteration numbers of the beginning and the end of the window to compute the average and
         the standard deviation of the gradient values and semi-positive definiteness of hessian matrix.
        Since in our experiments, we could not find any meaningful pattern or stopping value for this parameter,
        thus we excluded from the methods' description in our paper.
        However, for consistency issues we preserve it in our implementation and later version we will remove it.

    Attributes
    ----------

    """

    def __init__(
            self,
            n_clusters=10,
            *,
            p=2.,
            tau=1e-4,
            n_init=10,
            verbose=0,
            mu_1=45e-2,
            mu_2=95e-2,
            range_len=5,
            max_iter=100,
            init="random",
            batch_size=1,
            step_size=1e-2,
            centroids_idx=None,
            update_rule="vanilla",

    ):
        self.p = p
        self.tau = tau
        self.mu_1 = mu_1
        self.mu_2 = mu_2
        self.init = init
        self.n_init = n_init
        self.verbose = verbose
        self.max_iter = max_iter
        self.step_size = step_size
        self.range_len = range_len
        self.batch_size = batch_size
        self.n_clusters = n_clusters
        self.update_rule = update_rule
        self.centroids_idx = centroids_idx

        # Variable initialization
        self.moment_1 = []
        self.moment_2 = []
        self.ari = -jnp.inf
        self.nmi = -jnp.inf
        self.inertia = jnp.inf
        self.best_ari = -jnp.inf
        self.best_nmi = -jnp.inf
        self.data_scatter = jnp.inf
        self.best_inertia = jnp.inf
        self.batches = None
        self.clusters = None
        self.gdc_adam = False
        self.centroids = None
        self.stop_type = None
        self.gdc_vanilla = False
        self.gdc_nestrov = False
        self.aris_history = None
        self.nmis_history = None
        self.grads_history = None
        self.best_clusters = None
        self.grads_v_history = None
        self.inertias_history = None
        self.hessians_history = None
        self.best_centroids_idx = None
        self.best_iter_aris_history = None
        self.best_iter_nmis_history = None
        self.best_iter_hessians_history = None
        self.ari_or_inertia_ave_history = None
        self.best_iter_inertias_history = None
        self.best_iter_gradients_history = None

    @staticmethod
    def minkowski_fn(data_point, centroid, p):
        """Computes p-valued Minkowski distance between two 1-D vectors.
        datapoint : array of shape (1, n_features) instance to cluster.
        centers: array of shape (1, n_features) clusters centers.
        """

        if centroid.ndim > 1:
            raise ValueError("Input vector should be 1-D.")
        if p < 1:
            raise ValueError("p must be at least 1")

        if data_point.ndim > 1:
            return (jnp.sum((jnp.abs(data_point-centroid))**p, axis=1))**1/p

        else:
            return (jnp.sum((jnp.abs(data_point-centroid))**p))**1/p

    @staticmethod
    def cosine_fn(data_point, centroid, p=None):
        """Computes cosine distance between two 1-D vectors.
        datapoint : array of shape (1, n_features) instance to cluster.
        centers : array of shape (1, n_features) clusters centers.
        P value is added due to consistencies issues.
        """
        if data_point.ndim > 1:
            raise ValueError("Input vector should be 1-D.")
        if centroid.ndim > 1:
            raise ValueError("Input vector should be 1-D.")

        return 1 - jnp.divide(
            (jnp.inner(data_point, centroid) + 1e-10),
            (jnp.sqrt(sum(data_point**2)+1e-10) * jnp.sqrt(sum(centroid**2)+1e-10))
        )

    @staticmethod
    def is_positive_semidefinite(m):
        return jax.tree_leaves(jnp.all(jnp.linalg.eigvals(m) >= 0))

    @staticmethod
    def is_approximately_zero(v):
        return None

    @staticmethod
    def _get_subkey():
        seed = np.random.randint(low=0, high=1e10, size=1)[0]
        key = jax.random.PRNGKey(seed)
        _, subkey = jax.random.split(key)
        return subkey

    @staticmethod
    def compute_data_scatter(m):
        return jnp.sum(jnp.power(m, 2))

    @staticmethod
    def get_batches(n_samples, batch_size):
        """ returns a list of lists, containing the shuffled indexes of the entire data points,
            split into approximately equal chunks."""

        samples_indexed = np.arange(0, n_samples)
        np.random.shuffle(samples_indexed)

        remainder = n_samples % batch_size
        if remainder != 0:
            n_splits = (n_samples // batch_size) + 1
        else:
            n_splits = n_samples // batch_size

        batches = [
            samples_indexed[batch_size * i:batch_size * (i + 1)] for i in range(n_splits)
        ]
        return batches

    def compute_inertia(self, m, ):
        return sum(
            [jnp.sum(
                jnp.power(m[jnp.where(self.clusters == k)] - self.centroids[k], 2),
            ) for k in range(self.n_clusters)
            ]
        )

    def kmeans_plus_plus(self, x, distance_fn,):
        n_samples = x.shape[0]
        subkey = self._get_subkey()
        self.centroids_idx = jax.random.randint(
            subkey, minval=0, maxval=n_samples,
            shape=(1,)
        )
        for k in range(self.n_clusters-1):
            d_x = jnp.zeros((n_samples,), dtype=float)  # distances matrix
            for i in range(n_samples):
                if i not in self.centroids_idx:
                    d_x = d_x.at[i].set(
                        distance_fn(data_point=x[i], centroid=x[self.centroids_idx[k], :], p=self.p)
                    )
            next_center = jnp.argmax(d_x)
            self.centroids_idx = jnp.append(self.centroids_idx, next_center)

        return self.centroids_idx

    def _initiate_centroids_idx(self, x, distance_fn):
        """Generates array of indices to initiate centroids."""

        if self.init.lower() == "k-means++":
            print(
                "\n Centroids are initialized using K-Means++. \n"
            )
            self.centroids_idx = self.kmeans_plus_plus(x=x, distance_fn=distance_fn,)

        elif self.init.lower() == "random":
            print(
                "\n Centroids are initialized randomly. \n"
            )
            subkey = self._get_subkey()
            self.centroids_idx = jax.random.randint(
                subkey, minval=0, maxval=x.shape[0],
                shape=(self.n_clusters,)
            )
        elif self.init.lower() == "user":
            print(
                "\n User defined centroids indices are being applied.\n"
            )
            if isinstance(self.centroids_idx, (jax.numpy.ndarray,)):
                self.centroids_idx = jax.numpy.asarray(self.centroids_idx)
        else:
            assert False, "\n Ill-defined centroids initialization method."

        return self.centroids_idx

    def _initiate_clusters(self, n_samples):
        """ Generates array of shape (n_samples * 1) to store cluster recovery results.
        ---
        parameter
        n_samples : int , default = None
            Indicates the number of samples, i.e., number of rows of a feature data matrix or
             number of rows of an adjacency matrix.
        """

        subkey = self._get_subkey()
        self.clusters = jax.random.randint(
            subkey, minval=0, maxval=2*self.n_clusters,
            shape=(n_samples,)
        )

        return self.clusters

    def fit(self, x, distance_fn, y=None,):

        """Compute Various Gradient Descent Clustering Algorithm --with various update rules.
            Supported updates rules and algorithms:
                1) vanilla GDC (VGDC);
                2) Nestrov GDC (NGDC);
                3) Adam GDC (AGDC).

            Note:
             batch_size=1 implements stochastic gradient descent;
             batch_size=n_samples  implements batch gradient descent;
             1 < batch_size < n_samples  implements mini-batch gradient descent.

        Parameters
        ----------
        x : features array of shape (n_samples, n_features) or similarity data of shape
            (n_samples, n_samples) to cluster.
        distance_fn : callable to compute the distance
            between data points and cluster centroids.
        y :  array of shape (n_samples, 1), default = None
            Represents the ground truth.
        """

        if self.update_rule == "vgdc":
            self.gdc_vanilla = True
        elif self.update_rule == "ngdc":
            self.gdc_nestrov = True
        elif self.update_rule == "agdc":
            self.gdc_adam = True
        print("update rule: \n", self.gdc_vanilla , self.gdc_nestrov, self.gdc_adam)

        if not isinstance(x, (jax.numpy.ndarray,)):
            x = jnp.asarray(x)
        if y is not None:
            if not isinstance(y, (jax.numpy.ndarray,)):
                y = jnp.asarray(y)
            if len(y.shape) > 1:
                y = y.reval()

        search_for_best_nmi = True

        # Repeat the computations with different
        #   seeds initialization to select the best results.
        for _ in range(self.n_init):
            # Initialization:
            t = 0  # timestamp
            # Clusters and centroids
            n_samples = x.shape[0]
            n_features = x.shape[1]
            self.clusters = self._initiate_clusters(n_samples=n_samples)
            idx = self._initiate_centroids_idx(x=x, distance_fn=distance_fn)
            self.centroids = x[idx, :] + 1e-6  # white noise to prevent zero-gradient if 1st centroid = 1st data point
            self.centroids = self.centroids + 1e-6
            # Create batches
            self.batches = self.get_batches(n_samples=n_samples, batch_size=self.batch_size)
            # y = jnp.asarray([y[bb] for b in self.batches for bb in b])
            # Zero initialization of the moments vectors
            self.moment_1 = jnp.zeros(n_features)
            self.moment_2 = jnp.zeros(n_features)

            # Compute the data scatter
            self.data_scatter = self.compute_data_scatter(m=x)

            grads_history, grads_v_history, aris_history, nmis_history, inertias_history = [], [], [], [], []

            self.stop_type = None
            converged_in_data_iteration = False

            print(
                " Centroids_idx:", idx,
                f"\n Best ARI: {self.best_ari:.3f}"
                f" Best NMI: {self.best_nmi:.3f}"
                f" Data Scatter: {self.data_scatter:.3f} "
                f" Best Inertia: {self.best_inertia: .3f} "
            )

            for itr in range(1, self.max_iter+1):

                if converged_in_data_iteration:
                    break

                tmp_grads, tmp_grads_v, tmp_ari, tmp_nmi, tmp_inertia = [], [], [], [], []

                for batch in self.batches:
                    t += 1
                    # For-loop in the batch:
                    # If batch-size is larger than 2048, i.e, 256 bytes,
                    # use for loop since it MIGHT be faster; otherwise, use vectorization for smaller batch size.
                    # the threshold (1024) should be revised later.
                    # Iterating over each data point in a batch:
                    if batch.shape[0] >= 2048:
                        for i in batch:
                            # Cluster assignment:
                            data_point = x[i, :]  # .reshape(1, -1) # << For MLFN this section should be modified
                            # distances of size (1*K)
                            distances = jnp.asarray([
                                distance_fn(
                                    data_point=data_point, centroid=centroid, p=self.p) for centroid in self.centroids
                            ])
                            # update clusters per data point in batch
                            kk = jnp.argmin(distances, axis=0)  # << MLFN
                            self.clusters = self.clusters.at[i].set(kk)

                    # Batch computation
                    else:
                        batch_data = x[batch, :]
                        # distances of size N*K
                        distances = jnp.asarray([
                            distance_fn(
                                data_point=batch_data, centroid=centroid, p=self.p) for centroid in self.centroids
                        ])
                        batch_clusters = distances.argmin(axis=0)

                        # Update clusters in one-go (vectorized),
                        for i in batch:
                            self.clusters = self.clusters.at[i].set(batch_clusters[i])

                    previous_clusters = deepcopy(self.clusters)

                    # Compute gradients of the distance function w.r.t all centroids of the batch to update centroids
                    batch_data = x[batch, :]
                    batch_clusters = self.clusters[batch]

                    for k in range(self.n_clusters):
                        if len(jnp.where(batch_clusters == k)[0]) != 0:
                            to_update = True
                            mean_batch = jnp.mean(batch_data[jnp.where(batch_clusters == k)], axis=0)
                        else:
                            to_update = False
                            mean_batch = jnp.zeros(n_features)

                        if to_update:
                            if self.gdc_nestrov:
                                # print("Nestrov Momentum GDC")
                                previous_moment_1 = deepcopy(self.moment_1)  # momentum
                                mean_batch = mean_batch + self.mu_1 * previous_moment_1
                                grads = jax.jacfwd(distance_fn, argnums=(1,))(mean_batch, self.centroids[k, :], self.p)
                                grads = jax.tree_leaves(grads)[0]
                                self.moment_1 = (self.mu_1*previous_moment_1) - (self.step_size*grads)
                                updated_centroid = self.centroids[k, :] + self.moment_1
                                self.centroids = self.centroids.at[k].set(updated_centroid)
                            else:
                                print("Unsupported algorithm or update rule!")

                            if self.verbose > 3:
                                print("grads: \n", grads, "\n", )

                # Per iteration
                if y is not None:
                    self.ari = adjusted_rand_score(y, self.clusters)
                    self.nmi = normalized_mutual_info_score(y, self.clusters)
                    self.inertia = self.compute_inertia(m=x)
                    tmp_ari.append(self.ari)
                    tmp_nmi.append(self.nmi)
                    tmp_inertia.append(self.inertia)
                else:
                    self.ari = -jnp.inf
                    self.nmi = -jnp.inf
                    self.inertia = self.compute_inertia(m=x, )
                    tmp_ari.append(self.ari)
                    tmp_nmi.append(self.nmi)
                    tmp_inertia.append(self.inertia)

                # Convergence check:
                # First Order Necessary Condition (FONC)
                l1_norm_grads = jnp.sum(jnp.abs(grads))
                tmp_grads.append(l1_norm_grads)
                tmp_grads_v.append(grads)

                tmp_grads = jnp.asarray(tmp_grads)
                grads_history.append(tmp_grads.mean())
                tmp_grads_v = jnp.asarray(tmp_grads_v)
                grads_v_history.append(tmp_grads_v.mean(axis=0))
                tmp_ari = jnp.asarray(tmp_ari)
                aris_history.append(tmp_ari.mean())
                tmp_nmi = jnp.asarray(tmp_nmi)
                nmis_history.append(tmp_nmi.mean())
                tmp_nmi = jnp.asarray(tmp_nmi)
                tmp_inertia = jnp.asarray(tmp_inertia)
                inertias_history.append(tmp_inertia.mean())

                if self.verbose >= 2:
                    if y is not None and self.verbose != 0:
                        print(
                            f" N_itr = {itr} ARI = {self.ari:.3f} NMI = {self.nmi:.3f}"
                            f" Inertia = {self.inertia:.3f}"
                            f" L1-norm of grads = {tmp_grads.mean():.3f}"
                            f" gradient v = {tmp_grads_v.mean(axis=0)}"
                        )
                    elif y is None and self.verbose != 0:
                        print(
                            f" Inertia = {self.inertia:.3f}"
                            f" L1-norm of grads = {tmp_grads.mean():.3f}"
                            f" gradient v = {tmp_grads_v.mean(axis=0)}"
                        )

                    # Per iteration clusters' inertia convergence condition:
                    if self.compute_inertia(m=x) <= (self.tau*self.data_scatter)/100:
                        if self.verbose != 0:
                            print(
                                f"A stationary point has been found in the per data point convergence check,"
                                f" i.e., clusters' inertia is lower than the threshold {self.tau} % "
                                f"of data scatter ={self.data_scatter}. \n"
                            )
                            if y is not None and self.verbose != 0:
                                self.ari = adjusted_rand_score(y, self.clusters)
                                self.nmi = normalized_mutual_info_score(y, self.clusters)
                                self.inertia = self.compute_inertia(m=x)
                                print(
                                    f" N_itr = {itr} ARI = {self.ari:.3f} NMI = {self.nmi:.3f}"
                                    f" Inertia = {self.inertia:.3f}"
                                    f" L1-norm of grads = {tmp_grads.mean():.3f}"
                                    f" gradient v = {tmp_grads_v.mean(axis=0)}"
                                )
                            elif y is None and self.verbose != 0:
                                self.inertia = self.compute_inertia(m=x)
                                print(
                                    f" Inertia = {self.inertia:.3f}"
                                    f" L1-norm of grads = {tmp_grads.mean():.3f}"
                                    f" gradient v = {tmp_grads_v.mean(axis=0)}"
                                )
                        if jnp.all(previous_clusters == self.clusters):
                            converged_in_data_iteration = True
                            self.stop_type = "cc_per_iter"
                            print(
                                "\n Converged by two consecutive cluster recovering results "
                                "in the per data point convergence check \n",
                            )
                            break

                if y is not None and self.verbose != 0:
                    print(
                        f"N_iter = {itr}"
                        f" ARI = {self.ari:.3f}"
                        f" NMI = {self.nmi:.3f} "
                        f" Inertia = {self.inertia:.3f}\n"
                        f" Inertia: Ave. = {tmp_inertia.mean():.3f} Std. = {tmp_inertia.std():.3f} \n"
                        f" L-1 norm of grads: Ave. = {tmp_grads.mean():.3f} Std. = {tmp_grads.std():.3f} \n"
                        "\n"
                    )
                elif y is None and self.verbose != 0:
                    f"N_iter = {itr} Inertia = {self.inertia:.3f}\n"
                    f" Inertia: Ave. = {tmp_inertia.mean():.3f} Std. = {tmp_inertia.std():.3f} \n"
                    f" L-1 norm of grads: Ave. = {tmp_grads.mean():.3f} Std. = {tmp_grads.std():.3f} \n"
                    "\n"
                if self.verbose >= 3:
                    print(
                        f"N_iter = {itr}: \n"
                        "Inertia History for all previous iterations: \n",
                        inertias_history, "\n",
                        "ARI History for all previous iterations: \n",
                        aris_history, "\n",
                        "NMI History for all previous iterations: \n",
                        nmis_history, "\n",
                        "Gradient History for all previous iterations: \n",
                        grads_history, "\n",
                    )

            # Comparing the obtained results with the previous results and save the best one.
            if y is not None and search_for_best_nmi is True:
                self.ari = adjusted_rand_score(y, self.clusters)
                self.nmi = normalized_mutual_info_score(y, self.clusters)
                self.inertia = self.compute_inertia(m=x,)
                if self.best_nmi < self.nmi:
                    self.best_ari = deepcopy(self.ari)
                    self.best_nmi = deepcopy(self.nmi)
                    self.best_centroids_idx = deepcopy(idx)
                    self.best_inertia = deepcopy(self.inertia)
                    self.best_clusters = deepcopy(self.clusters)
                    self.aris_history = deepcopy(aris_history)
                    self.nmis_history = deepcopy(nmis_history)
                    self.inertias_history = deepcopy(inertias_history)
                    self.grads_history = deepcopy(grads_history)
                    self.grads_v_history = deepcopy(grads_v_history)
                    self.best_iter_aris_history = deepcopy(tmp_ari.mean())
                    self.best_iter_nmis_history = deepcopy(tmp_nmi.mean())
                    self.best_iter_inertias_history = deepcopy(tmp_inertia.mean())
                    self.best_iter_gradients_history = deepcopy(tmp_grads.mean())

            else:
                self.inertia = self.compute_inertia(m=x,)
                if self.inertia < self.best_inertia:
                    self.best_centroids_idx = deepcopy(idx)
                    self.best_inertia = deepcopy(self.inertia)
                    self.best_clusters = deepcopy(self.clusters)
                    self.grads_history = deepcopy(grads_history)
                    self.inertias_history = deepcopy(inertias_history)
                    self.grads_v_history = deepcopy(grads_v_history)
                    self.best_iter_inertias_history = deepcopy(tmp_inertia.mean())
                    self.best_iter_gradients_history = deepcopy(tmp_grads.mean())

        return self.best_clusters
