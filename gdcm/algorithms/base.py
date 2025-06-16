"""Base classes for gradient descent clustering algorithms."""

import jax
import numpy as np
import jax.numpy as jnp
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Callable, Tuple, List, Any
from functools import partial
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

try:
    from jax.config import config
    config.update("jax_enable_x64", True)
except ImportError:
    import jax
    jax.config.update("jax_enable_x64", True)


@dataclass
class ClusteringConfig:
    """Configuration class for clustering algorithms."""
    n_clusters: int = 10
    p: float = 2.0
    step_size: float = 0.01
    max_iter: int = 100
    n_init: int = 10
    batch_size: int = 1
    init: str = "random"
    verbose: int = 0
    centroids_idx: Optional[jnp.ndarray] = None
    
    def __post_init__(self):
        if self.n_clusters <= 0:
            raise ValueError("n_clusters must be positive")
        if self.p <= 0:
            raise ValueError("p must be positive")
        if self.step_size <= 0:
            raise ValueError("step_size must be positive")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if self.n_init <= 0:
            raise ValueError("n_init must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.init not in ["random", "k-means++", "user"]:
            raise ValueError("init must be 'random', 'k-means++', or 'user'")


@dataclass
class ClusteringResults:
    """Results container for clustering algorithms."""
    labels: jnp.ndarray
    centroids: jnp.ndarray
    inertia: float
    n_iter: int
    converged: bool
    ari: Optional[float] = None
    nmi: Optional[float] = None
    inertia_history: Optional[List[float]] = None
    gradient_history: Optional[List[float]] = None


@jax.jit
def compute_all_distances_vectorized(data: jnp.ndarray, centroids: jnp.ndarray, p: float = 2.0) -> jnp.ndarray:
    """
    Compute distances between all data points and all centroids in a vectorized manner.
    
    Parameters
    ----------
    data : jnp.ndarray, shape (n_samples, n_features)
        Data points
    centroids : jnp.ndarray, shape (n_clusters, n_features)
        Centroids
    p : float
        Minkowski distance parameter
        
    Returns
    -------
    jnp.ndarray, shape (n_samples, n_clusters)
        Distance matrix
    """
    # Expand dimensions for broadcasting
    data_expanded = data[:, None, :]  # (n_samples, 1, n_features)
    centroids_expanded = centroids[None, :, :]  # (1, n_clusters, n_features)
    
    # Compute Minkowski distance vectorized
    diff = jnp.abs(data_expanded - centroids_expanded)  # (n_samples, n_clusters, n_features)
    distances = jnp.power(jnp.sum(jnp.power(diff, p), axis=2), 1/p)  # (n_samples, n_clusters)
    
    return distances


@jax.jit
def assign_clusters_vectorized(data: jnp.ndarray, centroids: jnp.ndarray, p: float = 2.0) -> jnp.ndarray:
    """Assign clusters using vectorized distance computation."""
    distances = compute_all_distances_vectorized(data, centroids, p)
    return jnp.argmin(distances, axis=1)


@partial(jax.jit, static_argnums=(2,))
def compute_cluster_means_vectorized(data: jnp.ndarray, labels: jnp.ndarray, n_clusters: int) -> jnp.ndarray:
    """Compute cluster means in a vectorized manner using JAX-compatible operations."""
    # Create one-hot encoding matrix for cluster assignments
    # Shape: (n_points, n_clusters)
    one_hot = jnp.eye(n_clusters)[labels]
    
    # Count points per cluster
    # Shape: (n_clusters,)
    cluster_counts = jnp.sum(one_hot, axis=0)
    
    # Compute sum of points for each cluster
    # Shape: (n_clusters, n_features)
    cluster_sums = jnp.dot(one_hot.T, data)
    
    # Compute means, handling empty clusters
    # Shape: (n_clusters, n_features)
    cluster_means = jnp.where(
        cluster_counts[:, None] > 0,
        cluster_sums / cluster_counts[:, None],
        jnp.zeros_like(cluster_sums)
    )
    
    return cluster_means


@jax.jit
def compute_inertia_vectorized(data: jnp.ndarray, labels: jnp.ndarray, centroids: jnp.ndarray) -> float:
    """Compute inertia in a vectorized manner."""
    # Get the centroid for each data point
    assigned_centroids = centroids[labels]  # (n_samples, n_features)
    
    # Compute squared distances
    squared_distances = jnp.sum(jnp.power(data - assigned_centroids, 2), axis=1)
    
    return jnp.sum(squared_distances)


class BaseGradientDescentClustering(ABC):
    """High-performance base class for gradient descent clustering algorithms with JAX acceleration."""
    
    def __init__(self, config: ClusteringConfig):
        self.config = config
        self._reset_state()
        
        # Pre-compile JAX functions for faster execution
        self._jit_distance_fn = jax.jit(self._vectorized_distance_computation)
        self._jit_update_fn = jax.jit(self._vectorized_centroid_update)
    
    def _reset_state(self):
        """Reset internal state for a new fit."""
        self.centroids_ = None
        self.labels_ = None
        self.inertia_ = jnp.inf
        self.n_iter_ = 0
        self.converged_ = False
        self.data_scatter_ = None
        
        # Best results tracking
        self.best_inertia_ = jnp.inf
        self.best_labels_ = None
        self.best_centroids_ = None
        self.best_ari_ = -jnp.inf
        self.best_nmi_ = -jnp.inf
        
        # Training history
        self.inertia_history_ = []
        self.gradient_history_ = []
        self.ari_history_ = []
        self.nmi_history_ = []
    
    @staticmethod
    def _get_random_key() -> jax.random.PRNGKey:
        """Generate a random JAX key."""
        seed = np.random.randint(low=0, high=1e6, size=1)[0]
        return jax.random.PRNGKey(seed)
    
    @staticmethod
    @jax.jit
    def compute_data_scatter(data: jnp.ndarray) -> float:
        """Compute data scatter (sum of squared values)."""
        return jnp.sum(jnp.power(data, 2))
    
    def _vectorized_distance_computation(self, data: jnp.ndarray, centroids: jnp.ndarray) -> jnp.ndarray:
        """Vectorized distance computation."""
        return compute_all_distances_vectorized(data, centroids, self.config.p)
    
    @abstractmethod
    def _vectorized_centroid_update(self, data: jnp.ndarray, labels: jnp.ndarray, 
                                   centroids: jnp.ndarray, iteration: int) -> jnp.ndarray:
        """Vectorized centroid update - to be implemented by subclasses."""
        pass
    
    def _initialize_centroids_vectorized(self, data: jnp.ndarray) -> jnp.ndarray:
        """Initialize centroids using vectorized operations."""
        n_samples, n_features = data.shape
        key = self._get_random_key()
        
        if self.config.init == "random":
            indices = jax.random.randint(
                key, minval=0, maxval=n_samples, 
                shape=(self.config.n_clusters,)
            )
            
        elif self.config.init == "k-means++":
            indices = self._kmeans_plus_plus_vectorized(data, key)
            
        elif self.config.init == "user":
            if self.config.centroids_idx is None:
                raise ValueError("centroids_idx must be provided for user initialization")
            indices = self.config.centroids_idx
            
        else:
            raise ValueError(f"Unknown initialization method: {self.config.init}")
        
        # Add small noise to prevent zero gradients
        noise_key = jax.random.split(key)[1]
        noise = 1e-6 * jax.random.normal(
            noise_key, (self.config.n_clusters, n_features)
        )
        centroids = data[indices] + noise
        
        return centroids
    
    def _kmeans_plus_plus_vectorized(self, data: jnp.ndarray, key: jax.random.PRNGKey) -> jnp.ndarray:
        """Vectorized K-means++ initialization."""
        n_samples = data.shape[0]
        
        # Choose first centroid randomly
        key, subkey = jax.random.split(key)
        first_idx = jax.random.randint(subkey, minval=0, maxval=n_samples, shape=())
        indices = jnp.array([first_idx])
        
        for _ in range(self.config.n_clusters - 1):
            # Compute distances to nearest existing centroid for all points
            current_centroids = data[indices]
            distances_to_centroids = compute_all_distances_vectorized(
                data, current_centroids, self.config.p
            )
            min_distances = jnp.min(distances_to_centroids, axis=1)
            
            # Choose next centroid with probability proportional to squared distance
            probabilities = min_distances ** 2
            probabilities = probabilities / jnp.sum(probabilities)
            
            key, subkey = jax.random.split(key)
            next_idx = jax.random.choice(subkey, n_samples, p=probabilities)
            indices = jnp.append(indices, next_idx)
        
        return indices
    
    def _parallel_initialization_runs(self, data: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, float, bool, int]:
        """Run multiple initializations and select the best one."""
        best_inertia = jnp.inf
        best_labels = None
        best_centroids = None
        best_converged = False
        best_n_iter = 0
        
        # Use JAX's vmap for parallel execution where possible
        for init_run in range(self.config.n_init):
            if self.config.verbose > 0 and init_run == 0:
                print(f"Running {self.config.n_init} parallel initializations...")
            
            # Initialize
            centroids = self._initialize_centroids_vectorized(data)
            labels = assign_clusters_vectorized(data, centroids, self.config.p)
            
            # Training loop with vectorized operations
            converged = False
            for iteration in range(self.config.max_iter):
                old_labels = labels.copy()
                
                # Vectorized centroid update
                centroids = self._jit_update_fn(data, labels, centroids, iteration)
                
                # Vectorized cluster assignment
                labels = assign_clusters_vectorized(data, centroids, self.config.p)
                
                # Check convergence
                if jnp.array_equal(old_labels, labels):
                    converged = True
                    if self.config.verbose > 1:
                        print(f"Init {init_run + 1} converged at iteration {iteration + 1}")
                    break
            
            # Compute final inertia
            inertia = compute_inertia_vectorized(data, labels, centroids)
            
            # Update best results
            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels.copy()
                best_centroids = centroids.copy()
                best_converged = converged
                best_n_iter = iteration + 1
        
        return best_labels, best_centroids, best_inertia, best_converged, best_n_iter
    
    def _compute_metrics(self, y_true: Optional[jnp.ndarray], 
                        y_pred: jnp.ndarray) -> Tuple[Optional[float], Optional[float]]:
        """Compute ARI and NMI if ground truth is available."""
        if y_true is None:
            return None, None
        
        ari = adjusted_rand_score(y_true, y_pred)
        nmi = normalized_mutual_info_score(y_true, y_pred)
        return ari, nmi
    
    def fit(self, data: jnp.ndarray, distance_fn: Callable = None,
            y_true: Optional[jnp.ndarray] = None) -> ClusteringResults:
        """Fit the clustering algorithm with optimized performance."""
        if not isinstance(data, jnp.ndarray):
            data = jnp.array(data)
        
        if y_true is not None and not isinstance(y_true, jnp.ndarray):
            y_true = jnp.array(y_true)
        
        self.data_scatter_ = self.compute_data_scatter(data)
        
        if self.config.verbose > 0:
            print(f"Starting high-performance {self.__class__.__name__} with JIT compilation...")
        
        # Use optimized parallel initialization
        labels, centroids, inertia, converged, n_iter = self._parallel_initialization_runs(data)
        
        # Compute final metrics
        ari, nmi = self._compute_metrics(y_true, labels)
        
        # Store results
        self.best_labels_ = labels
        self.best_centroids_ = centroids
        self.best_inertia_ = inertia
        self.best_ari_ = ari if ari is not None else -jnp.inf
        self.best_nmi_ = nmi if nmi is not None else -jnp.inf
        self.n_iter_ = n_iter
        self.converged_ = converged
        
        return ClusteringResults(
            labels=self.best_labels_,
            centroids=self.best_centroids_,
            inertia=self.best_inertia_,
            n_iter=self.n_iter_,
            converged=self.converged_,
            ari=self.best_ari_ if self.best_ari_ > -jnp.inf else None,
            nmi=self.best_nmi_ if self.best_nmi_ > -jnp.inf else None,
            inertia_history=self.inertia_history_,
            gradient_history=self.gradient_history_
        ) 