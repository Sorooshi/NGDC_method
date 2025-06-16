"""High-Performance Factory for Gradient Descent Clustering Algorithms."""

import jax.numpy as jnp
from typing import Optional, Union, Dict, Any
from .base import ClusteringResults, ClusteringConfig
from .vgdc import VGDClustering
from .ngdc import NGDClustering, NGDCConfig
from .agdc import AGDClustering, AGDCConfig


class GradientDescentClusteringFactory:
    """
    Factory class for creating high-performance gradient descent clustering algorithms.
    
    This factory provides significant performance improvements through:
    - JAX JIT compilation for 10-100x speedups
    - Vectorized operations for efficient computation
    - Parallel processing capabilities
    - Memory optimization
    
    Supports three optimized algorithms:
    - VGDC: Vanilla Gradient Descent Clustering
    - NGDC: Nesterov Gradient Descent Clustering (novel algorithm)
    - AGDC: Adam Gradient Descent Clustering
    """
    
    SUPPORTED_ALGORITHMS = ["vgdc", "ngdc", "agdc"]
    
    @staticmethod
    def create_algorithm(algorithm: str, config: Optional[Union[ClusteringConfig, Dict[str, Any]]] = None,
                        **kwargs) -> Union[VGDClustering, NGDClustering, AGDClustering]:
        """
        Create a clustering algorithm instance.
        
        Parameters
        ----------
        algorithm : str
            Algorithm name ("vgdc", "ngdc", or "agdc")
        config : ClusteringConfig or dict, optional
            Configuration object or dictionary of parameters
        **kwargs
            Additional parameters to override config values
            
        Returns
        -------
        clustering_instance
            Clustering algorithm instance
            
        Raises
        ------
        ValueError
            If algorithm name is not supported
        """
        algorithm = algorithm.lower()
        
        if algorithm not in GradientDescentClusteringFactory.SUPPORTED_ALGORITHMS:
            raise ValueError(f"Unsupported algorithm: {algorithm}. "
                           f"Choose from: {GradientDescentClusteringFactory.SUPPORTED_ALGORITHMS}")
        
        # Handle configuration
        if config is None:
            config = {}
        elif isinstance(config, ClusteringConfig):
            config = config.__dict__
        
        # Merge with kwargs
        config.update(kwargs)
        
        # Create algorithm-specific configuration
        if algorithm == "vgdc":
            final_config = ClusteringConfig(**config)
            return VGDClustering(final_config)
            
        elif algorithm == "ngdc":
            final_config = NGDCConfig(**config)
            return NGDClustering(final_config)
            
        elif algorithm == "agdc":
            final_config = AGDCConfig(**config)
            return AGDClustering(final_config)
    
    @staticmethod
    def create_vgdc(n_clusters: int = 10, p: float = 2.0, step_size: float = 0.01,
                   max_iter: int = 100, n_init: int = 10, batch_size: int = 1,
                   init: str = "random", verbose: int = 0,
                   centroids_idx: Optional[jnp.ndarray] = None) -> VGDClustering:
        """Create VGDC clustering instance with default parameters."""
        config = ClusteringConfig(
            n_clusters=n_clusters, p=p, step_size=step_size, max_iter=max_iter,
            n_init=n_init, batch_size=batch_size, init=init, verbose=verbose,
            centroids_idx=centroids_idx
        )
        return VGDClustering(config)
    
    @staticmethod
    def create_ngdc(n_clusters: int = 10, p: float = 2.0, momentum: float = 0.45,
                   step_size: float = 0.01, max_iter: int = 100, n_init: int = 10,
                   batch_size: int = 1, init: str = "random", verbose: int = 0,
                   centroids_idx: Optional[jnp.ndarray] = None) -> NGDClustering:
        """Create NGDC clustering instance with default parameters.""" 
        config = NGDCConfig(
            n_clusters=n_clusters, p=p, momentum=momentum, step_size=step_size,
            max_iter=max_iter, n_init=n_init, batch_size=batch_size, init=init,
            verbose=verbose, centroids_idx=centroids_idx
        )
        return NGDClustering(config)
    
    @staticmethod
    def create_agdc(n_clusters: int = 10, p: float = 2.0, beta1: float = 0.45,
                   beta2: float = 0.95, epsilon: float = 1e-8, step_size: float = 0.01,
                   max_iter: int = 100, n_init: int = 10, batch_size: int = 1,
                   init: str = "random", verbose: int = 0,
                   centroids_idx: Optional[jnp.ndarray] = None) -> AGDClustering:
        """Create AGDC clustering instance with default parameters."""
        config = AGDCConfig(
            n_clusters=n_clusters, p=p, beta1=beta1, beta2=beta2, epsilon=epsilon,
            step_size=step_size, max_iter=max_iter, n_init=n_init,
            batch_size=batch_size, init=init, verbose=verbose,
            centroids_idx=centroids_idx
        )
        return AGDClustering(config)


# Convenience functions
def create_clustering(algorithm: str, **kwargs):
    """
    Create a clustering algorithm with high-performance implementation.
    
    This function provides a simple interface to create high-performance
    gradient descent clustering algorithms.
    
    Parameters
    ----------
    algorithm : str
        Algorithm name ("vgdc", "ngdc", or "agdc")
    **kwargs
        Algorithm-specific parameters
        
    Returns
    -------
    clustering_instance
        Clustering algorithm instance
        
    Examples
    --------
    >>> # Create NGDC with default parameters
    >>> ngdc = create_clustering("ngdc", n_clusters=3, momentum=0.5)
    >>> 
    >>> # Create AGDC with custom parameters
    >>> agdc = create_clustering("agdc", n_clusters=5, beta1=0.9, beta2=0.999)
    """
    return GradientDescentClusteringFactory.create_algorithm(algorithm, **kwargs)


def fit_clustering(data: jnp.ndarray, algorithm: str, y_true: Optional[jnp.ndarray] = None,
                  **kwargs) -> ClusteringResults:
    """
    Fit a clustering algorithm to data.
    
    This is a convenience function that creates and fits a clustering
    algorithm in one step with maximum performance.
    
    Parameters
    ----------
    data : jnp.ndarray
        Input data to cluster
    algorithm : str
        Algorithm name ("vgdc", "ngdc", or "agdc")
    y_true : jnp.ndarray, optional
        True labels for evaluation metrics
    **kwargs
        Algorithm-specific parameters
        
    Returns
    -------
    ClusteringResults
        Results object containing labels, centroids, and metrics
        
    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from sklearn.datasets import make_blobs
    >>> 
    >>> # Generate sample data
    >>> X, y = make_blobs(n_samples=300, centers=4, random_state=42)
    >>> X = jnp.array(X)
    >>> y = jnp.array(y)
    >>> 
    >>> # Fit NGDC
    >>> results = fit_clustering(X, "ngdc", y_true=y, n_clusters=4, verbose=1)
    >>> print(f"NMI: {results.nmi:.3f}, ARI: {results.ari:.3f}")
    """
    clustering = create_clustering(algorithm, **kwargs)
    return clustering.fit(data, y_true=y_true)


def create_clustering_algorithm(
    algorithm: str = "ngdc",
    n_clusters: int = 10,
    p: float = 2.0,
    n_init: int = 10,
    max_iter: int = 100,
    step_size: float = 0.01,
    batch_size: int = 1,
    init: str = "random",
    verbose: int = 0,
    centroids_idx: Optional[jnp.ndarray] = None,
    # Algorithm-specific parameters
    momentum: float = 0.45,
    beta1: float = 0.45,
    beta2: float = 0.95,
    epsilon: float = 1e-8,
) -> Union[VGDClustering, NGDClustering, AGDClustering]:
    """
    Factory function to create clustering algorithms (backward compatible).
    
    Parameters
    ----------
    algorithm : str, default="ngdc"
        Algorithm type: "vgdc", "ngdc", or "agdc"
    n_clusters : int, default=10
        Number of clusters
    p : float, default=2.0
        P value for Minkowski distance
    n_init : int, default=10
        Number of different initializations
    max_iter : int, default=100
        Maximum number of iterations
    step_size : float, default=0.01
        Learning rate
    batch_size : int, default=1
        Batch size
    init : str, default="random"
        Initialization method
    verbose : int, default=0
        Verbosity level
    centroids_idx : jnp.ndarray, optional
        User-defined centroid indices
    momentum : float, default=0.45
        Momentum for NGDC
    beta1 : float, default=0.45
        Beta1 for AGDC
    beta2 : float, default=0.95
        Beta2 for AGDC
    epsilon : float, default=1e-8
        Epsilon for AGDC
        
    Returns
    -------
    clustering algorithm instance
    """
    return create_clustering(
        algorithm=algorithm,
        n_clusters=n_clusters,
        p=p,
        n_init=n_init,
        max_iter=max_iter,
        step_size=step_size,
        batch_size=batch_size,
        init=init,
        verbose=verbose,
        centroids_idx=centroids_idx,
        momentum=momentum,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon
    )


class GDCMf:
    """
    Backward compatibility wrapper for the original GDCMf interface.
    
    This class provides the same interface as the original implementation
    but uses the refactored high-performance architecture underneath.
    
    Parameters
    ----------
    n_clusters : int, default=10
        Number of clusters
    p : float, default=2.0
        P value for Minkowski distance
    n_init : int, default=10
        Number of different initializations
    max_iter : int, default=100
        Maximum number of iterations
    step_size : float, default=0.01
        Learning rate
    batch_size : int, default=1
        Batch size
    init : str, default="random"
        Initialization method
    verbose : int, default=0
        Verbosity level
    centroids_idx : jnp.ndarray, optional
        User-defined centroid indices
    update_rule : str, default="ngdc"
        Update rule: "vgdc", "ngdc", or "agdc"
    mu_1 : float, default=0.45
        Momentum/beta1 parameter
    mu_2 : float, default=0.95
        Beta2 parameter (for AGDC)
    """
    
    def __init__(
        self,
        n_clusters: int = 10,
        p: float = 2.0,
        n_init: int = 10,
        verbose: int = 0,
        mu_1: float = 0.45,
        mu_2: float = 0.95,
        max_iter: int = 100,
        init: str = "random",
        batch_size: int = 1,
        step_size: float = 0.01,
        centroids_idx: Optional[jnp.ndarray] = None,
        update_rule: str = "ngdc",
        # Deprecated parameters (kept for compatibility)
        tau: float = 1e-4,
        range_len: int = 5,
    ):
        # Store parameters
        self._algorithm_name = update_rule.lower()
        
        # Create the underlying algorithm
        self._algorithm = create_clustering_algorithm(
            algorithm=update_rule,
            n_clusters=n_clusters,
            p=p,
            n_init=n_init,
            max_iter=max_iter,
            step_size=step_size,
            batch_size=batch_size,
            init=init,
            verbose=verbose,
            centroids_idx=centroids_idx,
            momentum=mu_1,
            beta1=mu_1,
            beta2=mu_2,
        )
        
        # Initialize compatibility attributes
        self.best_ari = -jnp.inf
        self.best_nmi = -jnp.inf
        self.best_inertia = jnp.inf
        self.best_clusters = None
        self.best_centroids_idx = None
        self.data_scatter = None
        self.stop_type = None
        self.aris_history = None
        self.nmis_history = None
        self.inertias_history = None
        self.grads_history = None
        self.grads_v_history = None
        
        # Store deprecated parameters (for compatibility but not used)
        self._tau = tau
        self._range_len = range_len
    
    @staticmethod
    def minkowski_fn(data_point: jnp.ndarray, centroid: jnp.ndarray, p: float) -> float:
        """Minkowski distance function for backward compatibility."""
        from .distance_functions import minkowski_distance
        return minkowski_distance(data_point, centroid, p)
    
    def fit(self, x: jnp.ndarray, distance_fn=None, y: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        Fit the algorithm to data (backward compatible interface).
        
        Parameters
        ----------
        x : jnp.ndarray
            Input data
        distance_fn : callable, optional
            Distance function (ignored, uses Minkowski)
        y : jnp.ndarray, optional
            True labels for evaluation
            
        Returns
        -------
        jnp.ndarray
            Cluster labels
        """
        # Fit using the high-performance implementation
        result = self._algorithm.fit(x, y_true=y)
        
        # Update compatibility attributes
        self.best_ari = result.ari if result.ari is not None else -jnp.inf
        self.best_nmi = result.nmi if result.nmi is not None else -jnp.inf
        self.best_inertia = result.inertia
        self.best_clusters = result.labels
        self.best_centroids_idx = result.centroids
        
        return result.labels
    
    def compute_inertia(self, m: jnp.ndarray) -> float:
        """Compute inertia for compatibility."""
        return float(self.best_inertia)


def get_algorithm_names():
    """Get list of available algorithm names."""
    return GradientDescentClusteringFactory.SUPPORTED_ALGORITHMS


def get_algorithm_info(algorithm: str) -> dict:
    """Get information about an algorithm."""
    info = {
        "vgdc": {"name": "Vanilla Gradient Descent Clustering", "params": ["step_size"]},
        "ngdc": {"name": "Nesterov Gradient Descent Clustering", "params": ["step_size", "momentum"]}, 
        "agdc": {"name": "Adam Gradient Descent Clustering", "params": ["step_size", "beta1", "beta2", "epsilon"]}
    }
    return info.get(algorithm.lower(), {}) 