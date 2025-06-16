"""Gradient Descent Clustering Methods (GDCM) Algorithms Package.

This package provides implementations of novel gradient descent clustering algorithms
introduced in the NGDC research, along with high-performance optimized versions.

Standard Algorithms:
- VGDClustering: Vanilla Gradient Descent Clustering
- NGDClustering: Nesterov Gradient Descent Clustering (novel algorithm)
- AGDClustering: Adam Gradient Descent Clustering

Optimized Algorithms (JAX-accelerated):
- OptimizedVGDClustering: High-performance VGDC with 10-100x speedup
- OptimizedNGDClustering: High-performance NGDC with JIT compilation
- OptimizedAGDClustering: High-performance AGDC with vectorization

Factory Classes:
- GradientDescentClusteringFactory: Standard algorithm factory
- OptimizedGradientDescentClusteringFactory: High-performance algorithm factory

Legacy Support:
- GDCMf: Backward compatibility wrapper for original implementation
"""

# High-performance implementations
from .vgdc import VGDClustering
from .ngdc import NGDClustering, NGDCConfig
from .agdc import AGDClustering, AGDCConfig
from .factory import (
    GradientDescentClusteringFactory,
    create_clustering_algorithm, 
    create_clustering,
    fit_clustering,
    GDCMf
)

# Base classes and data structures
from .base import (
    BaseGradientDescentClustering, 
    ClusteringConfig, 
    ClusteringResults
)

# Legacy support
from .gradient_descent_clustering_methods_features import GDCMf

# Distance functions
from .distance_functions import (
    minkowski_distance,
    euclidean_distance,
    manhattan_distance,
    cosine_distance,
    get_distance_function
)

__all__ = [
    # High-performance algorithms
    "VGDClustering",
    "NGDClustering", 
    "NGDCConfig",
    "AGDClustering",
    "AGDCConfig",
    
    # Factory and convenience functions
    "GradientDescentClusteringFactory",
    "create_clustering_algorithm",
    "create_clustering",
    "fit_clustering",
    
    # Base classes
    "BaseGradientDescentClustering",
    "ClusteringConfig",
    "ClusteringResults",
    
    # Legacy support
    "GDCMf",
    
    # Distance functions
    "minkowski_distance",
    "euclidean_distance",
    "manhattan_distance",
    "cosine_distance",
    "get_distance_function",
]

# Version information
__version__ = "2.0.0"
__author__ = "NGDC Research Team"
__description__ = "Novel Gradient Descent Clustering Methods" 