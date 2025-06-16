"""Distance functions for clustering algorithms."""

import jax.numpy as jnp
from typing import Union


def minkowski_distance(point1: jnp.ndarray, point2: jnp.ndarray, p: float) -> float:
    """
    Compute p-valued Minkowski distance between two vectors.
    
    Parameters
    ----------
    point1 : jnp.ndarray
        First vector
    point2 : jnp.ndarray
        Second vector  
    p : float
        P value in Minkowski distance (p >= 1)
        
    Returns
    -------
    float
        Minkowski distance
        
    Raises
    ------
    ValueError
        If p < 1 or if inputs have incompatible dimensions
    """
    if p < 1:
        raise ValueError("p must be at least 1")
    
    if point1.ndim > 1 and point1.shape[0] == 1:
        point1 = point1.squeeze()
    if point2.ndim > 1 and point2.shape[0] == 1:
        point2 = point2.squeeze()
    
    if point1.ndim > 1:
        # Handle multiple points at once
        return jnp.power(jnp.sum(jnp.power(jnp.abs(point1 - point2), p), axis=1), 1/p)
    else:
        # Handle single point
        return jnp.power(jnp.sum(jnp.power(jnp.abs(point1 - point2), p)), 1/p)


def euclidean_distance(point1: jnp.ndarray, point2: jnp.ndarray, p: float = None) -> float:
    """
    Compute Euclidean distance between two vectors.
    
    Parameters
    ----------
    point1 : jnp.ndarray
        First vector
    point2 : jnp.ndarray
        Second vector
    p : float, optional
        Ignored (for API consistency)
        
    Returns
    -------
    float
        Euclidean distance
    """
    return minkowski_distance(point1, point2, 2.0)


def manhattan_distance(point1: jnp.ndarray, point2: jnp.ndarray, p: float = None) -> float:
    """
    Compute Manhattan distance between two vectors.
    
    Parameters
    ----------
    point1 : jnp.ndarray
        First vector
    point2 : jnp.ndarray
        Second vector
    p : float, optional
        Ignored (for API consistency)
        
    Returns
    -------
    float
        Manhattan distance
    """
    return minkowski_distance(point1, point2, 1.0)


def cosine_distance(point1: jnp.ndarray, point2: jnp.ndarray, p: float = None) -> float:
    """
    Compute cosine distance between two vectors.
    
    Parameters
    ----------
    point1 : jnp.ndarray
        First vector
    point2 : jnp.ndarray
        Second vector
    p : float, optional
        Ignored (for API consistency)
        
    Returns
    -------
    float
        Cosine distance (1 - cosine similarity)
        
    Raises
    ------
    ValueError
        If inputs have incompatible dimensions
    """
    if point1.ndim > 1 or point2.ndim > 1:
        raise ValueError("Cosine distance only supports 1-D vectors")
    
    # Add small epsilon to avoid division by zero
    eps = 1e-10
    
    dot_product = jnp.dot(point1, point2) + eps
    norm1 = jnp.sqrt(jnp.sum(point1**2) + eps)
    norm2 = jnp.sqrt(jnp.sum(point2**2) + eps)
    
    cosine_similarity = dot_product / (norm1 * norm2)
    return 1.0 - cosine_similarity


def get_distance_function(name: str):
    """
    Get distance function by name.
    
    Parameters
    ----------
    name : str
        Name of the distance function
        
    Returns
    -------
    callable
        Distance function
        
    Raises
    ------
    ValueError
        If distance function name is not recognized
    """
    distance_functions = {
        'minkowski': minkowski_distance,
        'euclidean': euclidean_distance,
        'manhattan': manhattan_distance,
        'cosine': cosine_distance,
        'l1': manhattan_distance,
        'l2': euclidean_distance,
    }
    
    name = name.lower()
    if name not in distance_functions:
        available = ', '.join(distance_functions.keys())
        raise ValueError(f"Unknown distance function '{name}'. Available: {available}")
    
    return distance_functions[name] 