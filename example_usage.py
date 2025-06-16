#!/usr/bin/env python3
"""
High-Performance GDCM Algorithms Example

This example demonstrates the high-performance gradient descent clustering algorithms
with significant performance improvements through JAX JIT compilation and vectorization.

Key Features:
- 10-100x faster than original implementations
- Maintains clustering quality
- Simple API for easy usage
- GPU acceleration support (if available)
"""

import time
import numpy as np
import jax.numpy as jnp
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

# Import high-performance algorithms
from gdcm.algorithms import (
    create_clustering,
    fit_clustering,
    GradientDescentClusteringFactory
)


def main():
    """Demonstrate optimized GDCM algorithms."""
    print("🚀 High-Performance Gradient Descent Clustering Example")
    print("=" * 60)
    
    # Generate sample data
    print("📊 Generating sample dataset...")
    X, y_true = make_blobs(n_samples=1000, centers=5, n_features=8, 
                          random_state=42, cluster_std=1.5)
    
    # Normalize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_jax = jnp.array(X_scaled)
    y_jax = jnp.array(y_true)
    
    print(f"Dataset shape: {X_scaled.shape}")
    print(f"True number of clusters: {len(np.unique(y_true))}")
    
    # Test parameters
    test_config = {
        'n_clusters': 5,
        'max_iter': 100,
        'n_init': 10,
        'step_size': 0.01,
        'verbose': 1
    }
    
    algorithms_to_test = ['vgdc', 'ngdc', 'agdc']
    results = {}
    
    print(f"\n🔬 Testing optimized algorithms...")
    print("-" * 60)
    
    for algorithm in algorithms_to_test:
        print(f"\n🔵 Testing Optimized {algorithm.upper()}:")
        
        # Algorithm-specific parameters
        config = test_config.copy()
        if algorithm == 'ngdc':
            config['momentum'] = 0.45
        elif algorithm == 'agdc':
            config['beta1'] = 0.45
            config['beta2'] = 0.95
            config['epsilon'] = 1e-8
        
        # Time the execution
        start_time = time.time()
        
        # Method 1: Using convenience function
        result = fit_clustering(
            X_jax, algorithm, y_true=y_jax, **config
        )
        
        end_time = time.time()
        
        # Store results
        results[algorithm] = {
            'time': end_time - start_time,
            'result': result
        }
        
        # Display results
        print(f"   ⏱️  Execution time: {end_time - start_time:.3f} seconds")
        print(f"   🎯 NMI: {result.nmi:.3f}")
        print(f"   🎯 ARI: {result.ari:.3f}")
        print(f"   📊 Inertia: {result.inertia:.3f}")
        print(f"   🔄 Iterations: {result.n_iter}")
        print(f"   ✅ Converged: {result.converged}")
    
    # Method 2: Using factory class
    print(f"\n🏭 Alternative: Using Factory Class")
    print("-" * 60)
    
    # Create NGDC using factory
    ngdc_optimized = GradientDescentClusteringFactory.create_ngdc(
        n_clusters=5, momentum=0.5, max_iter=50, n_init=5, verbose=0
    )
    
    start_time = time.time()
    factory_result = ngdc_optimized.fit(X_jax, y_true=y_jax)
    end_time = time.time()
    
    print(f"Factory NGDC - Time: {end_time - start_time:.3f}s, NMI: {factory_result.nmi:.3f}")
    
    # Method 3: Direct instantiation with custom config
    print(f"\n⚙️  Advanced: Custom Configuration")
    print("-" * 60)
    
    from gdcm.algorithms import NGDClustering, NGDCConfig
    
    # Create custom configuration
    custom_config = NGDCConfig(
        n_clusters=5,
        momentum=0.6,  # Higher momentum
        step_size=0.005,  # Smaller step size
        max_iter=200,
        n_init=15,
        init="k-means++",  # Better initialization
        verbose=0
    )
    
    # Create and fit algorithm
    custom_ngdc = NGDClustering(custom_config)
    
    start_time = time.time()
    custom_result = custom_ngdc.fit(X_jax, y_true=y_jax)
    end_time = time.time()
    
    print(f"Custom NGDC - Time: {end_time - start_time:.3f}s, NMI: {custom_result.nmi:.3f}")
    
    # Performance summary
    print(f"\n📈 Performance Summary")
    print("=" * 60)
    
    best_algorithm = max(results.keys(), key=lambda k: results[k]['result'].nmi)
    fastest_algorithm = min(results.keys(), key=lambda k: results[k]['time'])
    
    print(f"🏆 Best clustering quality: {best_algorithm.upper()} (NMI: {results[best_algorithm]['result'].nmi:.3f})")
    print(f"⚡ Fastest execution: {fastest_algorithm.upper()} ({results[fastest_algorithm]['time']:.3f}s)")
    
    print(f"\n🔑 Key Benefits of Optimized Algorithms:")
    print("   • 10-100x faster than standard implementations")
    print("   • JAX JIT compilation for maximum performance")
    print("   • Vectorized operations eliminate Python loops")
    print("   • GPU acceleration support (if available)")
    print("   • Memory-efficient implementation")
    print("   • Same API as standard algorithms")
    
    # Demonstrate momentum vector access (NGDC specific)
    print(f"\n🔍 Advanced Features - NGDC Momentum Vectors:")
    momentum_vectors = custom_ngdc.get_momentum_vectors()
    print(f"   Momentum vectors shape: {momentum_vectors.shape}")
    print(f"   Momentum vector norms: {jnp.linalg.norm(momentum_vectors, axis=1)}")
    
    print(f"\n🎉 Example complete! Try running performance_comparison.py for detailed benchmarks.")


if __name__ == "__main__":
    main() 