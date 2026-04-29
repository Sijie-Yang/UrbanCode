__version__ = "0.2.2"
__author__ = "Sijie Yang"
__description__ = "A package for universal urban analysis"

# Make modules available at package level
__all__ = ['network', 'svi']

# Try to import network module (may fail due to pandas dependency issues)
# Suppress all exceptions during import - functions will be lazy-loaded on access
network = None
try:
    from . import network
    from .network.core import (
        download_network,
        save_network,
        load_saved_network,
        graph_to_gdf,
        graph_from_gdf
    )
    from .network.accessibility import (
        closeness_centrality_radius,
        betweenness_centrality_radius,
        reachability_radius,
        local_efficiency_radius,
        clustering_coefficient_radius,
        calculate_accessibility_metrics
    )
except Exception:
    # Silently fail - network module will be lazy-loaded when accessed
    # This prevents import errors from breaking the package import
    # Don't set any attributes - let __getattr__ handle them
    pass

# Lazy import svi module to avoid pandas dependency issues when only using network module
_svi_module = None

def __getattr__(name):
    """Lazy import modules when accessed."""
    global _svi_module
    
    if name == 'svi':
        if _svi_module is None:
            try:
                from . import svi as _svi_module
            except ImportError as e:
                raise ImportError(f"Failed to import svi module: {e}")
        return _svi_module
    
    # Check if this is a network-related attribute
    network_attrs = ['network', 'download_network', 'save_network', 'load_saved_network', 
                     'graph_to_gdf', 'graph_from_gdf', 'closeness_centrality_radius',
                     'betweenness_centrality_radius', 'reachability_radius',
                     'local_efficiency_radius', 'clustering_coefficient_radius',
                     'calculate_accessibility_metrics']
    
    if name in network_attrs:
        # Try to import network module (will be called if attribute doesn't exist)
            try:
                from . import network
                from .network.core import (
                    download_network,
                    save_network,
                    load_saved_network,
                    graph_to_gdf,
                    graph_from_gdf
                )
                from .network.accessibility import (
                    closeness_centrality_radius,
                    betweenness_centrality_radius,
                    reachability_radius,
                    local_efficiency_radius,
                    clustering_coefficient_radius,
                    calculate_accessibility_metrics
                )
                # Update globals
                globals()['network'] = network
                globals()['download_network'] = download_network
                globals()['save_network'] = save_network
                globals()['load_saved_network'] = load_saved_network
                globals()['graph_to_gdf'] = graph_to_gdf
                globals()['graph_from_gdf'] = graph_from_gdf
                globals()['closeness_centrality_radius'] = closeness_centrality_radius
                globals()['betweenness_centrality_radius'] = betweenness_centrality_radius
                globals()['reachability_radius'] = reachability_radius
                globals()['local_efficiency_radius'] = local_efficiency_radius
                globals()['clustering_coefficient_radius'] = clustering_coefficient_radius
                globals()['calculate_accessibility_metrics'] = calculate_accessibility_metrics
                # Return the requested attribute if it exists
                if name in globals():
                    return globals()[name]
                else:
                    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
            except (ImportError, AttributeError, ValueError) as e:
                error_msg = str(e)
                if 'numpy.dtype size changed' in error_msg or '_ARRAY_API' in error_msg:
                    raise ImportError(
                        f"Failed to import network module due to dependency version incompatibility: {e}\n"
                        f"This is likely due to pandas/numpy version mismatch. "
                        f"Try: pip install --upgrade pandas numpy"
                    ) from e
                else:
                    raise ImportError(f"Failed to import network module: {e}") from e
    
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")