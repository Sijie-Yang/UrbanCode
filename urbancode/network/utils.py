import networkx as nx
import osmnx as ox
import geopandas as gpd
from typing import Dict, Any

def clean_intersections(G: nx.MultiDiGraph, tolerance: float = 15, rebuild_graph: bool = True) -> nx.MultiDiGraph:
    """
    Clean the intersections in the graph by merging nearby nodes.
    
    Args:
        G (nx.MultiDiGraph): The input graph.
        tolerance (float): The tolerance in meters for merging nearby nodes.
        rebuild_graph (bool): Whether to rebuild the graph after cleaning.
    
    Returns:
        nx.MultiDiGraph: The cleaned graph.
    """
    return ox.consolidate_intersections(G, tolerance=tolerance, rebuild_graph=rebuild_graph)

def add_edge_bearings(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """
    Add edge bearings to the graph.
    
    Args:
        G (nx.MultiDiGraph): The input graph.
    
    Returns:
        nx.MultiDiGraph: The graph with edge bearings added.
    """
    return ox.add_edge_bearings(G)

def add_edge_grades(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """
    Add edge grades to the graph.
    
    Args:
        G (nx.MultiDiGraph): The input graph.
    
    Returns:
        nx.MultiDiGraph: The graph with edge grades added.
    """
    return ox.add_edge_grades(G)

def get_largest_component(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """
    Get the largest connected component of the graph.
    
    Args:
        G (nx.MultiDiGraph): The input graph.
    
    Returns:
        nx.MultiDiGraph: The largest connected component.
    """
    return ox.utils_graph.get_largest_component(G)

def graph_to_gdfs(G: nx.MultiDiGraph) -> Dict[str, gpd.GeoDataFrame]:
    """
    Convert the NetworkX graph to GeoDataFrames.
    
    Args:
        G (nx.MultiDiGraph): The input graph.
    
    Returns:
        Dict[str, gpd.GeoDataFrame]: A dictionary containing 'nodes' and 'edges' GeoDataFrames.
    """
    return ox.graph_to_gdfs(G)

def project_graph(G: nx.MultiDiGraph, to_crs: Any = None) -> nx.MultiDiGraph:
    """
    Project graph to a specified CRS.
    
    Args:
        G (nx.MultiDiGraph): The input graph.
        to_crs (Any): The CRS to project to. If None, project to UTM.
    
    Returns:
        nx.MultiDiGraph: The projected graph.
    """
    return ox.project_graph(G, to_crs=to_crs)

def calculate_graph_stats(G: nx.MultiDiGraph) -> Dict[str, Any]:
    """
    Calculate basic stats for the graph.
    
    Args:
        G (nx.MultiDiGraph): The input graph.
    
    Returns:
        Dict[str, Any]: A dictionary containing graph statistics.
    """
    return ox.basic_stats(G)

def calculate_extended_stats(G: nx.MultiDiGraph) -> Dict[str, Any]:
    """
    Calculate extended stats for the graph.
    
    Args:
        G (nx.MultiDiGraph): The input graph.
    
    Returns:
        Dict[str, Any]: A dictionary containing extended graph statistics.
    """
    return ox.extended_stats(G)