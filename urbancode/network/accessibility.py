"""
Accessibility calculation module for urban network analysis.

This module provides functions to calculate accessibility metrics based on radius,
including closeness centrality, betweenness centrality, and other urban network indicators.
"""

import networkx as nx
import numpy as np
from typing import Union, Dict, Optional, List
import warnings
from tqdm import tqdm


def closeness_centrality_radius(
    G: nx.MultiDiGraph,
    radius: float,
    weight: str = 'length',
    name: str = 'closeness_radius',
    distance: Optional[str] = None
) -> nx.MultiDiGraph:
    """
    Calculate closeness centrality within a specified radius for each node.
    
    Closeness centrality measures how close a node is to all other nodes within
    the radius. Higher values indicate better accessibility.
    
    Args:
        G (nx.MultiDiGraph): The input graph.
        radius (float): The radius in meters (or units of weight attribute) 
                       to consider for closeness calculation.
        weight (str): The edge attribute to use as weight (default: 'length').
        name (str): The name of the attribute to store the closeness centrality.
        distance (str, optional): The edge attribute to use as distance metric.
                                 If None, uses weight.
    
    Returns:
        nx.MultiDiGraph: The graph with closeness centrality added as node attributes.
    
    Raises:
        ValueError: If radius is negative or graph is empty.
    """
    if radius < 0:
        raise ValueError("Radius must be non-negative")
    
    if len(G.nodes) == 0:
        raise ValueError("Graph must contain at least one node")
    
    if distance is None:
        distance = weight
    
    # Convert to undirected graph for distance calculation if needed
    G_undirected = G.to_undirected() if G.is_directed() else G
    
    closeness_dict = {}
    
    # Add progress bar for closeness centrality calculation
    for node in tqdm(G.nodes(), desc="Calculating closeness centrality", unit="nodes"):
        # Find all nodes within radius
        distances = nx.single_source_dijkstra_path_length(
            G_undirected, node, cutoff=radius, weight=distance
        )
        
        # Remove the node itself from distances
        if node in distances:
            del distances[node]
        
        # Calculate closeness: 1 / (sum of distances) if reachable nodes exist
        if len(distances) > 0:
            total_distance = sum(distances.values())
            closeness_dict[node] = len(distances) / total_distance if total_distance > 0 else 0.0
        else:
            closeness_dict[node] = 0.0
    
    nx.set_node_attributes(G, closeness_dict, name)
    return G


def betweenness_centrality_radius(
    G: nx.MultiDiGraph,
    radius: float,
    weight: str = 'length',
    name: str = 'betweenness_radius',
    distance: Optional[str] = None,
    normalized: bool = True
) -> nx.MultiDiGraph:
    """
    Calculate betweenness centrality within a specified radius for each node.
    
    Betweenness centrality measures how often a node lies on the shortest path
    between other nodes within the radius. Higher values indicate more important
    nodes for connectivity.
    
    Args:
        G (nx.MultiDiGraph): The input graph.
        radius (float): The radius in meters (or units of weight attribute) 
                       to consider for betweenness calculation.
        weight (str): The edge attribute to use as weight (default: 'length').
        name (str): The name of the attribute to store the betweenness centrality.
        distance (str, optional): The edge attribute to use as distance metric.
                                 If None, uses weight.
        normalized (bool): If True, normalize the betweenness values.
    
    Returns:
        nx.MultiDiGraph: The graph with betweenness centrality added as node attributes.
    
    Raises:
        ValueError: If radius is negative or graph is empty.
    """
    if radius < 0:
        raise ValueError("Radius must be non-negative")
    
    if len(G.nodes) == 0:
        raise ValueError("Graph must contain at least one node")
    
    if distance is None:
        distance = weight
    
    # Convert to undirected graph for distance calculation if needed
    G_undirected = G.to_undirected() if G.is_directed() else G
    
    betweenness_dict = {}
    
    # Add progress bar for betweenness centrality calculation
    for node in tqdm(G.nodes(), desc="Calculating betweenness centrality", unit="nodes"):
        # Find all nodes within radius
        try:
            distances = nx.single_source_dijkstra_path_length(
                G_undirected, node, cutoff=radius, weight=distance
            )
        except nx.NetworkXNoPath:
            betweenness_dict[node] = 0.0
            continue
        
        # Remove the node itself
        if node in distances:
            del distances[node]
        
        # Get subgraph of nodes within radius
        nodes_in_radius = set(distances.keys())
        nodes_in_radius.add(node)
        
        if len(nodes_in_radius) < 2:
            betweenness_dict[node] = 0.0
            continue
        
        # Create subgraph
        subgraph_nodes = [n for n in G.nodes() if n in nodes_in_radius]
        subgraph = G_undirected.subgraph(subgraph_nodes)
        
        # Calculate betweenness for this subgraph
        try:
            subgraph_betweenness = nx.betweenness_centrality(
                subgraph, weight=distance, normalized=normalized
            )
            betweenness_dict[node] = subgraph_betweenness.get(node, 0.0)
        except (nx.NetworkXError, ZeroDivisionError):
            betweenness_dict[node] = 0.0
    
    nx.set_node_attributes(G, betweenness_dict, name)
    return G


def reachability_radius(
    G: nx.MultiDiGraph,
    radius: float,
    weight: str = 'length',
    name: str = 'reachability',
    distance: Optional[str] = None
) -> nx.MultiDiGraph:
    """
    Calculate the number of reachable nodes within a specified radius.
    
    Reachability measures how many nodes can be reached from each node within
    the given radius. This is a simple but effective accessibility metric.
    
    Args:
        G (nx.MultiDiGraph): The input graph.
        radius (float): The radius in meters (or units of weight attribute).
        weight (str): The edge attribute to use as weight (default: 'length').
        name (str): The name of the attribute to store the reachability.
        distance (str, optional): The edge attribute to use as distance metric.
                                 If None, uses weight.
    
    Returns:
        nx.MultiDiGraph: The graph with reachability added as node attributes.
    """
    if radius < 0:
        raise ValueError("Radius must be non-negative")
    
    if distance is None:
        distance = weight
    
    G_undirected = G.to_undirected() if G.is_directed() else G
    
    reachability_dict = {}
    
    # Add progress bar for reachability calculation
    for node in tqdm(G.nodes(), desc="Calculating reachability", unit="nodes"):
        try:
            distances = nx.single_source_dijkstra_path_length(
                G_undirected, node, cutoff=radius, weight=distance
            )
            # Count reachable nodes (excluding self)
            reachability_dict[node] = len(distances) - (1 if node in distances else 0)
        except nx.NetworkXNoPath:
            reachability_dict[node] = 0
    
    nx.set_node_attributes(G, reachability_dict, name)
    return G


def local_efficiency_radius(
    G: nx.MultiDiGraph,
    radius: float,
    weight: str = 'length',
    name: str = 'local_efficiency',
    distance: Optional[str] = None
) -> nx.MultiDiGraph:
    """
    Calculate local efficiency within a specified radius.
    
    Local efficiency measures the average efficiency of information transfer
    in the subgraph around each node. It's related to clustering but considers
    path lengths.
    
    Args:
        G (nx.MultiDiGraph): The input graph.
        radius (float): The radius in meters (or units of weight attribute).
        weight (str): The edge attribute to use as weight (default: 'length').
        name (str): The name of the attribute to store the local efficiency.
        distance (str, optional): The edge attribute to use as distance metric.
                                 If None, uses weight.
    
    Returns:
        nx.MultiDiGraph: The graph with local efficiency added as node attributes.
    """
    if radius < 0:
        raise ValueError("Radius must be non-negative")
    
    if distance is None:
        distance = weight
    
    G_undirected = G.to_undirected() if G.is_directed() else G
    
    efficiency_dict = {}
    
    # Add progress bar for local efficiency calculation
    for node in tqdm(G.nodes(), desc="Calculating local efficiency", unit="nodes"):
        try:
            # Find nodes within radius
            distances = nx.single_source_dijkstra_path_length(
                G_undirected, node, cutoff=radius, weight=distance
            )
            
            nodes_in_radius = set(distances.keys())
            nodes_in_radius.add(node)
            
            if len(nodes_in_radius) < 2:
                efficiency_dict[node] = 0.0
                continue
            
            # Create subgraph
            subgraph_nodes = [n for n in G.nodes() if n in nodes_in_radius]
            subgraph = G_undirected.subgraph(subgraph_nodes)
            
            # Calculate local efficiency
            if len(subgraph.nodes()) < 2:
                efficiency_dict[node] = 0.0
                continue
            
            # Calculate efficiency: average of 1/distance for all pairs
            total_efficiency = 0.0
            pair_count = 0
            
            for u in subgraph.nodes():
                for v in subgraph.nodes():
                    if u != v:
                        try:
                            path_length = nx.shortest_path_length(
                                subgraph, u, v, weight=distance
                            )
                            if path_length > 0:
                                total_efficiency += 1.0 / path_length
                                pair_count += 1
                        except (nx.NetworkXNoPath, nx.NetworkXError):
                            pass
            
            efficiency_dict[node] = (
                total_efficiency / pair_count if pair_count > 0 else 0.0
            )
        except (nx.NetworkXNoPath, nx.NetworkXError):
            efficiency_dict[node] = 0.0
    
    nx.set_node_attributes(G, efficiency_dict, name)
    return G


def clustering_coefficient_radius(
    G: nx.MultiDiGraph,
    radius: float,
    weight: str = 'length',
    name: str = 'clustering_radius',
    distance: Optional[str] = None
) -> nx.MultiDiGraph:
    """
    Calculate clustering coefficient within a specified radius.
    
    Clustering coefficient measures the degree to which nodes in a neighborhood
    tend to cluster together. Higher values indicate more interconnected neighborhoods.
    
    Args:
        G (nx.MultiDiGraph): The input graph.
        radius (float): The radius in meters (or units of weight attribute).
        weight (str): The edge attribute to use as weight (default: 'length').
        name (str): The name of the attribute to store the clustering coefficient.
        distance (str, optional): The edge attribute to use as distance metric.
                                 If None, uses weight.
    
    Returns:
        nx.MultiDiGraph: The graph with clustering coefficient added as node attributes.
    """
    if radius < 0:
        raise ValueError("Radius must be non-negative")
    
    if distance is None:
        distance = weight
    
    # Convert to undirected graph and ensure it's not a multigraph for clustering calculation
    if G.is_directed():
        G_undirected = G.to_undirected()
    else:
        G_undirected = G
    
    # If it's a multigraph, convert to simple graph for clustering coefficient calculation
    if G_undirected.is_multigraph():
        G_undirected = nx.Graph(G_undirected)
    
    clustering_dict = {}
    
    # Add progress bar for clustering coefficient calculation
    for node in tqdm(G.nodes(), desc="Calculating clustering coefficient", unit="nodes"):
        try:
            # Find nodes within radius
            distances = nx.single_source_dijkstra_path_length(
                G_undirected, node, cutoff=radius, weight=distance
            )
            
            nodes_in_radius = set(distances.keys())
            nodes_in_radius.add(node)
            
            if len(nodes_in_radius) < 2:
                clustering_dict[node] = 0.0
                continue
            
            # Create subgraph
            subgraph_nodes = [n for n in G.nodes() if n in nodes_in_radius]
            subgraph = G_undirected.subgraph(subgraph_nodes)
            
            # Calculate clustering coefficient for the subgraph
            if len(subgraph.nodes()) < 2:
                clustering_dict[node] = 0.0
                continue
            
            try:
                clustering_coeff = nx.clustering(subgraph, node)
                clustering_dict[node] = clustering_coeff
            except (nx.NetworkXError, KeyError):
                clustering_dict[node] = 0.0
        except nx.NetworkXNoPath:
            clustering_dict[node] = 0.0
    
    nx.set_node_attributes(G, clustering_dict, name)
    return G


def calculate_accessibility_metrics(
    G: nx.MultiDiGraph,
    radius: float,
    metrics: Optional[List[str]] = None,
    weight: str = 'length',
    distance: Optional[str] = None
) -> nx.MultiDiGraph:
    """
    Calculate multiple accessibility metrics within a specified radius.
    
    This is a convenience function that calculates multiple metrics at once.
    
    Args:
        G (nx.MultiDiGraph): The input graph.
        radius (float): The radius in meters (or units of weight attribute).
        metrics (List[str], optional): List of metrics to calculate. Options:
            - 'closeness': Closeness centrality
            - 'betweenness': Betweenness centrality
            - 'reachability': Number of reachable nodes
            - 'local_efficiency': Local efficiency
            - 'clustering': Clustering coefficient
            If None, calculates all metrics.
        weight (str): The edge attribute to use as weight (default: 'length').
        distance (str, optional): The edge attribute to use as distance metric.
                                 If None, uses weight.
    
    Returns:
        nx.MultiDiGraph: The graph with all requested metrics added as node attributes.
    """
    if metrics is None:
        metrics = ['closeness', 'betweenness', 'reachability', 
                   'local_efficiency', 'clustering']
    
    available_metrics = {
        'closeness': closeness_centrality_radius,
        'betweenness': betweenness_centrality_radius,
        'reachability': reachability_radius,
        'local_efficiency': local_efficiency_radius,
        'clustering': clustering_coefficient_radius
    }
    
    for metric in metrics:
        if metric not in available_metrics:
            warnings.warn(f"Unknown metric '{metric}' skipped. Available: {list(available_metrics.keys())}")
            continue
        
        G = available_metrics[metric](
            G, radius=radius, weight=weight, distance=distance
        )
    
    return G

