import networkx as nx
import momepy

def closeness_centrality(G: nx.MultiDiGraph, weight: str = 'length', name: str = 'closeness') -> nx.MultiDiGraph:
    """
    Calculate closeness centrality for each node in the graph.
    
    Args:
        G (nx.MultiDiGraph): The input graph.
        weight (str): The edge attribute to use as weight.
        name (str): The name of the attribute to store the closeness centrality.
    
    Returns:
        nx.MultiDiGraph: The graph with closeness centrality added as node attributes.
    """
    return momepy.closeness_centrality(G, weight=weight, name=name)

def betweenness_centrality(G: nx.MultiDiGraph, weight: str = 'length', name: str = 'betweenness') -> nx.MultiDiGraph:
    """
    Calculate betweenness centrality for each node in the graph.
    
    Args:
        G (nx.MultiDiGraph): The input graph.
        weight (str): The edge attribute to use as weight.
        name (str): The name of the attribute to store the betweenness centrality.
    
    Returns:
        nx.MultiDiGraph: The graph with betweenness centrality added as node attributes.
    """
    return momepy.betweenness_centrality(G, weight=weight, name=name)

def straightness_centrality(G: nx.MultiDiGraph, weight: str = 'length', name: str = 'straightness') -> nx.MultiDiGraph:
    """
    Calculate straightness centrality for each node in the graph.
    
    Args:
        G (nx.MultiDiGraph): The input graph.
        weight (str): The edge attribute to use as weight.
        name (str): The name of the attribute to store the straightness centrality.
    
    Returns:
        nx.MultiDiGraph: The graph with straightness centrality added as node attributes.
    """
    return momepy.straightness_centrality(G, weight=weight, name=name)

def subgraph_centrality(G: nx.MultiDiGraph, radius: int = 5, distance: str = None, **kwargs) -> nx.MultiDiGraph:
    """
    Calculate various centrality measures for subgraphs around each node.
    
    Args:
        G (nx.MultiDiGraph): The input graph.
        radius (int): The radius of the subgraph to consider.
        distance (str): The edge attribute to use as distance.
        **kwargs: Additional arguments to pass to momepy.subgraph function.
    
    Returns:
        nx.MultiDiGraph: The graph with subgraph centrality measures added as node attributes.
    """
    return momepy.subgraph(G, radius=radius, distance=distance, **kwargs)