import networkx as nx
import momepy

def node_degree(G: nx.MultiDiGraph, name: str = 'degree') -> nx.MultiDiGraph:
    """
    Calculate node degree for each node in the graph.
    
    Args:
        G (nx.MultiDiGraph): The input graph.
        name (str): The name of the attribute to store the degree.
    
    Returns:
        nx.MultiDiGraph: The graph with node degrees added as node attributes.
    """
    return momepy.node_degree(G, name=name)

def mean_node_degree(G: nx.MultiDiGraph, radius: int = 5, name: str = 'mean_node_degree') -> nx.MultiDiGraph:
    """
    Calculate mean node degree for a subgraph around each node.
    
    Args:
        G (nx.MultiDiGraph): The input graph.
        radius (int): The radius of the subgraph to consider.
        name (str): The name of the attribute to store the mean node degree.
    
    Returns:
        nx.MultiDiGraph: The graph with mean node degrees added as node attributes.
    """
    return momepy.mean_node_degree(G, radius=radius, name=name)

def cyclomatic(G: nx.MultiDiGraph, radius: int = 5, name: str = 'cyclomatic') -> nx.MultiDiGraph:
    """
    Calculate cyclomatic complexity for a subgraph around each node.
    
    Args:
        G (nx.MultiDiGraph): The input graph.
        radius (int): The radius of the subgraph to consider.
        name (str): The name of the attribute to store the cyclomatic complexity.
    
    Returns:
        nx.MultiDiGraph: The graph with cyclomatic complexity added as node attributes.
    """
    return momepy.cyclomatic(G, radius=radius, name=name)

def edge_node_ratio(G: nx.MultiDiGraph, radius: int = 5, name: str = 'edge_node_ratio') -> nx.MultiDiGraph:
    """
    Calculate edge/node ratio for a subgraph around each node.
    
    Args:
        G (nx.MultiDiGraph): The input graph.
        radius (int): The radius of the subgraph to consider.
        name (str): The name of the attribute to store the edge/node ratio.
    
    Returns:
        nx.MultiDiGraph: The graph with edge/node ratio added as node attributes.
    """
    return momepy.edge_node_ratio(G, radius=radius, name=name)

def meshedness(G: nx.MultiDiGraph, radius: int = 5, name: str = 'meshedness') -> nx.MultiDiGraph:
    """
    Calculate meshedness for a subgraph around each node.
    
    Args:
        G (nx.MultiDiGraph): The input graph.
        radius (int): The radius of the subgraph to consider.
        name (str): The name of the attribute to store the meshedness.
    
    Returns:
        nx.MultiDiGraph: The graph with meshedness added as node attributes.
    """
    return momepy.meshedness(G, radius=radius, name=name)

def cds_length(G: nx.MultiDiGraph, radius: int = 5, mode: str = 'sum', name: str = 'cds_length') -> nx.MultiDiGraph:
    """
    Calculate cul-de-sac length for a subgraph around each node.
    
    Args:
        G (nx.MultiDiGraph): The input graph.
        radius (int): The radius of the subgraph to consider.
        mode (str): The mode of calculation ('sum' or 'mean').
        name (str): The name of the attribute to store the cul-de-sac length.
    
    Returns:
        nx.MultiDiGraph: The graph with cul-de-sac length added as node attributes.
    """
    return momepy.cds_length(G, radius=radius, mode=mode, name=name)