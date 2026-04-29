import osmnx as ox
import networkx as nx
import momepy
import geopandas as gpd
import os
import time
import hashlib
from typing import Union, Dict, Optional

# Default cache directory
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".urbancode", "cache")

def _get_cache_path(place: str, network_type: str, output_type: str) -> str:
    """Generate cache file path based on place, network_type, and output_type."""
    # Create cache directory if it doesn't exist
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    # Create a hash of the place and network_type for filename
    cache_key = f"{place}_{network_type}_{output_type}"
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
    
    if output_type.lower() == 'graph':
        return os.path.join(CACHE_DIR, f"{cache_hash}.graphml")
    else:
        return os.path.join(CACHE_DIR, f"{cache_hash}.gpkg")

def download_network(
    output_type: str, 
    network_type: str, 
    place: str,
    use_cache: bool = True,
    cache_dir: Optional[str] = None
) -> Union[nx.MultiDiGraph, gpd.GeoDataFrame]:
    """
    Load a street network from OpenStreetMap with progress indication and caching support.
    
    This function supports downloading networks for various cities worldwide. Common place formats:
    - City name: "Singapore", "New York, USA", "London, UK"
    - City, Country: "Paris, France", "Tokyo, Japan"
    - Address: "Manhattan, New York, USA"
    
    Args:
        output_type (str): The type of output ('graph' or 'gdf'). This is a required parameter.
        network_type (str): The type of network to download ('drive', 'walk', 'bike', 'all', etc.). 
                           This is a required parameter.
        place (str): The place to download the network for. Can be city name, city with country,
                    or more specific address.
        use_cache (bool): Whether to use cached network if available (default: True).
        cache_dir (str, optional): Custom cache directory. If None, uses default cache location.
    
    Returns:
        Union[nx.MultiDiGraph, gpd.GeoDataFrame]: A NetworkX graph or GeoDataFrame representing 
                                                 the street network.
    
    Examples:
        >>> G = download_network('graph', 'drive', 'Singapore')
        >>> G = download_network('graph', 'walk', 'New York, USA')
        >>> gdf = download_network('gdf', 'bike', 'London, UK')
    """
    if output_type.lower() not in ['graph', 'gdf']:
        raise ValueError("The first parameter, output_type, must be either 'graph' or 'gdf'")
    
    if not network_type:
        raise ValueError("The second parameter, network_type, must be specified")
    
    if not place:
        raise ValueError("The place parameter must be specified")
    
    # Set cache directory
    global CACHE_DIR
    if cache_dir:
        CACHE_DIR = cache_dir
        os.makedirs(CACHE_DIR, exist_ok=True)
    
    # Check cache first
    if use_cache:
        cache_path = _get_cache_path(place, network_type, output_type)
        if os.path.exists(cache_path):
            print(f"Loading cached network for {place} ({network_type})...")
            try:
                if output_type.lower() == 'graph':
                    G = ox.load_graphml(cache_path)
                    print(f"Loaded cached network with {len(G.nodes)} nodes and {len(G.edges)} edges.")
                    return G
                else:
                    gdf = gpd.read_file(cache_path)
                    print(f"Loaded cached GeoDataFrame with {len(gdf)} rows.")
                    return gdf
            except Exception as e:
                print(f"Warning: Failed to load cache ({e}). Downloading fresh network...")
    
    print(f"Starting to download the {network_type} network for {place}...")
    start_time = time.time()
    
    try:
        # Download the network
        G = ox.graph_from_place(place, network_type=network_type)
    except Exception as e:
        raise ValueError(
            f"Failed to download network for '{place}'. "
            f"Please check the place name format. Error: {str(e)}"
        )
    
    end_time = time.time()
    print(f"Download completed in {end_time - start_time:.2f} seconds.")
    print(f"Network graph has {len(G.nodes)} nodes and {len(G.edges)} edges.")
    
    # Save to cache
    if use_cache:
        cache_path = _get_cache_path(place, network_type, output_type)
        try:
            if output_type.lower() == 'graph':
                ox.save_graphml(G, cache_path)
            else:
                gdf_edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
                gdf_edges.to_file(cache_path, driver='GPKG')
            print(f"Network cached to {cache_path}")
        except Exception as e:
            print(f"Warning: Failed to save cache ({e})")
    
    if output_type.lower() == 'gdf':
        # Convert the graph to GeoDataFrame
        gdf_edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
        print(f"Converted to GeoDataFrame with {len(gdf_edges)} rows.")
        return gpd.GeoDataFrame(geometry=gdf_edges.geometry)
    else:
        return G

def save_network(data: Union[nx.MultiDiGraph, gpd.GeoDataFrame], filename: str):
    """
    Save the network to a file, auto-detecting the data format.
    
    Args:
    data (Union[nx.MultiDiGraph, gpd.GeoDataFrame]): The graph or GeoDataFrame to save.
    filename (str): The filename to save the data to.
    """
    # Auto-detect the data format
    if isinstance(data, nx.MultiDiGraph):
        # It's a graph
        _, file_extension = os.path.splitext(filename)
        if file_extension.lower() != '.graphml':
            raise ValueError("For graph data, the filename must have a .graphml extension")
        
        print(f"Saving graph to GraphML file: {filename}")
        ox.save_graphml(data, filename)
    
    elif isinstance(data, gpd.GeoDataFrame):
        # It's a GeoDataFrame
        _, file_extension = os.path.splitext(filename)
        if file_extension.lower() not in ['.gpkg', '.shp']:
            raise ValueError("For GeoDataFrame, the filename must have either .gpkg (GeoPackage) or .shp (Shapefile) extension")
        
        if file_extension.lower() == '.gpkg':
            print(f"Saving GeoDataFrame to GeoPackage: {filename}")
            data.to_file(filename, driver='GPKG')
        else:  # .shp
            print(f"Saving GeoDataFrame to Shapefile: {filename}")
            data.to_file(filename, driver='ESRI Shapefile')
    
    else:
        raise ValueError("Input data must be either a NetworkX MultiDiGraph or a GeoDataFrame")

    print(f"Data successfully saved to {filename}")

def load_saved_network(filename: str) -> Union[nx.MultiDiGraph, gpd.GeoDataFrame]:
    """
    Load a saved network from a file.
    
    Args:
    filename (str): The filename to load the data from.
    
    Returns:
    Union[nx.MultiDiGraph, gpd.GeoDataFrame]: The loaded graph or GeoDataFrame.
    
    Raises:
    ValueError: If the file format is not supported or the file doesn't exist.
    """
    if not os.path.exists(filename):
        raise ValueError(f"File not found: {filename}")

    _, file_extension = os.path.splitext(filename)
    
    if file_extension.lower() == '.graphml':
        print(f"Loading GraphML file: {filename}")
        return ox.load_graphml(filename)
    
    elif file_extension.lower() in ['.gpkg', '.shp']:
        print(f"Loading {'GeoPackage' if file_extension.lower() == '.gpkg' else 'Shapefile'}: {filename}")
        return gpd.read_file(filename)
    
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

def graph_to_gdf(G: nx.MultiDiGraph, element: str = 'edges') -> Union[gpd.GeoDataFrame, Dict[str, gpd.GeoDataFrame]]:
    """
    Convert the NetworkX graph to GeoDataFrame(s) of edges, nodes, or both using OSMnx.
    
    Args:
    G (nx.MultiDiGraph): The input graph.
    element (str): Which elements to convert. Options are 'edges' (default), 'nodes', or 'both'.
    
    Returns:
    Union[gpd.GeoDataFrame, Dict[str, gpd.GeoDataFrame]]: 
        If 'edges' or 'nodes': A GeoDataFrame containing edges or nodes respectively.
        If 'both': A dictionary containing 'nodes' and 'edges' GeoDataFrames.
    
    Raises:
    ValueError: If an invalid element type is specified.
    """
    if element.lower() not in ['edges', 'nodes', 'both']:
        raise ValueError("Invalid element type. Choose 'edges', 'nodes', or 'both'.")
    
    if element.lower() == 'both':
        nodes, edges = ox.graph_to_gdfs(G)
        return {'nodes': nodes, 'edges': edges}
    elif element.lower() == 'edges':
        return ox.graph_to_gdfs(G, nodes=False, edges=True)
    else:  # nodes
        return ox.graph_to_gdfs(G, nodes=True, edges=False)

def graph_from_gdf(gdf: Union[gpd.GeoDataFrame, Dict[str, gpd.GeoDataFrame]]) -> nx.MultiDiGraph:
    """
    Convert GeoDataFrame(s) to a NetworkX MultiDiGraph using momepy's primal approach.

    Args:
    gdf (Union[gpd.GeoDataFrame, Dict[str, gpd.GeoDataFrame]]): 
        Either a single GeoDataFrame of edges, or a dictionary containing 'edges' GeoDataFrame.
        If a dictionary is provided, only the 'edges' GeoDataFrame will be used.

    Returns:
    nx.MultiGraph: A NetworkX MultiGraph representing the primal graph.

    Raises:
    ValueError: If the input is not a GeoDataFrame or a dictionary with the 'edges' key.
    """
    if isinstance(gdf, dict):
        if 'edges' not in gdf:
            raise ValueError("The input dictionary must contain an 'edges' key.")
        edges_gdf = gdf['edges']
    elif isinstance(gdf, gpd.GeoDataFrame):
        edges_gdf = gdf
    else:
        raise ValueError("Input must be either a GeoDataFrame of edges or a dictionary with an 'edges' GeoDataFrame.")

    # Use momepy to convert the edges GeoDataFrame to a NetworkX graph
    G = momepy.gdf_to_nx(edges_gdf, approach="primal")

    return G