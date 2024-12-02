import networkx as nx
import osmnx as ox
import matplotlib.pyplot as plt
from typing import Union, Optional, Dict
import geopandas as gpd
import contextily as cx

def plot_network(
    network: Union[nx.Graph, gpd.GeoDataFrame, Dict[str, gpd.GeoDataFrame]],
    figsize: tuple = (10, 10),
    bgcolor: str = 'w',
    plot_nodes: bool = True,
    plot_edges: bool = True,
    node_color: Union[str, Dict[int, str]] = '#66ccff',
    node_size: int = 15,
    node_alpha: float = 1,
    node_feature: Optional[str] = None,
    node_cmap: str = 'viridis',
    edge_color: Union[str, Dict[tuple, str]] = '#999999',
    edge_linewidth: float = 1,
    edge_alpha: float = 1,
    edge_feature: Optional[str] = None,
    edge_cmap: str = 'viridis',
    show: bool = True,
    close: bool = False,
    save: bool = False,
    filepath: Optional[str] = None,
    dpi: int = 300,
    add_basemap: bool = False
) -> None:
    """
    Plot the network using various input formats and customization options.

    Args:
    network (Union[nx.Graph, gpd.GeoDataFrame, Dict[str, gpd.GeoDataFrame]]): 
        The input network as a NetworkX graph, GeoDataFrame, or a dictionary of GeoDataFrames.
    figsize (tuple): Figure size.
    bgcolor (str): Background color.
    plot_nodes (bool): Whether to plot nodes.
    plot_edges (bool): Whether to plot edges.
    node_color (Union[str, Dict[int, str]]): Color of the nodes or a dictionary mapping node IDs to colors.
    node_size (int): Size of the nodes.
    node_alpha (float): Alpha (transparency) of the nodes.
    node_feature (Optional[str]): Node feature to use for coloring.
    node_cmap (str): Colormap for node feature.
    edge_color (Union[str, Dict[tuple, str]]): Color of the edges or a dictionary mapping edge tuples to colors.
    edge_linewidth (float): Width of the edges.
    edge_alpha (float): Alpha (transparency) of the edges.
    edge_feature (Optional[str]): Edge feature to use for coloring.
    edge_cmap (str): Colormap for edge feature.
    show (bool): Whether to show the plot.
    close (bool): Whether to close the plot after showing.
    save (bool): Whether to save the plot to a file.
    filepath (Optional[str]): Path to save the file (if save is True).
    dpi (int): DPI for saving the figure.
    add_basemap (bool): Whether to add a basemap to the plot.
    """
    # Convert input to GeoDataFrames
    if isinstance(network, nx.Graph):
        gdf_nodes, gdf_edges = ox.graph_to_gdfs(network)
    elif isinstance(network, gpd.GeoDataFrame):
        if 'geometry' in network.columns and network.geometry.type.isin(['Point', 'MultiPoint']).all():
            gdf_nodes = network
            gdf_edges = None
        else:
            gdf_edges = network
            gdf_nodes = None
    elif isinstance(network, dict):
        gdf_nodes = network.get('nodes')
        gdf_edges = network.get('edges')
    else:
        raise ValueError("Invalid input type. Expected NetworkX graph, GeoDataFrame, or dictionary of GeoDataFrames.")

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize, facecolor=bgcolor)

    # Plot edges
    if plot_edges and gdf_edges is not None:
        if edge_feature:
            edge_colors = gdf_edges[edge_feature]
            gdf_edges.plot(ax=ax, column=edge_feature, cmap=edge_cmap, linewidth=edge_linewidth, alpha=edge_alpha)
        else:
            gdf_edges.plot(ax=ax, color=edge_color, linewidth=edge_linewidth, alpha=edge_alpha)

    # Plot nodes
    if plot_nodes and gdf_nodes is not None:
        if node_feature:
            node_colors = gdf_nodes[node_feature]
            gdf_nodes.plot(ax=ax, column=node_feature, cmap=node_cmap, markersize=node_size, alpha=node_alpha)
        else:
            gdf_nodes.plot(ax=ax, color=node_color, markersize=node_size, alpha=node_alpha)

    # Add basemap if requested
    if add_basemap:
        cx.add_basemap(ax, source=cx.providers.OpenStreetMap.Mapnik)

    # Remove axis
    ax.axis('off')

    # Save, show, and close
    if save:
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight', format='png', facecolor=fig.get_facecolor(), edgecolor='none')
    if show:
        plt.show()
    if close:
        plt.close()


def plot_graph_route(G: nx.MultiDiGraph, 
                     route: list,
                     figsize: tuple = (10, 10),
                     bgcolor: str = 'w',
                     node_color: str = '#999999',
                     node_size: int = 15,
                     edge_color: str = '#999999',
                     route_color: str = '#66ccff',
                     route_linewidth: float = 4,
                     route_alpha: float = 0.5,
                     show: bool = True,
                     close: bool = False,
                     save: bool = False,
                     filepath: Optional[str] = None,
                     dpi: int = 300) -> None:
    """
    Plot a route on the graph using OSMnx.
    
    Args:
        G (nx.MultiDiGraph): The input graph.
        route (list): List of nodes representing the route.
        figsize (tuple): Figure size.
        bgcolor (str): Background color.
        node_color (str): Color of the nodes.
        node_size (int): Size of the nodes.
        edge_color (str): Color of the edges.
        route_color (str): Color of the route.
        route_linewidth (float): Width of the route.
        route_alpha (float): Alpha (transparency) of the route.
        show (bool): Whether to show the plot.
        close (bool): Whether to close the plot after showing.
        save (bool): Whether to save the plot to a file.
        filepath (Optional[str]): Path to save the file (if save is True).
        dpi (int): DPI for saving the figure.
    """
    ox.plot_graph_route(G, route, figsize=figsize, bgcolor=bgcolor,
                        node_color=node_color, node_size=node_size,
                        edge_color=edge_color, route_color=route_color,
                        route_linewidth=route_linewidth, route_alpha=route_alpha,
                        show=show, close=close, save=save, filepath=filepath, dpi=dpi)

def plot_figure_ground(G: nx.MultiDiGraph,
                       figsize: tuple = (10, 10),
                       edge_color: str = '0.2',
                       bgcolor: str = 'w',
                       dpi: int = 300,
                       show: bool = True,
                       save: bool = False,
                       filepath: Optional[str] = None) -> None:
    """
    Plot a figure-ground diagram of the street network.
    
    Args:
        G (nx.MultiDiGraph): The input graph.
        figsize (tuple): Figure size.
        edge_color (str): Color of the edges.
        bgcolor (str): Background color.
        dpi (int): DPI for saving the figure.
        show (bool): Whether to show the plot.
        save (bool): Whether to save the plot to a file.
        filepath (Optional[str]): Path to save the file (if save is True).
    """
    ox.plot_figure_ground(G, figsize=figsize, edge_color=edge_color,
                          bgcolor=bgcolor, dpi=dpi, show=show,
                          save=save, filepath=filepath)

def plot_centrality(G: nx.MultiDiGraph,
                    centrality_metric: str,
                    figsize: tuple = (10, 10),
                    cmap: str = 'viridis',
                    show: bool = True,
                    save: bool = False,
                    filepath: Optional[str] = None,
                    dpi: int = 300) -> None:
    """
    Plot the graph with nodes colored by a centrality metric.
    
    Args:
        G (nx.MultiDiGraph): The input graph.
        centrality_metric (str): The name of the centrality metric to visualize.
        figsize (tuple): Figure size.
        cmap (str): Colormap for the centrality values.
        show (bool): Whether to show the plot.
        save (bool): Whether to save the plot to a file.
        filepath (Optional[str]): Path to save the file (if save is True).
        dpi (int): DPI for saving the figure.
    """
    nc = [G.nodes[node][centrality_metric] for node in G.nodes()]
    ns = [20 * G.nodes[node][centrality_metric] for node in G.nodes()]
    
    fig, ax = ox.plot_graph(G, figsize=figsize, node_color=nc, node_size=ns, 
                            edge_color='#999999', edge_linewidth=0.2, 
                            node_zorder=2, cmap=cmap, show=False)
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(nc), vmax=max(nc)))
    cbar = plt.colorbar(sm, ax=ax, shrink=0.7)
    cbar.set_label(centrality_metric.replace('_', ' ').title())
    
    if show:
        plt.show()
    if save:
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close()