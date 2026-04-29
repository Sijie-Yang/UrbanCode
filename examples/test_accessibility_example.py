"""
Example script demonstrating accessibility calculation functionality.

This script shows how to:
1. Download a city network
2. Convert network to graph
3. Calculate accessibility metrics with radius
4. Use various network indicators
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import urbancode as uc
    import networkx as nx
    import matplotlib.pyplot as plt
    import numpy as np
    import geopandas as gpd
    print("Successfully imported urbancode module")
except ImportError as e:
    print(f"Warning: Could not import urbancode: {e}")
    print("Please install required dependencies: pip install -r requirements.txt")
    sys.exit(1)


def visualize_single_metric(G, metric_name, title, output_filename, cmap='viridis', node_size=3):
    """
    Helper function to visualize a single accessibility metric.
    
    Parameters:
    -----------
    G : networkx.Graph
        Graph with calculated metrics
    metric_name : str
        Name of the metric attribute in nodes
    title : str
        Title for the plot
    output_filename : str
        Output filename for saving the plot
    cmap : str
        Colormap for visualization
    node_size : int
        Size of nodes in the plot
    """
    try:
        print(f"\nCreating visualization for {metric_name}...")
        
        # Take a subset for better visualization performance
        nodes_subset = list(G.nodes())[:1000]  # First 1000 nodes
        G_small = G.subgraph(nodes_subset).copy()
        
        # Convert to GeoDataFrames
        gdf_nodes = uc.graph_to_gdf(G_small, element='nodes')
        gdf_edges = uc.graph_to_gdf(G_small, element='edges')
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(title, fontsize=16)
        
        # Plot 1: Network structure
        ax1 = axes[0]
        gdf_edges.plot(ax=ax1, color='gray', linewidth=0.5, alpha=0.7)
        gdf_nodes.plot(ax=ax1, color='red', markersize=1, alpha=0.8)
        ax1.set_title('Network Structure')
        ax1.axis('off')
        
        # Plot 2: Metric visualization
        ax2 = axes[1]
        gdf_edges.plot(ax=ax2, color='lightgray', linewidth=0.3, alpha=0.5)
        
        # Check if metric exists in nodes
        if metric_name in gdf_nodes.columns:
            gdf_nodes.plot(ax=ax2, column=metric_name, cmap=cmap, 
                          markersize=node_size, alpha=0.8, legend=True)
            ax2.set_title(f'{metric_name.replace("_", " ").title()}')
            
            # Print statistics
            values = gdf_nodes[metric_name].values
            print(f"\n{metric_name} Statistics:")
            print(f"  Mean: {np.mean(values):.6f}")
            print(f"  Max: {np.max(values):.6f}")
            print(f"  Min: {np.min(values):.6f}")
            print(f"  Std: {np.std(values):.6f}")
        else:
            ax2.text(0.5, 0.5, f'Metric "{metric_name}" not found', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Metric Not Available')
        
        ax2.axis('off')
        
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"Visualization saved as: {output_filename}")
        plt.close()  # Close to avoid memory issues
        
    except Exception as e:
        print(f"Error creating visualization: {e}")
        import traceback
        traceback.print_exc()


def visualize_multiple_metrics(G, metrics_info, title, output_filename):
    """
    Helper function to visualize multiple accessibility metrics in one plot.
    
    Parameters:
    -----------
    G : networkx.Graph
        Graph with calculated metrics
    metrics_info : list of tuples
        List of (metric_name, display_name, cmap) tuples
    title : str
        Title for the plot
    output_filename : str
        Output filename for saving the plot
    """
    try:
        print(f"\nCreating multi-metric visualization...")
        
        # Take a subset for better visualization performance
        nodes_subset = list(G.nodes())[:1000]  # First 1000 nodes
        G_small = G.subgraph(nodes_subset).copy()
        
        # Convert to GeoDataFrames
        gdf_nodes = uc.graph_to_gdf(G_small, element='nodes')
        gdf_edges = uc.graph_to_gdf(G_small, element='edges')
        
        # Calculate number of subplots needed
        n_metrics = len(metrics_info)
        n_cols = min(3, n_metrics)  # Max 3 columns
        n_rows = (n_metrics + n_cols - 1) // n_cols  # Ceiling division
        
        # Create figure with subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
        fig.suptitle(title, fontsize=16)
        
        # Handle single subplot case
        if n_metrics == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
        else:
            axes = axes.flatten()
        
        # Plot each metric
        for i, (metric_name, display_name, cmap) in enumerate(metrics_info):
            ax = axes[i]
            
            # Plot edges first
            gdf_edges.plot(ax=ax, color='lightgray', linewidth=0.3, alpha=0.5)
            
            # Check if metric exists in nodes
            if metric_name in gdf_nodes.columns:
                gdf_nodes.plot(ax=ax, column=metric_name, cmap=cmap, 
                              markersize=2, alpha=0.8, legend=True)
                ax.set_title(display_name)
                
                # Print statistics
                values = gdf_nodes[metric_name].values
                print(f"\n{metric_name} Statistics:")
                print(f"  Mean: {np.mean(values):.6f}")
                print(f"  Max: {np.max(values):.6f}")
                print(f"  Min: {np.min(values):.6f}")
            else:
                ax.text(0.5, 0.5, f'Metric "{metric_name}" not found', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{display_name} (Not Available)')
            
            ax.axis('off')
        
        # Hide unused subplots
        for i in range(n_metrics, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"Multi-metric visualization saved as: {output_filename}")
        plt.close()  # Close to avoid memory issues
        
    except Exception as e:
        print(f"Error creating multi-metric visualization: {e}")
        import traceback
        traceback.print_exc()


def example_1_download_and_convert():
    """Example 1: Download network and convert to graph."""
    print("\n" + "="*60)
    print("Example 1: Download network and convert to graph")
    print("="*60)
    
    try:
        # Download network for a city
        print("\nDownloading network for Singapore...")
        G = uc.download_network(
            output_type='graph',
            network_type='drive',
            place='Singapore',
            use_cache=True  # Use cache if available
        )
        
        print(f"Network downloaded: {len(G.nodes)} nodes, {len(G.edges)} edges")
        
        # Convert to GeoDataFrame
        print("\nConverting graph to GeoDataFrame...")
        gdf = uc.graph_to_gdf(G, element='edges')
        print(f"GeoDataFrame created: {len(gdf)} edges")
        
        # Convert back to graph
        print("\nConverting GeoDataFrame back to graph...")
        G_from_gdf = uc.graph_from_gdf(gdf)
        print(f"Graph recreated: {len(G_from_gdf.nodes)} nodes, {len(G_from_gdf.edges)} edges")
        
        return G
        
    except Exception as e:
        print(f"Error in example 1: {e}")
        return None


def example_2_closeness_centrality():
    """Example 2: Calculate closeness centrality with radius."""
    print("\n" + "="*60)
    print("Example 2: Calculate closeness centrality with radius")
    print("="*60)
    
    try:
        # Download or use existing graph
        G = uc.download_network(
            output_type='graph',
            network_type='drive',
            place='Singapore',
            use_cache=True
        )
        
        # Calculate closeness centrality with 500m radius
        radius = 500.0  # meters
        print(f"\nCalculating closeness centrality with radius={radius}m...")
        
        G_result = uc.closeness_centrality_radius(
            G, radius=radius, weight='length', name='closeness_500m'
        )
        
        # Get some statistics
        closeness_values = [
            G_result.nodes[node].get('closeness_500m', 0)
            for node in G_result.nodes()
        ]
        
        import numpy as np
        print(f"\nCloseness Centrality Statistics (radius={radius}m):")
        print(f"  Mean: {np.mean(closeness_values):.6f}")
        print(f"  Max: {np.max(closeness_values):.6f}")
        print(f"  Min: {np.min(closeness_values):.6f}")
        print(f"  Std: {np.std(closeness_values):.6f}")
        
        # Visualize closeness centrality
        visualize_single_metric(
            G_result, 
            'closeness_500m', 
            f'Closeness Centrality (radius={radius}m)',
            'closeness_centrality_visualization.png',
            cmap='viridis'
        )
        
        return G_result
        
    except Exception as e:
        print(f"Error in example 2: {e}")
        return None


def example_3_betweenness_centrality():
    """Example 3: Calculate betweenness centrality with radius."""
    print("\n" + "="*60)
    print("Example 3: Calculate betweenness centrality with radius")
    print("="*60)
    
    try:
        G = uc.download_network(
            output_type='graph',
            network_type='drive',
            place='Singapore',
            use_cache=True
        )
        
        radius = 500.0
        print(f"\nCalculating betweenness centrality with radius={radius}m...")
        
        G_result = uc.betweenness_centrality_radius(
            G, radius=radius, weight='length', name='betweenness_500m'
        )
        
        betweenness_values = [
            G_result.nodes[node].get('betweenness_500m', 0)
            for node in G_result.nodes()
        ]
        
        import numpy as np
        print(f"\nBetweenness Centrality Statistics (radius={radius}m):")
        print(f"  Mean: {np.mean(betweenness_values):.6f}")
        print(f"  Max: {np.max(betweenness_values):.6f}")
        print(f"  Min: {np.min(betweenness_values):.6f}")
        
        # Visualize betweenness centrality
        visualize_single_metric(
            G_result, 
            'betweenness_500m', 
            f'Betweenness Centrality (radius={radius}m)',
            'betweenness_centrality_visualization.png',
            cmap='plasma'
        )
        
        return G_result
        
    except Exception as e:
        print(f"Error in example 3: {e}")
        return None


def example_4_reachability():
    """Example 4: Calculate reachability."""
    print("\n" + "="*60)
    print("Example 4: Calculate reachability")
    print("="*60)
    
    try:
        G = uc.download_network(
            output_type='graph',
            network_type='walk',
            place='Singapore',
            use_cache=True
        )
        
        radius = 1000.0  # 1km walking distance
        print(f"\nCalculating reachability with radius={radius}m...")
        
        G_result = uc.reachability_radius(
            G, radius=radius, weight='length', name='reachability_1km'
        )
        
        reachability_values = [
            G_result.nodes[node].get('reachability_1km', 0)
            for node in G_result.nodes()
        ]
        
        import numpy as np
        print(f"\nReachability Statistics (radius={radius}m):")
        print(f"  Mean: {np.mean(reachability_values):.2f} nodes")
        print(f"  Max: {np.max(reachability_values)} nodes")
        print(f"  Min: {np.min(reachability_values)} nodes")
        
        # Visualize reachability
        visualize_single_metric(
            G_result, 
            'reachability_1km', 
            f'Reachability (radius={radius}m)',
            'reachability_visualization.png',
            cmap='coolwarm'
        )
        
        return G_result
        
    except Exception as e:
        print(f"Error in example 4: {e}")
        return None


def example_5_all_metrics():
    """Example 5: Calculate all accessibility metrics at once."""
    print("\n" + "="*60)
    print("Example 5: Calculate all accessibility metrics")
    print("="*60)
    
    try:
        G = uc.download_network(
            output_type='graph',
            network_type='drive',
            place='Singapore',
            use_cache=True
        )
        
        radius = 500.0
        print(f"\nCalculating all accessibility metrics with radius={radius}m...")
        
        G_result = uc.calculate_accessibility_metrics(
            G,
            radius=radius,
            metrics=['closeness', 'betweenness', 'reachability', 
                    'local_efficiency', 'clustering'],
            weight='length'
        )
        
        print("\nMetrics calculated:")
        print("  - Closeness centrality")
        print("  - Betweenness centrality")
        print("  - Reachability")
        print("  - Local efficiency")
        print("  - Clustering coefficient")
        
        # Show sample node with all metrics
        sample_node = list(G_result.nodes())[0]
        print(f"\nSample node {sample_node} metrics:")
        for attr in ['closeness_radius', 'betweenness_radius', 'reachability',
                     'local_efficiency', 'clustering_radius']:
            value = G_result.nodes[sample_node].get(attr, 'N/A')
            print(f"  {attr}: {value}")
        
        # Visualize all calculated metrics
        metrics_info = [
            ('closeness_radius', 'Closeness Centrality', 'viridis'),
            ('betweenness_radius', 'Betweenness Centrality', 'plasma'),
            ('reachability', 'Reachability', 'coolwarm'),
            ('local_efficiency', 'Local Efficiency', 'cividis'),
            ('clustering_radius', 'Clustering Coefficient', 'inferno')
        ]
        
        visualize_multiple_metrics(
            G_result,
            metrics_info,
            f'All Accessibility Metrics (radius={radius}m)',
            'all_metrics_visualization.png'
        )
        
        # Create a simple histogram of reachability values
        reachability_values = [
            G_result.nodes[node].get('reachability', 0)
            for node in G_result.nodes()
        ]
        
        plt.figure(figsize=(10, 6))
        plt.hist(reachability_values, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title(f'Distribution of Reachability (radius={radius}m)')
        plt.xlabel('Number of Reachable Nodes')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # Add statistics text
        mean_reach = np.mean(reachability_values)
        plt.axvline(mean_reach, color='red', linestyle='--', 
                   label=f'Mean: {mean_reach:.1f}')
        plt.legend()
        
        # Save the plot
        output_file = 'reachability_distribution.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\nReachability distribution saved as: {output_file}")
        plt.close()  # Close to avoid memory issues
        
        return G_result
        
    except Exception as e:
        print(f"Error in example 5: {e}")
        return None


def example_6_different_cities():
    """Example 6: Download networks for different cities."""
    print("\n" + "="*60)
    print("Example 6: Download networks for different cities")
    print("="*60)
    
    cities = [
        'Singapore',
        'New York, USA',
        'London, UK',
        'Tokyo, Japan',
        'Paris, France'
    ]
    
    print("\nSupported city formats:")
    for city in cities:
        print(f"  - {city}")
    
    print("\nNote: You can use any place name that OpenStreetMap recognizes.")
    print("Examples:")
    print("  - City name: 'Singapore'")
    print("  - City, Country: 'New York, USA'")
    print("  - Specific area: 'Manhattan, New York, USA'")


def example_7_visualization():
    """Example 7: Visualize accessibility metrics."""
    print("\n" + "="*60)
    print("Example 7: Visualizing accessibility metrics")
    print("="*60)
    
    try:
        # Download a small network for visualization
        G = uc.download_network(
            output_type='graph',
            network_type='drive',
            place='Singapore',
            use_cache=True
        )
        
        # Take a subset for better visualization performance
        nodes_subset = list(G.nodes())[:500]  # First 500 nodes
        G_small = G.subgraph(nodes_subset).copy()
        
        print(f"\nUsing subset of {len(G_small.nodes)} nodes for visualization...")
        
        # Calculate accessibility metrics
        radius = 1000.0
        print(f"Calculating accessibility metrics with radius={radius}m...")
        
        G_result = uc.calculate_accessibility_metrics(
            G_small, 
            radius=radius,
            metrics=['closeness', 'reachability'],
            weight='length'
        )
        
        # Convert to GeoDataFrames for visualization
        print("\nConverting to GeoDataFrames...")
        gdf_nodes = uc.graph_to_gdf(G_result, element='nodes')
        gdf_edges = uc.graph_to_gdf(G_result, element='edges')
        
        # Create visualization
        print("Creating visualizations...")
        
        # Plot 1: Network structure
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Accessibility Metrics Visualization', fontsize=16)
        
        # Basic network
        ax1 = axes[0, 0]
        gdf_edges.plot(ax=ax1, color='gray', linewidth=0.5, alpha=0.7)
        gdf_nodes.plot(ax=ax1, color='red', markersize=1, alpha=0.8)
        ax1.set_title('Network Structure')
        ax1.axis('off')
        
        # Closeness centrality
        ax2 = axes[0, 1]
        gdf_edges.plot(ax=ax2, color='lightgray', linewidth=0.3, alpha=0.5)
        gdf_nodes.plot(ax=ax2, column='closeness', cmap='viridis', 
                      markersize=3, alpha=0.8, legend=True)
        ax2.set_title('Closeness Centrality')
        ax2.axis('off')
        
        # Reachability
        ax3 = axes[1, 0]
        gdf_edges.plot(ax=ax3, color='lightgray', linewidth=0.3, alpha=0.5)
        gdf_nodes.plot(ax=ax3, column='reachability', cmap='plasma', 
                      markersize=3, alpha=0.8, legend=True)
        ax3.set_title('Reachability')
        ax3.axis('off')
        
        # Statistics comparison
        ax4 = axes[1, 1]
        closeness_values = gdf_nodes['closeness'].values
        reachability_values = gdf_nodes['reachability'].values
        
        ax4.scatter(closeness_values, reachability_values, alpha=0.6, s=10)
        ax4.set_xlabel('Closeness Centrality')
        ax4.set_ylabel('Reachability')
        ax4.set_title('Closeness vs Reachability')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        output_file = 'accessibility_visualization.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved as: {output_file}")
        
        # Show statistics
        print("\nAccessibility Statistics:")
        print(f"Closeness Centrality:")
        print(f"  Mean: {np.mean(closeness_values):.6f}")
        print(f"  Max: {np.max(closeness_values):.6f}")
        print(f"  Min: {np.min(closeness_values):.6f}")
        
        print(f"\nReachability:")
        print(f"  Mean: {np.mean(reachability_values):.2f} nodes")
        print(f"  Max: {np.max(reachability_values)} nodes")
        print(f"  Min: {np.min(reachability_values)} nodes")
        
        # Optional: Show the plot (comment out if running headless)
        # plt.show()
        
        return G_result
        
    except Exception as e:
        print(f"Error in visualization example: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':
    print("="*60)
    print("UrbanCode Accessibility Calculation Examples")
    print("="*60)
    
    # Run examples
    try:
        example_1_download_and_convert()
        example_2_closeness_centrality()
        example_3_betweenness_centrality()
        example_4_reachability()
        example_5_all_metrics()
        example_6_different_cities()
        example_7_visualization()
        
        print("\n" + "="*60)
        print("All examples completed!")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n\nExamples interrupted by user.")
    except Exception as e:
        print(f"\n\nError running examples: {e}")
        import traceback
        traceback.print_exc()

