Network Analysis
===============

The network module provides tools for analyzing urban street networks using OSMnx and NetworkX.

Network Extraction
-----------------

.. automodule:: urbancode.network
   :members:
   :undoc-members:
   :show-inheritance:

The network module provides functions for:

* Downloading street networks from OpenStreetMap
* Analyzing network topology
* Calculating network metrics
* Visualizing network structures

Network Metrics
--------------

The module includes various network metrics:

* Connectivity
* Centrality measures
* Network density
* Street segment characteristics
* Intersection analysis

Examples
--------

Here are some examples of using the network module:

.. code-block:: python

   from urbancode.network import extract_network

   # Download street network for a city
   G = extract_network('San Francisco, California')

   # Calculate basic network metrics
   metrics = calculate_network_metrics(G)

   # Analyze street segments
   segments = analyze_street_segments(G)

   # Visualize the network
   plot_network(G, metrics) 