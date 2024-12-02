import unittest
import sys
import os
import networkx as nx
import geopandas as gpd

# Add the parent directory of 'urbancode' to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import urbancode as uc

class TestNetwork(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures once for all test methods."""
        try:
            # Test graph download
            cls.G = uc.download_network(output_type='graph', network_type='drive', place='Singapore')
            # Test gdf download
            cls.gdf = uc.download_network('gdf', 'drive', 'Singapore')
            
            # Create temporary filenames for saving tests
            cls.filename_1 = './urbancode/network/tests/singapore_network.graphml'
            cls.filename_2 = './urbancode/network/tests/singapore_network.gpkg'
            cls.filename_3 = './urbancode/network/tests/singapore_network.shp'

            # Save networks for later tests
            uc.save_network(cls.G, cls.filename_1)
            uc.save_network(cls.gdf, cls.filename_2)
            uc.save_network(cls.gdf, cls.filename_3)

            print("Network downloaded and saved successfully.")
        except Exception as e:
            raise unittest.SkipTest(f"setUpClass failed: {str(e)}")

    @classmethod
    def tearDownClass(cls):
        """Clean up after all test methods."""
        cls.cleanup_files()

    @classmethod
    def cleanup_files(cls):
        for filename in [cls.filename_1, cls.filename_2, cls.filename_3]:
            if os.path.exists(filename):
                os.remove(filename)
            # For shapefiles, also remove associated files
            if filename.endswith('.shp'):
                for ext in ['.shx', '.dbf', '.prj', 'cpg']:
                    associated_file = filename[:-4] + ext
                    if os.path.exists(associated_file):
                        os.remove(associated_file)

    def test_1_download_network(self):
        """Test the download_network function."""
        print("\nTesting download_network function...")
        self.assertIsInstance(self.G, nx.Graph)
        print(f"Network in Graph format loaded successfully. It has {len(self.G.nodes)} nodes and {len(self.G.edges)} edges.")

        self.assertIsInstance(self.gdf, gpd.GeoDataFrame)
        print(f"Network in GeoDataFrame(gdf) format loaded successfully. It has {len(self.gdf)} edge geometry objects as streets.")
        print(self.gdf.info())

        print("------download_network tests passed.------")

    def test_2_save_network(self):
        """Test the save_network function."""
        print("\nTesting save_network function...")
        self.assertTrue(os.path.exists(self.filename_1))
        self.assertTrue(os.path.exists(self.filename_2))
        self.assertTrue(os.path.exists(self.filename_3))

        print("------save_network tests passed.------")

    def test_3_load_saved_network(self):
        """Test the load_saved_network function."""
        print("\nTesting load_saved_network function...")
        try:
            # Load the saved network files (.graphml)
            print("Testing .graphml reading")
            loaded_G = uc.load_saved_network(self.filename_1)
            self.assertIsInstance(loaded_G, nx.Graph)
            self.assertEqual(len(self.G.nodes), len(loaded_G.nodes))
            self.assertEqual(len(self.G.edges), len(loaded_G.edges))
            print("Network in the .graphml format loaded successfully. Node and edge counts match the original graph.")

            print("Testing .gpkg reading")
            # Load the saved network files (.gpkg)
            loaded_gdf = uc.load_saved_network(self.filename_2)
            self.assertIsInstance(loaded_gdf, gpd.GeoDataFrame)
            self.assertEqual(len(self.gdf), len(loaded_gdf))
            self.assertEqual(len(self.G.edges), len(loaded_gdf))
            print("Network in the .gpkg format loaded successfully. Geometry counts match the original graph.")

            print("Testing .shp reading")
            # Load the saved network files (.shp)
            loaded_gdf = uc.load_saved_network(self.filename_3)
            self.assertIsInstance(loaded_gdf, gpd.GeoDataFrame)
            self.assertEqual(len(self.gdf), len(loaded_gdf))
            self.assertEqual(len(self.G.edges), len(loaded_gdf))
            print("Network in the .shp format loaded successfully. Geometry counts match the original graph.")

            print("------load_saved_network tests passed.------")

        except Exception as e:
            self.fail(f"load_saved_network raised an exception: {str(e)}")

    def test_4_graph_to_gdf(self):
        print("\nTesting graph_to_gdf function...")
        
        # Test edges conversion
        edges_gdf = uc.graph_to_gdf(self.G, element='edges')
        self.assertIsInstance(edges_gdf, gpd.GeoDataFrame)
        self.assertEqual(len(edges_gdf), len(self.G.edges))
        self.assertTrue('geometry' in edges_gdf.columns)
        print("Edges conversion successful.")

        # Test nodes conversion
        nodes_gdf = uc.graph_to_gdf(self.G, element='nodes')
        self.assertIsInstance(nodes_gdf, gpd.GeoDataFrame)
        self.assertEqual(len(nodes_gdf), len(self.G.nodes))
        self.assertTrue('geometry' in nodes_gdf.columns)
        print("Nodes conversion successful.")

        # Test both conversion
        both_gdf = uc.graph_to_gdf(self.G, element='both')
        self.assertIsInstance(both_gdf, dict)
        self.assertTrue('nodes' in both_gdf and 'edges' in both_gdf)
        self.assertEqual(len(both_gdf['nodes']), len(self.G.nodes))
        self.assertEqual(len(both_gdf['edges']), len(self.G.edges))
        print("Both nodes and edges conversion successful.")

        # Test invalid element type
        with self.assertRaises(ValueError):
            uc.graph_to_gdf(self.G, element='invalid')
        print("Invalid element type handled correctly.")

        print("------graph_to_gdf tests passed.------")

    def test_5_graph_from_gdf(self):
        print("\nTesting graph_from_gdf function...")

        # Convert graph to GeoDataFrame for testing
        edges_gdf = uc.graph_to_gdf(self.G, element='edges')

        # Test conversion from GeoDataFrame
        G_from_gdf = uc.graph_from_gdf(edges_gdf)
        self.assertIsInstance(G_from_gdf, nx.MultiGraph)
        self.assertEqual(len(G_from_gdf.edges), len(edges_gdf))
        print("Conversion from GeoDataFrame successful.")

        # Test conversion from dictionary
        gdf_dict = {'edges': edges_gdf}
        G_from_dict = uc.graph_from_gdf(gdf_dict)
        self.assertIsInstance(G_from_dict, nx.MultiGraph)
        self.assertEqual(len(G_from_dict.edges), len(edges_gdf))
        print("Conversion from dictionary successful.")

        # Test invalid input
        with self.assertRaises(ValueError):
            uc.graph_from_gdf({'invalid': edges_gdf})
        print("Invalid input handled correctly.")

        print("------graph_from_gdf tests passed.------")

if __name__ == '__main__':
    unittest.main(verbosity=2)

