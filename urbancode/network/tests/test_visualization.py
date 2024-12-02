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

            print("Network downloaded successfully.")

        except Exception as e:
            raise unittest.SkipTest(f"setUpClass failed: {str(e)}")

    def test_1_plot_network(self):
        """Test the plot_network function."""
        print("\nTesting plot_network function...")
        uc.plot_network(self.G)
        uc.plot_network(self.gdf, add_basemap=True)

        print("------plot_network tests passed.------")

if __name__ == '__main__':
    unittest.main(verbosity=2)