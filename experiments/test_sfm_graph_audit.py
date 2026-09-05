import csv
from pathlib import Path
import tempfile
import unittest

from sfm_graph_audit import audit


class GraphAuditTest(unittest.TestCase):
    def run_graph(self, count, edges, unregistered=()):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (root/'images.csv').open('w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['image_id', 'name', 'registered'])
                writer.writerows((i, str(i), int(i not in unregistered)) for i in range(count))
            with (root/'pairs.csv').open('w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['image_id1', 'image_id2', 'active', 'composite_weight'])
                writer.writerows((a, b, 1, 1) for a, b in edges)
            return audit(root/'images.csv', root/'pairs.csv')

    def test_two_loops_joined_by_bridge(self):
        result = self.run_graph(6, [(0,1),(1,2),(2,0),(2,3),(3,4),(4,5),(5,3)])
        self.assertEqual(result['bridge_count'], 1)
        self.assertEqual([b['size'] for b in result['blocks']], [3,3])

    def test_long_chain_and_isolated_camera(self):
        result = self.run_graph(3001, [(i,i+1) for i in range(2999)])
        self.assertEqual(result['bridge_count'], 2999)
        self.assertEqual(result['connected_components'], 2)

    def test_unregistered_and_duplicate_edges(self):
        result = self.run_graph(4, [(0,1),(1,0),(1,2),(2,0),(2,3)], (3,))
        self.assertEqual(result['registered'], 3)
        self.assertEqual(result['bridge_count'], 0)


if __name__ == '__main__':
    unittest.main()
