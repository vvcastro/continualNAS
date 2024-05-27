from ofa import OFASearchSpace
import unittest


class TestOFASearchSpace(unittest.TestCase):
    def setUp(self):
        self.ofa_search_space = OFASearchSpace(family="mobilenetv3")

    def test_sampling(self):
        samples = self.ofa_search_space.sample(n_samples=10)
        self.assertEqual(len(samples), 10)
        for sample in samples:
            self.assertTrue("depths" in sample)
            self.assertTrue("ksizes" in sample)
            self.assertTrue("widths" in sample)
            self.assertTrue("resolution" in sample)
            self.assertEqual(len(sample["depths"]), self.ofa_search_space.num_blocks)

    def test_encoding_decoding(self):
        samples = self.ofa_search_space.sample(n_samples=10)
        for sample in samples:
            encoded = self.ofa_search_space.encode(sample)
            decoded = self.ofa_search_space.decode(encoded)
            self.assertEqual(decoded, sample)

    def test_zero_padding(self):
        depths = [2, 3, 1]
        values = [3, 5, 3, 5, 7, 3]
        expected = [3, 5, 0, 0, 3, 5, 7, 0, 3, 0, 0, 0]
        padded = self.ofa_search_space.zero_padding(values, depths)
        self.assertEqual(padded, expected)


if __name__ == "__main__":
    unittest.main()
