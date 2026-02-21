import unittest
import sys
import os

# Add paths to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from anylabeling.services.auto_labeling.lru_cache import LRUCache

class TestLRUCache(unittest.TestCase):
    def test_cache_put_get(self):
        cache = LRUCache(maxsize=2)
        cache.put("a", 1)
        cache.put("b", 2)
        self.assertEqual(cache.get("a"), 1)
        self.assertEqual(cache.get("b"), 2)

    def test_cache_eviction(self):
        cache = LRUCache(maxsize=2)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)  # Should evict "a"
        
        self.assertIsNone(cache.get("a"))
        self.assertEqual(cache.get("b"), 2)
        self.assertEqual(cache.get("c"), 3)

    def test_cache_move_to_end(self):
        cache = LRUCache(maxsize=2)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.get("a")     # Access "a", so "b" becomes the oldest
        cache.put("c", 3)  # Should evict "b"
        
        self.assertEqual(cache.get("a"), 1)
        self.assertIsNone(cache.get("b"))
        self.assertEqual(cache.get("c"), 3)

    def test_cache_find(self):
        cache = LRUCache(maxsize=2)
        cache.put("a", 1)
        self.assertTrue(cache.find("a"))
        self.assertFalse(cache.find("b"))

    def test_cache_clear(self):
        cache = LRUCache(maxsize=2)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.clear()
        self.assertIsNone(cache.get("a"))
        self.assertIsNone(cache.get("b"))
        self.assertFalse(cache.find("a"))

if __name__ == '__main__':
    unittest.main()
