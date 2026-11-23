"""Server-side response cache checks; no model or MCP server is started."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from exp_core.adapters.server import ResponseCache, build_parser, is_cacheable


class CacheKey(unittest.TestCase):
    def test_key_is_stable_across_dict_ordering(self):
        left = ResponseCache.key("search", {"query": "x", "limit": 2})
        right = ResponseCache.key("search", {"limit": 2, "query": "x"})
        self.assertEqual(left, right)
        self.assertEqual(left, '["search",{"limit":2,"query":"x"}]')

    def test_name_and_arguments_both_scope_the_key(self):
        self.assertNotEqual(ResponseCache.key("read", {"id": 1}),
                            ResponseCache.key("read", {"id": 2}))
        self.assertNotEqual(ResponseCache.key("read", {"id": 1}),
                            ResponseCache.key("list", {"id": 1}))


class CacheBehaviour(unittest.TestCase):
    def test_miss_then_hit_without_touching_the_value_semantics(self):
        cache = ResponseCache(ttl_seconds=3600)
        payload = {"ok": True, "value": {"items": [1, 2]}}
        self.assertIsNone(cache.get("read", {"id": 1}, now=10))
        self.assertEqual(cache.stats, {"hits": 0, "misses": 1,
                                       "entries": 0, "ttl_seconds": 3600})
        cache.put("read", {"id": 1}, payload, now=10)
        self.assertEqual(cache.get("read", {"id": 1}, now=20), payload)
        self.assertEqual(cache.stats, {"hits": 1, "misses": 1,
                                       "entries": 1, "ttl_seconds": 3600})

    def test_different_arguments_do_not_share_an_entry(self):
        cache = ResponseCache(ttl_seconds=3600)
        cache.put("read", {"id": 1}, {"ok": True, "value": 1}, now=0)
        self.assertIsNone(cache.get("read", {"id": 2}, now=1))
        self.assertEqual(cache.get("read", {"id": 1}, now=1), {"ok": True, "value": 1})
        self.assertEqual(cache.stats["entries"], 1)

    def test_entry_expires_at_the_ttl_boundary_and_is_deleted(self):
        cache = ResponseCache(ttl_seconds=3600)
        cache.put("read", {}, {"ok": True, "value": 1}, now=100)
        self.assertEqual(cache.get("read", {}, now=3699), {"ok": True, "value": 1})
        self.assertIsNone(cache.get("read", {}, now=3700))
        self.assertEqual(cache.stats, {"hits": 1, "misses": 1,
                                       "entries": 0, "ttl_seconds": 3600})

    def test_hits_return_a_deep_copy(self):
        cache = ResponseCache(ttl_seconds=3600)
        cache.put("read", {}, {"ok": True, "value": {"items": [1]}}, now=0)
        cached = cache.get("read", {}, now=1)
        cached["value"]["items"].append(2)
        self.assertEqual(cache.get("read", {}, now=2)["value"], {"items": [1]})

    def test_zero_ttl_disables_storage(self):
        cache = ResponseCache()
        cache.put("read", {}, {"ok": True, "value": 1}, now=0)
        self.assertEqual(cache.stats, {"hits": 0, "misses": 0,
                                       "entries": 0, "ttl_seconds": 0})
        self.assertIsNone(cache.get("read", {}, now=0))


class Cacheability(unittest.TestCase):
    def test_successful_payload_is_cacheable(self):
        self.assertTrue(is_cacheable({"ok": True, "value": 1}))

    def test_failure_payload_is_not_cacheable(self):
        self.assertFalse(is_cacheable({"ok": False, "error": "boom"}))
        self.assertFalse(is_cacheable({"ok": False, "error_code": "injected_transient"}))
        self.assertFalse(is_cacheable({"error": "boom"}))
        self.assertFalse(is_cacheable(None))


class CommandLine(unittest.TestCase):
    def test_response_cache_ttl_defaults_to_disabled(self):
        args = build_parser().parse_args(["--port", "0"])
        self.assertEqual(args.response_cache_ttl, 0)

    def test_response_cache_ttl_parses_the_policy_value(self):
        args = build_parser().parse_args(["--port", "0", "--response-cache-ttl", str(3600)])
        self.assertEqual(args.response_cache_ttl, 3600)
        cache = ResponseCache(args.response_cache_ttl)
        self.assertEqual(cache.stats["ttl_seconds"], 3600)


if __name__ == "__main__":
    unittest.main()
