import unittest

from stockfish_homonym.bridge.client import BridgeClient, BridgeConfig


class BridgeTest(unittest.TestCase):
    def test_bridge_reset_and_step(self) -> None:
        client = BridgeClient(BridgeConfig(target_inventory=20, horizon=8, warmup_steps=5))
        try:
            payload = client.reset(seed=123, calm_only=True)
            self.assertIn("obs", payload)
            self.assertEqual(len(payload["obs"]), 58)
            self.assertEqual(payload["info"]["inventory_remaining"], 20)

            payload = client.step(2)
            self.assertEqual(len(payload["obs"]), 58)
            self.assertIsInstance(payload["reward"], float)
            self.assertIn("current_price", payload["info"])
        finally:
            client.close()


if __name__ == "__main__":
    unittest.main()
