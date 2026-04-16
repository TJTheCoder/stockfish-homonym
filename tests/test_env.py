import unittest

import numpy as np

from stockfish_homonym.env.platform_execution_env import OBS_DIM, PlatformEnvConfig, PlatformExecutionEnv


def make_env() -> PlatformExecutionEnv:
    config = PlatformEnvConfig(target_inventory=20, horizon=10, warmup_steps=5)
    return PlatformExecutionEnv(seed=0, config=config)


class PlatformEnvTest(unittest.TestCase):
    def test_reset_returns_expected_shape(self) -> None:
        env = make_env()
        try:
            obs, info = env.reset(seed=0)
            self.assertEqual(obs.shape, (OBS_DIM,))
            self.assertTrue(np.all(np.isfinite(obs)))
            self.assertIn("target_symbol_index", info)
        finally:
            env.close()

    def test_step_returns_gymnasium_tuple(self) -> None:
        env = make_env()
        env.reset(seed=1)
        try:
            obs, reward, terminated, truncated, info = env.step(2)
            self.assertEqual(obs.shape, (OBS_DIM,))
            self.assertIsInstance(reward, float)
            self.assertIsInstance(terminated, bool)
            self.assertIsInstance(truncated, bool)
            self.assertIn("shortfall_so_far", info)
        finally:
            env.close()

    def test_aggressive_action_eventually_finishes_inventory(self) -> None:
        env = make_env()
        env.reset(seed=2)
        try:
            done = False
            for _ in range(20):
                _, _, terminated, truncated, _ = env.step(5)
                done = terminated or truncated
                if done:
                    break
            self.assertTrue(done)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
