import unittest

from stockfish_homonym.baselines.twap import PlatformTwapAgent
from stockfish_homonym.env.platform_execution_env import PlatformEnvConfig, PlatformExecutionEnv
from stockfish_homonym.eval.evaluator import EvalMetrics, Evaluator


def make_env(seed: int) -> PlatformExecutionEnv:
    config = PlatformEnvConfig(target_inventory=20, horizon=12, warmup_steps=5)
    return PlatformExecutionEnv(seed=seed, config=config)


class EvaluatorTest(unittest.TestCase):
    def test_evaluator_runs_and_returns_metrics(self) -> None:
        evaluator = Evaluator(env_factory=make_env, n_episodes=3, seed_offset=100)
        metrics = evaluator.evaluate(PlatformTwapAgent(target_inventory=20, horizon=12))

        self.assertIsInstance(metrics, EvalMetrics)
        self.assertGreaterEqual(metrics.fill_rate, 0.0)
        self.assertLessEqual(metrics.fill_rate, 1.0)
        self.assertAlmostEqual(sum(metrics.action_dist.values()), 1.0, places=6)
        self.assertEqual(set(metrics.regime_shortfall.keys()), {"calm", "normal", "stressed"})


if __name__ == "__main__":
    unittest.main()
