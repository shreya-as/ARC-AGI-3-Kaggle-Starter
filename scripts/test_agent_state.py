"""Regression checks for observation hashing and clean evaluation memory."""
import unittest
import random
from collections import Counter
from types import SimpleNamespace
from arcengine import GameState
from replay_json import load_my_agent_class


class AgentStateTests(unittest.TestCase):
    def setUp(self):
        self.agent_class = load_my_agent_class()

    def test_small_object_change_survives_hashing(self):
        grid = [[0] * 64 for _ in range(64)]
        changed = [row[:] for row in grid]
        changed[2][2] = 1
        self.assertEqual(self.agent_class._coarse_pixels_hash(grid),
                         self.agent_class._coarse_pixels_hash(changed))
        self.assertNotEqual(self.agent_class._exact_pixels_hash(grid),
                            self.agent_class._exact_pixels_hash(changed))

    def test_grid_wrapper_and_dimensions(self):
        hash_grid = self.agent_class._exact_pixels_hash
        self.assertEqual(hash_grid([[1, 2]]), hash_grid([[[1, 2]]]))
        self.assertNotEqual(hash_grid([[1, 2]]), hash_grid([[1], [2]]))

    def test_exploration_keeps_coarse_abstraction(self):
        grid = [[0] * 64 for _ in range(64)]
        changed = [row[:] for row in grid]
        changed[2][2] = 1
        self.assertEqual(self.agent_class._pixels_hash(grid),
                         self.agent_class._pixels_hash(changed))

    def test_fresh_agent_defaults(self):
        self.assertFalse(self.agent_class.USE_PERSISTENT_MEMORY)
        self.assertFalse(self.agent_class.USE_ENGINE_PLANNER)
        self.assertFalse(getattr(self.agent_class, "OPENING_BOOK", {}))
        agent = object.__new__(self.agent_class)
        self.assertEqual(agent._opening_memory_path(), [])

    def test_observed_moves_are_remembered(self):
        agent = object.__new__(self.agent_class)
        agent.success_paths = {}
        agent.best_success_path = []
        agent.memory_success_paths = []
        path = [("ACTION1", {}), ("ACTION2", {})]
        agent._remember_success_path("observed-start", path)
        self.assertEqual(agent.success_paths["observed-start"], path)
        agent._remember_success_path("observed-start", path[:1])
        self.assertEqual(agent.success_paths["observed-start"], path[:1])
        self.assertNotIn("unseen-start", agent.success_paths)

    def model_agent(self):
        agent = object.__new__(self.agent_class)
        agent.observed_nodes = {}
        agent.observed_pending = None
        agent.observed_effects = {}
        agent.observed_mismatches = 0
        agent.rng = random.Random(0)
        return agent

    def model_frame(self, x=1, state=GameState.NOT_FINISHED, level=0):
        grid = [[0]*8 for _ in range(8)]
        grid[2][x] = 3
        return SimpleNamespace(frame=[grid], levels_completed=level,
                               available_actions=[1, 2], state=state)

    def test_observed_transition_and_prediction_mismatch(self):
        agent = self.model_agent()
        frame = self.model_frame()
        source = agent._observe_model(frame)
        agent._observed_action(frame, source)
        pending = agent.observed_pending
        target = agent._observe_model(self.model_frame(x=2))
        edge = agent.observed_nodes[source]["edges"][pending[1]]
        self.assertEqual(edge["outcomes"][target], 1)
        self.assertEqual(agent.observed_effects[pending[2]]["changed"], 1)
        self.assertEqual(agent.observed_effects[pending[2]][("translation", (3, ((0, 0),)), 1, 0)], 1)
        agent.observed_pending = (*pending[:3], target)
        agent._observe_model(self.model_frame(x=3))
        self.assertEqual(agent.observed_mismatches, 1)
        self.assertEqual(len(edge["outcomes"]), 2)
        self.assertIsNone(agent.observed_pending)

    def test_terminal_win_is_recorded_before_loop_exits(self):
        agent = self.model_agent()
        frame = self.model_frame()
        source = agent._observe_model(frame)
        agent._observed_action(frame, source)
        key = agent.observed_pending[1]
        self.assertTrue(agent.is_done([], self.model_frame(state=GameState.WIN, level=1)))
        self.assertEqual(agent.observed_nodes[source]["edges"][key]["progress"], 1)

    def test_death_is_not_progress(self):
        agent = self.model_agent()
        frame = self.model_frame()
        source = agent._observe_model(frame)
        agent._observed_action(frame, source)
        key = agent.observed_pending[1]
        agent._observe_model(self.model_frame(state=GameState.GAME_OVER))
        edge = agent.observed_nodes[source]["edges"][key]
        self.assertEqual(edge["dead"], 1)
        self.assertEqual(edge["progress"], 0)

    def test_frontier_route_uses_observed_edges(self):
        agent = self.model_agent()
        frame = self.model_frame()
        source = agent._observe_model(frame)
        target_frame = self.model_frame(x=2)
        target = agent._observe_model(target_frame)
        a = ("ACTION1", ())
        b = ("ACTION2", ())
        candidates = [(a, ("ACTION1",)), (b, ("ACTION2",))]
        agent.observed_nodes[source] = {"candidates": candidates, "edges": {
            a: {"outcomes": Counter({target: 1}), "dead": 0, "progress": 0},
            b: {"outcomes": Counter({source: 1}), "dead": 0, "progress": 0}}}
        agent.observed_nodes[target] = {"candidates": candidates, "edges": {}}
        action = agent._observed_action(frame, source)
        self.assertEqual(action.name, "ACTION1")
        self.assertEqual(agent.last_decision_mode, "observed-frontier")

    def test_disabled_memory_never_touches_disk(self):
        class ForbiddenPath:
            def exists(self):
                raise AssertionError("Disk memory accessed")
        agent = object.__new__(self.agent_class)
        agent.USE_PERSISTENT_MEMORY = False
        agent.MEMORY_FILE = ForbiddenPath()
        agent.memory_dirty = True
        self.assertEqual(agent._read_memory_file(), {})
        agent._save_persistent_memory()


if __name__ == "__main__":
    unittest.main()
