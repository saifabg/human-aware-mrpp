"""Unit tests. Run from the repository root:  python -m unittest discover -s tests -v"""
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hamrpp import DEFAULT_CONFIG, map1, map2  # noqa: E402
from hamrpp.costs import turn_cost, social_terms  # noqa: E402
from hamrpp.planners import social_cost_dijkstra, plan_classic, plan_human_aware, hard_exclusion_nodes  # noqa: E402
from hamrpp.spacetime import plan_human_aware_st, plan_hard_exclusion_st  # noqa: E402
from hamrpp.cbs import plan_cbs, find_conflicts  # noqa: E402
from hamrpp.metrics import collision_pairs, turn_count, min_human_distance, robot_rows  # noqa: E402
from hamrpp.scenarios import illustrative, random_scenario, trial_seed  # noqa: E402
from hamrpp.stats import holm, mcnemar_exact_p, wilcoxon_p  # noqa: E402


class TestMaps(unittest.TestCase):
    def test_map1_counts(self):
        g = map1()
        self.assertEqual(g.n_nodes, 72)
        self.assertEqual(g.n_directed_edges, 296)
        self.assertTrue(g.is_connected())

    def test_map2_counts(self):
        g = map2()
        self.assertEqual(g.n_nodes, 133)
        self.assertEqual(g.n_directed_edges, 376)
        self.assertTrue(g.is_connected())

    def test_node_numbering(self):
        g = map1()   # nodes used by the illustrative scenario (Fig. 4)
        self.assertEqual(g.pos[1], (0.0, 0.0))
        self.assertAlmostEqual(g.pos[50][1], 5.6)
        self.assertAlmostEqual(g.pos[60][0], 4.0)


class TestCosts(unittest.TestCase):
    def test_turn_cost(self):
        self.assertEqual(turn_cost((0, 0), (1, 0), (2, 0)), 0.0)
        self.assertAlmostEqual(turn_cost((0, 0), (1, 0), (1, 1)), 1.0)
        self.assertAlmostEqual(turn_cost((0, 0), (1, 0), (2, 1)), 0.5)
        self.assertEqual(turn_cost(None, (1, 0), (2, 1)), 0.0)

    def test_uncertain_cost_reduces_to_eq2_eq3(self):
        from hamrpp.costs import social_terms_uncertain
        for pu, pv, hum in (((0, 0), (0.8, 0), [(0.3, 0.2)]), ((1, 1), (1.8, 1.8), [(1.5, 1.0), (0, 0)])):
            a = social_terms(pu, pv, hum)
            b = social_terms_uncertain(pu, pv, hum, 0.0)
            self.assertAlmostEqual(a[0], b[0]); self.assertAlmostEqual(a[1], b[1])
        s0 = social_terms_uncertain((0, 0), (0, 0), [(0, 0)], 0.0)[0]
        s1 = social_terms_uncertain((0, 0), (0, 0), [(0, 0)], 0.6)[0]
        self.assertLess(s1, s0)                          # peak falls as uncertainty grows

    def test_social_terms(self):
        s, c = social_terms((0, 0), (0, 0), [(0, 0)])
        self.assertAlmostEqual(s, 1.0)
        self.assertAlmostEqual(c, 3.0)      # three samples, each (1 - 0)^2
        s, c = social_terms((5, 5), (5, 6), [(0, 0)])
        self.assertLess(s, 1e-6)
        self.assertEqual(c, 0.0)


class TestPlanners(unittest.TestCase):
    def test_dijkstra_shortest_without_humans(self):
        g = map1()
        cfg = DEFAULT_CONFIG.with_(turn_weight=0.0, social_weight=0.0)
        p = social_cost_dijkstra(g, 1, 11, [], cfg)
        self.assertEqual(len(p), 11)      # straight bottom corridor

    def test_table1_reproduced(self):
        g = map1(); robots, humans = illustrative()
        paths, _ = plan_human_aware(g, robots, humans, DEFAULT_CONFIG)
        rows = robot_rows(g, paths, robots, humans, DEFAULT_CONFIG)
        got = [(r["path_len"], r["turns"], round(r["min_human_dist"], 2)) for r in rows]
        self.assertEqual(got, [(8, 0, 2.52), (13, 2, 1.0), (12, 6, 0.51)])

    def test_hard_exclusion_removes_nodes(self):
        g = map1()
        allowed = hard_exclusion_nodes(g, [(0.0, 0.0)], 1.0)
        self.assertNotIn(1, allowed)

    def test_spacetime_is_conflict_free(self):
        g = map1()
        for i in range(25):
            robots, humans = random_scenario(g, 5, 4, trial_seed(99, i))
            paths, _ = plan_human_aware_st(g, robots, humans)
            self.assertEqual(collision_pairs(paths), 0)
            self.assertEqual(find_conflicts(paths), [])
            for r in robots:
                p = paths[r["id"]]
                if p:
                    self.assertEqual(p[0], r["start"]); self.assertEqual(p[-1], r["goal"])
                    for a, b in zip(p, p[1:]):
                        self.assertTrue(a == b or b in dict(g.adj[a]))

    def test_cbs_solves_small_instances(self):
        g = map1()
        for i in range(10):
            robots, humans = random_scenario(g, 3, 4, trial_seed(7, i))
            paths, info = plan_cbs(g, robots, humans)
            if info["status"] == "solved":
                self.assertEqual(find_conflicts(paths), [])

    def test_pbs_and_ecbs_valid_and_conflict_free(self):
        from hamrpp.modern import plan_pbs, plan_ecbs
        g = map1()
        for fn in (plan_pbs, plan_ecbs):
            for i in range(8):
                robots, humans = random_scenario(g, 5, 4, trial_seed(11, i))
                paths, info = fn(g, robots, humans)
                for r in robots:
                    p = paths[r["id"]]
                    self.assertTrue(p, fn.__name__)
                    self.assertEqual((p[0], p[-1]), (r["start"], r["goal"]))
                    for a, b in zip(p, p[1:]):
                        self.assertTrue(a == b or b in dict(g.adj[a]))
                if info["status"] == "solved":
                    self.assertEqual(find_conflicts(paths), [], fn.__name__)

    def test_pibt_and_lacam_valid_and_conflict_free(self):
        from hamrpp.lacam import plan_pibt, plan_lacam
        g = map1()
        for fn in (plan_pibt, plan_lacam):
            for i in range(8):
                robots, humans = random_scenario(g, 6, 4, trial_seed(13, i))
                paths, info = fn(g, robots, humans)
                for r in robots:
                    p = paths[r["id"]]
                    if fn is plan_lacam:
                        self.assertTrue(p, "LaCAM is complete on these instances")
                    if p:
                        self.assertEqual((p[0], p[-1]), (r["start"], r["goal"]))
                        for a, b in zip(p, p[1:]):
                            self.assertTrue(a == b or b in dict(g.adj[a]))
                self.assertEqual(find_conflicts({k: v for k, v in paths.items() if v}), [], fn.__name__)

    def test_hard_exclusion_st_respects_personal_space(self):
        g = map1()
        robots, humans = random_scenario(g, 3, 4, trial_seed(5, 0))
        paths, _ = plan_hard_exclusion_st(g, robots, humans)
        for p in paths.values():
            if p:
                self.assertGreater(min_human_distance(g, p, humans), 1.0)


class TestDynamic(unittest.TestCase):
    def test_closed_loop_collision_free_and_deterministic(self):
        from hamrpp.dynamic import closed_loop
        g = map1()
        robots, _ = random_scenario(g, 4, 0, trial_seed(21, 0))
        for v in ("snapshot", "predictive", "hard_exclusion"):
            a = closed_loop(g, robots, 6, v, 123)
            b = closed_loop(g, robots, 6, v, 123)
            self.assertEqual(a["collision_events"], 0, v)
            self.assertEqual(a["intrusion_steps"], b["intrusion_steps"])     # same seed -> same run

    def test_constant_velocity_prediction(self):
        from hamrpp.dynamic import Crowd, frames_for
        import numpy as np
        c = Crowd(map1(), 2, np.random.default_rng(0)); c.step(); c.step()
        fr = frames_for("predictive", c, DEFAULT_CONFIG, 3)
        (x0, y0), (vx, vy) = c.observed()[0], c.velocities()[0]
        self.assertAlmostEqual(fr[2][0][0], x0 + 2 * vx); self.assertAlmostEqual(fr[2][0][1], y0 + 2 * vy)


class TestMetrics(unittest.TestCase):
    def test_collisions_wait_at_goal(self):
        self.assertEqual(collision_pairs({0: [1, 2, 3], 1: [5, 4, 3, 2]}), 1)  # 1 waits at 3, 2 arrives at t=2
        self.assertEqual(collision_pairs({0: [1, 2], 1: [3, 4]}), 0)

    def test_turn_count(self):
        g = map1()
        self.assertEqual(turn_count(g, [1, 2, 3], DEFAULT_CONFIG), 0)


class TestStats(unittest.TestCase):
    def test_holm(self):
        adj = holm([0.01, 0.04, 0.03])
        self.assertAlmostEqual(adj[0], 0.03)
        self.assertAlmostEqual(adj[2], 0.06)
        self.assertAlmostEqual(adj[1], 0.06)

    def test_mcnemar(self):
        p, (a, b) = mcnemar_exact_p([1, 1, 1, 1, 0], [0, 0, 0, 0, 0])
        self.assertEqual((a, b), (4, 0)); self.assertAlmostEqual(p, 0.125)

    def test_wilcoxon_identical(self):
        self.assertEqual(wilcoxon_p([1, 2, 3], [1, 2, 3])[0], 1.0)


if __name__ == "__main__":
    unittest.main()
