"""
evac_env.py  --  Evacuation environment compliant with PettingZoo Parallel API

CTDE (Centralized Training, Decentralized Execution):
  Actor  : Uses local observations only (decentralized execution)
  Critic : Uses global state features (centralized training)

Per-agent random start / destination:
  Every reset(), each agent gets a random start node and a random destination
  shelter (chosen uniformly among evac_nodes, capacity is NOT considered when
  assigning destinations). Each agent also gets its own fixed walking speed
  for the episode.

Reward Design:
  Potential-based shaping (now per-agent, based on distance to THAT agent's
  own assigned destination shelter, not a global "nearest shelter" field)
    phi(node, agent) = -dist_to_own_target(node) / max_dist  in [-1, 0]
    r_shape = gamma_shape * phi(next) - phi(curr)   (dense guiding reward)
    r_safe  = +1.0  (terminal reward for successful arrival)

  Congestion penalty (thresholded)
    density      = n_agents_on_link / edge_area
    ratio        = density / DENSITY_MAX
    excess       = max(0, (ratio - CONGESTION_THRESHOLD) / (1 - CONGESTION_THRESHOLD))
    r_congestion = -CONGESTION_PENALTY * min(excess, 1.0)
    Below CONGESTION_THRESHOLD (as a fraction of DENSITY_MAX), the penalty is
    exactly zero -- lightly-used edges impose no avoidance pressure at all, so
    r_shape (distance-to-destination progress) is the only signal driving the
    choice among uncongested candidates. Above the threshold, the penalty
    ramps back up to the same max magnitude as before at density=DENSITY_MAX.
    This is meant to stop the policy from taking unnecessary detours around
    mild/negligible congestion, which was hurting arrival time/success rate
    relative to the shortest-path baseline despite scoring a better reward.
    Shelter capacity is NOT enforced (destinations are assigned ignoring
    capacity, so arrival always succeeds).

  Time penalty
    r_time = -TIME_PENALTY, applied every step an agent is still evacuating
    (including the terminal step, whether it arrives or times out). Small
    fixed penalty (same order of magnitude as r_shape/r_congestion, NOT the
    raw STEP_TIME in seconds) that discourages dawdling/looping in place
    beyond what the distance-based shaping alone penalizes.

Movement / speed model:
  Each agent has its own basic_speed ~ N(AGENT_SPEED_MEAN, AGENT_SPEED_STD),
  clipped to [AGENT_SPEED_MIN, AGENT_SPEED_MAX], sampled once per episode.
  Actual speed on an edge slows down with congestion (Greenshields-style
  linear model), using the PREVIOUS step's observed density on that edge:
    speed = basic_speed * max(1 - density/DENSITY_MAX, V_MIN_RATIO)
  MOVE_DIST per step = speed * STEP_TIME.

Artificial congestion (optional, off by default -- see __init__):
  When artificial_congestion=True, reset() randomly selects a fraction of
  edges (artificial_congestion_fraction) and gives each a permanent density
  FLOOR drawn from artificial_congestion_levels (fractions of DENSITY_MAX,
  e.g. 0.5 or 0.8). Effective density used everywhere (speed, reward,
  observation, global state) is max(real agent density, floor): it rises
  above the floor normally as agents pile on, but never drops below it even
  with zero agents present. Lets you stress-test whether the policy learns
  to route around persistently bad roads (e.g. permanent damage/debris)
  rather than only reacting to transient crowding. Re-randomized every
  reset() so the affected edges differ episode to episode.

  artificial_congestion_target_shortest_path=True switches edge selection
  from uniformly-random to "on the shortest-path baseline's actual route":
  the greedy action-0 path from the cluster center (or a random node) to
  every shelter. This specifically punishes the shortest-path baseline
  (whose fixed route is blocked) while a trained policy that can detour may
  do better -- useful for constructing scenarios where MAPPO should beat the
  baseline by design, rather than just matching it.

Observation Vector (dim = max_degree * 7 * 2 + 1), all features in [0, 1]:
  neighbor ordering, per agent, is sorted by ascending distance-to-THAT-
  AGENT'S-OWN-target-shelter (precomputed per shelter in __init__), so
  action index 0 always means "move toward my own destination" at every
  node, for every agent (regardless of which shelter it is heading to).

  Block A [0 : 7*max_degree]            : the max_degree edges selectable
                                           from the agent's CURRENT node.
  Block B [7*d : 14*d]                  : the max_degree edges incident to
                                           the agent's DESTINATION node
                                           (same distance-based ordering
                                           rule; static per shelter).
  Each edge block packs 7 features per edge, grouped by edge (not by
  feature type), in this order:
    0: edge_area / max_edge_area                       in [0,1]
    1: agent's own basic_speed / AGENT_SPEED_MAX        in [0,1]
    2: n_agents_on_edge / n_agents                      in [0,1]
    3: density (n_agents_on_edge / edge_area) / DENSITY_MAX   in [0,1]
    4: max(1 - density/DENSITY_MAX, V_MIN_RATIO)   (avg edge speed / BASE_SPEED)
    5: density / DENSITY_MAX                            in [0,1]  (duplicate
       of feature 3, kept as a separate explicit feature per spec)
    6: closeness_to_target = 1 - dist_to_target(neighbor)/global_max_dist,
       in [0,1] (1 = neighbor IS the destination, 0 = as far as possible).
       This is the old phi(next) signal, rescaled to [0,1] and shown
       PER CANDIDATE EDGE (unlike the hop-progress scalar below, which is
       identical across all edges at a given step and so cannot by itself
       tell the policy which of the current options is actually closer).
  Missing edges (node degree < max_degree) are zero-padded.

  [-1] (last element) : hop-based progress toward destination
        progress = 1 - remaining_hops / initial_hops_at_episode_start, in [0,1]
        (a single scalar summarizing overall journey completion so far;
        complements feature 6 above, which discriminates BETWEEN options)

Global State Vector (dim = 9, Critic input only):
  [0:6] the 6 edge features above, each averaged over ALL edges in the graph
        (feature 1 "own basic_speed" becomes the population mean speed)
  [6] fraction of edges with density > 80% of DENSITY_MAX
  [7] fraction of edges with density > 50% of DENSITY_MAX
  [8] fraction of agents still traveling (evacuating), in [0, 1]
"""

import numpy as np
from collections import defaultdict
from common import (BASE_SPEED, ROAD_WIDTH, compute_shelter_distances,
                     compute_distances_from_node)
from pettingzoo import ParallelEnv as _Base
from gymnasium import spaces

# No-op @profile decorator when not running under a profiler
import builtins
if not hasattr(builtins, 'profile'):
    builtins.profile = lambda f: f

# ── Reward / physics constants ─────────────────────────────────────────────────
GAMMA_SHAPE        = 0.99
CONGESTION_PENALTY = 5.00   # max per-step penalty for very congested links
CONGESTION_THRESHOLD = 0.5  # fraction of DENSITY_MAX below which congestion
                             # is ignored entirely (no avoidance pressure);
                             # penalty ramps from 0 at this threshold up to
                             # -CONGESTION_PENALTY at density == DENSITY_MAX
TIME_PENALTY       = 0.01   # small fixed penalty applied every step an agent
                             # is still evacuating, to discourage dawdling /
                             # taking unnecessarily long routes (same order of
                             # magnitude as r_shape/r_congestion, not raw STEP_TIME)
STEP_TIME          = 5.0    # seconds per simulation step

DENSITY_MAX        = 1.4    # jam density (agents/m^2); used for both the
                             # Greenshields speed model and density normalization
V_MIN_RATIO        = 0.1    # minimum speed fraction of an agent's basic_speed,
                             # even at jam density (prevents total gridlock)
AGENT_SPEED_MEAN   = 1.2    # per-agent basic walking speed distribution (m/s)
AGENT_SPEED_STD    = 0.2
AGENT_SPEED_MIN    = 0.6
AGENT_SPEED_MAX    = 1.8

# Agent status codes
STATUS_EVACUATING = np.int8(0)
STATUS_SAFE       = np.int8(1)
STATUS_FAILED     = np.int8(2)


class EvacuationEnv(_Base):
    metadata = {"render_modes": [], "name": "evacuation_kochi_v1"}

    def __init__(self, node_coords, adj, road_nodes, evac_nodes, evac_capacity,
                 node_list, node_to_idx, neighbor_lists, max_degree,
                 n_agents=500, max_steps=600, reward_dest=1.0,
                 cluster_start=False, cluster_radius_hops=6,
                 cluster_center_pool=None,
                 artificial_congestion=False,
                 artificial_congestion_levels=(0.5, 0.8),
                 artificial_congestion_fraction=0.05,
                 artificial_congestion_target_shortest_path=False,
                 artificial_congestion_sp_fraction=1.0):
        """
        cluster_start: if True, reset() picks one random node each episode as
            a "cluster center" and samples every agent's start position from
            nodes within `cluster_radius_hops` hops of it, instead of
            uniformly over the whole map. Useful for stress-testing
            congestion (e.g. visualize_congestion.py) — a whole neighborhood
            evacuating at once funnels through much narrower road capacity
            than agents spread uniformly across the city. Defaults to False
            so normal training behavior (uniform random start) is unchanged.
        cluster_radius_hops: hop-count radius (unweighted, via
            common.compute_distances_from_node) used to build the candidate
            start-node set when cluster_start=True.
        cluster_center_pool: optional list of node ids to restrict WHERE the
            cluster center itself may be chosen from (e.g. only nodes in the
            eastern part of the map). Defaults to None, meaning any
            non-shelter node is eligible.
        artificial_congestion: if True, reset() randomly picks a subset of
            (undirected) edges and gives each one a permanent density FLOOR
            (e.g. 50% or 80% of DENSITY_MAX), independent of how many agents
            are actually using it. Effective density on that edge is
            max(real agent density, floor): as agents pile onto it the
            density (and resulting slowdown/congestion penalty/observation
            features) rises above the floor exactly as normal, but once
            agents leave it drops back down to the floor, never to zero. Off
            by default so normal behavior is unchanged.
        artificial_congestion_levels: tuple of candidate floor levels, each a
            fraction of DENSITY_MAX (e.g. 0.5 = "always at least 50%
            congested"). One is picked at random per selected edge, per
            episode.
        artificial_congestion_fraction: fraction of the map's (undirected)
            edges that get an artificial floor each episode. Which edges are
            selected is re-randomized every reset(). Only used when
            artificial_congestion_target_shortest_path is False.
        artificial_congestion_target_shortest_path: if True, instead of
            picking edges uniformly at random, congestion is placed ON the
            route the greedy shortest-path (action-0) baseline would
            actually take -- from the current cluster center (or a random
            node if cluster_start is off) to every shelter. Use this to
            construct scenarios that specifically punish the shortest-path
            baseline (its one fixed route is blocked) and see whether a
            trained policy that can detour comes out ahead.
        artificial_congestion_sp_fraction: fraction of the identified
            shortest-path-route edges that actually get floored (1.0 =
            all of them). Only used when
            artificial_congestion_target_shortest_path is True.
        """
        self.node_coords    = node_coords
        self.adj            = adj
        self.road_nodes     = road_nodes
        self.evac_nodes     = evac_nodes
        self.evac_capacity  = evac_capacity
        self.node_list      = node_list
        self.node_to_idx    = node_to_idx
        self.neighbor_lists = neighbor_lists   # kept for reference (unused for movement now)
        self.max_degree     = max_degree
        self.n_nodes        = len(node_list)
        self.n_agents       = n_agents
        self.max_steps      = max_steps
        self.reward_dest    = reward_dest
        self.cluster_start        = cluster_start
        self.cluster_radius_hops  = cluster_radius_hops
        self.cluster_center_pool  = cluster_center_pool
        self.artificial_congestion          = artificial_congestion
        self.artificial_congestion_levels   = artificial_congestion_levels
        self.artificial_congestion_fraction = artificial_congestion_fraction
        self.artificial_congestion_target_shortest_path = artificial_congestion_target_shortest_path
        self.artificial_congestion_sp_fraction           = artificial_congestion_sp_fraction
        self._congestion_floor = {}   # (cidx, nidx) -> density floor; populated in reset()
        self._non_shelter   = [n for n in node_list if n not in evac_nodes]

        # ── Precomputed road lengths (src_idx, dst_idx) -> metres, and edge areas ──
        self.road_lengths = {}
        for src_node, neighbors in adj.items():
            if src_node not in node_to_idx: continue
            src_idx = node_to_idx[src_node]
            for dst_node, w in neighbors.items():
                if dst_node not in node_to_idx: continue
                self.road_lengths[(src_idx, node_to_idx[dst_node])] = float(w)
        self._n_edges = max(len(self.road_lengths), 1)
        self.edge_area = {k: v * ROAD_WIDTH for k, v in self.road_lengths.items()}
        self._max_edge_area = max(self.edge_area.values()) if self.edge_area else 1.0

        # ── Per-shelter distance / hop / neighbor-order tables ───────────────
        # Each agent is assigned its own destination shelter each episode, so
        # phi shaping and "action 0 = toward my destination" ordering must be
        # computed per-shelter (not a single global field as before).
        raw_neighbors = [[] for _ in range(self.n_nodes)]
        for node in node_list:
            ni = node_to_idx[node]
            raw_neighbors[ni] = [node_to_idx[nb] for nb in adj.get(node, {}) if nb in node_to_idx]

        self._shelter_list     = list(evac_nodes)
        self._n_shelters       = max(len(self._shelter_list), 1)
        self._shelter_node_idx = np.array(
            [node_to_idx[s] for s in self._shelter_list], dtype=np.int32)
        self._shelter_dist = np.full((self._n_shelters, self.n_nodes), 1e9, dtype=np.float32)
        self._shelter_hops = np.full((self._n_shelters, self.n_nodes), 10**6, dtype=np.int32)
        self._shelter_neighbor_order = []
        for s_i, shelter in enumerate(self._shelter_list):
            dist_map, hop_map = compute_distances_from_node(shelter, adj, road_nodes)
            for ni, node in enumerate(node_list):
                self._shelter_dist[s_i, ni] = dist_map.get(node, 1e9)
                self._shelter_hops[s_i, ni] = hop_map.get(node, 10**6)
            order = [sorted(raw_neighbors[ni], key=lambda nb: self._shelter_dist[s_i, nb])
                     for ni in range(self.n_nodes)]
            self._shelter_neighbor_order.append(order)
        finite_dist = self._shelter_dist[self._shelter_dist < 1e9]
        self._global_max_dist = float(finite_dist.max()) if finite_dist.size else 1.0

        # Static "edges incident to the destination node" block: the neighbor
        # list of each shelter's own node, using that same shelter's distance
        # ordering rule (trivially local, but keeps a consistent rule).
        self._shelter_own_neighbors = [
            self._shelter_neighbor_order[s_i][int(self._shelter_node_idx[s_i])]
            for s_i in range(self._n_shelters)]

        # ── Observation / action spaces ───────────────────────────────────────
        # obs_dim = max_degree * 7 * 2 + 1:
        #   current-node edge block (7*max_degree) + destination-node edge
        #   block (7*max_degree) + hop-based progress scalar (1)
        self._obs_dim   = max_degree * 7 * 2 + 1
        self._obs_space = spaces.Box(low=0.0, high=1.0,
                                     shape=(self._obs_dim,), dtype=np.float32)
        self._act_space = spaces.Discrete(max_degree)
        self.possible_agents = [f"agent_{i}" for i in range(n_agents)]
        self.agents          = []

        # ── Per-agent state arrays ────────────────────────────────────────────
        self._agent_node_idx       = np.zeros(n_agents, dtype=np.int32)
        self._agent_status         = np.full(n_agents, STATUS_EVACUATING, dtype=np.int8)
        self._agent_on_link        = np.zeros(n_agents, dtype=bool)
        self._agent_link_src       = np.zeros(n_agents, dtype=np.int32)
        self._agent_link_dst       = np.zeros(n_agents, dtype=np.int32)
        self._agent_link_progress  = np.zeros(n_agents, dtype=np.float32)
        self._agent_target_shelter = np.zeros(n_agents, dtype=np.int32)
        self._agent_speed          = np.full(n_agents, AGENT_SPEED_MEAN, dtype=np.float32)
        self._agent_init_hops      = np.ones(n_agents, dtype=np.float32)
        self._agent_travel_steps   = np.zeros(n_agents, dtype=np.int32)
        self._step_count           = 0
        self._rng                  = np.random.default_rng(42)
        self._arrival_times        = []

        self._current_link_use = {}
        self._actions_arr      = np.zeros(n_agents, dtype=np.int32)

        # Pre-allocated output arrays (rebuilt each step for active agents only)
        self._obs_mat     = np.zeros((n_agents, self._obs_dim), dtype=np.float32)
        self._active_mask = np.zeros(n_agents, dtype=bool)

    # ── Spaces ────────────────────────────────────────────────────────────────
    def observation_space(self, agent): return self._obs_space
    def action_space(self, agent):      return self._act_space

    # ── Reset ─────────────────────────────────────────────────────────────────
    def _sample_congestion_floor(self):
        """Pick a random subset of undirected edges and assign each a
        permanent density floor (see artificial_congestion docstring in
        __init__). Both directions of a chosen physical edge get the same
        floor. Re-sampled fresh every reset()."""
        seen = set()
        undirected = []
        for (a, b) in self.edge_area.keys():
            key = frozenset((a, b))
            if key in seen:
                continue
            seen.add(key)
            undirected.append((a, b))
        if not undirected:
            return {}
        n_select = min(len(undirected), max(
            1, int(round(self.artificial_congestion_fraction * len(undirected)))))
        chosen_idx = self._rng.choice(len(undirected), size=n_select, replace=False)
        levels = self.artificial_congestion_levels
        floor = {}
        for idx in np.atleast_1d(chosen_idx):
            a, b = undirected[int(idx)]
            level = levels[int(self._rng.integers(0, len(levels)))]
            floor_density = level * DENSITY_MAX
            floor[(a, b)] = floor_density
            floor[(b, a)] = floor_density
        return floor

    def _shortest_path_edges_from(self, start_idx, shelter_idx):
        """Walk the greedy shortest-path chain from start_idx to the given
        shelter (i.e. always taking action 0 = self._shelter_neighbor_order[
        shelter_idx][node][0]), the same route the shortest-path baseline
        would actually take. Returns the list of (cidx, nidx) edges along
        that route. Stops early on cycles/dead ends (shouldn't normally
        happen since this follows a valid Dijkstra shortest-path tree)."""
        edges = []
        target_idx = int(self._shelter_node_idx[shelter_idx])
        cur = start_idx
        visited = {cur}
        for _ in range(self.n_nodes):
            if cur == target_idx:
                break
            nb = self._shelter_neighbor_order[shelter_idx][cur]
            if not nb:
                break
            nxt = nb[0]
            edges.append((cur, nxt))
            if nxt in visited:
                break
            visited.add(nxt)
            cur = nxt
        return edges

    def _sample_congestion_floor_on_shortest_path(self):
        """Instead of random edges, target the roads that the shortest-path
        (greedy action-0) baseline would actually walk: the route from the
        current cluster center (or a random node if cluster_start is off) to
        every shelter. This is meant to construct scenarios the shortest-path
        policy specifically struggles with -- its route is congested by
        construction -- to see whether a trained policy that can detour does
        better. Floors every edge on these routes (or a random subset of them
        if artificial_congestion_sp_fraction < 1.0)."""
        if self.cluster_start and self._cluster_center is not None:
            ref_idx = self.node_to_idx[self._cluster_center]
        else:
            ref_node = self._non_shelter[int(self._rng.integers(0, len(self._non_shelter)))]
            ref_idx = self.node_to_idx[ref_node]

        path_edges = set()
        for s_i in range(self._n_shelters):
            for edge in self._shortest_path_edges_from(ref_idx, s_i):
                path_edges.add(edge)
        path_edges = list(path_edges)
        if not path_edges:
            return {}

        frac = self.artificial_congestion_sp_fraction
        n_select = len(path_edges) if frac >= 1.0 else min(
            len(path_edges), max(1, int(round(frac * len(path_edges)))))
        if n_select < len(path_edges):
            chosen_idx = self._rng.choice(len(path_edges), size=n_select, replace=False)
            path_edges = [path_edges[int(i)] for i in np.atleast_1d(chosen_idx)]

        levels = self.artificial_congestion_levels
        floor = {}
        for (a, b) in path_edges:
            level = levels[int(self._rng.integers(0, len(levels)))]
            floor_density = level * DENSITY_MAX
            floor[(a, b)] = floor_density
            floor[(b, a)] = floor_density
        return floor

    def _eff_density(self, cidx, nidx, cnt, area):
        """Effective density on edge (cidx,nidx): the real agent-count-based
        density, floored at this edge's artificial congestion level (0.0 if
        artificial_congestion is off or this edge wasn't selected)."""
        d = cnt / area
        floor = self._congestion_floor.get((cidx, nidx), 0.0)
        return d if d > floor else floor

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.agents                   = self.possible_agents[:]
        self._step_count              = 0
        self._agent_status[:]         = STATUS_EVACUATING
        self._agent_on_link[:]        = False
        self._agent_link_progress[:]  = 0.0
        self._current_link_use        = {}
        self._actions_arr[:]          = 0
        self._agent_travel_steps[:]   = 0
        self._arrival_times           = []

        if self.cluster_start:
            # Pick one random cluster center (from cluster_center_pool if
            # given, so callers can bias WHERE the cluster forms, e.g. only
            # the eastern part of the map) and restrict start nodes to those
            # within cluster_radius_hops of it (falls back to the full
            # non-shelter node set if the neighborhood is too small).
            center_pool = self.cluster_center_pool if self.cluster_center_pool else self._non_shelter
            center = center_pool[int(self._rng.integers(0, len(center_pool)))]
            _, hop_map = compute_distances_from_node(center, self.adj, self.road_nodes)
            candidates = [n for n in self._non_shelter
                          if hop_map.get(n, 10**6) <= self.cluster_radius_hops]
            if len(candidates) < 5:
                candidates = self._non_shelter
            self._cluster_center = center
        else:
            candidates = self._non_shelter
            self._cluster_center = None

        if self.artificial_congestion:
            if self.artificial_congestion_target_shortest_path:
                self._congestion_floor = self._sample_congestion_floor_on_shortest_path()
            else:
                self._congestion_floor = self._sample_congestion_floor()
        else:
            self._congestion_floor = {}

        start_idxs = self._rng.integers(0, len(candidates), size=self.n_agents)
        for i in range(self.n_agents):
            self._agent_node_idx[i] = self.node_to_idx[candidates[start_idxs[i]]]

        # Random destination shelter per agent (capacity NOT considered).
        target_slots = self._rng.integers(0, self._n_shelters, size=self.n_agents)
        self._agent_target_shelter[:] = target_slots

        # Random per-agent basic walking speed for this episode.
        self._agent_speed[:] = np.clip(
            self._rng.normal(AGENT_SPEED_MEAN, AGENT_SPEED_STD, size=self.n_agents),
            AGENT_SPEED_MIN, AGENT_SPEED_MAX).astype(np.float32)

        for i in range(self.n_agents):
            s = int(target_slots[i])
            self._agent_init_hops[i] = max(float(self._shelter_hops[s, self._agent_node_idx[i]]), 1.0)

        self._build_obs_matrix({})
        self._active_mask[:] = (self._agent_status == STATUS_EVACUATING) & (~self._agent_on_link)

        infos = {a: {} for a in self.agents}
        return self._obs_mat, self._active_mask, infos

    # ── Step ──────────────────────────────────────────────────────────────────
    def step(self, actions_arr: np.ndarray):
        """
        actions_arr: int32 numpy array shape (n_agents,)
            actions_arr[i] = chosen neighbor index for agent i, in the order
            of self._shelter_neighbor_order[agent's target][current_node]
            (index 0 = move toward that agent's own destination; ignored if
            not evacuating)
        """
        self._step_count += 1
        rewards      = {}
        terminations = {}
        truncations  = {}
        infos        = {}
        link_use     = defaultdict(int)

        # ── Phase 1: Move ─────────────────────────────────────────────────────
        evac_idxs = np.where(self._agent_status == STATUS_EVACUATING)[0]
        self._agent_travel_steps[evac_idxs] += 1

        for i in evac_idxs:
            target = int(self._agent_target_shelter[i])
            if not self._agent_on_link[i]:
                cidx = int(self._agent_node_idx[i])
                nb   = self._shelter_neighbor_order[target][cidx]
                if not nb:
                    continue
                act  = min(int(actions_arr[i]), len(nb) - 1)
                self._agent_link_src[i]      = cidx
                self._agent_link_dst[i]      = nb[act]
                self._agent_on_link[i]       = True
                self._agent_link_progress[i] = 0.0

            cidx = int(self._agent_link_src[i])
            nidx = int(self._agent_link_dst[i])

            # Speed depends on the PREVIOUS step's observed congestion on this
            # edge (this step's link_use isn't fully known until every agent
            # has moved, so using it here would make speed order-dependent).
            prev_cnt = self._current_link_use.get((cidx, nidx), 0)
            area     = self.edge_area.get((cidx, nidx), ROAD_WIDTH)
            density  = self._eff_density(cidx, nidx, prev_cnt, area)
            speed    = float(self._agent_speed[i]) * max(1.0 - density / DENSITY_MAX, V_MIN_RATIO)
            self._agent_link_progress[i] += speed * STEP_TIME
            link_use[(cidx, nidx)] += 1

            if self._agent_link_progress[i] >= self.road_lengths.get((cidx, nidx), 1.0):
                self._agent_node_idx[i]      = nidx
                self._agent_on_link[i]       = False
                self._agent_link_progress[i] = 0.0

        self._current_link_use = dict(link_use)

        # ── Phase 2: Rewards ──────────────────────────────────────────────────
        for i in evac_idxs:
            a = self.possible_agents[i]
            target = int(self._agent_target_shelter[i])
            if self._agent_on_link[i]:
                cidx = int(self._agent_link_src[i])
                nidx = int(self._agent_link_dst[i])
            else:
                nidx = int(self._agent_node_idx[i])
                cidx = nidx

            phi_next     = -float(self._shelter_dist[target, nidx]) / self._global_max_dist
            phi_curr     = -float(self._shelter_dist[target, cidx]) / self._global_max_dist
            r_shape      = GAMMA_SHAPE * phi_next - phi_curr
            r_shape      = 10.0 * r_shape
            # r_shape      = 10.0 * (phi_next - phi_curr)
            area         = self.edge_area.get((cidx, nidx), ROAD_WIDTH)
            density      = self._eff_density(cidx, nidx, link_use.get((cidx, nidx), 0), area)
            density_ratio = density / DENSITY_MAX
            congestion_excess = max(0.0, (density_ratio - CONGESTION_THRESHOLD) / (1.0 - CONGESTION_THRESHOLD))
            r_congestion = -CONGESTION_PENALTY * min(congestion_excess, 1.0)
            r_time       = -0.1 * TIME_PENALTY

            target_node_idx = int(self._shelter_node_idx[target])
            if not self._agent_on_link[i] and nidx == target_node_idx:
                # Destination reached. Capacity is not enforced (destinations
                # were assigned ignoring capacity), so arrival always succeeds.
                self._agent_status[i] = STATUS_SAFE
                rewards[a]      = 10.0 + r_shape + r_congestion + r_time
                terminations[a] = True
                truncations[a]  = False
                self._arrival_times.append(int(self._agent_travel_steps[i]) * STEP_TIME)
            elif self._step_count >= self.max_steps:
                self._agent_status[i] = STATUS_FAILED
                rewards[a]      = r_shape + r_congestion + r_time
                terminations[a] = False
                truncations[a]  = True
            else:
                rewards[a]      = r_shape + r_congestion + r_time
                terminations[a] = False
                truncations[a]  = False

        # ── Phase 3: Global state ─────────────────────────────────────────────
        n_evac = int(np.sum(self._agent_status == STATUS_EVACUATING))
        gstate = self.get_global_state(n_evac, link_use)
        for a in list(self.agents):
            if a not in infos: infos[a] = {}
            infos[a]['global_state'] = gstate

        # ── Phase 4: Obs + mask ───────────────────────────────────────────────
        self._build_obs_matrix(link_use)
        self._active_mask[:] = (self._agent_status == STATUS_EVACUATING) & (~self._agent_on_link)

        self.agents = [a for a in self.agents
                       if not terminations.get(a, False)
                       and not truncations.get(a, False)]
        return self._obs_mat, self._active_mask, rewards, terminations, truncations, infos

    # ── Observation matrix ────────────────────────────────────────────────────
    def _fill_edge_block(self, i, offset, src_idx, neighbor_idxs, link_use, own_speed, target):
        """Fill 7*max_degree observation slots (starting at `offset`) with the
        7 normalized features of each of `neighbor_idxs`' outgoing edges from
        `src_idx`, grouped per edge: [area, own_speed, n_agents, density,
        avg_speed_ratio, density_ratio, closeness_to_target]."""
        d = self.max_degree
        for k, nb_idx in enumerate(neighbor_idxs[:d]):
            cnt     = link_use.get((src_idx, nb_idx), 0)
            area    = self.edge_area.get((src_idx, nb_idx), ROAD_WIDTH)
            density = self._eff_density(src_idx, nb_idx, cnt, area)
            # Per-edge directional signal: how close does taking THIS edge get
            # you to your own target, in [0,1] (1 = at the destination, 0 =
            # as far as possible). This is the old phi(next) value rescaled
            # to [0,1], shown per-candidate-edge (unlike the hop-progress
            # scalar, which is the same for every edge at a given step and
            # so can't by itself discriminate between options).
            closeness = 1.0 - min(float(self._shelter_dist[target, nb_idx]) / self._global_max_dist, 1.0)
            base = offset + k * 7
            self._obs_mat[i, base + 0] = min(area / self._max_edge_area, 1.0)
            self._obs_mat[i, base + 1] = min(own_speed / AGENT_SPEED_MAX, 1.0)
            self._obs_mat[i, base + 2] = min(cnt / self.n_agents, 1.0)
            self._obs_mat[i, base + 3] = min(density / DENSITY_MAX, 1.0)
            self._obs_mat[i, base + 4] = max(1.0 - density / DENSITY_MAX, V_MIN_RATIO)
            self._obs_mat[i, base + 5] = min(density / DENSITY_MAX, 1.0)
            self._obs_mat[i, base + 6] = max(closeness, 0.0)

    def _build_obs_matrix(self, link_use):
        """
        Rebuild observation rows for all agents that are evacuating and at a node.
        obs[i] = [ 7 features per edge, for each of the up-to-max_degree edges
                   selectable from the agent's current node (sorted by distance
                   to the agent's OWN destination),
                   7 features per edge, for each of the up-to-max_degree edges
                   incident to the agent's destination node (same ordering rule),
                   hop-based progress toward destination ]
        """
        d = self.max_degree
        self._obs_mat[:] = 0.0
        active_idxs = np.where(
            (self._agent_status == STATUS_EVACUATING) & (~self._agent_on_link))[0]
        for i in active_idxs:
            cidx      = int(self._agent_node_idx[i])
            target    = int(self._agent_target_shelter[i])
            own_speed = float(self._agent_speed[i])
            shelter_node_idx = int(self._shelter_node_idx[target])

            # Block A: edges selectable from the current node.
            nb_cur = self._shelter_neighbor_order[target][cidx]
            self._fill_edge_block(i, 0, cidx, nb_cur, link_use, own_speed, target)

            # Block B: edges incident to the destination node (static per shelter).
            nb_dst = self._shelter_own_neighbors[target]
            self._fill_edge_block(i, 7 * d, shelter_node_idx, nb_dst, link_use, own_speed, target)

            # Progress: 1 - (remaining hops to destination / hops at episode start)
            remaining_hops = float(self._shelter_hops[target, cidx])
            init_hops      = float(self._agent_init_hops[i])
            progress       = 1.0 - min(remaining_hops / max(init_hops, 1.0), 1.0)
            self._obs_mat[i, -1] = max(progress, 0.0)

    # ── Global state ──────────────────────────────────────────────────────────
    def get_global_state(self, n_evac, link_use):
        """
        Compute the 9-dim global state vector used by the Critic:
          [0:6] the 6 edge features (area, basic_speed, n_agents, density,
                avg_speed_ratio, density_ratio), each averaged over every
                edge in the graph. Feature 1 becomes the population's mean
                basic_speed (since there is no single "current agent" here).
          [6]   fraction of edges with density > 80% of DENSITY_MAX
          [7]   fraction of edges with density > 50% of DENSITY_MAX
          [8]   fraction of agents still evacuating (traveling)
        """
        if not self.road_lengths:
            edge_feat_avg = np.zeros(6, dtype=np.float32)
            hi80 = hi50 = 0
        else:
            mean_speed = float(np.mean(self._agent_speed)) if self.n_agents else BASE_SPEED
            areas = []; speeds = []; counts = []; densities = []
            speed_ratios = []; density_ratios = []
            hi80 = hi50 = 0
            for (cidx, nidx) in self.road_lengths:
                area    = self.edge_area[(cidx, nidx)]
                cnt     = link_use.get((cidx, nidx), 0)
                density = self._eff_density(cidx, nidx, cnt, area)
                areas.append(min(area / self._max_edge_area, 1.0))
                speeds.append(min(mean_speed / AGENT_SPEED_MAX, 1.0))
                counts.append(min(cnt / self.n_agents, 1.0))
                densities.append(min(density / DENSITY_MAX, 1.0))
                speed_ratios.append(max(1.0 - density / DENSITY_MAX, V_MIN_RATIO))
                density_ratios.append(min(density / DENSITY_MAX, 1.0))
                if density >= 0.8 * DENSITY_MAX: hi80 += 1
                if density >= 0.5 * DENSITY_MAX: hi50 += 1
            edge_feat_avg = np.array([
                np.mean(areas), np.mean(speeds), np.mean(counts),
                np.mean(densities), np.mean(speed_ratios), np.mean(density_ratios)],
                dtype=np.float32)

        return np.concatenate([
            edge_feat_avg,
            np.array([hi80 / self._n_edges, hi50 / self._n_edges, n_evac / self.n_agents],
                      dtype=np.float32)])

    # ── Utilities ─────────────────────────────────────────────────────────────
    def summary(self):
        """
        Return a dict with final counts and arrival-time statistics for logging.
        Replaces the old mortality-rate summary: destinations are assigned
        without capacity limits, so every arrival succeeds, and the metric of
        interest is how long it took each agent to reach its own destination.
        """
        n_safe = int(np.sum(self._agent_status == STATUS_SAFE))
        n_fail = int(np.sum(self._agent_status == STATUS_FAILED))
        n_evac = int(np.sum(self._agent_status == STATUS_EVACUATING))
        arrived = list(self._arrival_times)
        avg_arrival_arrived_only = float(np.mean(arrived)) if arrived else float('nan')
        all_times = arrived + [self.max_steps * STEP_TIME] * (self.n_agents - len(arrived))
        avg_arrival_with_timeout = float(np.mean(all_times)) if all_times else float('nan')
        return dict(safe=n_safe, failed=n_fail, evacuating=n_evac,
                    n_arrived=len(arrived), n_agents=self.n_agents,
                    avg_arrival_time_arrived_only=avg_arrival_arrived_only,
                    avg_arrival_time_with_timeout=avg_arrival_with_timeout)
