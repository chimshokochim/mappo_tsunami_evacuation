"""
evac_env.py  --  Evacuation environment compliant with PettingZoo Parallel API

CTDE (Centralized Training, Decentralized Execution):
  Actor  : Uses local observations only (decentralized execution)
  Critic : Uses global state features (centralized training)

Reward Design:
  Potential-based shaping
    phi(node) = -dist_to_shelter(node) / max_dist  in [-1, 0]
    r_shape = gamma_shape * phi(next) - phi(curr)   (dense guiding reward)
    r_safe  = +1.0  (terminal reward for successful arrival)

  Congestion penalty
    density      = n_agents_on_link / ROAD_WIDTH
    r_congestion = -CONGESTION_PENALTY * min(density / DENSITY_THRESHOLD, 1.0)
    Teaches agents to spread across routes when the nearest path is congested.
    Shelter CAPACITY is respected (full shelters reject arrivals).

Observation Vector (dim = max_degree * 2 + 1):
  neighbor_lists MUST be pre-sorted by ascending distance-to-nearest-shelter
  (done in common.build_graph_index) so action index 0 always means
  "move toward the nearest shelter" at every node.

  [0 : max_degree]            : normalized congestion density on each outgoing
                                 link, in shelter-proximity order. 0.0 = padding.
  [max_degree : 2*max_degree] : phi value of each neighboring node, same order.
                                 phi in [-1, 0]; padding = -1.0 (worst case).
  [2*max_degree]              : real-time fullness in [0,1] of the shelter
                                 nearest to the agent's current node. Included
                                 to resolve state aliasing: agents at the same
                                 node with the same congestion/phi but different
                                 shelter fullness now receive distinct observations.

Global State Vector (dim = 5, Critic input only):
  [0] fraction of edges with density > 80%   (normalized by total edges)
  [1] fraction of edges with density > 50%   (normalized by total edges)
  [2] average shelter fullness               (in [0, 1])
  [3] fraction of full shelters              (in [0, 1])
  [4] fraction of agents still evacuating    (in [0, 1])
"""

import numpy as np
from collections import defaultdict
from common import BASE_SPEED, ROAD_WIDTH, compute_shelter_distances
from pettingzoo import ParallelEnv as _Base
from gymnasium import spaces

# No-op @profile decorator when not running under a profiler
import builtins
if not hasattr(builtins, 'profile'):
    builtins.profile = lambda f: f

# ── Reward / physics constants ─────────────────────────────────────────────────
GAMMA_SHAPE        = 0.99
CONGESTION_PENALTY = 0.05   # max per-step penalty for very congested links
DENSITY_THRESHOLD  = 3.0    # density (agents/m) at which full congestion penalty applies
STEP_TIME          = 5.0    # seconds per simulation step

# Agent status codes
STATUS_EVACUATING = np.int8(0)
STATUS_SAFE       = np.int8(1)
STATUS_FAILED     = np.int8(2)


class EvacuationEnv(_Base):
    metadata = {"render_modes": [], "name": "evacuation_kochi_v1"}

    def __init__(self, node_coords, adj, road_nodes, evac_nodes, evac_capacity,
                 node_list, node_to_idx, neighbor_lists, max_degree,
                 n_agents=500, max_steps=600, reward_dest=1.0):

        self.node_coords    = node_coords
        self.adj            = adj
        self.road_nodes     = road_nodes
        self.evac_nodes     = evac_nodes
        self.evac_capacity  = evac_capacity
        self.node_list      = node_list
        self.node_to_idx    = node_to_idx
        self.neighbor_lists = neighbor_lists   # must be pre-sorted by shelter proximity
        self.max_degree     = max_degree
        self.n_nodes        = len(node_list)
        self.n_agents       = n_agents
        self.max_steps      = max_steps
        self.reward_dest    = reward_dest
        self._non_shelter   = [n for n in node_list if n not in evac_nodes]

        # ── Potential field ───────────────────────────────────────────────────
        # phi[i] = -dist_to_nearest_shelter[i] / max_dist, in [-1, 0].
        # return_source=True also returns which shelter each node is closest to,
        # used below to observe real-time shelter fullness per agent.
        dist_map, max_dist, nearest_shelter_node = compute_shelter_distances(
            evac_nodes, adj, road_nodes, return_source=True)
        max_dist = max(max_dist, 1.0)
        self._phi = np.array(
            [-dist_map.get(node_list[i], max_dist * 2) / max_dist
             for i in range(self.n_nodes)], dtype=np.float32)

        # ── Nearest shelter index per node ────────────────────────────────────
        # Integer array: _nearest_shelter_idx[node_idx] = node_idx of that node's
        # nearest shelter. Stored as indices (not string ids) for fast array lookup
        # in the hot per-step loop. node_list[idx] recovers the string id when needed.
        self._nearest_shelter_idx = np.array(
            [node_to_idx.get(nearest_shelter_node.get(node_list[i], node_list[i]), 0)
             for i in range(self.n_nodes)], dtype=np.int32)

        # ── Shelter bookkeeping ───────────────────────────────────────────────
        self._evac_nodes_list  = list(evac_nodes)
        self._initial_capacity = dict(evac_capacity)
        self._n_shelters       = max(len(evac_nodes), 1)
        self._node_is_evac     = np.array(
            [node_list[i] in evac_nodes for i in range(self.n_nodes)], dtype=bool)

        # ── Observation / action spaces ───────────────────────────────────────
        # obs_dim = max_degree * 2 + 1:
        #   congestion block (max_degree) + phi block (max_degree) + shelter fullness (1)
        self._obs_dim   = max_degree * 2 + 1
        self._obs_space = spaces.Box(low=-1.0, high=1.0,
                                     shape=(self._obs_dim,), dtype=np.float32)
        self._act_space = spaces.Discrete(max_degree)
        self.possible_agents = [f"agent_{i}" for i in range(n_agents)]
        self.agents          = []

        # ── Per-agent state arrays ────────────────────────────────────────────
        self._agent_node_idx      = np.zeros(n_agents, dtype=np.int32)
        self._agent_status        = np.full(n_agents, STATUS_EVACUATING, dtype=np.int8)
        self._agent_on_link       = np.zeros(n_agents, dtype=bool)
        self._agent_link_src      = np.zeros(n_agents, dtype=np.int32)
        self._agent_link_dst      = np.zeros(n_agents, dtype=np.int32)
        self._agent_link_progress = np.zeros(n_agents, dtype=np.float32)
        self._cap_remain          = {}
        self._step_count          = 0
        self._rng                 = np.random.default_rng(42)

        # ── Precomputed road lengths (src_idx, dst_idx) → metres ─────────────
        self.road_lengths = {}
        for src_node, neighbors in adj.items():
            if src_node not in node_to_idx: continue
            src_idx = node_to_idx[src_node]
            for dst_node, w in neighbors.items():
                if dst_node not in node_to_idx: continue
                self.road_lengths[(src_idx, node_to_idx[dst_node])] = float(w)
        self._n_edges = max(len(self.road_lengths), 1)

        self._current_link_use = {}
        self._actions_arr      = np.zeros(n_agents, dtype=np.int32)

        # ── Static phi block of observations ─────────────────────────────────
        # obs[max_degree : 2*max_degree] = phi of each neighbor.
        # This never changes during an episode, so it is precomputed once here.
        # Only the congestion block and shelter fullness scalar are updated per step.
        self._neighbor_phi_static = np.full((self.n_nodes, max_degree), -1.0, dtype=np.float32)
        for nidx in range(self.n_nodes):
            nb = self.neighbor_lists[nidx]
            for k, nb_idx in enumerate(nb[:max_degree]):
                self._neighbor_phi_static[nidx, k] = self._phi[nb_idx]

        # Pre-allocated output arrays (rebuilt each step for active agents only)
        self._obs_mat     = np.zeros((n_agents, self._obs_dim), dtype=np.float32)
        self._active_mask = np.zeros(n_agents, dtype=bool)

    # ── Spaces ────────────────────────────────────────────────────────────────
    def observation_space(self, agent): return self._obs_space
    def action_space(self, agent):      return self._act_space

    # ── Shelter fullness helper ───────────────────────────────────────────────
    def _shelter_fullness(self, shelter_node):
        """Current occupancy ratio in [0,1] for a given shelter node id."""
        if shelter_node is None:
            return 0.0
        cap = self._initial_capacity.get(shelter_node, 1)
        remain = self._cap_remain.get(shelter_node, 0)
        return 1.0 - remain / max(cap, 1)

    # ── Reset ─────────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.agents                   = self.possible_agents[:]
        self._cap_remain              = dict(self.evac_capacity)
        self._step_count              = 0
        self._agent_status[:]         = STATUS_EVACUATING
        self._agent_on_link[:]        = False
        self._agent_link_progress[:]  = 0.0
        self._current_link_use        = {}
        self._actions_arr[:]          = 0

        start_idxs = self._rng.integers(0, len(self._non_shelter), size=self.n_agents)
        for i in range(self.n_agents):
            self._agent_node_idx[i] = self.node_to_idx[self._non_shelter[start_idxs[i]]]

        self._build_obs_matrix({})
        self._active_mask[:] = (self._agent_status == STATUS_EVACUATING) & (~self._agent_on_link)

        infos = {a: {} for a in self.agents}
        return self._obs_mat, self._active_mask, infos

    # ── Step ──────────────────────────────────────────────────────────────────
    def step(self, actions_arr: np.ndarray):
        """
        actions_arr: int32 numpy array shape (n_agents,)
            actions_arr[i] = chosen neighbor index for agent i
            (index 0 = move toward nearest shelter; ignored if not evacuating)
        """
        self._step_count += 1
        rewards      = {}
        terminations = {}
        truncations  = {}
        infos        = {}
        link_use     = defaultdict(int)
        MOVE_DIST    = BASE_SPEED * STEP_TIME

        # ── Phase 1: Move ─────────────────────────────────────────────────────
        evac_idxs = np.where(self._agent_status == STATUS_EVACUATING)[0]
        for i in evac_idxs:
            if not self._agent_on_link[i]:
                cidx = int(self._agent_node_idx[i])
                nb   = self.neighbor_lists[cidx]
                if not nb:
                    continue
                act  = min(int(actions_arr[i]), len(nb) - 1)
                self._agent_link_src[i]      = cidx
                self._agent_link_dst[i]      = nb[act]
                self._agent_on_link[i]       = True
                self._agent_link_progress[i] = 0.0

            self._agent_link_progress[i] += MOVE_DIST
            cidx = int(self._agent_link_src[i])
            nidx = int(self._agent_link_dst[i])
            link_use[(cidx, nidx)] += 1

            if self._agent_link_progress[i] >= self.road_lengths.get((cidx, nidx), 1.0):
                self._agent_node_idx[i]      = nidx
                self._agent_on_link[i]       = False
                self._agent_link_progress[i] = 0.0

        self._current_link_use = dict(link_use)

        # ── Phase 2: Rewards ──────────────────────────────────────────────────
        for i in evac_idxs:
            a = self.possible_agents[i]
            if self._agent_on_link[i]:
                cidx = int(self._agent_link_src[i])
                nidx = int(self._agent_link_dst[i])
            else:
                nidx = int(self._agent_node_idx[i])
                cidx = nidx

            r_shape      = GAMMA_SHAPE * float(self._phi[nidx]) - float(self._phi[cidx])
            density      = link_use.get((cidx, nidx), 0) / ROAD_WIDTH
            r_congestion = -CONGESTION_PENALTY * min(density / DENSITY_THRESHOLD, 1.0)

            if not self._agent_on_link[i] and self._node_is_evac[nidx]:
                nnode = self.node_list[nidx]
                if self._cap_remain.get(nnode, 0) > 0:
                    self._agent_status[i]    = STATUS_SAFE
                    self._cap_remain[nnode] -= 1
                    rewards[a]      = 1.0 + r_shape + r_congestion
                    terminations[a] = True
                    truncations[a] = False
                else:
                    # Shelter full: agent fails
                    self._agent_status[i] = STATUS_FAILED
                    rewards[a]      = r_shape + r_congestion
                    terminations[a] = True
                    truncations[a] = False
            elif self._step_count >= self.max_steps:
                self._agent_status[i] = STATUS_FAILED
                rewards[a]      = r_shape + r_congestion
                terminations[a] = False
                truncations[a] = True
            else:
                rewards[a]      = r_shape + r_congestion
                terminations[a] = False
                truncations[a] = False

        # ── Phase 3: Global state ─────────────────────────────────────────────
        n_safe = int(np.sum(self._agent_status == STATUS_SAFE))
        n_evac = int(np.sum(self._agent_status == STATUS_EVACUATING))
        gstate = self.get_global_state(n_evac, link_use)
        for a in list(self.agents):
            if a not in infos: infos[a] = {}
            infos[a]['global_state']     = gstate
            infos[a]['global_mortality'] = (self.n_agents - n_safe) / self.n_agents

        # ── Phase 4: Obs + mask ───────────────────────────────────────────────
        self._build_obs_matrix(link_use)
        self._active_mask[:] = (self._agent_status == STATUS_EVACUATING) & (~self._agent_on_link)

        self.agents = [a for a in self.agents
                       if not terminations.get(a, False)
                       and not truncations.get(a, False)]
        return self._obs_mat, self._active_mask, rewards, terminations, truncations, infos

    # ── Observation matrix ────────────────────────────────────────────────────
    def _build_obs_matrix(self, link_use):
        """
        Rebuild observation rows for all agents that are evacuating and at a node.
        obs[i] = [congestion_0..max_degree-1,   # normalized link density per neighbor
                  phi_0..max_degree-1,            # static; copied from precomputed table
                  nearest_shelter_fullness]        # live occupancy of closest shelter
        """
        norm        = ROAD_WIDTH * DENSITY_THRESHOLD
        active_idxs = np.where(
            (self._agent_status == STATUS_EVACUATING) & (~self._agent_on_link))[0]
        for i in active_idxs:
            cidx = int(self._agent_node_idx[i])
            nb   = self.neighbor_lists[cidx]

            # Congestion block: normalized agent count on each outgoing link
            self._obs_mat[i, :self.max_degree] = 0.0
            for k, nidx in enumerate(nb[:self.max_degree]):
                cnt = link_use.get((cidx, nidx), 0)
                self._obs_mat[i, k] = min(cnt / norm, 1.0)

            # Phi block: static neighbor potentials (bounded to avoid overwriting fullness slot)
            self._obs_mat[i, self.max_degree:2*self.max_degree] = self._neighbor_phi_static[cidx]

            # Fullness scalar: live occupancy of this agent's nearest shelter
            shelter_idx  = int(self._nearest_shelter_idx[cidx])
            shelter_node = self.node_list[shelter_idx]
            self._obs_mat[i, -1] = self._shelter_fullness(shelter_node)

    # ── Global state ──────────────────────────────────────────────────────────
    def get_global_state(self, n_evac, link_use):
        """
        Compute the 5-dim global state vector used by the Critic.
        Aggregates network-wide congestion and shelter fullness statistics.
        """
        hi80 = sum(1 for cnt in link_use.values()
                   if cnt / ROAD_WIDTH >= 0.8 * DENSITY_THRESHOLD)
        hi50 = sum(1 for cnt in link_use.values()
                   if cnt / ROAD_WIDTH >= 0.5 * DENSITY_THRESHOLD)
        fullness_list = [
            1.0 - self._cap_remain.get(s, 0) / max(self._initial_capacity.get(s, 1), 1)
            for s in self._evac_nodes_list]
        avg_fullness = sum(fullness_list) / len(fullness_list) if fullness_list else 0.0
        frac_full    = sum(1 for f in fullness_list if f >= 1.0) / self._n_shelters
        return np.array([hi80 / self._n_edges, hi50 / self._n_edges,
                         avg_fullness, frac_full, n_evac / self.n_agents],
                        dtype=np.float32)

    # ── Utilities ─────────────────────────────────────────────────────────────
    def mortality_rate(self):
        """Fraction of agents that did not reach a shelter."""
        return (self.n_agents - int(np.sum(self._agent_status == STATUS_SAFE))) / self.n_agents

    def summary(self):
        """Return a dict with final counts and mortality rate for logging."""
        n_safe = int(np.sum(self._agent_status == STATUS_SAFE))
        n_fail = int(np.sum(self._agent_status == STATUS_FAILED))
        n_evac = int(np.sum(self._agent_status == STATUS_EVACUATING))
        return dict(safe=n_safe, failed=n_fail, evacuating=n_evac,
                    mortality=self.mortality_rate())