"""
training.py  --  MA-PPO Tsunami evacuation training phase
Shared parameters across all agents (actor + critic).

Each episode, every agent gets a random destination shelter (capacity is not
considered when assigning destinations) and its own randomly-sampled walking
speed. Start positions are clustered (CLUSTER_START): all agents start
within CLUSTER_RADIUS_HOPS hops of one random cluster center chosen
uniformly from the whole map, so a whole neighborhood evacuates at once and
congestion actually forms. See evac_env.py's module docstring for the full
observation / global-state layout.

Observation per agent (dim = max_degree*7*2 + 1): edge-grouped features
  (area, own basic_speed, n_agents, density, avg_speed_ratio, density_ratio,
  closeness_to_target) for the max_degree edges selectable from the current
  node, then the same 7 features for the max_degree edges incident to the
  agent's destination node, then a hop-based progress-to-destination scalar.
Neighbor lists are sorted per-destination-shelter, so action index 0 always
means "move toward MY OWN destination."
Global state (Critic input, dim=9): [avg edge features (6), hi80, hi50, frac_evacuating]
"""

import os, sys, time, random, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Use GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── Hyperparameters ────────────────────────────────────────────────────────────
HIDDEN_SIZE    = 64     # neurons per hidden layer (both Actor and Critic)
LR_ACTOR       = 1e-4  # Adam learning rate for Actor -- orthogonal init removed
                        # (accelerated instability instead of taming it, see note
                        # near Actor/Critic classes), so back to LR=1e-4 + rollout=4
                        # + entropy anneal(ep9000) as the best-confirmed combo so far.
# LR_ACTOR       = 3e-4  # original
# LR_ACTOR       = 3e-5  # screening-test value, no longer needed w/o orthoinit
LR_CRITIC      = 1e-3  # Adam learning rate for Critic (higher: faster value learning)
GAE_LAMBDA     = 0.95  # GAE smoothing factor (0=TD, 1=MC)
CLIP_EPSILON   = 0.2   # PPO clipping range for policy ratio
UPDATE_EPOCHS  = 4     # number of gradient passes per rollout buffer
ENTROPY_COEF   = 0.05  # starting weight on entropy bonus (encourages exploration)

# ── Entropy coefficient annealing ───────────────────────────────────────
# Diagnosed via plot_actor_loss_decomposition.py: once policy entropy
# saturates near its ceiling (ln(2) for a 2-action policy) mid-training,
# the PPO surrogate term (the actual advantage-driven improvement signal)
# shrinks toward ~0, so the FIXED entropy bonus (ENTROPY_COEF * H) comes to
# dominate the total actor_loss almost entirely -- the network keeps being
# pushed toward max-entropy (near-uniform) behavior long after the
# advantage signal that justified high entropy has died down, which we
# believe drives the slow, unexplained far_frac drift back into the
# congestion-danger zone seen late in training.
#
# Fix: keep ENTROPY_COEF_START = ENTROPY_COEF unchanged (this preserves
# the early-training exploration that let the good far_frac trajectory
# emerge in the first place -- do not touch the starting value, only the
# decay), and linearly anneal it down to ENTROPY_COEF_END by a fixed
# ABSOLUTE episode (ENTROPY_ANNEAL_EPISODE), then hold flat at
# ENTROPY_COEF_END for the remainder.
#
# NOTE: originally this was expressed as a FRACTION of TOTAL_EPISODES
# (e.g. anneal over the first 60%), reasoning that this would scale
# correctly for short test runs. That was wrong: across every run so far
# (LR=3e-4 and LR=1e-5 alike), the far_frac self-correction from its
# mid-training overshoot (~0.58) back down to the true optimum (~0.4) has
# consistently happened around ep8000-9000 in ABSOLUTE episode terms,
# seemingly independent of TOTAL_EPISODES or LR -- likely tied to how long
# the Critic itself takes to catch up, not to the Actor's LR. A
# fraction-based schedule on a short TOTAL_EPISODES run (e.g. 4000) then
# anneals to completion at ep2400, well before that ep8000-9000
# self-correction point, locking the policy into the bad overshoot state.
# Using a fixed absolute anchor avoids this mismatch regardless of how
# long the run is.
ENTROPY_COEF_START   = ENTROPY_COEF
ENTROPY_COEF_END     = 0.01
ENTROPY_ANNEAL_EPISODE = 9000   # absolute episode by which to reach ENTROPY_COEF_END


def current_entropy_coef(ep):
    """Linear anneal from ENTROPY_COEF_START to ENTROPY_COEF_END by absolute
    episode ENTROPY_ANNEAL_EPISODE, then hold at the end value."""
    if ENTROPY_ANNEAL_EPISODE <= 0:
        return ENTROPY_COEF_END
    frac = min(1.0, ep / ENTROPY_ANNEAL_EPISODE)
    return ENTROPY_COEF_START + frac * (ENTROPY_COEF_END - ENTROPY_COEF_START)

# Number of episodes accumulated into the rollout buffer before each
# PPO update, instead of updating after every single episode. A single
# episode is a small, high-variance sample of the multi-agent dynamics --
# if that one episode happens to be unusually lopsided (e.g. nearly every
# agent's decision points to the same direction, whether or not that
# turns out well), the resulting gradient is strongly and uniformly
# directional even though each individual (normalized, ratio-clipped)
# advantage is bounded -- clipping protects against any ONE sample
# dominating, but not against MANY samples in one episode agreeing by
# chance. Accumulating several episodes' worth of more varied experience
# before each update dilutes that effect. Set to 1 to recover the
# original "update every episode" behavior.
ROLLOUT_EPISODES = 4   # SCREENING TEST: with LR=1e-4 this collapsed hard around
                        # ep2280-2300 (danger zone lands at ~update #2300 since
                        # rollout=1 means 1 episode = 1 update). Now testing whether
                        # a much lower LR_ACTOR (3e-5) survives that same update-count
                        # danger zone -- if not, rollout=1's noise floor can't be fixed
                        # by LR alone and we go back to ROLLOUT_EPISODES=4 for good.

# ── Clustered start (see evac_env.EvacuationEnv) ────────────────────────────────
# Every episode, all agents start within CLUSTER_RADIUS_HOPS hops of one
# random cluster center (itself chosen uniformly from anywhere on the map,
# no geographic bias) instead of being spread uniformly over the whole
# city. This makes congestion a real, learnable phenomenon: the policy now
# has to learn to route AROUND agents packed into the same neighborhood,
# not just walk toward its own destination in isolation. Set to False to
# go back to the original uniform-random-start task.
CLUSTER_START               = True
CLUSTER_RADIUS_HOPS         = 6
CLUSTER_CENTER_POOL         = None   # None = no geographic restriction

# ── Artificial congestion (permanent density floor on random edges) ────────────
# When True, a random subset of edges gets a permanent minimum density each
# episode (e.g. "always at least 50% congested"), regardless of actual agent
# count -- it can rise above the floor as agents pile on, but never drops
# below it. Simulates persistently bad roads (damage/debris/bottlenecks)
# rather than only transient crowding. See EvacuationEnv's docstring for
# details. Off by default.
ARTIFICIAL_CONGESTION           = True
ARTIFICIAL_CONGESTION_LEVELS    = (1.0,)   # fractions of DENSITY_MAX
ARTIFICIAL_CONGESTION_FRACTION  = 0.50         # fraction of edges affected (random mode only)

# If True, instead of random edges, congestion is placed ON the route the
# greedy shortest-path (action-0) baseline would actually take (cluster
# center -> every shelter). Purpose-built to make scenarios the shortest-path
# baseline specifically struggles with, so a trained policy that can detour
# has room to actually beat it (rather than just match it).
ARTIFICIAL_CONGESTION_TARGET_SHORTEST_PATH = True
ARTIFICIAL_CONGESTION_SP_FRACTION          = 1.0   # fraction of that route's edges floored

# ── Small synthetic grid map (instead of the real Kochi OSM map) ───────────────
# Swap in a tiny NxN grid road network for quick, easy-to-reason-about
# experiments. When True, parse_osm()/load_evac_data() are skipped entirely
# and common.build_grid_graph()/make_grid_evac_data() are used instead.
# N_AGENTS/MAX_STEPS/CLUSTER_RADIUS_HOPS are scaled way down from the real
# 5750-node map's defaults since a 5x5 grid only has 25 nodes.
USE_GRID_MAP             = True
GRID_ROWS                = 5
GRID_COLS                = 5
GRID_CELL_SIZE_M         = 80.0
GRID_CONNECT_DIAGONALS   = True     # 8-connected (incl. diagonals)
GRID_N_SHELTERS          = 2        # randomly placed among the 25 nodes
GRID_SEED                = 42       # which shelters get chosen (independent of training SEED)
GRID_N_AGENTS            = 3000
GRID_MAX_STEPS           = 200
GRID_CLUSTER_RADIUS_HOPS = 2         # grid diameter is only ~4 hops (8-connected)

# ── Minimal "line" topology: one center node, one edge to a near shelter, ──────
# one edge to a far shelter (common.build_line_graph). Simplest possible
# setting to test whether MAPPO learns to split traffic across both routes
# under congestion, vs. shortest-path which always sends everyone down the
# near (shorter) edge. Takes priority over USE_GRID_MAP when True. All 3000
# agents start at 'center' (forced via cluster_start with a single-node
# pool below) and depart with staggered timing (see STAGGERED_DEPARTURE).
USE_LINE_MAP             = True
LINE_DIST_NEAR           = 150.0    # meters, center -> shelter_near
LINE_DIST_FAR            = 300.0    # meters, center -> shelter_far (~1:2 ratio)
LINE_N_AGENTS            = 3000
# NOTE: with only 2 narrow edges carrying all 3000 agents (no alternate
# routes, no capacity limit other than the Greenshields speed slowdown),
# congestion is severe: under the shortest-path baseline it takes ~800+
# steps for everyone to fully clear the edges (verified by direct
# simulation). A short max_steps (e.g. 150) would time out ~85% of agents
# every single episode, giving no useful learning signal. 900 gives the
# baseline enough room to mostly finish while still leaving real headroom
# for a smarter (or worse) policy to differ from it.
LINE_MAX_STEPS           = 900

# Staggered departure: each agent gets its own random departure step instead
# of everyone leaving 'center' at step 0, so agents arrive at the fork
# spread out over time rather than all at once. Only wired up for the line
# map for now (see EvacuationEnv's staggered_departure docstring).
STAGGERED_DEPARTURE      = True
DEPARTURE_WINDOW_FRAC    = 0.25     # everyone has departed by 25% of max_steps

# If True, everyone's initial target is the single nearest shelter (here:
# shelter_near, since everyone starts at 'center') instead of a random
# 50/50 split -- so the shortest-path baseline piles 100% of agents onto
# one route, giving MAPPO room to learn to send some agents to the
# farther-but-less-congested shelter instead (re-targeting happens
# automatically in evac_env.py's step() whenever an agent chooses to walk
# onto an edge leading to a shelter -- see nearest_shelter_target
# docstring). Only wired up for the line map for now.
LINE_NEAREST_SHELTER_TARGET = True

# If True, the Critic's global state is the minimal 3-dim
# [density_near, density_far, frac_evacuating] instead of the default
# 9-dim state (6 edge features averaged over ALL edges + hi80 + hi50 +
# frac_evacuating). Averaging over just 2 edges hides exactly which one
# is congested, which is the "Critic is too coarse-grained" issue --
# giving it the two densities separately should produce a less noisy
# value estimate (and thus a less noisy advantage/PPO gradient) right
# when it matters most: whenever near and far have different congestion
# levels. See EvacuationEnv's line_critic_state docstring for why the
# other 4 raw features and the hi80/hi50 flags were dropped rather than
# just also split per-edge. Only wired up for the line map for now.
LINE_CRITIC_STATE = True

# The shortest-path baseline re-runs TOTAL_EPISODES full episodes with
# env.step() called unconditionally every tick (see
# evaluate_shortest_path_baseline()'s docstring) -- on the line map this
# routinely takes as long as (or longer than) training itself, especially
# once congestion is severe enough that it times out every episode. Set to
# False to skip it entirely and just train MAPPO -- arrival_time.png /
# mappo_reward_analysis.png / shelter_choice.png will only show the
# trained policy (no green/purple baseline lines), and
# plot_line_zoom.py works unchanged either way since it never reads the
# baseline_* fields. Re-enable when you actually need the baseline number
# for comparison.
RUN_BASELINE = False

# The baseline has no learning, so its per-episode results are just noise
# around a fixed level -- there's no trend to capture by running it for as
# many episodes as training. BASELINE_N_EPISODES lets you run only a small
# number of episodes (e.g. 10), average their results, and use that flat
# average as the baseline value for every episode on the plots (instead of
# 4000 separately-simulated, individually noisy baseline episodes). Set to
# None (or >= TOTAL_EPISODES) to run the baseline for the full
# TOTAL_EPISODES instead, e.g. if you want to see its own episode-to-episode
# variance rather than just a flat reference line.
BASELINE_N_EPISODES = 10

from common import (
    OSM_FILE, EXCEL_FILE,
    N_TRAIN_AGENTS, TOTAL_EPISODES, MAX_STEPS_EP,
    REWARD_DEST, GAMMA, SEED,
    parse_osm, load_evac_data, build_graph_index, compute_shelter_distances,
    build_grid_graph, make_grid_evac_data,
    build_line_graph, make_line_evac_data,
)
from evac_env import EvacuationEnv

# Fix all random seeds for reproducibility
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ═══════════════════════════════════════════════════════════════════════════════
#  Networks
# ═══════════════════════════════════════════════════════════════════════════════
# NOTE: orthogonal weight initialization was tried here and removed.
# Across two different (rollout, LR) combinations, adding it made the
# periodic instability arrive MUCH earlier (rollout=1+LR=3e-5: crisis at
# ep900 instead of ep2400; rollout=4+LR=3e-4: crisis at ep1300 instead of
# ep7800), consistent with orthogonal init preserving gradient/activation
# norm more faithfully across layers than PyTorch's default init -- i.e.
# stronger, more consistent gradient flow, which for this problem acts like
# an effective LR increase and accelerates the instability rather than
# damping it. Back to plain default init.

class Actor(nn.Module):
    """
    Decentralized policy network.
    Each agent runs its own copy with shared weights.
    Input:  local observation (max_degree*2+1,)
    Output: probability distribution over neighbor choices (max_degree,)
    """
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, HIDDEN_SIZE), nn.Tanh(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE), nn.Tanh(),
            nn.Linear(HIDDEN_SIZE, action_dim), nn.Softmax(dim=-1))
    def forward(self, x): return self.net(x)

class Critic(nn.Module):
    """
    Centralized value network (CTDE: Centralized Training, Decentralized Execution).
    Sees the full global state during training to estimate V(s).
    Input:  global state (9,) by default, or (3,) when the env's
            line_critic_state=True (see EvacuationEnv.global_state_dim /
            LINE_CRITIC_STATE below).
    Output: scalar state value V(s)
    """
    def __init__(self, global_state_dim=9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_state_dim, HIDDEN_SIZE), nn.Tanh(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE), nn.Tanh(),
            nn.Linear(HIDDEN_SIZE, 1))
    def forward(self, x): return self.net(x)

# ═══════════════════════════════════════════════════════════════════════════════
#  MAPPO Agent
# ═══════════════════════════════════════════════════════════════════════════════
class MAPPOAgent:
    """Holds Actor + Critic networks and their optimizers."""
    def __init__(self, n_agents, n_nodes, max_degree, neighbor_lists, global_state_dim=9):
        self.n_agents       = n_agents
        self.n_nodes        = n_nodes
        self.max_degree     = max_degree
        self.neighbor_lists = neighbor_lists

        obs_dim = max_degree * 7 * 2 + 1   # current-node edge block(7*d) + destination-node edge block(7*d) + progress
        self.actor  = Actor(obs_dim, max_degree).to(device)
        self.critic = Critic(global_state_dim).to(device)
        self.optimizer_actor  = optim.Adam(self.actor.parameters(),  lr=LR_ACTOR)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

    def get_actions_batch(self, obs_arr: np.ndarray):
        """
        Inference for all active agents in a single GPU forward pass.
        obs_arr: (N, obs_dim) — observations for N active agents.
        Returns: actions (N,) int32, logprobs (N,) float32.
        logprobs are stored in the rollout buffer for PPO ratio computation.
        """
        if len(obs_arr) == 0:
            return [], []
        obs_tensor = torch.from_numpy(obs_arr).to(device)
        with torch.no_grad():
            probs = self.actor(obs_tensor)
        
        actions_t  = torch.multinomial(probs, 1).squeeze(1)
        logprobs_t = torch.log(probs.gather(1, actions_t.unsqueeze(1)).squeeze(1) + 1e-8)
        
        cpu_np = torch.stack([actions_t.float(), logprobs_t], dim=1).cpu().numpy()
        return cpu_np[:, 0].astype(np.int32), cpu_np[:, 1]

    def evaluate_actions(self, obs_batch, actions_batch):
        """
        Re-evaluate stored actions under the CURRENT (updated) policy.
        Called during ppo_update() — gradients are active here.
        Returns: log_prob of each action, entropy of the policy distribution.
        Entropy bonus discourages premature convergence to a single action.
        """
        probs            = self.actor(obs_batch)
        dist             = torch.distributions.Categorical(probs)
        action_log_probs = dist.log_prob(actions_batch)
        dist_entropy     = dist.entropy()
        return action_log_probs, dist_entropy

# ═══════════════════════════════════════════════════════════════════════════════
#  GAE + PPO update
# ═══════════════════════════════════════════════════════════════════════════════
def compute_gae(next_value, rewards, masks, values):
    """
    Generalized Advantage Estimation (Schulman et al. 2016).
    Computes returns = advantage + value baseline for each timestep.
      delta = r + γ*V(s') - V(s)       (TD error)
      A_t   = delta + γ*λ * A_{t+1}    (GAE recursion, backwards)
    masks: 0 if episode ended at this step, 1 otherwise (cuts bootstrap at terminal).
    Returns a list of per-agent per-step return targets for the Critic loss.
    """
    rew_arr  = np.array(rewards, dtype=np.float32)
    mask_arr = np.array(masks, dtype=np.float32)
    val_arr  = np.array(values, dtype=np.float32)
    n = len(rew_arr)
    returns = np.empty(n, dtype=np.float32)
    gae = 0.0
    last_val = next_value
    for step in reversed(range(n)):
        delta = rew_arr[step] + GAMMA * last_val * mask_arr[step] - val_arr[step]
        gae   = delta + GAMMA * GAE_LAMBDA * mask_arr[step] * gae
        returns[step] = gae + val_arr[step]
        last_val = val_arr[step]
    return returns.tolist()

def _far_shelter_fraction(summary):
    """
    Diagnostic helper (line-map only): fraction of THIS episode's arrivals
    that ended up at 'shelter_far' rather than 'shelter_near', using the
    arrival_shelter_counts / shelter_list fields evac_env.py's summary()
    now returns. Returns nan if this env has no 'shelter_far' (e.g. grid/
    real map) or nobody arrived this episode. Used to check whether the
    trained policy's near/far split actually changes over training, or
    stays fixed at whatever an untrained (near-uniform-softmax) policy's
    initial split happens to be.
    """
    shelter_list = summary.get('shelter_list')
    counts       = summary.get('arrival_shelter_counts')
    if not shelter_list or 'shelter_far' not in shelter_list:
        return float('nan')
    total = sum(counts)
    if total == 0:
        return float('nan')
    far_idx = shelter_list.index('shelter_far')
    return counts[far_idx] / total

def evaluate_shortest_path_baseline(env, total_episodes, base_seed):
    """
    Run a no-learning baseline: every agent always picks action 0, i.e. the
    edge toward its own destination shelter (neighbor lists are sorted by
    distance-to-own-target, so this is greedy shortest-path routing). Uses
    the SAME per-episode seeds as the training loop (base_seed + ep), so the
    start/target assignments line up 1:1 with the corresponding training
    episode and the comparison is apples-to-apples. Congestion still slows
    agents down exactly as it does for the trained policy, so this measures
    "what if agents never learned to route around congestion at all."
    """
    baseline_arrived_only = []
    baseline_with_timeout = []
    baseline_counts       = []
    baseline_rewards      = []   # mean per-agent return per episode, same metric as history_rewards
    baseline_far_frac     = []   # diagnostic: fraction of arrivals at shelter_far (line map only)
    for ep in range(total_episodes):
        t0 = time.time()
        env.reset(seed=base_seed + ep)
        zero_actions = np.zeros(env.n_agents, dtype=np.int32)
        ep_reward = 0.0
        for _ in range(env.max_steps):
            _, _, rewards, _, truncations, _ = env.step(zero_actions)
            ep_reward += sum(rewards.values())
            if not env.agents or any(truncations.values()):
                break
        summary = env.summary()
        baseline_arrived_only.append(summary['avg_arrival_time_arrived_only'])
        baseline_with_timeout.append(summary['avg_arrival_time_with_timeout'])
        baseline_counts.append((summary['n_arrived'], summary['n_agents']))
        baseline_rewards.append(ep_reward / env.n_agents)
        baseline_far_frac.append(_far_shelter_fraction(summary))
        ep_duration = time.time() - t0

        recent_arrived = [t for t in baseline_arrived_only[-50:] if not np.isnan(t)]
        avg_arr = np.mean(recent_arrived) if recent_arrived else float('nan')
        avg_rew = np.mean(baseline_rewards[-50:])
        n_arr, n_tot = baseline_counts[-1]
        print(f"[Baseline] Episode {ep:4d} | Avg Reward (per-agent): {avg_rew:7.2f} | "
              f"Avg Arrival Time (arrived, 50ep): {avg_arr:6.1f}s | "
              f"Arrived: {n_arr}/{n_tot} | Speed: {ep_duration:.1f}s/ep")
    return (baseline_arrived_only, baseline_with_timeout, baseline_counts, baseline_rewards,
            baseline_far_frac)

def ppo_update(agent_system, memory, entropy_coef=ENTROPY_COEF):
    """
    One PPO update pass over the full rollout buffer collected this episode.
    Runs UPDATE_EPOCHS gradient steps on both Critic and Actor.

    Critic loss: MSE between V(s) and GAE returns (baseline regression).
    Actor loss:  clipped surrogate objective — prevents large policy updates.
      ratio     = π_new(a|s) / π_old(a|s)   (re-computed via evaluate_actions)
      surr1     = ratio * A
      surr2     = clip(ratio, 1±ε) * A
      loss      = -min(surr1, surr2) - entropy_coef * H(π)
    Advantages are normalized per batch to stabilize gradient scale.

    entropy_coef: the ENTROPY_COEF value to use for THIS update. Defaults to
    the (fixed) module constant ENTROPY_COEF for backward compatibility, but
    the training loop now passes current_entropy_coef(ep) so the entropy
    bonus can anneal over training (see ENTROPY_COEF_START/_END above).

    Returns a dict of this episode's loss/entropy diagnostics (mean and
    last-epoch value across the UPDATE_EPOCHS passes), for logging --
    lets you check whether the Critic is actually converging (loss
    trending down) or diverging/oscillating, and whether the Actor's
    policy entropy is collapsing too fast or staying too high.
    """
    obs_batch      = torch.FloatTensor(np.array(memory['obs'])).to(device)
    state_per_step = torch.FloatTensor(np.array(memory['state_per_step'])).to(device)
    step_index     = torch.LongTensor(memory['step_index'])
    # Expand stored global states: one state per step → one per agent-step
    state_batch    = state_per_step[step_index].to(device)
    actions_batch  = torch.LongTensor(np.array(memory['actions'])).to(device)
    logprobs_batch = torch.FloatTensor(np.array(memory['logprobs'])).to(device)
    returns_batch  = torch.FloatTensor(np.array(memory['returns'])).to(device)

    # Per-epoch loss/entropy history, returned to the caller for logging/
    # diagnostics -- lets us see whether the Critic is actually converging
    # (loss_critic trending down within/across episodes) or diverging/
    # oscillating, and whether the Actor's entropy is collapsing too fast
    # (policy locking in early) or staying high (never committing).
    critic_losses  = []
    actor_losses   = []
    entropies      = []

    for _ in range(UPDATE_EPOCHS):
        # ── Critic update ──────────────────────────────────────────────────
        current_values = agent_system.critic(state_batch).squeeze(-1)
        loss_critic    = 0.5 * nn.MSELoss()(current_values, returns_batch)
        agent_system.optimizer_critic.zero_grad()
        loss_critic.backward()
        agent_system.optimizer_critic.step()

        # ── Actor update ───────────────────────────────────────────────────
        advantages = returns_batch - current_values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
        action_log_probs, dist_entropy = agent_system.evaluate_actions(obs_batch, actions_batch)
        ratios = torch.exp(action_log_probs - logprobs_batch)
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON) * advantages
        loss_actor = -torch.min(surr1, surr2).mean() - entropy_coef * dist_entropy.mean()
        agent_system.optimizer_actor.zero_grad()
        loss_actor.backward()
        agent_system.optimizer_actor.step()

        critic_losses.append(float(loss_critic.item()))
        actor_losses.append(float(loss_actor.item()))
        entropies.append(float(dist_entropy.mean().item()))

    return dict(
        critic_loss_mean=float(np.mean(critic_losses)),
        critic_loss_last=critic_losses[-1],
        actor_loss_mean=float(np.mean(actor_losses)),
        actor_loss_last=actor_losses[-1],
        entropy_mean=float(np.mean(entropies)),
        entropy_last=entropies[-1],
    )

# ═══════════════════════════════════════════════════════════════════════════════
#  Utilities
# ═══════════════════════════════════════════════════════════════════════════════
def _save_graph(node_list, node_to_idx, neighbor_lists, max_degree,
                evac_nodes, evac_capacity, node_coords, adj, path='graph_data.pkl'):
    """Serialize graph data so execution scripts can skip OSM re-parsing."""
    with open(path, 'wb') as f:
        pickle.dump(dict(node_list=node_list, node_to_idx=node_to_idx,
                         neighbor_lists=neighbor_lists, max_degree=max_degree,
                         evac_nodes=list(evac_nodes), evac_capacity=evac_capacity,
                         node_coords=node_coords, adj=dict(adj)), f)
    print(f"  Graph Data Saved: {path}")

def _save_model(mappo_brain, path_prefix='mappo'):
    """Save Actor and Critic weights separately for later execution/evaluation."""
    torch.save(mappo_brain.actor.state_dict(),  f'{path_prefix}_actor.pt')
    torch.save(mappo_brain.critic.state_dict(), f'{path_prefix}_critic.pt')
    print(f"  Model saved: {path_prefix}_actor.pt / {path_prefix}_critic.pt")

# ═══════════════════════════════════════════════════════════════════════════════
#  Training
# ═══════════════════════════════════════════════════════════════════════════════
def train():
    print("=" * 60)
    print(" Training Phase [Multi-Agent PPO]")
    print("=" * 60)

    # ── Map source: real Kochi OSM map, a small synthetic grid, or the ───────
    # minimal center/near-shelter/far-shelter "line" topology ────────────────
    train_staggered_departure    = False
    train_departure_window_frac  = DEPARTURE_WINDOW_FRAC
    train_cluster_center_pool    = CLUSTER_CENTER_POOL
    train_nearest_shelter_target = False
    train_line_critic_state      = False
    if USE_LINE_MAP:
        print(f"Using minimal line topology (center -> shelter_near @ "
              f"{LINE_DIST_NEAR:.0f}m, center -> shelter_far @ {LINE_DIST_FAR:.0f}m) "
              f"instead of the OSM/grid map.")
        node_coords, adj, road_nodes = build_line_graph(
            dist_near=LINE_DIST_NEAR, dist_far=LINE_DIST_FAR)
        evac_nodes, evac_capacity = make_line_evac_data()
        train_n_agents       = LINE_N_AGENTS
        train_max_steps      = LINE_MAX_STEPS
        train_cluster_radius = 0   # forces all agents to start exactly at 'center'
        train_cluster_center_pool   = ['center']
        train_staggered_departure   = STAGGERED_DEPARTURE
        train_nearest_shelter_target = LINE_NEAREST_SHELTER_TARGET
        train_line_critic_state      = LINE_CRITIC_STATE
        graph_save_path       = 'graph_data_line.pkl'
        # MAPPO_RUN_TAG lets a multi-seed runner (see run_multiseed.py) give
        # each run's actor/critic/history files a distinct name (e.g.
        # 'mappo_line_seed43_...') so parallel runs with different SEEDs
        # don't overwrite each other's output. Empty by default -- normal
        # single-run behavior (file names) is unchanged.
        model_prefix          = 'mappo_line' + os.environ.get('MAPPO_RUN_TAG', '')
    elif USE_GRID_MAP:
        print(f"Using synthetic {GRID_ROWS}x{GRID_COLS} grid map "
              f"({'8' if GRID_CONNECT_DIAGONALS else '4'}-connected) instead of the OSM map.")
        node_coords, adj, road_nodes = build_grid_graph(
            rows=GRID_ROWS, cols=GRID_COLS, cell_size_m=GRID_CELL_SIZE_M,
            connect_diagonals=GRID_CONNECT_DIAGONALS)
        evac_nodes, evac_capacity = make_grid_evac_data(
            road_nodes, n_shelters=GRID_N_SHELTERS, seed=GRID_SEED)
        train_n_agents       = GRID_N_AGENTS
        train_max_steps      = GRID_MAX_STEPS
        train_cluster_radius = GRID_CLUSTER_RADIUS_HOPS
        graph_save_path       = 'graph_data_grid.pkl'
        model_prefix          = 'mappo_grid'
    else:
        # ── Parse OSM map and evacuation building data ───────────────────────
        node_coords, adj, road_nodes = parse_osm(OSM_FILE)
        evac_nodes, evac_capacity    = load_evac_data(EXCEL_FILE, node_coords, road_nodes)
        train_n_agents       = N_TRAIN_AGENTS
        train_max_steps      = MAX_STEPS_EP
        train_cluster_radius = CLUSTER_RADIUS_HOPS
        graph_save_path       = 'graph_data.pkl'
        model_prefix          = 'mappo'

    # Compute shelter distances first so neighbor_lists can be sorted by them.
    # This guarantees action index 0 = "move toward nearest shelter" at every node.
    shelter_dist, _ = compute_shelter_distances(evac_nodes, adj, road_nodes)
    node_list, node_to_idx, neighbor_lists, max_degree = build_graph_index(
        adj, road_nodes, shelter_dist=shelter_dist)
    n_nodes = len(node_list)
    print(f"Nodes:{n_nodes}  Max Degree:{max_degree}  Destinations:{len(evac_nodes)}")
    print("  neighbor_lists sorted by proximity-to-shelter "
          "(action 0 = move toward nearest shelter)")

    _save_graph(node_list, node_to_idx, neighbor_lists, max_degree,
                evac_nodes, evac_capacity, node_coords, adj, path=graph_save_path)

    # ── Build environment and agent ──────────────────────────────────────────
    env = EvacuationEnv(
        node_coords=node_coords, adj=adj, road_nodes=road_nodes,
        evac_nodes=evac_nodes, evac_capacity=evac_capacity,
        node_list=node_list, node_to_idx=node_to_idx,
        neighbor_lists=neighbor_lists, max_degree=max_degree,
        n_agents=train_n_agents, max_steps=train_max_steps, reward_dest=REWARD_DEST,
        cluster_start=(CLUSTER_START or USE_LINE_MAP), cluster_radius_hops=train_cluster_radius,
        cluster_center_pool=train_cluster_center_pool,
        artificial_congestion=(ARTIFICIAL_CONGESTION and not USE_LINE_MAP),
        artificial_congestion_levels=ARTIFICIAL_CONGESTION_LEVELS,
        artificial_congestion_fraction=ARTIFICIAL_CONGESTION_FRACTION,
        artificial_congestion_target_shortest_path=ARTIFICIAL_CONGESTION_TARGET_SHORTEST_PATH,
        artificial_congestion_sp_fraction=ARTIFICIAL_CONGESTION_SP_FRACTION,
        staggered_departure=train_staggered_departure,
        departure_window_frac=train_departure_window_frac,
        nearest_shelter_target=train_nearest_shelter_target,
        line_critic_state=train_line_critic_state,
    )

    mappo_brain = MAPPOAgent(train_n_agents, n_nodes, max_degree, neighbor_lists,
                              global_state_dim=env.global_state_dim)

    # Sanity-check network sizes at startup
    actor_params  = sum(p.numel() for p in mappo_brain.actor.parameters())
    critic_params = sum(p.numel() for p in mappo_brain.critic.parameters())
    print(f"  Actor : {actor_params:,} params = {actor_params*4/1e6:.2f}MB "
          f"(obs_dim={max_degree*7*2+1})")
    print(f"  Critic: {critic_params:,} params = {critic_params*4/1e6:.2f}MB "
          f"(global_state_dim={env.global_state_dim})")

    # ── Training loop ────────────────────────────────────────────────────────
    arrival_arrived_only = []   # avg arrival time (s), arrived agents only, per episode
    arrival_with_timeout = []   # avg arrival time (s), timeouts counted as max_steps, per episode
    arrival_counts       = []   # (n_arrived, n_agents) per episode
    history_rewards       = []   # mean per-agent return per episode (for plotting)
    history_far_frac      = []   # diagnostic: fraction of arrivals at shelter_far (line map only)
    history_critic_loss   = []   # diagnostic: mean Critic MSE loss this episode's PPO update
    history_actor_loss    = []   # diagnostic: mean Actor clipped-surrogate loss this episode
    history_entropy       = []   # diagnostic: mean policy entropy this episode
    history_entropy_coef  = []   # diagnostic: ENTROPY_COEF actually used this episode's
                                  # update (varies now that it's annealed -- see
                                  # current_entropy_coef() above; needed to reconstruct
                                  # entropy_term = entropy_coef * H exactly downstream,
                                  # e.g. in plot_actor_loss_decomposition.py)
    total_t0 = time.time()

    # Rollout buffer accumulated across ROLLOUT_EPISODES episodes before each
    # PPO update (see ROLLOUT_EPISODES docstring above). None when empty.
    rollout_buffer = None
    episodes_since_update = 0

    for ep in range(TOTAL_EPISODES):
        t0 = time.time()
        obs_mat, active_mask, infos = env.reset(seed=SEED + ep)
        state      = env.get_global_state(n_evac=train_n_agents, link_use={})
        next_state = state

        # Rollout buffer: collects one full episode of experience for PPO update
        memory = {'obs': [], 'state_per_step': [], 'step_index': [],
                  'actions': [], 'logprobs': [], 'rewards': [], 'masks': [], 'values': []}
        ep_reward = 0.0

        for step in range(train_max_steps):
            active_idxs = np.where(active_mask)[0]

            if len(active_idxs) == 0:
                # All evacuating agents are mid-link; advance simulation with no new decisions.
                # Rewards are still generated for every evacuating agent this
                # step (env.step() doesn't gate them on active_idxs), so they
                # must still be added to ep_reward here -- otherwise every
                # step where nobody happens to need a fresh decision silently
                # drops its reward, making ep_reward an undercount of the
                # real per-agent time penalty (this was the previous bug).
                next_obs_mat, active_mask, rewards, _, truncations, infos = \
                    env.step(env._actions_arr)
                if infos:
                    next_state = next(iter(infos.values()))['global_state']
                obs_mat = next_obs_mat; state = next_state
                ep_reward += sum(rewards.values())
                if not env.agents or any(truncations.values()): break
                continue

            # ── Batch inference: one GPU call for all active agents ───────────
            actions_arr_active, logprobs_arr_active = mappo_brain.get_actions_batch(
                obs_mat[active_idxs])
            env._actions_arr[active_idxs] = actions_arr_active

            # Critic evaluates current global state (shared across all agents)
            with torch.no_grad():
                value = mappo_brain.critic(
                    torch.from_numpy(state).to(device)).item()

            next_obs_mat, active_mask, rewards, terminations, truncations, infos = \
                env.step(env._actions_arr)
            if infos:
                next_state = next(iter(infos.values()))['global_state']

            # Store global state once per step; agents reference it by index
            current_step_idx = len(memory['state_per_step'])
            memory['state_per_step'].append(state)
            for k, i in enumerate(active_idxs):
                a = env.possible_agents[i]
                memory['obs'].append(obs_mat[i])
                memory['step_index'].append(current_step_idx)
                memory['actions'].append(int(actions_arr_active[k]))
                memory['logprobs'].append(float(logprobs_arr_active[k]))
                memory['rewards'].append(rewards.get(a, 0.0))
                memory['masks'].append(0 if terminations.get(a, False) else 1)
                memory['values'].append(value)

            obs_mat = next_obs_mat; state = next_state
            ep_reward += sum(rewards.values())
            if not env.agents or any(truncations.values()): break

        # ── End of episode: compute this episode's returns, then accumulate ────
        # into the rollout buffer. The PPO update itself only runs every
        # ROLLOUT_EPISODES episodes (see ROLLOUT_EPISODES docstring), using
        # the combined, more varied experience of several episodes rather
        # than just this one -- reduces the chance that one unusually
        # lopsided episode (e.g. nearly every agent's decision pointing the
        # same direction that episode, whether or not it turns out well)
        # dominates a single gradient update on its own. GAE/returns are
        # still computed per-episode (bootstrapping must respect episode
        # boundaries), only the timing of the actual network update changes.
        loss_info = None
        ep_entropy_coef = None
        if len(memory['rewards']) > 0:
            next_value        = mappo_brain.critic(
                torch.from_numpy(next_state).to(device)).item()
            memory['returns'] = compute_gae(next_value, memory['rewards'],
                                            memory['masks'], memory['values'])

            if rollout_buffer is None:
                rollout_buffer = {'obs': [], 'state_per_step': [], 'step_index': [],
                                   'actions': [], 'logprobs': [], 'returns': []}
            # step_index values are local to this episode's own
            # state_per_step list (start at 0) -- offset them by how much
            # is already in the buffer so they still point at the right
            # entry once this episode's state_per_step is appended after it.
            offset = len(rollout_buffer['state_per_step'])
            rollout_buffer['obs'].extend(memory['obs'])
            rollout_buffer['state_per_step'].extend(memory['state_per_step'])
            rollout_buffer['step_index'].extend(si + offset for si in memory['step_index'])
            rollout_buffer['actions'].extend(memory['actions'])
            rollout_buffer['logprobs'].extend(memory['logprobs'])
            rollout_buffer['returns'].extend(memory['returns'])
            episodes_since_update += 1

        is_last_episode = (ep == TOTAL_EPISODES - 1)
        if rollout_buffer is not None and (episodes_since_update >= ROLLOUT_EPISODES or is_last_episode):
            ep_entropy_coef = current_entropy_coef(ep)
            loss_info = ppo_update(mappo_brain, rollout_buffer, entropy_coef=ep_entropy_coef)
            rollout_buffer = None
            episodes_since_update = 0

        history_critic_loss.append(loss_info['critic_loss_mean'] if loss_info else float('nan'))
        history_actor_loss.append(loss_info['actor_loss_mean'] if loss_info else float('nan'))
        history_entropy.append(loss_info['entropy_mean'] if loss_info else float('nan'))
        history_entropy_coef.append(ep_entropy_coef if loss_info else float('nan'))

        summary = env.summary()
        arrival_arrived_only.append(summary['avg_arrival_time_arrived_only'])
        arrival_with_timeout.append(summary['avg_arrival_time_with_timeout'])
        arrival_counts.append((summary['n_arrived'], summary['n_agents']))
        mean_ep_reward = ep_reward / train_n_agents
        history_rewards.append(mean_ep_reward)
        far_frac = _far_shelter_fraction(summary)
        history_far_frac.append(far_frac)
        ep_duration = time.time() - t0

        recent_arrived = [t for t in arrival_arrived_only[-50:] if not np.isnan(t)]
        avg_arr = np.mean(recent_arrived) if recent_arrived else float('nan')
        avg_rew = np.mean(history_rewards[-50:])
        n_arr, n_tot = arrival_counts[-1]
        far_frac_txt = f" | Far-shelter frac: {far_frac:.2f}" if not np.isnan(far_frac) else ""
        loss_txt = (f" | Critic loss: {loss_info['critic_loss_mean']:.4f} | "
                    f"Actor loss: {loss_info['actor_loss_mean']:.4f} | "
                    f"Entropy: {loss_info['entropy_mean']:.3f}") if loss_info else ""
        print(f"Episode {ep:4d} | Avg Reward (per-agent): {avg_rew:7.2f} | "
              f"Avg Arrival Time (arrived, 50ep): {avg_arr:6.1f}s | "
              f"Arrived: {n_arr}/{n_tot} | Speed: {ep_duration:.1f}s/ep{far_frac_txt}{loss_txt}")

    total_duration = time.time() - total_t0
    print(f"Training Complete in {total_duration / 60:.2f} minutes.")
    _save_model(mappo_brain, path_prefix=model_prefix)

    # ── Shortest-path (no-learning) baseline, same per-episode seeds ─────────
    if RUN_BASELINE:
        n_baseline_ep = (BASELINE_N_EPISODES
                          if BASELINE_N_EPISODES and BASELINE_N_EPISODES < TOTAL_EPISODES
                          else TOTAL_EPISODES)
        flat = n_baseline_ep < TOTAL_EPISODES
        print(f"Running shortest-path baseline for comparison "
              f"({n_baseline_ep} episode{'s' if n_baseline_ep != 1 else ''}"
              f"{', averaged into a flat line' if flat else ''})...")
        t0 = time.time()
        baseline_arrived_only, baseline_with_timeout, baseline_counts, baseline_rewards, baseline_far_frac = \
            evaluate_shortest_path_baseline(env, n_baseline_ep, SEED)
        print(f"  Baseline Complete in {(time.time() - t0) / 60:.2f} minutes.")

        if flat:
            # No learning happens in the baseline, so its per-episode
            # results are pure noise around a fixed level -- average the
            # small sample and broadcast that flat value across every
            # training episode, instead of running/plotting TOTAL_EPISODES
            # separately-simulated (and individually noisy) baseline runs.
            mean_arrived  = float(np.nanmean(baseline_arrived_only))
            mean_timeout  = float(np.nanmean(baseline_with_timeout))
            mean_reward   = float(np.nanmean(baseline_rewards))
            mean_far_frac = float(np.nanmean(baseline_far_frac))
            mean_n_arr    = float(np.mean([c[0] for c in baseline_counts]))
            mean_n_tot    = baseline_counts[0][1]
            baseline_arrived_only = [mean_arrived] * TOTAL_EPISODES
            baseline_with_timeout = [mean_timeout] * TOTAL_EPISODES
            baseline_rewards      = [mean_reward] * TOTAL_EPISODES
            baseline_far_frac     = [mean_far_frac] * TOTAL_EPISODES
            baseline_counts       = [(mean_n_arr, mean_n_tot)] * TOTAL_EPISODES
    else:
        print("RUN_BASELINE=False -- skipping shortest-path baseline.")
        baseline_arrived_only = baseline_with_timeout = baseline_counts = None
        baseline_rewards      = baseline_far_frac                       = None

    # ── Persist raw per-episode histories so they can be re-plotted later ────
    # (e.g. a custom moving-average window) without re-running training.
    history_path = f'{model_prefix}_training_history.pkl'
    with open(history_path, 'wb') as f:
        pickle.dump(dict(
            arrival_arrived_only=arrival_arrived_only,
            arrival_with_timeout=arrival_with_timeout,
            arrival_counts=arrival_counts,
            history_rewards=history_rewards,
            history_far_frac=history_far_frac,
            history_critic_loss=history_critic_loss,
            history_actor_loss=history_actor_loss,
            history_entropy=history_entropy,
            history_entropy_coef=history_entropy_coef,
            baseline_arrived_only=baseline_arrived_only,
            baseline_with_timeout=baseline_with_timeout,
            baseline_counts=baseline_counts,
            baseline_rewards=baseline_rewards,
            baseline_far_frac=baseline_far_frac,
            total_episodes=TOTAL_EPISODES,
        ), f)
    print(f"  Raw per-episode history saved: {history_path}")

    return (mappo_brain, arrival_arrived_only, arrival_with_timeout, arrival_counts,
            history_rewards, baseline_arrived_only, baseline_with_timeout, baseline_counts,
            baseline_rewards, history_far_frac, baseline_far_frac,
            history_critic_loss, history_actor_loss, history_entropy)

def plot_training_results(arrival_arrived_only, arrival_with_timeout, arrival_counts,
                           mean_episode_rewards,
                           baseline_arrived_only=None, baseline_with_timeout=None,
                           baseline_counts=None, baseline_rewards=None,
                           history_far_frac=None, baseline_far_frac=None,
                           history_critic_loss=None, history_actor_loss=None,
                           history_entropy=None):
    """Plot and save two training curves: average arrival time (vs. a
    shortest-path/no-learning baseline run with the same per-episode
    start/target seeds) and mean episode return."""
    # Moving-average window scales with how many episodes were actually run,
    # so the trend stays readable instead of getting drowned in per-episode
    # noise on long runs (e.g. 5 episodes barely smooths anything over 2000
    # episodes; 100 does). Still never smaller than 5.
    window = max(5, len(arrival_arrived_only) // 20)

    # ── Average arrival-time curve ───────────────────────────────────────────
    # Two series for the trained policy: arrived-agents-only average, and an
    # average that also counts timed-out agents as having taken max_steps.
    # Capacity is not enforced, so every non-timeout agent is a success.
    # If provided, the shortest-path baseline (action 0 every step, i.e.
    # greedy routing with no learned congestion-avoidance) is overlaid using
    # the same episode axis for a direct comparison.
    fig, ax = plt.subplots(figsize=(9, 4.5))
    eps_x = list(range(1, len(arrival_arrived_only) + 1))
    ax.plot(eps_x, arrival_arrived_only, 'o-', color='steelblue', lw=1.5, ms=3,
            label='Trained policy: avg arrival time (arrived only)')
    ax.plot(eps_x, arrival_with_timeout, 'o-', color='darkorange', lw=1.5, ms=3,
            label='Trained policy: avg arrival time (with timeout)')
    if len(arrival_arrived_only) >= window:
        valid = np.array([t if not np.isnan(t) else np.nan for t in arrival_arrived_only])
        if np.sum(~np.isnan(valid)) >= window:
            ma = np.convolve(np.nan_to_num(valid, nan=np.nanmean(valid)),
                              np.ones(window) / window, mode='valid')
            ax.plot(range(window, len(arrival_arrived_only) + 1), ma,
                    '--', color='navy', lw=2, label=f'{window}-ep moving avg (arrived only)')
        ma2 = np.convolve(arrival_with_timeout, np.ones(window) / window, mode='valid')
        ax.plot(range(window, len(arrival_with_timeout) + 1), ma2,
                '--', color='firebrick', lw=2, label=f'{window}-ep moving avg (with timeout)')

    baseline_last_txt = ""
    if baseline_arrived_only is not None:
        eps_bx = list(range(1, len(baseline_arrived_only) + 1))
        ax.plot(eps_bx, baseline_arrived_only, 'o-', color='seagreen', lw=1.2, ms=2, alpha=0.7,
                label='Shortest-path baseline: avg arrival time (arrived only)')
        ax.plot(eps_bx, baseline_with_timeout, 'o-', color='purple', lw=1.2, ms=2, alpha=0.7,
                label='Shortest-path baseline: avg arrival time (with timeout)')
        if len(baseline_arrived_only) >= window:
            valid_b = np.array([t if not np.isnan(t) else np.nan for t in baseline_arrived_only])
            if np.sum(~np.isnan(valid_b)) >= window:
                ma_b = np.convolve(np.nan_to_num(valid_b, nan=np.nanmean(valid_b)),
                                    np.ones(window) / window, mode='valid')
                ax.plot(range(window, len(baseline_arrived_only) + 1), ma_b,
                        '--', color='darkgreen', lw=2,
                        label=f'Baseline: {window}-ep moving avg (arrived only)')
            ma2_b = np.convolve(baseline_with_timeout, np.ones(window) / window, mode='valid')
            ax.plot(range(window, len(baseline_with_timeout) + 1), ma2_b,
                    '--', color='indigo', lw=2,
                    label=f'Baseline: {window}-ep moving avg (with timeout)')
        if baseline_counts:
            n_arr_b, n_tot_b = baseline_counts[-1]
            baseline_last_txt = (f" | Baseline arrival rate: {n_arr_b}/{n_tot_b} "
                                  f"({100*n_arr_b/max(n_tot_b,1):.1f}%)")

    n_arr_last, n_tot_last = arrival_counts[-1] if arrival_counts else (0, 0)
    ax.set_xlabel('Episode'); ax.set_ylabel('Avg time to reach destination (s)')
    ax.set_title('Training Curve - MAPPO\n(Decentralized Actor with Centralized Critic)\n'
                  f'Trained policy arrival rate: {n_arr_last}/{n_tot_last} '
                  f'({100*n_arr_last/max(n_tot_last,1):.1f}%){baseline_last_txt}',
                  fontsize=10)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig('arrival_time.png', dpi=150); plt.close()
    print("  Training Curve Saved: arrival_time.png")

    # ── Mean episode return curve ────────────────────────────────────────────
    eps_x = np.arange(1, len(mean_episode_rewards) + 1)
    fig, ax = plt.subplots(figsize=(9, 4), dpi=150)
    ax.plot(eps_x, mean_episode_rewards, 'o-', color='steelblue', lw=2, ms=3, label='Trained policy: mean episode return')
    if len(mean_episode_rewards) >= window:
        ma = np.convolve(mean_episode_rewards, np.ones(window) / window, mode='valid')
        ax.plot(np.arange(window, len(mean_episode_rewards) + 1), ma,
                '--', color='tomato', lw=2, label=f'Trained policy: {window}-ep moving avg')
    if baseline_rewards is not None:
        eps_bx = np.arange(1, len(baseline_rewards) + 1)
        ax.plot(eps_bx, baseline_rewards, 'o-', color='seagreen', lw=1.2, ms=2, alpha=0.7,
                label='Shortest-path baseline: mean episode return')
        if len(baseline_rewards) >= window:
            ma_b = np.convolve(baseline_rewards, np.ones(window) / window, mode='valid')
            ax.plot(np.arange(window, len(baseline_rewards) + 1), ma_b,
                    '--', color='darkgreen', lw=2, label=f'Baseline: {window}-ep moving avg')
    ax.set_xlabel('Episode'); ax.set_ylabel('Mean Episode Return')
    ax.set_title('Mean Episode Return per Episode averaged over all agents')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig('mappo_reward_analysis.png', dpi=150); plt.close()
    print("  Mean Episode Return Plot Saved: mappo_reward_analysis.png")

    # ── Diagnostic: fraction of arrivals at shelter_far, per episode ─────────
    # (line map only -- nan/absent on grid/real map). Lets you see directly
    # whether the near/far split is actually changing as training
    # progresses, vs. staying fixed at whatever an untrained policy's
    # initial (near-uniform-softmax) action distribution happens to
    # produce -- which the reward/arrival-time curves alone can't show,
    # since a flat curve there is ambiguous between "converged" and
    # "never moved from its random initialization."
    if history_far_frac is not None and any(not np.isnan(v) for v in history_far_frac):
        fig, ax = plt.subplots(figsize=(9, 4), dpi=150)
        eps_x = np.arange(1, len(history_far_frac) + 1)
        ax.plot(eps_x, history_far_frac, 'o-', color='steelblue', lw=1.5, ms=3,
                label='Trained policy: fraction of arrivals at shelter_far')
        if len(history_far_frac) >= window:
            valid = np.array([v if not np.isnan(v) else np.nan for v in history_far_frac])
            if np.sum(~np.isnan(valid)) >= window:
                ma = np.convolve(np.nan_to_num(valid, nan=np.nanmean(valid)),
                                  np.ones(window) / window, mode='valid')
                ax.plot(np.arange(window, len(history_far_frac) + 1), ma,
                        '--', color='navy', lw=2, label=f'Trained policy: {window}-ep moving avg')
        if baseline_far_frac is not None:
            eps_bx = np.arange(1, len(baseline_far_frac) + 1)
            ax.plot(eps_bx, baseline_far_frac, 'o-', color='seagreen', lw=1.2, ms=2, alpha=0.7,
                    label='Shortest-path baseline: fraction of arrivals at shelter_far')
        ax.axhline(0.5, color='gray', lw=1, ls=':', label='50/50 split')
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel('Episode'); ax.set_ylabel('Fraction of arrivals at shelter_far')
        ax.set_title('Near/Far Shelter Choice per Episode')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig('shelter_choice.png', dpi=150); plt.close()
        print("  Shelter Choice Plot Saved: shelter_choice.png")

    # ── Diagnostic: Critic/Actor loss + policy entropy per episode ───────────
    # Lets you tell "Critic is converging (loss trending down/flat)" from
    # "Critic is diverging/oscillating" -- if the Critic's value estimates
    # are noisy or biased, the Actor's advantage-weighted updates inherit
    # that noise, which can show up as a policy that drifts to a WORSE
    # place over training instead of improving (see: reward/arrival-time
    # getting worse over hundreds of episodes despite reward being the
    # literal PPO objective -- a real red flag, not "expected" behavior).
    # Entropy trending toward 0 too fast = policy locking in a choice
    # before it's had enough exploration to find a good one.
    if history_critic_loss is not None and any(not np.isnan(v) for v in history_critic_loss):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), dpi=150, sharex=True)
        eps_x = np.arange(1, len(history_critic_loss) + 1)

        ax1.plot(eps_x, history_critic_loss, 'o-', color='steelblue', lw=1, ms=2, alpha=0.5,
                 label='Critic loss (MSE vs. GAE returns)')
        if len(history_critic_loss) >= window:
            valid = np.array([v if not np.isnan(v) else np.nan for v in history_critic_loss])
            if np.sum(~np.isnan(valid)) >= window:
                ma = np.convolve(np.nan_to_num(valid, nan=np.nanmean(valid)),
                                  np.ones(window) / window, mode='valid')
                ax1.plot(np.arange(window, len(history_critic_loss) + 1), ma,
                          '--', color='navy', lw=2, label=f'{window}-ep moving avg')
        ax1.set_ylabel('Critic loss'); ax1.set_yscale('log')
        ax1.set_title('Critic loss per episode (log scale)')
        ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

        if history_actor_loss is not None:
            ax2.plot(eps_x, history_actor_loss, 'o-', color='darkorange', lw=1, ms=2, alpha=0.5,
                      label='Actor loss (clipped surrogate)')
            if len(history_actor_loss) >= window:
                valid_a = np.array([v if not np.isnan(v) else np.nan for v in history_actor_loss])
                if np.sum(~np.isnan(valid_a)) >= window:
                    ma_a = np.convolve(np.nan_to_num(valid_a, nan=np.nanmean(valid_a)),
                                        np.ones(window) / window, mode='valid')
                    ax2.plot(np.arange(window, len(history_actor_loss) + 1), ma_a,
                              '--', color='firebrick', lw=2, label=f'{window}-ep moving avg (actor loss)')
        if history_entropy is not None:
            ax2b = ax2.twinx()
            ax2b.plot(eps_x, history_entropy, 'o-', color='seagreen', lw=1, ms=2, alpha=0.4,
                       label='Policy entropy')
            ax2b.set_ylabel('Policy entropy (nats)', color='seagreen')
        ax2.set_xlabel('Episode'); ax2.set_ylabel('Actor loss')
        ax2.set_title('Actor loss + policy entropy per episode')
        ax2.legend(fontsize=8, loc='upper left'); ax2.grid(True, alpha=0.3)

        plt.tight_layout(); plt.savefig('ppo_losses.png', dpi=150); plt.close()
        print("  PPO Loss/Entropy Plot Saved: ppo_losses.png")

if __name__ == '__main__':
    brain, res_arrival_arrived, res_arrival_timeout, res_arrival_counts, res_rewards, \
        res_baseline_arrived, res_baseline_timeout, res_baseline_counts, res_baseline_rewards, \
        res_history_far_frac, res_baseline_far_frac, \
        res_history_critic_loss, res_history_actor_loss, res_history_entropy = train()
    plot_training_results(arrival_arrived_only=res_arrival_arrived,
                           arrival_with_timeout=res_arrival_timeout,
                           arrival_counts=res_arrival_counts,
                           mean_episode_rewards=res_rewards,
                           baseline_arrived_only=res_baseline_arrived,
                           baseline_with_timeout=res_baseline_timeout,
                           baseline_counts=res_baseline_counts,
                           baseline_rewards=res_baseline_rewards,
                           history_far_frac=res_history_far_frac,
                           baseline_far_frac=res_baseline_far_frac,
                           history_critic_loss=res_history_critic_loss,
                           history_actor_loss=res_history_actor_loss,
                           history_entropy=res_history_entropy)