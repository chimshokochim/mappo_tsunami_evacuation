# Shared-Actor MAPPO-Lagrangian Design

## Constraint model

All agents use one shared policy, `pi_theta`, and one shared multiplier per
cost type, `lambda_j`. Agent indices are therefore sample indices, not separate
policy parameters. The constrained surrogate at policy update `k` is

```text
max_theta min_{lambda_j >= 0}
    E[A_{pi_k}]
    - sum_j lambda_j (J_j(pi_k) - c_j + E[A_{j,pi_k}]),
```

with the usual PPO trust-region approximation. `A` is the reward advantage;
`A_j` is the advantage for cost `j`. The implementation keeps the lower cost
index `j` and does not introduce reward/cost superscripts.

## Sampled implementation

For decision sample `n`,

```text
A_hat_n     = R_hat_n - V_phi(s_n)
A_hat_jn    = G_hat_jn - V_{j,psi}(s_n)
M_n(lambda) = A_hat_n - sum_j lambda_j A_hat_jn.
```

The actor maximizes the PPO-Clip surrogate formed with `M_n(lambda)`:

```text
r_n(theta) = pi_theta(a_n|o_n) / pi_old(a_n|o_n)

L_clip = mean_n min(
    r_n M_n,
    clip(r_n, 1-epsilon, 1+epsilon) M_n
)

actor_loss = -L_clip - entropy_coefficient * mean_entropy.
```

The reward critic and the multi-output cost critic minimize

```text
reward_critic_loss = 0.5 * mean_n (V_phi(s_n) - R_hat_n)^2
cost_critic_loss   = 0.5 * mean_{n,j} (V_{j,psi}(s_n) - G_hat_jn)^2.
```

Once per complete rollout, rather than once per minibatch or PPO epoch,

```text
J_hat_j  = mean_n G_hat_jn
lambda_j = clip(
    lambda_j + alpha_lambda_j * (J_hat_j - c_j),
    0,
    lambda_max
).
```

This multiplier update is a direct on-policy primal-dual estimate of the
observed constraint violation. It is intentionally simpler than the paper's
importance-ratio form in Equation (28).

## First cost

The environment currently provides one cost:

```text
G_hat_1n = 1 if agent n failed, otherwise 0
c_1       = 0
```

Because the cost is non-negative, an expected mean cost no greater than zero
means zero failure probability. Every agent makes one shelter decision, so no
temporal cost GAE is required for this first implementation.

The arrays and networks still use shape `[num_decisions, num_costs]`, allowing
future columns such as capacity rejection and tsunami death to be represented
as separate constraints.

## Reward/cost separation

In `constraint_mode=lagrangian`, the training reward target contains the team
efficiency reward-to-go but not the failure penalty. `per_agent_failed` is
stored separately as the cost return. This prevents one failure from being
penalized simultaneously through both reward and constraint channels.

The legacy failure-penalty-inclusive episode return remains available as an
evaluation metric, and an efficiency-only episode return is logged separately.

## Relation to paper Equations (25)-(31)

- Equation (25): shared-policy Lagrangian advantage, summed over cost types.
- Equation (26): PPO-Clip actor update using that advantage.
- Equation (27): constraint violation; here estimated directly from rollout
  cost returns.
- Equations (28)-(29): replaced by the projected observed-violation update
  above.
- Equation (30): unnecessary because there is no sequential update of separate
  per-agent actors.
- Equation (31): retained as the reward critic MSE; the cost critic uses the
  analogous MSE for cost returns.

`constraint_mode=none` preserves the original PPO objective and does not create
a cost critic or multipliers.
