# Group Reward-Decoupled Normalization Policy Optimization (GDPO)

Last updated: 11/26/2026.

GDPO normalizes each configured reward component independently within a rollout group,
then aggregates the normalized components with the configured weights. This preserves
contrast from a component that would be hidden by summing raw rewards first.

This filter draft depends on the V1 GDPO transport prerequisite in
[verl #8424](https://github.com/verl-project/verl/pull/8424): the V1 advantage path
must provide GDPO component scores, prompts, and attention masks to the production
estimator. This change intentionally adds no transport implementation and is not an
end-to-end GDPO training fix on an unpatched main checkout.

## Configuration

GDPO training uses the existing algorithm settings:

```yaml
algorithm:
  adv_estimator: gdpo
  gdpo_reward_keys: [format_reward, accuracy_reward]
  gdpo_reward_weights: [1.0, 1.0] # optional; equal weights are the default
  norm_adv_by_std_in_grpo: true # required for the V1 filter's current caller path
```

The built-in V1 sampler can opt into GDPO-aware DAPO filtering with:

```yaml
algorithm:
  filter_groups:
    enable: true
    metric: gdpo
```

With this opt-in metric, each finished group is filtered only when its weighted,
per-component GRPO-normalized signal is constant within the group. The check uses
the same FP32 arithmetic and `1e-6` epsilon as the GRPO helper. The built-in V1
filter currently requires `norm_adv_by_std_in_grpo: true`, matching the production
GDPO caller; the `false` path is not forwarded by that caller yet (see
[verl #8754](https://github.com/verl-project/verl/pull/8754)). It happens before
GDPO's final batch-level whitening, which is applied only after sampling. Single-
trajectory groups are retained. The supported layout is one output per session;
if existing trajectory keys show multiple output indices for one `uid/session`,
the GDPO filter fails closed. It does not claim session-level correctness for
multi-output groups. Missing configured component fields or a weight/key length
mismatch is an explicit configuration/data error. Weights may be signed, but must
be finite with at least one non-zero value; all-zero and non-finite weights are
rejected.

The existing `reward`, `acc`, `score`, and other named filtering metrics keep their
current semantics. Custom samplers are not changed by this option; the built-in V1
ReplayBuffer and ReplayBufferAsync are the supported implementations.

GPU-scale convergence and end-to-end training quality have not been validated by
this CPU filter change.
