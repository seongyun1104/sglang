# I1 provider preflight: 2026-08-23 through 2026-09-01

Status: `BLOCKED_ON_PROVIDER_PRIVILEGES`

No I1 HBM/L2 profile or I2a row-order timing was executed.

## Checks

Three available H100 environments were checked without modifying unrelated
workspace data:

| Environment | GPU | Result | Disposition |
|---|---|---|---|
| pre-existing instance 48461971 | H100 NVL | `ERR_NVGPUCTRPERM`; clock lock denied | left untouched |
| rented instance 48463603 | single H100 NVL | clock lock denied | destroyed immediately |
| rented instance 48463967 | single H100 PCIe | clock lock denied | destroyed immediately |
| rented instance 49506714 | single H100 NVL | `ERR_NVGPUCTRPERM`; 1410 MHz clock lock denied | artifacts copied; destroyed immediately |
| Lium node `d56a8f27` / pod `07fcae4c` | single H100 PCIe, `ncu_profiling_enabled=true` | custom-image pod creation failed | no experiment; failed pod removed |
| Lium node `178b63ef` / pod `810da6a9` | single H100 PCIe, `ncu_profiling_enabled=true` | custom-image pod creation failed | no experiment; failed pod deleted through API |
| Lium node `d56a8f27` / pod `fe0e61ce` | single H100 PCIe, `ncu_profiling_enabled=true` | verified SGLang template pod creation failed | no experiment; failed pod deleted through API |

The three newly rented instances were stopped at the capability gate. They did not
run the latency anchor, I1 profiler target, or I2a matrix. Only the pre-existing
environment and instance 49506714 reached an NCU query that explicitly reported
counter denial. Instance 49506714 also attempted the real Torch kernel capture and
failed with `ERR_NVGPUCTRPERM`; no readable `.ncu-rep` was created. Its preflight
artifacts are stored outside this source checkout under:

```text
results/fa3_kv_aliasing_i1/provider_preflight_20260901_instance_49506714
```

The 2026-09-01 instance ran for approximately 9.6 minutes at $2.782/hour, for an
estimated compute and disk cost of about $0.45. Vast reported zero active instances
after teardown.

On 2026-09-01, Lium's authenticated `/executors` API independently confirmed that
the two single-H100 PCIe candidates had `ncu_profiling_enabled=true`. Both nodes
failed before exposing SSH, including one attempt with Lium's `VERIFY_SUCCESS`
SGLang template. No preflight script or CUDA workload ran. The Lium account balance
remained exactly $5.00 after all failed pods were deleted, and `lium ps --format
json` returned an empty list. These failures are provider provisioning failures,
not I1 capability results.

## Next acceptable environment

Proceed only when the provider confirms both capabilities and the local preflight
creates `PRECHECK_PASS`:

```text
nvidia-smi application-clock control is permitted
an actual CUDA kernel produces a readable NCU report
```

Do not rent another marketplace host based only on the GPU model or a generic
"verified" label. A provider must explicitly confirm both profiling capabilities,
or a host administrator must enable them, before another paid attempt.

For Lium specifically, require both the profiling API field and a pod that reaches
SSH readiness. Do not treat a `CREATION_FAILED` pod as a counter-permission test.
