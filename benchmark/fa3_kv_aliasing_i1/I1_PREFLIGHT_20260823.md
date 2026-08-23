# I1 provider preflight: 2026-08-23

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

The two newly rented instances were stopped at the capability gate. They did not
run the latency anchor, I1 profiler target, or I2a matrix. Only the pre-existing
environment reached an NCU query that explicitly reported counter denial. The
new candidates failed clock control before a kernel counter capture was attempted.

## Next acceptable environment

Proceed only when the provider confirms both capabilities and the local preflight
creates `PRECHECK_PASS`:

```text
nvidia-smi application-clock control is permitted
an actual CUDA kernel produces a readable NCU report
```

Randomly renting more marketplace hosts is not authorized by this receipt. A
provider with explicit profiling privileges or a host administrator who enables
the counters is required.
