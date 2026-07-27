# MolmoAct2 → MolmoSpaces Legacy Workline

## Summary

This workline connected MolmoAct2-BimanualYAM to MolmoSpaces and tested whether the official MolmoAct2 YAM policy could run closed-loop through a MolmoSpaces adapter. It is included because it is part of the broader MolmoSpaces integration effort, but it is a legacy/ended line rather than the current public success claim.

## What worked

- MolmoAct2 `/act` request/response plumbing was connected to MolmoSpaces.
- Three-camera packets, simulator stepping, action decoding, HDF5/MP4 artifact generation, and partial approach/contact/grasp/lift behavior were observed.
- Official MolmoAct2 SAPIEN `sim_eval` produced one genuine successful episode under its own official environment. That proves official MolmoAct2 sim_eval capability, not MolmoSpaces adapter success.

## What did not hold

- Stable complete task success inside MolmoSpaces was not established.
- Later evidence showed policy-side flow-matching randomness was not fully controlled by the simulator seed, so repeated nominally identical environment seeds could produce different policy actions.
- Fixes to action-frame and execution semantics improved partial behavior, but did not establish reliable placement success.

## Public evidence boundary

Valid public claim: MolmoAct2 was connected to MolmoSpaces as a diagnostic adapter, and the integration exposed embodiment/distribution/action-frame/policy-RNG issues.

Invalid claim: do not state that MolmoAct2 was successfully reproduced inside MolmoSpaces as a stable complete bimanual task solution.

## Source material retained from the workdir

The legacy runtime archive contained adapter overlays, diagnostic scripts, tests, scene examples, reports, and many runtime artifacts. Public Git keeps only the reusable code/docs/lightweight summaries or links them from this index. Large videos, PPM frames, HDF5 files, raw action JSONL traces, and private runtime logs are intentionally excluded from normal Git.
