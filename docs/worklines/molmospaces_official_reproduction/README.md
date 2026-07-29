# Bounded Official MolmoSpaces Reproduction Workline

## Purpose

This workline preserves bounded evidence from earlier official MolmoSpaces entrypoint and asset reproduction attempts. It is kept separate from MimicGen Pick-and-Place, MolmoAct2, and bimanual YAM worklines.

## Public code and evidence

This repository keeps the current MolmoSpaces source snapshot and a lightweight inventory of the bounded reproduction artifacts:

- Source snapshot: repository root, especially `molmo_spaces/`, `molmo_spaces_isaac/`, and `molmo_spaces_maniskill/`.
- Inventory: [`results/workline_index/molmospaces_official_reproduction.md`](../../../results/workline_index/molmospaces_official_reproduction.md)

## Minimal run sequence

After standard setup, use the upstream-style package entrypoints and documentation from the root repository. This workline does not add a separate public script directory because it was a bounded check of official entrypoints/assets rather than a new integration pipeline.

For official RB-Y1 scripted/planner data generation, the relevant entrypoint is `python -m molmo_spaces.data_generation.main ...` with configs such as `RBY1PickDataGenConfig` and `RBY1PickAndPlaceDataGenConfig`. Those configs require `curobo`; a plain `pip install -e ".[mujoco]"` environment is not sufficient. This official datagen path is distinct from the repository's public `src/pnp/` integration scripts.

In addition to successful `curobo` installation, the planner-backed policy path also requires a working planner-server configuration for the target environment. Reproduction guidance should therefore verify both config construction and planner-server connectivity, rather than assuming one universal host or one universal local-only setup.

Basic local checks:

```bash
python -m py_compile $(find molmo_spaces -name '*.py' -type f | sort)
python -m py_compile $(find molmo_spaces_isaac molmo_spaces_maniskill -name '*.py' -type f | sort)
```

Full benchmark reproduction requires official assets, simulator setup, and benchmark-scale runs; those raw artifacts are not committed in normal Git.

## Evidence boundary

Valid claim: bounded official-entrypoint/assets evidence was preserved and indexed.

Invalid claim: do not describe this as a complete MolmoSpaces benchmark reproduction, a full paper reproduction, or bimanual YAM behavior evidence.
