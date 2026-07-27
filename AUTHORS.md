# Authors and Attribution

This repository is a research integration snapshot based on MolmoSpaces.

## Upstream MolmoSpaces

- Project: MolmoSpaces
- Upstream repository: https://github.com/allenai/molmospaces
- Original copyright: Copyright 2026 Allen Institute for AI
- License: Apache License 2.0, retained in `LICENSE`

The MolmoSpaces source tree, examples, documentation, resources, package metadata, and simulator infrastructure originate from the upstream Allen Institute for AI project unless otherwise noted.

## Upstream MimicGen

- Project: MimicGen
- Upstream repository: https://github.com/NVlabs/mimicgen
- Organization: NVIDIA / NVlabs
- Paper authors: Ajay Mandlekar, Soroush Nasiriany, Bowen Wen, Iretiayo Akinola, Yashraj Narang, Linxi Fan, Yuke Zhu, and Dieter Fox
- Paper: "MimicGen: A Data Generation System for Scalable Robot Learning using Human Demonstrations", CoRL 2023
- Code license: NVIDIA Source Code License
- Dataset license: CC-BY 4.0

The MimicGen method, upstream codebase, documentation, and datasets are credited to the original NVIDIA / NVlabs project and paper authors. This repository does not vendor the MimicGen source tree; `tools/setup_mimicgen_dependency.sh` fetches it as an external dependency.

## MolmoSpaces x MimicGen Integration

- Integration author: Kunyu Yang
- GitHub: https://github.com/yanggoumao2
- Affiliation: Institute of Trustworthy Embodied Intelligence, Fudan University

Kunyu Yang authored and organized the integration layer connecting MolmoSpaces Pick-and-Place rollouts with MimicGen-style source selection, datagen-info extraction, robomimic/MimicGen HDF5 conversion, rollout generation, action-hash deduplicated collection, bimanual YAM browser/keyboard teleoperation utilities, result summaries, public demo media, and release documentation contained under `src/`, `results/`, `media/`, `docs/mimicgen_integration_readme.md`, `docs/experiments.md`, and `docs/repository_layout.md`.

## Data and External Dependencies

MolmoBot-Data, MimicGen, robomimic, MuJoCo, and other third-party dependencies retain their own licenses and terms. Large datasets, generated HDF5 files, full rollout video directories, and local runtime logs are intentionally excluded from this repository.
