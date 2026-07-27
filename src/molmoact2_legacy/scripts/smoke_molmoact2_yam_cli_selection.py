"""No-rollout smoke test for MolmoAct2 YAM CLI/config selection."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/molmospaces-matplotlib")
os.environ.setdefault("NLTK_DATA", "/tmp/molmospaces-nltk-data")

try:
    import nltk

    nltk.download = lambda *args, **kwargs: True
except ImportError:
    pass

SCRIPTS_DATAGEN = Path(__file__).resolve().parent / "datagen"
sys.path.insert(0, str(SCRIPTS_DATAGEN))

import run_pipeline  # noqa: E402

from molmo_spaces.configs.policy_configs_baselines import (  # noqa: E402
    BimanualYamPiPolicyConfig,
    MolmoAct2YamPolicyConfig,
    PiPolicyConfig,
)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    run_pipeline.add_run_pipeline_args(parser)
    return parser


def test_molmoact2_yam_policy_choice_is_available() -> None:
    args = make_parser().parse_args(["--robot", "bimanual_yam", "--policy", "molmoact2_yam"])
    assert args.robot == "bimanual_yam"
    assert args.policy == "molmoact2_yam"
    assert run_pipeline.resolve_task_type(args) == "packing"

    default_args = make_parser().parse_args([])
    assert run_pipeline.resolve_task_type(default_args) == "pick"


def test_policy_config_selection_preserves_existing_pi_routes() -> None:
    assert isinstance(run_pipeline.get_policy_config("pi", robot="droid"), PiPolicyConfig)
    assert isinstance(
        run_pipeline.get_policy_config("pi", robot="bimanual_yam"),
        BimanualYamPiPolicyConfig,
    )


def test_policy_config_selection_allows_only_bimanual_yam() -> None:
    config = run_pipeline.get_policy_config("molmoact2_yam", robot="bimanual_yam")
    assert isinstance(config, MolmoAct2YamPolicyConfig)
    assert config.endpoint_url is not None or config.remote_config["host"]
    assert config.remote_config["path"] == "/act"
    assert config.timeout > 0
    assert config.num_steps > 0
    assert config.gripper_max == 0.041
    assert config.gripper_scale_source == "molmospaces_bimanual_yam_xml_ctrlrange"
    assert config.instruction_override == "Put everything into the box."
    assert config.debug_dump_max_calls == 3
    assert "molmoact2_yam_debug" in config.debug_dump_dir
    assert config.camera_mapping == {
        "top_cam": "exo_camera",
        "left_cam": "left_wrist_camera",
        "right_cam": "right_wrist_camera",
    }

    try:
        run_pipeline.get_policy_config("molmoact2_yam", robot="droid")
    except ValueError as exc:
        message = str(exc)
        assert "--policy molmoact2_yam" in message
        assert "--robot bimanual_yam" in message
    else:
        raise AssertionError("molmoact2_yam must reject non-bimanual_yam robots")


def test_tabletop_pick_config_uses_official_like_yam_camera_targets() -> None:
    from molmo_spaces.data_generation.config.molmoact2_yam_tabletop_config import (
        MolmoAct2YamTabletopNearPathPickDataGenConfig,
        MolmoAct2YamTabletopPackingDataGenConfig,
        MolmoAct2YamTabletopPickDataGenConfig,
    )

    for config_cls in (
        MolmoAct2YamTabletopPickDataGenConfig,
        MolmoAct2YamTabletopNearPathPickDataGenConfig,
        MolmoAct2YamTabletopPackingDataGenConfig,
    ):
        config = config_cls()
        assert config.camera_config.img_resolution == (640, 360)

        cameras = {camera.name: camera for camera in config.camera_config.cameras}
        assert set(cameras) == {"top_cam", "left_cam", "right_cam"}
        assert cameras["top_cam"].reference_body_names == ["robot_0/bimanual_base", "robot_0/base"]
        assert cameras["top_cam"].camera_offset == [0.15, 0.0, 0.8]
        assert cameras["top_cam"].fov == 69.4
        assert cameras["left_cam"].reference_body_names == ["robot_0/left_link_6"]
        assert cameras["right_cam"].reference_body_names == ["robot_0/right_link_6"]
        assert cameras["left_cam"].camera_offset == [0.0, 0.09, 0.06]
        assert cameras["right_cam"].camera_offset == [0.0, 0.09, 0.06]
        assert cameras["left_cam"].fov == 87.0
        assert cameras["right_cam"].fov == 87.0
        assert config.policy_config.camera_mapping == {
            "top_cam": "top_cam",
            "left_cam": "left_cam",
            "right_cam": "right_cam",
        }


def main() -> None:
    test_molmoact2_yam_policy_choice_is_available()
    test_policy_config_selection_preserves_existing_pi_routes()
    test_policy_config_selection_allows_only_bimanual_yam()
    test_tabletop_pick_config_uses_official_like_yam_camera_targets()
    print("molmoact2_yam_cli_selection smoke passed")


if __name__ == "__main__":
    main()
