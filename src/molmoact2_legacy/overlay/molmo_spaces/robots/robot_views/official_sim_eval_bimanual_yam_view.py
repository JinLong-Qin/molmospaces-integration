"""Robot view for the official MolmoAct2 sim_eval BimanualYAM MJCF."""

from typing import Literal

import numpy as np
from mujoco import MjData

from molmo_spaces.robots.robot_views.abstract import MJCFFrameMixin, GripperGroup
from molmo_spaces.robots.robot_views.bimanual_yam_view import (
    BimanualYamArmGroup,
    BimanualYamBaseGroup,
    BimanualYamRobotView,
)
from molmo_spaces.utils.mj_model_and_data_utils import body_pose


OFFICIAL_YAM_GRIPPER_OPEN_COMMAND = -0.0475
OFFICIAL_YAM_GRIPPER_CLOSED_COMMAND = 0.0


class OfficialSimEvalBimanualYamGripperGroup(MJCFFrameMixin, GripperGroup):
    """Official sim_eval YAM gripper with two directly actuated finger tips."""

    def __init__(
        self,
        mj_data: MjData,
        side: Literal["left", "right"],
        base_group: BimanualYamBaseGroup,
        namespace: str = "",
    ) -> None:
        model = mj_data.model
        self._namespace = namespace
        self._side = side
        self._gripper_prefix = f"{namespace}{side}_"

        joint_ids = [
            model.joint(f"{self._gripper_prefix}left_finger").id,
            model.joint(f"{self._gripper_prefix}right_finger").id,
        ]
        act_ids = [
            model.actuator(f"{self._gripper_prefix}gripper_left_tip").id,
            model.actuator(f"{self._gripper_prefix}gripper_right_tip").id,
        ]
        root_body_id = model.body(f"{self._gripper_prefix}link_6").id
        super().__init__(mj_data, joint_ids, act_ids, root_body_id, base_group)
        self._ee_site_id = model.site(f"{self._gripper_prefix}grasp_site").id

    @property
    def leaf_frame_id(self) -> int:
        return self._ee_site_id

    @property
    def leaf_frame_type(self):
        return "site"

    @property
    def side(self) -> str:
        return self._side

    def set_gripper_ctrl_open(self, open: bool) -> None:
        command = OFFICIAL_YAM_GRIPPER_OPEN_COMMAND if open else OFFICIAL_YAM_GRIPPER_CLOSED_COMMAND
        self.ctrl = np.array([command, command], dtype=np.float32)

    @property
    def inter_finger_dist_range(self) -> tuple[float, float]:
        return 0.0, abs(OFFICIAL_YAM_GRIPPER_OPEN_COMMAND) * 2.0

    @property
    def inter_finger_dist(self) -> float:
        return float(np.abs(self.joint_pos[0]) + np.abs(self.joint_pos[1]))

    @property
    def root_frame_to_world(self) -> np.ndarray:
        return body_pose(self.mj_data, self._root_body_id)


class OfficialSimEvalBimanualYamRobotView(BimanualYamRobotView):
    """Bimanual YAM view compatible with official MolmoAct2 sim_eval actuator names."""

    def __init__(self, mj_data: MjData, namespace: str = "") -> None:
        self._namespace = namespace
        base = BimanualYamBaseGroup(mj_data, namespace=namespace)
        move_groups = {
            "base": base,
            "left_arm": BimanualYamArmGroup(mj_data, "left", base, namespace=namespace),
            "right_arm": BimanualYamArmGroup(mj_data, "right", base, namespace=namespace),
            "left_gripper": OfficialSimEvalBimanualYamGripperGroup(
                mj_data, "left", base, namespace=namespace
            ),
            "right_gripper": OfficialSimEvalBimanualYamGripperGroup(
                mj_data, "right", base, namespace=namespace
            ),
        }
        super(BimanualYamRobotView, self).__init__(mj_data, move_groups)

    @property
    def name(self) -> str:
        return f"{self._namespace}official_sim_eval_bimanual_yam"
