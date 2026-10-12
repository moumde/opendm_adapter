"""Dobot CR3 + LinkerHand O6 dataset registration."""

from opendm.constants.robot import ROBOT_STATE_DESCS, RobotType
from opendm.dataset.register import register_dataset


register_dataset(
    {
        "motor_recognition": {
            "jsonl_dir": "./data/cr5_o6_20250917/jsonl",
            "image_dir": "./data/cr5_o6_20250917/videos",

            "image_keys": [
                "images_1",
                "images_2",
                "images_3",
            ],

            "image_prompts": [
                "Head",
                "Left wrist",
                "Right wrist",
            ],

            "robot_type": RobotType.CR3_O6,

            "state_desc": ROBOT_STATE_DESCS[
                RobotType.CR3_O6
            ],
        },
    },
    prefix="cr5_o6",
)