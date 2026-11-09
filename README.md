# OpenDM on Dobot CR3 and LinkerHand O6

This repository records an experimental adaptation of [OpenDM](https://github.com/dexmal/opendm) for a Dobot CR3 arm with a LinkerHand O6. My work focused on the robot-specific data path, fine-tuning experiments, and deployment integration.

## Project Work

- Registered the CR3/O6 motor-recognition dataset in [`opendm/dataset/cr5_o6.py`](opendm/dataset/cr5_o6.py).
- Added a task configuration in [`playground/dm05_cr3_o6_sft.py`](playground/dm05_cr3_o6_sft.py).
- Implemented CR3/O6 robot and camera interfaces and an inference deployment entry point under [`deploy/`](deploy/).

LoRA fine-tuning was explored during debugging. Grasping performance on the physical setup remained inconsistent and below the desired level; this work should be viewed as an experiment, not as a validated or production-ready grasping system.

> **Configuration note:** The checked-in task configuration currently sets `use_lora=False`. It does not, by itself, reproduce an LoRA run. No quantitative success-rate benchmark is claimed here.

## Current Limitations

The observed grasping result is not yet satisfactory. Potential contributors include the limited task-specific demonstrations, the gap between the pretrained model's data and the CR3/O6 setup, and the fine-tuning choices. These are hypotheses; the experiments in this repository do not isolate their individual effects.

## Upstream Framework

For OpenDM model details, general installation, training, and inference documentation, see the [upstream OpenDM repository](https://github.com/dexmal/opendm). This repository focuses on the CR3/O6 experiment and integration.

## License

See [LICENSE](LICENSE).
