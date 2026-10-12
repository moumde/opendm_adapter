script/dm05_launcher.sh   --exp playground/dm05_cr3_o6_sft.py   --task inference   --model-config.model-name-or-path ./user_checkpoints/dm05_cr3_o6_100k_fresh/checkpoint-80000   --model-config.chunk-size 50   --model-config.precision-policy bf16_mixed   --model-config.llm-attn-implementation sdpa   --model-config.vision-attn-implementation sdpa   --model-config.action-attn-implementation sdpa   --inference-config.output-action-dim 12   --inference-config.image-prompts "Head" "Left wrist" "Right wrist"   --inference-config.port 7891


LD_LIBRARY_PATH=/home/je/anaconda3/envs/openpi/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} \
PYTHONPATH=/home/je/code/linkerhand-python-sdk:/home/je/code/opendm/deploy${PYTHONPATH:+:$PYTHONPATH} \
/home/je/anaconda3/envs/openpi/bin/python \
deploy/deploy_dm05_cr3_o6.py \
  --server-host 127.0.0.1 \
  --server-port 7891 \
  --robot-ip 192.168.2.14 \
  --transport rs485 \
  --modbus /dev/ttyUSB0 \
  --confirm-motion \
  --power-on \
  --duration-s 60


PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True script/dm05_launcher.sh   --exp playground/dm05_cr3_o6_sft.py   --nproc_per_node 1   --task train   --model-config.model-name-or-path ./checkpoints/DM05   --model-config.precision-policy bf16_mixed   --model-config.llm-attn-implementation sdpa   --use-lora True   --trainer-config.per-device-train-batch-size 4   --trainer-config.num-train-steps 100000   --trainer-config.save-steps 20000   --trainer-config.output-dir ./user_checkpoints/dm05_cr3_o6_100k_fresh   2>&1 | tee /tmp/dm05_cr3_o6_100k_fresh.log


{'loss': '0.01642', 'grad_norm': '0.6406', 'learning_rate': '4.673e-06', 'fm_loss': '0.01642', 'step': 80085, 'epoch': '6.597'}
{'loss': '0.009717', 'grad_norm': '0.4648', 'learning_rate': '4.673e-06', 'fm_loss': '0.009717', 'step': 80086, 'epoch': '6.597'}
{'loss': '0.01855', 'grad_norm': '0.5938', 'learning_rate': '4.673e-06', 'fm_loss': '0.01855', 'step': 80087, 'epoch': '6.597'}
{'loss': '0.02462', 'grad_norm': '0.4785', 'learning_rate': '4.672e-06', 'fm_loss': '0.02462', 'step': 80088, 'epoch': '6.597'}
{'loss': '0.01142', 'grad_norm': '0.5078', 'learning_rate': '4.672e-06', 'fm_loss': '0.01142', 'step': 80089, 'epoch': '6.597'}
{'loss': '0.0107', 'grad_norm': '0.4141', 'learning_rate': '4.672e-06', 'fm_loss': '0.0107', 'step': 80090, 'epoch': '6.597'}
