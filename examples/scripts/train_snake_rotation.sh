


set -x
export NODE_RANK=0
PET_NODE_RANK=${PET_NODE_RANK:-0}
export RAY_MASTER_PORT=6379
export RAY_DASHBOARD_PORT=8264
export RAY_health_check_timeout_ms=20000
# export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=0
# export VLLM_LOGGING_LEVEL=DEBUG
# export CUDA_LAUNCH_BLOCKING=1
# export NCCL_DEBUG=TRACE
# export VLLM_TRACE_FUNCTION=1
# export NCCL_P2P_DISABLE=1 
pkill -u root python

SCRIPT_DIR=$(dirname "$0")
PARENT_DIR=$(dirname "$SCRIPT_DIR")
cd "$PARENT_DIR"
FILENAME=$(basename "$0")

if [ -z "$1" ]; then
  echo "Error: EXP argument is missing."
  echo "Usage: $0 <EXP_VALUE>"
  exit 1
fi
EXP="$1"

if [ -n "$2" ]; then
  MODEL_PATH="$2"
else
  MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"
fi
OUTPUT_DIR=""

export REWARD_LOG_PATH="${OUTPUT_DIR}/reward_test.log"
export WORKING_DIR=$PWD


ray stop
if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

if [ "$NODE_RANK" -eq 0 ]; then
    ray start --head  --port=$RAY_MASTER_PORT --dashboard-host=0.0.0.0 --dashboard-port=$RAY_DASHBOARD_PORT --num-gpus 6 --dashboard-grpc-port=30000 --metrics-export-port=30001 --num-cpus=32
else
    sleep 30
    ray start --address="$MASTER_ADDR:$RAY_MASTER_PORT" --num-gpus 6 --block
fi

sleep 30

BASE_DIR=""



if [ "$PET_NODE_RANK" -eq 0 ]; then
  RAY_ADDRESS="http://127.0.0.1:$RAY_DASHBOARD_PORT" ray job submit \
    --working-dir $WORKING_DIR \
    -- python3 -m openrlhf.cli.train_ppo_ray \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 6 \
    --ref_num_gpus_per_node 6 \
    --vllm_num_engines 6 \
    --vllm_tensor_parallel_size 1 \
    --vllm_gpu_memory_utilization 0.5 \
    --vllm_enable_sleep \
    --colocate_all_models \
    --pretrain ${MODEL_PATH}  \
    --remote_rm_url ./examples/scripts/reward_func_qwen_snake_rotation.py \
    --save_path ${OUTPUT_DIR} \
    --micro_train_batch_size 4 \
    --train_batch_size 120 \
    --micro_rollout_batch_size 8 \
    --rollout_batch_size 120 \
    --temperature 1.0 \
    --n_samples_per_prompt 6 \
    --max_epochs 1 \
    --num_episodes 3 \
    --prompt_max_len 3000 \
    --max_samples 1000000 \
    --generate_max_len 3000 \
    --advantage_estimator rloo \
    --zero_stage 3 \
    --bf16 \
    --actor_learning_rate 1e-6 \
    --init_kl_coef 0.0 \
    --prompt_data ${BASE_DIR}/metadata_${EXP}.jsonl \
    --input_key message \
    --normalize_reward \
    --flash_attn \
    --gradient_checkpointing \
    --save_steps 5  \
    --ckpt_path "${OUTPUT_DIR}/ckpt" \
    --max_ckpt_num 3 \
    --save_hf_ckpt \
    --lambd 1.00 \
    --limit_mm_per_prompt 4 \
    --wandb_run_name "${FILENAME}_${EXP}" \
    --freeze_prefix visual \
    --load_checkpoint | tee ${OUTPUT_DIR}/training.log
fi
# --runtime-env-json='{"setup_commands": ["pip install openrlhf[vllm]"]}' [Install deps]
# --ref_reward_offload [Offload to CPU]
# --remote_rm_url http://localhost:5000/get_reward
