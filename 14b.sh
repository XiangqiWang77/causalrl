# 用1,2,3三张卡（0被占），这里不改也行
export CUDA_VISIBLE_DEVICES=1,2,3
set -x
export WANDB_API_KEY=3f95fb3793cb54b9d6431d66d888927fb8b6d782

ray stop
# 【改1】Ray 只注册 2 张 GPU，确保 rollout world_size=2
ray start --head --node-ip-address=0.0.0.0 --port=6378 \
  --dashboard-host=0.0.0.0 --dashboard-port=8265 --ray-debugger-external \
  --num-gpus 2

data_dir=./data
model_path=/groups/xzhang33/xwang76/qwen3-14b
cur_task=082614b_tp3
save_model_checkpoint=/groups/xzhang33/xwang76/train_models/$cur_task

nohup python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.kl_ctrl.kl_coef=0.001 \
  data.train_files=$data_dir/train.parquet \
  data.val_files=$data_dir/test.parquet \
  data.train_batch_size=1 \
  data.max_prompt_length=512 \
  data.truncation=right \
  data.max_response_length=1560 \
  actor_rollout_ref.model.path=$model_path \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-5 \
  actor_rollout_ref.actor.ppo_mini_batch_size=2 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=2100 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.rollout.enable_chunked_prefill=True \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_dtype=fp8 \
  +actor_rollout_ref.rollout.swap_space=24 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.n=2 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  reward_model.reward_api=http://0.0.0.0:6009/get_reward2 \
  trainer.critic_warmup=1 \
  trainer.logger=['console','wandb'] \
  trainer.project_name='GRPO-qwen-14b-CEGRPO' \
  trainer.experiment_name=$cur_task \
  trainer.n_gpus_per_node=2 \
  trainer.nnodes=1 \
  trainer.save_freq=400 \
  trainer.default_local_dir=$save_model_checkpoint \
  trainer.test_freq=20 \
  trainer.total_epochs=10 $@ > GRPOqwen14B_tp3.out 2>&1 &
