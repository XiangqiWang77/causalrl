
#reward_model.reward_api=http://10.xxx.0.xxx:6009/get_reward2 \

export CUDA_VISIBLE_DEVICES=0,1,2,3

set -x

# export RAY_DEBUG=1
export WANDB_API_KEY=3f95fb3793cb54b9d6431d66d888927fb8b6d782
ray stop
ray start --head --node-ip-address=0.0.0.0 --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8266 --ray-debugger-external --num-gpus 4



data_dir=./data

model_path=/path/to/llama-3-8B
cur_task=0906AllJacobLLAMA
save_model_checkpoint=/path/to/train_models/$cur_task

echo "在主节点上启动训练..."
nohup python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.001 \
    data.train_files=$data_dir/train.parquet \
    data.val_files=$data_dir/test.parquet \
    data.train_batch_size=32 \
    data.max_prompt_length=512 \
    data.truncation=right \
    data.max_response_length=2048\
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.actor.optim.lr=2e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=3\
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=2600 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.68 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.n=6 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.reward_api=http://0.0.0.0:6010/get_reward2 \
    trainer.critic_warmup=1 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='GRPO-llama-8b-AllJacob' \
    trainer.experiment_name=$cur_task \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.default_local_dir=$save_model_checkpoint \
    trainer.test_freq=10 \
    trainer.total_epochs=10 $@ > AllJacobllama8B.out 2>&1 &
