# Outcome-based Batch Exploration baseline (Song et al. 2025, arXiv:2509.06941).
# =============================================================================
# IDENTICAL to train_grpo.sh in every training knob. The ONLY differences:
#   adv_estimator   grpo          -> grpo_batch     (A_i = GRPO_adv_i + c·b_i)
#   +combine_mode   em_fmt_doc    -> batch_outcome  (stash R_acc, log final answer)
#   +batch_explore_c (new)        -> 0.2            (Eq.2 coefficient c; Qwen setting)
# within-batch answer-repetition penalty:
#   b_i = −(1/N) Σ_{j≠i} 1{a_i = a_j}   (a = normalized final answer)
# added UN-NORMALIZED at the advantage level (faithful Eq.2), NOT correct-gated
# (diverse answers on unsolved questions is the method's point). R_fmt dropped
# via use_format_reward=false, matching the rest of the baseline family.
# See main_ppo.py combine_mode=='batch_outcome' + ray_trainer.py adv_estimator=='grpo_batch'.

export CUDA_VISIBLE_DEVICES=0,1,2,3
export DATA_DIR=/data/dongwan/nq_hotpotqa_train
export TRAIN_DATA_DIR=$DATA_DIR
export TEST_DATA_DIR=$DATA_DIR

WAND_PROJECT='RDPO-3B'

export BASE_MODEL='Qwen/Qwen2.5-3B'
export EXPERIMENT_NAME=Batch-3B

export VLLM_ATTENTION_BACKEND=XFORMERS
export RAY_TMPDIR=/data/dongwan/ray_tmp
export TMPDIR=/data/dongwan/ray_tmp
export WANDB_DIR=/data/dongwan/wandb
export WANDB_CACHE_DIR=/data/dongwan/wandb_cache
export HF_HOME=/data/huggingface
export NCCL_P2P_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=$TRAIN_DATA_DIR/train.parquet \
    data.val_files=$TEST_DATA_DIR/test.parquet \
    data.train_data_num=null \
    data.val_data_num=null \
    data.train_batch_size=128 \
    data.val_batch_size=256 \
    data.max_prompt_length=6656 \
    data.max_response_length=512 \
    data.max_start_length=2048 \
    data.max_obs_length=512 \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=grpo_batch \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size=16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=64 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=64 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.n_agent=5 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.actor.state_masking=true \
    algorithm.no_think_rl=false \
    trainer.logger=['wandb'] \
    +trainer.val_only=false \
    +trainer.val_before_train=false \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=0 \
    trainer.project_name=\'$WAND_PROJECT\' \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=15 \
    trainer.total_training_steps=201 \
    trainer.default_local_dir=/data/dongwan/verl_checkpoints/$EXPERIMENT_NAME \
    max_turns=4 \
    retriever.url="http://127.0.0.1:8000/retrieve" \
    retriever.topk=3 \
    +combine_mode=batch_outcome \
    +batch_explore_c=0.2 \
    +use_format_reward=false \
    +mask_answer_span=true \
    2>&1 | tee /data/dongwan/$EXPERIMENT_NAME.log
