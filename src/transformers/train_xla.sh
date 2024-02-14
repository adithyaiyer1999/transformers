# Setup envs
export PJRT_DEVICE=TPU
export XLA_USE_BF16=1
export XLA_IR_DEBUG=1
export XLA_HLO_DEBUG=1

export PROFILE_EPOCH=0
export PROFILE_STEP=3
export PROFILE_DURATION_MS=20000
export PROFILE_LOGDIR=/tmp/home/
# Some dependency 
# Run
cd ~/transformers
sudo pip3 install -e . --user
git checkout llama2-google-next-training
git pull origin llama2-google-next-training
python examples/pytorch/language-modeling/run_clm.py \
 --tokenizer_name hf-internal-testing/llama-tokenizer \
 --dataset_name wikitext \
 --dataset_config_name wikitext-2-raw-v1 \
 --per_device_train_batch_size 256 \
 --per_device_eval_batch_size 1 \
 --num_train_epochs 1 \
 --do_train \
 --output_dir /tmp/output \
 --overwrite_output_dir \
 --config_name "meta-llama/Llama-2-7b-hf" \
 --token "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" \
 --save_strategy no \
 --logging_strategy no \
 --remove_unused_columns no \
 --optim adafactor \
 --torch_dtype bfloat16 \
 --dataloader_drop_last yes \
 --block_size 1024 \
 --spmd_2d_sharding 1 \
 --spmd_grad_chkpt \
 --spmd_fsdp_sharding False \
