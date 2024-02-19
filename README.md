# Running LLAMA-2 7B/13B/70B on TPU-pods with SPMD

We'll just use the huggingface wikitext data as an example, and train for 3 epochs to see speed for different batch sizes

## Step 1 : Create VM

Modify the name, and the accelerator type based on the model size. Traditionally - 7B runs on 32 pods well, 13B requires 128 pod. 70B as of now seems to go OOM on even 256 pods.

```
export TPU_NAME=adi-128-1
export ZONE=us-central2-b
export ACCELERATOR_TYPE=v4-128
export RUNTIME_VERSION=tpu-ubuntu2204-base
export PROJECT=nyu-vision-lab


gcloud alpha compute tpus queued-resources create $TPU_NAME \
 --node-id=$TPU_NAME \
 --zone=$ZONE \
 --project=$PROJECT \
 --accelerator-type=$ACCELERATOR_TYPE \
 --runtime-version=$RUNTIME_VERSION \
 --best-effort
```

## Step 2: Install libraryies, and build huggingface from the github branch

It is important to use the `spmd_tutorial` branch to install the libraries, there are changes to the huggingface `Trainer.py` class and `run_clm.py` file.

```
gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} \
	--zone ${ZONE} \
	--project us-central2-b \
	--worker=all \
	--command='
# Step 1: install torch, torch-xla, libtpu
pip install torch~=2.2.0 torch_xla[tpu]~=2.2.0 -f https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-nightly-cp38-cp38-linux_x86_64.whl

# Step 2: install HF
git clone -b spmd_tutorial https://github.com/adithyaiyer1999/transformers.git
cd transformers
sudo pip3 install -e . --user
pip3 install datasets accelerate evaluate scikit-learn
'
```

## Step 3 : Train the model

```
gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} \
--zone us-central2-b \
--project ${PROJECT} \
--worker=all \
--command='
export PJRT_DEVICE=TPU
export XLA_USE_BF16=1
export XLA_IR_DEBUG=1
export XLA_HLO_DEBUG=1

export PROFILE_EPOCH=0
export PROFILE_STEP=3
export PROFILE_DURATION_MS=20000
export PROFILE_LOGDIR=/tmp/home/

# Install
pip install torch~=2.2.0 torch_xla[tpu]~=2.2.0 -f https://storage.googleapis.com/libtpu-releases/index.html
pip3 install regex tokenizers==0.14.1
pip3 install -U huggingface-hub==0.19.4

# Run
cd transformers
git checkout spmd_tutorial
git pull origin spmd_tutorial
sudo pip3 install -e . --user
python examples/pytorch/language-modeling/run_clm.py \
 --tokenizer_name hf-internal-testing/llama-tokenizer \
 --dataset_name wikitext \
 --dataset_config_name wikitext-2-raw-v1 \
 --per_device_train_batch_size 256 \
 --per_device_eval_batch_size 1 \
 --num_train_epochs 3 \
 --do_train \
 --output_dir /tmp/output \
 --overwrite_output_dir \
 --config_name "meta-llama/Llama-2-13b-hf" \
 --token "add your hf token" \
 --save_strategy no \
 --logging_strategy no \
 --remove_unused_columns no \
 --optim adafactor \
 --torch_dtype bfloat16 \
 --dataloader_drop_last Yes \
 --block_size 1024 \
 --spmd_2d_sharding 1 \
 --spmd_dcn_parallelism 1 \
 --spmd_grad_chkpt
'

```

Some parameters to keep in mind : 
0. Replace with your HF token with llama2 access
1. `dataset_name` and `dataset_config_name` can be modified based on which data you want to train on
2. VERY IMP : `per_device_train_batch_size` argument actually refers to the *GLOBAL BATCH SIZE*, and not local per device. This seems to be a bug in HF training [Link](https://github.com/huggingface/transformers/issues/21444). 
3. Replace `config_name` with the model of your liking
4. `spmd_2d_sharding` refers to how you want to shard the grid. Keeping it 1 makes data and model shard across all devices. For larger models, using a value of 4 is recommended for 256 pods+ devices. More details on this sharding logic coming soon. Ref : https://pytorch.org/blog/high-performance-llama-2/


## Results : 

![Image](./Screenshot%202024-02-18%20at%209.14.16%20PM.png)