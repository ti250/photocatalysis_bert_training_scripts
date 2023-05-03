module load conda; conda activate /grand/projects/SolarWindowsADSP/taketomo/polaris_conda

export HTTP_PROXY="http://proxy-01.pub.alcf.anl.gov:3128"
export HTTPS_PROXY="http://proxy-01.pub.alcf.anl.gov:3128"
export http_proxy="http://proxy-01.pub.alcf.anl.gov:3128"
export https_proxy="http://proxy-01.pub.alcf.anl.gov:3128"
export ftp_proxy="http://proxy-01.pub.alcf.anl.gov:3128"
export no_proxy="admin,polaris-adminvm-01,localhost,*.cm.polaris.alcf.anl.gov,polaris-*,*.polaris.alcf.anl.gov,*.alcf.anl.gov"


echo Using Python `which python`

echo Logging into WandB...

export WANDB__SERVICE_WAIT=300
wandb login $WANDB_KEY

echo WandB login successful

NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE=4
NPROCESSES=$(( NNODES * NRANKS_PER_NODE ))
BASE_MODEL_NAME=`echo "${MODEL_NAME##*/}"`
RUN_NAME="$BASE_MODEL_NAME"-squad2-"$SUFFIX"
PYTHONPATH=$PBS_O_WORKDIR
OUTNAME=/grand/projects/SolarWindowsADSP/taketomo/polaris_outputs/$RUN_NAME
MAIN_IP_ADDR=$(hostname -i)
WANDB_PROJECT=finetune-existing
HF_HOME=/grand/projects/SolarWindowsADSP/taketomo/huggingface/
NCCL_COLLNET_ENABLE=1
NCCL_NET_GDR_LEVEL=PHB

export MPICH_GPU_SUPPORT_ENABLED=1

echo $RUN_NAME
echo $OUTNAME
echo Working directory is $PBS_O_WORKDIR

echo Jobid: $PBS_JOBID
echo Running on host `hostname`
echo Running on nodes `cat $PBS_NODEFILE`

ds_report


HOSTFILE=hostfile-`echo $MAIN_IP_ADDR | sed 's/ //g'`

rm $HOSTFILE
touch $HOSTFILE
cat $PBS_NODEFILE > $HOSTFILE
sed -e 's/$/ slots=4/' -i $HOSTFILE

echo Hostfile path $HOSTFILE
cat $HOSTFILE

rm .deepspeed_env
touch .deepspeed_env
echo "PATH=${PATH}" >> .deepspeed_env
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" >> .deepspeed_env
echo "http_proxy=${http_proxy}" >> .deepspeed_env
echo "https_proxy=${https_proxy}" >> .deepspeed_env
echo "no_proxy=${no_proxy}'" >> .deepspeed_env
echo "ftp_proxy=${ftp_proxy}'" >> .deepspeed_env
echo "HTTP_PROXY=${HTTP_PROXY}'" >> .deepspeed_env
echo "HTTPS_PROXY=${HTTPS_PROXY}'" >> .deepspeed_env
echo "HF_HOME=${HF_HOME}" >> .deepspeed_env
echo "WANDB_PROJECT=${WANDB_PROJECT}" >> .deepspeed_env

deepspeed \
    --hostfile=$HOSTFILE \
polaris_scripts/run_qa.py \
    --model_name_or_path "$MODEL_NAME" \
    --dataset_name squad_v2 \
    --report_to wandb \
    --do_train \
    --do_eval \
    --learning_rate $LR \
    --num_train_epochs $NUM_EPOCHS \
    --max_seq_length $MAX_SEQ_LENGTH \
    --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
    --doc_stride 128 \
    --seed 0 \
    --output_dir "$OUTNAME" \
    --run_name $RUN_NAME \
    --change_cls_token $CHANGE_CLS_TOKEN \
    --no_pad_to_max_length \
    --version_2_with_negative \
    --lr_scheduler_type $LR_SCHEDULER_TYPE \
    --save_total_limit 5 \
    --save_steps 10000 \
    --warmup_ratio $WARMUP_RATIO \
    --deepspeed polaris_scripts/ds_config.json

rm $HOSTFILE
