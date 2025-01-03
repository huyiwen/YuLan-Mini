# setup env on each node in the slurm job

LOG_PREFIX=log/"$SLURM_JOB_NAME-$SLURM_JOB_ID"
LOG_DIR=/home/u20140041/pretrain-mini/${LOG_PREFIX}
echo $(date +%Y-%m-%d-%H:%M:%S) > $LOG_FILE
echo Setup hostname: $(hostname) >> $LOG_FILE
LOG_FILE=/home/u20140041/pretrain-mini/${LOG_PREFIX}/part0.log
echo "========================" >> $LOG_FILE
FILES_TO_LOG=($0 train.py train_utils.py model/modeling_miniyulan.py model/configuration_miniyulan.py torchrun_wrapper.sh)
mkdir -p $LOG_DIR/artifacts
for file in ${FILES_TO_LOG[@]}; do
  echo $file >> $LOG_FILE
  cat $file >> $LOG_FILE
  cat $file >> $LOG_DIR/artifacts/$(echo $file | tr '/' '-')
  echo "========================" >> $LOG_FILE
done

set -x

source ~/.bashrc
source .venv/bin/activate  # venvbashrc

# 传递参数
FETCH_TIME=$1  # 没有默认值，需要在 submit_to_slurm.sh 中填写
PER_DEVICE_TRAIN_BATCH_SIZE=$2  # 默认值为 18（对应 7 节点）
DATASET_MODEL_NAME=$3  # 默认值为 myl

# 计算相关环境变量
NNODES=$SLURM_JOB_NUM_NODES
export WORLD_SIZE=$(expr $NNODES \* 8)
hostnames=$(scontrol show hostnames $SLURM_JOB_NODELIST)
comma_hostnames=$(echo $hostnames | tr ' ' ',')
export MASTER_ADDR=$(echo $hostnames | cut -d ' ' -f 1)  # MASTER节点对应RANK 0
MASTER_ADDR=$(getent ahosts $MASTER_ADDR | awk '{ print $1 }' | tail -n 1)
JOB_NAME=$SLURM_JOB_NAME
JOB_ID=$SLURM_JOB_ID
export MASTER_PORT=$(expr 11450 + $(expr $RANDOM % 10000))  # 随机选择一个端口

trap 'cleanup' SIGTERM  # handle scancel gracefully

# cleanup 函数：在捕获到 SIGTERM 信号时，清理所有由 pdsh 启动的远程进程
cleanup() {
  echo "Received SIGTERM at $(date +%Y-%m-%d-%H:%M:%S), cleaning up remote processes..."
  pdsh -w $comma_hostnames "kill \$(ps aux | grep '$SLURM_JOB_NAME-$SLURM_JOB_ID' | grep -v grep | awk '{print \$2}')"
  kill $(ps aux | grep '$SLURM_JOB_NAME-$SLURM_JOB_ID' | grep -v grep | awk '{print $2}')
  kill $(ps aux | grep '$SLURM_JOB_NAME $SLURM_JOB_ID' | grep -v grep | awk '{print $2}')
  curl -H "Content-Type: application/json" -X POST https://wxpusher.zjiecode.com/api/send/message --data '{"appToken": "xxx", "content": "canceled job '$SLURM_JOB_NAME-$SLURM_JOB_ID'", "topicIds": [32270]}'
  exit 15
}

############################### 上面没有需要更改的地方 ###############################