# 将本脚本所有输出重定向到文件log/$SLURM_JOB_NAME-$SLURM_JOB_ID/part$SLURM_PROCID.log:
cd xxx
comma_hostnames=$1
shift
PROCID=$(expr $(echo $comma_hostnames | tr "," "\n" | grep -n `hostname` | cut -c-1) - 1)  # 仅适用9个节点以内
SLURM_JOB_NAME=$1
shift
SLURM_JOB_ID=$1
shift
if [ -z "$PROCID" ]; then
    echo "torchrun_wrapper.sh: PROCID is empty, exit"
    exit 1
fi
if [ -z "$SLURM_JOB_NAME" ]; then
    echo "torchrun_wrapper.sh: SLURM_JOB_NAME is empty, exit"
    exit 1
fi
if [ -z "$SLURM_JOB_ID" ]; then
    echo "torchrun_wrapper.sh: SLURM_JOB_ID is empty, exit"
    exit 1
fi
echo "$(date +%Y-%m-%d %H:%M:%S) torchrun_wrapper.sh: SLURM_JOB_NAME=$SLURM_JOB_NAME, SLURM_JOB_ID=$SLURM_JOB_ID, PROCID=$PROCID; hostname=`hostname`" >> log/$SLURM_JOB_NAME-$SLURM_JOB_ID/part$PROCID.log
exec &>> log/$SLURM_JOB_NAME-$SLURM_JOB_ID/part$PROCID.log

export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
export NCCL_MIN_CHANNELS=32

source ~/.bashrc

module load /opt/app/spack/share/spack/modules/gcc/11.3.0
module load /opt/app/spack/share/spack/modules/cuda/12.5.1
module load /opt/app/spack/share/spack/modules/libaio/0.3.113-gcc_13.1.0

source .venv/bin/activate  # venv

# export NCCL_SOCKET_IFNAME=vpapvn  # https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html
# export NCCL_IB_DISABLE=1  # https://github.com/NVIDIA/nccl/issues/451
export LDFLAGS="-L/usr/lib64"
export CFLAGS="-I/usr/include"
export PYTHONPATH=.
export CUTLASS_PATH=~/cutlass
export LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8  # https://stackoverflow.com/questions/74367207/segmentation-fault-core-dumped-when-launching-python-in-anaconda
export OPENBLAS_NUM_THREADS=24  # https://stackoverflow.com/questions/52026652/openblas-blas-thread-init-pthread-create-resource-temporarily-unavailable
export OMP_NUM_THREADS=24  # https://stackoverflow.com/questions/53351194/openmp-libgomp-thread-creation-failed-resource-temporarily-unavailable-when

export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# DEBUG
export TRANSFORMERS_VERBOSITY=debug
export NCCL_DEBUG=DEBUG  # https://stackoverflow.com/questions/61075390/pytorch-nccl-error-unhandled-system-error-nccl-version-2-4-8
export NCCL_DEBUG_SUBSYS=GRAPH # https://pytorch.org/docs/stable/distributed.html
# export TORCH_LOGS=+all
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export TORCH_CPP_LOG_LEVEL=INFO


CACHE_PATH='/fs/archive/share/yulan/data/aa_hf_cache'
export TMPDIR=${CACHE_PATH}/tmp
export HF_DATASETS_CACHE=${CACHE_PATH}/hf_datasets_cache
export HF_HOME=${CACHE_PATH}/hf_home
mkdir -p ${CACHE_PATH}
mkdir -p ${TMPDIR}
mkdir -p ${HF_DATASETS_CACHE}
mkdir -p ${HF_HOME}

# 打印所有环境变量
env

# 输出
echo "torchrun_wrapper.sh: SLURM_JOB_NAME=$SLURM_JOB_NAME, SLURM_JOB_ID=$SLURM_JOB_ID, PROCID=$PROCID; hostname=`hostname`"
echo -e "torchrun_wrapper.sh: torchrun --node_rank $PROCID $@\n"

# 设置 -e 选项，这会使脚本在任何命令失败时立即退出
set -e

# 设置 -o pipefail，这确保管道中的任何命令失败都会导致整个管道失败
set -o pipefail

torchrun --node_rank $PROCID $@

if [ $PROCID -eq 0 ]; then
    curl -H "Content-Type: application/json" -X POST https://wxpusher.zjiecode.com/api/send/message --data '{"appToken": "xxx", "content": "'$SLURM_JOB_NAME-$SLURM_JOB_ID' done ", "topicIds": [32270]}'
fi
