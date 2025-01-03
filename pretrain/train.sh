source ~/.bashrc

# 将作业提交给SLURM

# 参数：--time=30:00:00 最大运行时间24小时
# 参数：--job-name=xxx 作业名称
# 参数：--nodes=1 使用1个节点（注意调节batch size!!!）

function decay_train() {
    # 保存数据集并启动训练
    SCRIPT=$1
    FETCH_TIME=$2
    if [[ ${#FETCH_TIME} -ne 18 ]]; then
        echo "FETCH_TIME格式错误：$FETCH_TIME"
        exit 1
    fi
    RUN_REASON=$3
    if [[ ${#RUN_REASON} -lt 10 ]]; then
        echo "RUN_REASON 至少大于10个字：$RUN_REASON"
        exit 1
    fi
    PER_DEVICE_TRAIN_BATCH_SIZE=${4:-18}
    NNODES=${5:-7}
    MODEL_NAME=${6:-"myl_new_no_math"}
    JOB_NAME=$(basename $SCRIPT .sh)-$FETCH_TIME
    if [  -z /fs/archive/share/yulan/data/aa_mini/output/${JOB_NAME} ]; then
        echo "已有checkpoint！请注意是否会覆盖：/fs/archive/share/yulan/data/aa_mini/output/${JOB_NAME}"
        exit 1
    fi
    echo "JOB_NAME: $JOB_NAME"
    echo "请检查总BATCH_SIZE: $PER_DEVICE_TRAIN_BATCH_SIZE * $NNODES * 8 = $((PER_DEVICE_TRAIN_BATCH_SIZE * NNODES * 8))"
    echo "等价于BATCH_SIZE：$((PER_DEVICE_TRAIN_BATCH_SIZE * NNODES * 8 * 4096)) Tokens"
    if [ -d /fs/archive/share/yulan/data/aa_mini/hf_dataset/$MODEL_NAME/$FETCH_TIME ]; then
        echo "数据集已存在 /fs/archive/share/yulan/data/aa_mini/hf_dataset/$MODEL_NAME/$FETCH_TIME"
    else
        python preprocess/fetch_data/distributed_save.py $FETCH_TIME $MODEL_NAME
    fi

    JOB_ID=$(sbatch --time=36:00:00 --job-name=$JOB_NAME --nodes=$NNODES $SCRIPT $FETCH_TIME $PER_DEVICE_TRAIN_BATCH_SIZE $MODEL_NAME | grep -o -P '\d+')
    echo "JOB_ID: $JOB_ID"
    if [ -z $JOB_ID ]; then
        echo "启动失败"
        exit 1
    fi
    mkdir -p "log/$JOB_NAME-$JOB_ID"
    touch "log/$JOB_NAME-$JOB_ID/reason-$RUN_REASON"

    sleep 5
    nohup new_start_monitor $JOB_NAME $JOB_ID > "log/$JOB_NAME-$JOB_ID/monitor.log" 2>&1 &
    LOF_FILE="log/$JOB_NAME-$JOB_ID/part0.log"
    squeue -o "%.6i %.35j %t %8M  %.R"
    exit 0
}


function main_train() {
    # 保存数据集并启动训练
    SCRIPT=$1
    FETCH_TIME=$2
    if [[ ${#FETCH_TIME} -ne 18 ]]; then
        echo "FETCH_TIME格式错误：$FETCH_TIME"
        exit 1
    fi
    RUN_REASON=$3
    if [[ ${#RUN_REASON} -lt 10 ]]; then
        echo "RUN_REASON 至少大于10个字：$RUN_REASON"
        exit 1
    fi
    PER_DEVICE_TRAIN_BATCH_SIZE=${4:-18}
    NNODES=${5:-7}
    MODEL_NAME=${6:-"myl_new_no_math"}
    JOB_NAME=$(basename $SCRIPT .sh)
    if [  -z /fs/archive/share/yulan/data/aa_mini/output/${JOB_NAME} ]; then
        echo "已有checkpoint！请注意是否会覆盖：/fs/archive/share/yulan/data/aa_mini/output/${JOB_NAME}"
        exit 1
    fi
    echo "JOB_NAME: $JOB_NAME"
    echo "请检查总BATCH_SIZE: $PER_DEVICE_TRAIN_BATCH_SIZE * $NNODES * 8 = $((PER_DEVICE_TRAIN_BATCH_SIZE * NNODES * 8))"
    echo "等价于BATCH_SIZE：$((PER_DEVICE_TRAIN_BATCH_SIZE * NNODES * 8 * 4096)) Tokens"
    if [ -d /fs/archive/share/yulan/data/aa_mini/hf_dataset/$MODEL_NAME/$FETCH_TIME ]; then
        echo "数据集已存在 /fs/archive/share/yulan/data/aa_mini/hf_dataset/$MODEL_NAME/$FETCH_TIME"
    else
        python preprocess/fetch_data/distributed_save.py $FETCH_TIME $MODEL_NAME
    fi

    JOB_ID=$(sbatch --time=36:00:00 --job-name=$JOB_NAME --nodes=$NNODES $SCRIPT $FETCH_TIME $PER_DEVICE_TRAIN_BATCH_SIZE $MODEL_NAME | grep -o -P '\d+')
    echo "JOB_ID: $JOB_ID"
    if [ -z $JOB_ID ]; then
        echo "启动失败"
        exit 1
    fi
    mkdir -p "log/$JOB_NAME-$JOB_ID"
    touch "log/$JOB_NAME-$JOB_ID/reason-$RUN_REASON"

    sleep 5
    nohup new_start_monitor $JOB_NAME $JOB_ID > "log/$JOB_NAME-$JOB_ID/monitor.log" 2>&1 &
    LOF_FILE="log/$JOB_NAME-$JOB_ID/part0.log"
    squeue -o "%.6i %.35j %t %8M  %.R"
    exit 0
}


# Note: Due to subsequent modifications to the training code, this launch script may require re-adaptation.

main_train yulanmini-2B-final-phase1.sh 20241017_013512 "2B-model-phase1,lm_head_alpha=1+deepspeed1+norm_alpha=True+rms_type=llama+emb_alpha=False, " 18 7 myl_new_no_math

main_train yulanmini-2B-final-phase2.sh 02_20241017_013401 "2B-model-phase2,lm_head_alpha=1+deepspeed1+norm_alpha=True+rms_type=llama+emb_alpha=False, " 18 7 myl_new_no_math

main_train yulanmini-2B-final-phase3.sh 03_20241020_001556 "2B-model-phase3,lm_head_alpha=1+deepspeed1+norm_alpha=True+rms_type=llama+emb_alpha=False, " 18 7 myl_new_no_math

main_train yulanmini-2B-final-phase4.sh 04_20241021_170901 "2B-model-phase4,lm_head_alpha=1+deepspeed1+norm_alpha=True+rms_type=llama+emb_alpha=False, " 18 7 myl_new_no_math

main_train yulanmini-2B-final-phase5.sh 05_20241022_221453 "2B-model-phase5,lm_head_alpha=1+deepspeed1+norm_alpha=True+rms_type=llama+emb_alpha=False, " 18 7 myl_new_no_math

main_train yulanmini-2B-final-phase6.sh 06_20241024_013137 "2B-model-phase6,lm_head_alpha=1+deepspeed1+norm_alpha=True+rms_type=llama+emb_alpha=False, " 18 7 myl_new_no_math

main_train yulanmini-2B-final-phase7-dp2.sh 07_20241025_022032 "2B-model-phase7,lm_head_alpha=1+deepspeed2+norm_alpha=True+rms_type=llama+emb_alpha=False, " 18 7 myl_new_no_math

main_train yulanmini-2B-final-phase8.sh 08_20241026_151354 "2B-model-phase8,lm_head_alpha=1+deepspeed2+norm_alpha=True+rms_type=llama+emb_alpha=False, " 18 7 myl_new_no_math

main_train yulanmini-2B-final-phase9.sh 09_20241027_190948 "2B-model-phase9,lm_head_alpha=1+deepspeed2+norm_alpha=True+rms_type=llama+emb_alpha=False, " 18 7 myl_new_no_math

main_train yulanmini-2B-final-phase10.sh 10_20241028_225112 "2B-model-phase10,lm_head_alpha=1+deepspeed2+norm_alpha=True+rms_type=llama+emb_alpha=False, " 18 7 myl_new_no_math

main_train yulanmini-2B-final-phase11.sh 11_20241030_124814 "2B-model-phase11,lm_head_alpha=1+deepspeed2+norm_alpha=True+rms_type=llama+emb_alpha=False, " 21 6 myl_new_no_math

main_train yulanmini-2B-final-phase12.sh 12_20241101_002827 "2B-model-phase12,lm_head_alpha=1+deepspeed2+norm_alpha=True+rms_type=llama+emb_alpha=False, " 21 6 myl_new_no_math

main_train yulanmini-2B-final-phase13.sh 13_20241102_160534 "2B-model-phase13,lm_head_alpha=1+deepspeed2+norm_alpha=True+rms_type=llama+emb_alpha=False, " 21 6 myl_new_no_math

main_train yulanmini-2B-final-phase14.sh 14_20241104_000454 "2B-model-phase14,lm_head_alpha=1+deepspeed2+norm_alpha=True+rms_type=llama+emb_alpha=False, " 21 6 myl_new_no_math

main_train yulanmini-2B-final-phase15.sh 15_20241105_023029 "2B-model-phase15, lm_head_alpha=1+deepspeed2+norm_alpha=True+rms_type=llama+emb_alpha=False, " 21 6 myl_new_no_math

main_train yulanmini-2B-final-phase16.sh 16_20241106_180613 "2B-model-phase16, lm_head_alpha=1+deepspeed2+norm_alpha=True+rms_type=llama+emb_alpha=False, " 21 6 myl_new_no_math

main_train yulanmini-2B-final-phase17.sh 17_20241108_004951 "2B-model-phase17, lm_head_alpha=1+deepspeed2+norm_alpha=True+rms_type=llama+emb_alpha=False, " 21 6 myl_new_no_math

main_train yulanmini-2B-final-phase18-hyw.sh 18_20241113_034017 "2B-model-phase18-remake, lm_head_alpha=1+deepspeed2+norm_alpha=True+rms_type=llama+emb_alpha=False, " 21 6 myl_mix890

main_train yulanmini-2B-final-phase19-hyw.sh 19_20241114_115241 "2B-model-phase19-remake, lm_head_alpha=1+deepspeed2+norm_alpha=True+rms_type=llama+emb_alpha=False, " 21 6 myl_mix890

main_train yulanmini-2B-final-phase20-remake.sh 20_20241115_234357 "2B-model-phase20-remake, lm_head_alpha=1+deepspeed2+norm_alpha=True+rms_type=llama+emb_alpha=False, " 21 6 myl_mix890

main_train yulanmini-2B-final-phase21.sh 21_20241117_021115 "2B-model-phase21, lm_head_alpha=1+deepspeed2+norm_alpha=True+rms_type=llama+emb_alpha=False, " 21 6 myl_mix890

main_train yulanmini-2B-final-phase22.sh 22_20241118_155407 "2B-model-phase22, lm_head_alpha=1+deepspeed2+norm_alpha=True+rms_type=llama+emb_alpha=False, " 21 6 myl_mix890

main_train yulanmini-2B-final-phase23.sh 23_20241120_033942 "2B-model-phase23, lm_head_alpha=1+deepspeed2+norm_alpha=True+rms_type=llama+emb_alpha=False, " 21 6 myl_mix890

main_train yulanmini-2B-final-phase24.sh 24_20241121_133110 "2B-model-phase23, lm_head_alpha=1+deepspeed2+norm_alpha=True+rms_type=llama+emb_alpha=False, " 21 6 myl_mix890

main_train yulanmini-2B-final-phase25.sh 25_20241123_030124 "2B-model-phase23, lm_head_alpha=1+deepspeed2+norm_alpha=True+rms_type=llama+emb_alpha=False, " 21 6 myl_mix890

decay_train yulanmini-2B-s25d-decay80-1sqrt-long-28k-final-phase26.sh 26_20241211_015209 "decay-80B-phase26 " 26 5 myl_mix890_long_28k

decay_train yulanmini-2B-s25d-decay80-1sqrt-long-28k-final-phase27.sh 27_20241213_051741 "decay-80B-phase27 " 26 5 myl_mix890_long_28k
