#!/bin/bash

data_path=$1
dir_path=$(dirname $(realpath $0))
job_id=$RANDOM

source /data/pretrain-mini/.venv/bin/activate

tokenizer_path=/data/pretrain-mini/preprocess/modify_tokenizer/1731-dropout
num_file=10000
num_worker=16
# num_file means how many jsonl/json/parquet files to tokenize at once (to avoid memory overflow). If num_file < actual number of files, simply run the script multiple times to tokenize all files.

export RAW_DATA_PREFIX="/data/aa_raw"
export INPUT_IDS_PREFIX="/data/aa_input_ids"
# target save path for tokenized data. The tokenized data will retain the same directory structure as the raw data.

echo num_file=$num_file num_worker=$num_worker

# check if data_path exists
if [ ! -d "$data_path" ]; then
    echo "$data_path does not exist."
    exit
else
    echo $data_path
fi

python $dir_path/tokenize_text.py \
    --tokenizer_path $tokenizer_path \
    --data_path $data_path \
    --model_name mini \
    --num_file $num_file \
    --text_key text \
    --num_worker $num_worker \
    --skip_exist True

# split data by 0.01B tokens
datasets_to_delete=$dir_path/cache_$job_id.txt
python $dir_path/split_data.py $datasets_to_delete $data_path

# delete intermediate tokenization files
cat $datasets_to_delete | xargs -I {} rm {}
rm $datasets_to_delete

# incase of missing deletion
if [ -n "$(find . -type f -regex '.*part-[0-9]+\.jsonl')" ]; then
    find . -type f -regex '.*part-[0-9]+\.jsonl'
    echo "Please check the intermediate part-xx.jsonl files listed above and delete them manually."
fi
