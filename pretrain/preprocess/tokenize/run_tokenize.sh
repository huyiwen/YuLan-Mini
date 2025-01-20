#!/bin/bash

data_path=$1

tokenizer_path=<hf-tokenizer-path>
num_file=10000
num_worker=8
# num_file means how many jsonl/json/parquet files to tokenize at once (to avoid memory overflow). If num_file < actual number of files, simply run the script multiple times to tokenize all files.

export RAW_DATA_PREFIX="/data/raw"
export INPUT_IDS_PREFIX="/data/input_ids"
# target save path for tokenized data. The tokenized data will retain the same directory structure as the raw data.

echo num_file=$num_file num_worker=$num_worker

# check if data_path exists
if [ ! -d "$data_path" ]; then
    echo "$data_path does not exist."
    exit
else
    echo $data_path
fi


python tokenize/tokenize_text.py \
    --tokenizer_path $tokenizer_path \
    --data_path $data_path \
    --model_name mini \
    --num_file $num_file \
    --text_key text \
    --num_worker $num_worker \
    --skip_exist True

# split data by 0.01B tokens
python tokenize/split_data.py $data_path

# delete intermediate tokenization files
cat datasets_to_delete.txt | xargs -I {} rm {}
rm datasets_to_delete.txt

# incase of missing deletion
if [ -n "$(find . -type f -regex '.*part-[0-9]+\.jsonl')" ]; then
    find . -type f -regex '.*part-[0-9]+\.jsonl'
    echo "Please check the intermediate part-xx.jsonl files listed above and delete them manually."
fi
