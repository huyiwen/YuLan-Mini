# Data Preprocess

这部分介绍了数据的预处理方法。

## Data Filtering and Cleaning

> We will release the related code soon.

## Training Data Preparation

This part introduces the process of tokenization, data mixing, etc.

### Preliminary

We store the data before tokenization and after tokenization in two different folders, e.g., `/data/raw` and `/data/input_ids`.

```txt
/data
├── raw
│   ├── dataset-collection-1
│   │   ├── dataset-1
│   │   │   ├── file-1.jsonl
│   │   │   ├── file-2.jsonl
│   │   │   └── ...
│   │   ├── dataset-2
│   │   │   ├── file-1.parquet
│   │   └── ...
│   └── dataset-collection-2
│       ├── dataset-1
│       └── ...
└── input_ids
    └── dataset-collection-2
        ├── dataset-1
        │   └── wo_ppl
        │       ├── splitted_part-0001.parquet
        │       ├── splitted_part-0001-metadata.json
        │       └── ...
        └── ...
```

The data before tokenization is stored in parquet (recommended) or jsonl format, which looks like:

```json
{"text": "This is a sentence."}
{"text": "This is another sentence."}
```

### Tokenization

The following script will tokenize the files in the target path (including all subfolders), and then split the tokenized data into multiple files according to 0.01B Tokens for more fine-grained data mixing.

```bash
bash tokenize/run_tokenize.sh /data/raw/dataset-collection-2/dataset-1
```

> Please modify the script to set:
> 1. default key for text dataset, e.g. "text",
> 2. tokenizer path,
> 3. the save location of the split files,
> 4. context window size, e.g. 4096,
> 5. the number of threads,


### Data Mixing


