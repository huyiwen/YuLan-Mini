import argparse
import json
import multiprocessing as mp
import os
import pathlib
import random
import re
import signal
from copy import deepcopy

import numpy as np
import pyarrow
from pyarrow import parquet as pq
from tqdm import tqdm, trange

from transformers import AutoTokenizer

random.seed(42)

# Max line per file to tokenize. In case of OOM
MAX_DATA = int(1e7)

# replace raw data path with input_ids path
raw_data_prefix = os.environ["RAW_DATA_PREFIX"]
input_ids_prefix = os.environ["INPUT_IDS_PREFIX"]


def get_tgt_folder(file_path, model_name):
    """Each jsonl or parquet file will generate a folder with the same name."""

    # token id folder directory
    file_path = file_path.replace(raw_data_prefix,
                                  input_ids_prefix)

    # remove the file extension
    tgt_folder = file_path[:file_path.rfind(".")]
    tgt_folder = os.path.join(tgt_folder, "wo_ppl")
    if os.path.exists(tgt_folder) == True:
        is_exists = True
    else:
        is_exists = False
    pathlib.Path(tgt_folder).mkdir(parents=True, exist_ok=True)
    return tgt_folder, is_exists


def warn(msg):
    print("\033[0;33m" + str(msg) + "\033[0m")


def tokenize_text(dataset,
                  src_folder,
                  file_nos,
                  tgt_folder,
                  idx,
                  text_key,
                  is_first,
                  skip_exists: bool = False):
    tgt_path = os.path.join(tgt_folder, "part-{}.jsonl".format(idx))
    if is_first == False:
        write_mode = "a"
    else:
        if skip_exists and os.path.exists(tgt_path):
            warn(f"skip {tgt_path}")
            return
        write_mode = "w"

    batch_size = 1000
    with open(tgt_path, write_mode) as fout:
        for batch_st in tqdm(range(0, len(dataset), batch_size)):
            batch_data = dataset[batch_st:batch_st + batch_size]
            batch_file_nos = file_nos[batch_st:batch_st + batch_size]
            input_ids = tokenizer([data[text_key] for data in batch_data],
                                add_special_tokens=False)["input_ids"]
            for ipts, no in zip(input_ids, batch_file_nos):
                new_data = {"input_ids": ipts, "source": f"{src_folder}:{no}"}
                fout.write(json.dumps(new_data, ensure_ascii=False) + "\n")


wanna_exit = False


def interrupt_handler(signum, frame, ask=True):
    print("Ctrl+C pressed. Waiting for the current process to be finished.")
    global wanna_exit
    wanna_exit = True


def start_mp(dataset, is_first, src_folder, file_nos):
    """dataset: List[Dict[str, str]]"""

    if len(dataset) == 0:
        warn("len(dataset) == 0")
        return
    if not isinstance(dataset, list):
        warn("not isinstance(dataset, list)")
        return
    try:
        assert args.text_key in dataset[0]
        text_key = args.text_key
    except AssertionError:
        warn(f"Available Keys: {dataset[0].keys()}")
        raise Exception("Unknown Key!")

    seed = random.random()
    def sample_seed():
        return seed

    # random.shuffle(dataset, sample_seed)
    # random.shuffle(file_nos, sample_seed)
    random.shuffle(dataset)
    random.shuffle(file_nos)


    part_num = args.num_worker
    slice_idx = np.linspace(0, len(dataset), part_num + 1).astype("int")
    p = mp.Pool(part_num)
    for start_id in range(part_num):
        start, end = slice_idx[start_id], slice_idx[start_id + 1]
        new_lines = dataset[start:end]
        p.apply_async(tokenize_text,
                      args=(new_lines, src_folder, file_nos, tgt_folder, start_id, text_key,
                            is_first))
    p.close()
    p.join()
    print("All of the child processes over!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_path", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--num_files", type=int)
    parser.add_argument("--text_key", type=str)
    parser.add_argument("--num_worker", type=int)
    parser.add_argument("--skip_exist", type=bool, default=False)
    parser.add_argument("--skip_exists", type=bool, default=False)
    args = parser.parse_args()

    # load tokenizer
    kwargs = {}
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, **kwargs)

    signal.signal(signal.SIGINT, interrupt_handler)
    for root, _, files in os.walk(args.data_path, topdown=False):
        step = 0
        random.shuffle(files)
        for fp in tqdm(files):
            if wanna_exit:
                print("Process done.")
                break

            file_path = os.path.join(root, fp)
            if ".git" in file_path:
                continue
            tgt_folder, is_exists = get_tgt_folder(file_path, args.model_name)
            if is_exists == True and args.skip_exist == True:
                warn(f"skip {fp}")
                continue

            print("Process {}".format(file_path))
            print("Target Folder: {}".format(tgt_folder))

            fin = open(file_path, "r")
            is_jsonl = False

            # this is shit code
            if os.path.exists(file_path + "/dataset_info.json"):
                import datasets
                ds = datasets.load_from_disk(file_path, streaming=True)
                started = 0
                for i in trange(MAX_DATA, desc="Reading Data"):
                    try:
                        dataset = [next(ds) for _ in range(320000)]
                        file_nos = [started + i for i in range(len(dataset))]

                        start_mp(dataset, True, file_path, file_nos)

                        started += len(dataset)
                        step = step + 1
                        if step >= args.num_files:
                            break
                    except StopIteration:
                        break

            if file_path.endswith(".json") == True:
                try:
                    dataset = json.load(fin)
                    file_nos = [i for i in range(len(dataset))]

                    start_mp(dataset, True, file_path, file_nos)
                    step = step + 1
                    if step >= args.num_files:
                        break
                    continue
                except json.decoder.JSONDecodeError:
                    is_jsonl = True
                    fin.close()
                    # reopen for jsonl
                    fin = open(file_path, "r")

            if file_path.endswith(".jsonl") == True or is_jsonl == True:
                is_finish = False
                is_first = True
                started = 0
                while True:
                    dataset = []
                    for i in trange(MAX_DATA, desc="Reading Data"):
                        tmp_data = fin.readline()
                        if not tmp_data:
                            is_finish = True
                            break
                        try:
                            tmp_data = json.loads(tmp_data)
                            tmp_data["text"] = str(tmp_data["text"]).replace(
                                "\u3000", " ")  ###清洗数据
                            url_pattern = re.compile(r'https?://\S+|www\.\S+')
                            tmp_data["text"] = url_pattern.sub(
                                '<url>', tmp_data["text"])
                            dataset.append(tmp_data)
                        except json.decoder.JSONDecodeError as e:
                            warn(str(e) + tmp_data)
                            continue

                    file_nos = [started + i for i in range(len(dataset))]
                    start_mp(dataset, is_first, file_path, file_nos)
                    is_first = False  # append mode
                    if is_finish == True:
                        break
            elif file_path.endswith(".parquet"):
                try:
                    table = pq.read_table(file_path)
                    file_nos = [i for i in range(len(table))]
                    start_mp(table.to_pylist(), True, file_path, file_nos)
                except pyarrow.lib.ArrowInvalid as e:
                    warn(str(e))
                    continue
            else:
                continue

            fin.close()
            step = step + 1
            if step >= args.num_files:
                break
