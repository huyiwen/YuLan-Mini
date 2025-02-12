import math
import sys

import torch
from safetensors import safe_open


def calc_norm(model_path: str):

    with safe_open(f"{model_path}/model.safetensors", framework="pt") as f:
        for k in f.keys():
            v = f.get_tensor(k)
            vnorm = torch.norm(v).item()
            vnum = torch.numel(v)
            print(k, vnorm, vnorm / vnum, vnorm / math.sqrt(vnum))


if __name__ == "__main__":
    calc_norm(sys.argv[1])
