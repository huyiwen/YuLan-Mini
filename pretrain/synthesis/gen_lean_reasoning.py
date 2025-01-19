import json
import os
import re
import time
from random import random, sample
from typing import Tuple

import datasets
import pandas as pd
import sglang as sgl
from sglang import (RuntimeEndpoint, assistant, function, gen,
                    set_default_backend, system, user)
from tqdm import tqdm

# 设置默认的运行时端点
set_default_backend(RuntimeEndpoint("http://localhost:30000"))


# Deepseek-Prover-V1
@function
def analyze_deepseek(s, natural_language_statement, formal_statement, state_before, state_after, tactic, explanation="", **kwargs) -> str:
    if os.path.exists("/home/huyiwen/monorepo/projects/stop_signal"):
        return None

    input_template = """I am a mathematician unfamiliar with Lean. Please explain the tactics used in a proof, as if you are in the process of trying to prove a theorem and haven't yet completed it. Explain the reasoning and logic behind choosing those specific tactics.

**Statement:**
{natural_language_statement}
```lean4
{formal_statement}
```

**Current state:**
```lean4
{state_before}
```

**Proof:**
```lean4
{tactic}
```
"""

    assistant_prefix = """**Reasoning:**
{explanation}"""

    s += user(input_template.format(
        natural_language_statement=natural_language_statement,
        formal_statement=formal_statement,
        state_before=state_before,
        tactic=tactic,
    ))

    s += assistant(assistant_prefix.format(explanation="") + gen("explanation", max_tokens=600))
    return None


# Lean-Github
@function
def analyze_github(s, state_before, tactic, state_after, **kwargs) -> str:
    if os.path.exists("/home/huyiwen/monorepo/projects/stop_signal"):
        return None

    input_template = """I am a mathematician unfamiliar with Lean. Please explain the tactics used in a proof, as if you are in the process of trying to prove a theorem and haven't yet completed it. Explain the reasoning and logic behind choosing those specific tactics.

**Current state:**
```lean4
{state_before}
```

**Proof:**
```lean4
{tactic}
```
"""

    assistant_prefix = """**State after:**
{state_after}

**Reasoning:**
"""

    s += user(input_template.format(
        state_before=state_before,
        tactic=tactic,
    ))

    s += assistant(assistant_prefix.format(state_after=state_after) + gen("explanation", max_tokens=600))
    return None


# Lean-Workbook
# State Before + Tactic -> State After
@function
def analyze_workbook_a(s, natural_language_statement, formal_statement, state_before, state_after, tactic, explanation="", **kwargs) -> str:
    if os.path.exists("/home/huyiwen/monorepo/projects/stop_signal"):
        return None

    input_template = """Given a Lean tactic at a intermediate step in a proof and the goal state before the tactic, predict the resulting goal state after the tactic's application and provide a detailed explanation. You do not need to consider whether the tactic is sufficient to complete the proof; simply explain why the goal state changes to your predicted state after the tactic's execution.

**Statement:**
{natural_language_statement}
```lean4
{formal_statement}
```

**Goal state before:**
```lean4
{state_before}
```

**Tactic to execute:**
```lean4
{tactic}
```
"""

    assistant_prefix = """**State after:**
{state_after}

**Explanation:**
"""

    s += user(input_template.format(
        natural_language_statement=natural_language_statement,
        formal_statement=formal_statement,
        state_before=state_before,
        tactic=tactic,
    ))

    s += assistant(assistant_prefix.format(state_after=state_after) + gen("explanation", max_tokens=600))
    return None


# State After + Tactic -> State Before
@function
def analyze_workbook_b(s, natural_language_statement, formal_statement, state_before, state_after, tactic, explanation="", **kwargs) -> str:
    if os.path.exists("/home/huyiwen/monorepo/projects/stop_signal"):
        return None

    input_template = """Given a tactic applied at an intermediate step of a Lean proof and the resulting goal state **after** applying the tactic, predict one possible goal state **before** the tactic was applied, and provide a detailed explanation  You don't need to consider whether the tactic is sufficient to complete the proof; simply explain why the **pre-tactic goal state** would have resulted in the given post-tactic state.

**Statement:**
{natural_language_statement}
```lean4
{formal_statement}
```

**Tactic applied:**
```lean4
{tactic}
```

**Resulting state after:**
```lean4
{state_after}
```
"""

    assistant_prefix = """**Goal state before:**
{state_before}

**Explanation:**
"""

    s += user(input_template.format(
        natural_language_statement=natural_language_statement,
        formal_statement=formal_statement,
        state_after=state_after,
        tactic=tactic,
    ))

    s += assistant(assistant_prefix.format(state_before=state_before) + gen("explanation", max_tokens=600))
    return None



@function
def analyze_workbook(s, natural_language_statement, formal_statement, state_before, tactic, state_after, **kwargs) -> str:
    if os.path.exists("/home/huyiwen/monorepo/projects/stop_signal"):
        return None

    input_template = """Give the next tactic in the proof with explanatory comments.

Statement: {natural_language_statement}

```lean4
{formal_statement}
```

**Current state:**

{state_before}
"""

    assistant_prefix = """**Next tactic:**
{tactic}
/-State:
{state_after}-/

**Explanatory comments:**
"""

    s += user(input_template.format(
        natural_language_statement=natural_language_statement,
        formal_statement=formal_statement,
        state_before=state_before,
        tactic=tactic,
        state_after=state_after,
    ))

    s += assistant(assistant_prefix.format(tactic=tactic, state_after=state_after) + gen("explanation", max_tokens=400))
    return None


def analyze(lines, analyze_fn, name):

    # 使用batch处理多个文本
    states = analyze_fn.run_batch(
        lines,
        progress_bar=True,
        num_threads=256,
        temperature=0.3,
        top_p=0.4,
    )

    answers = []
    for line, state in zip(lines, states):
        # extract the explanation from the state
        try:
            line["explanation"] = state["explanation"]
        except Exception:
            line["explanation"] = ""
        # extract the stop reason from the state
        try:
            line["stop_reason"] = state.get_meta_info("explanation").get("finish_reason", {}).get("type", "")
        except:
            line["stop_reason"] = ""
        answers.append(line)

    print(f"/home/huyiwen/monorepo/projects/miniyulan/gen_lean/lean_explain_{name}.jsonl")
    with open(f"/home/huyiwen/monorepo/projects/miniyulan/gen_lean/lean_explain_{name}.jsonl", "w") as f:
        for line in answers:
            f.write(json.dumps(line) + "\n")


def get_data(repo="workbook"):
    if repo == "workbook":  # Not used
        lines = datasets.load_dataset("/home/huyiwen/lean-tactics/Lean-Workbook", split="train").to_list()
        return lines
    elif repo == "github":  # Not used
        lean_github = pd.read_parquet('/home/huyiwen/lean-tactics/Lean-Github/lean-github.parquet')

        # dedup
        lean_github = lean_github.drop_duplicates(subset=['url', 'commit', 'file_path', 'start', 'end', 'tactic', 'state_before', 'state_after'])

        # convert string to real tuple
        lean_github['start'] = lean_github['start'].apply(lambda x: tuple(map(int, x[1:-1].split(','))))
        lean_github['end'] = lean_github['end'].apply(lambda x: tuple(map(int, x[1:-1].split(','))))
        return lean_github.to_dict(orient='records')
    elif repo == "deepseek":
        lines = datasets.load_dataset("/home/huyiwen/lean-tactics/DeepSeek-Prover-V1", split="train").to_list()
        return lines
    elif repo == "workbook-c":
        with open("/home/huyiwen/lean-tactics/Lean-Workbook/c.jsonl") as f:
            lines = [json.loads(line) for line in f]
        return lines
    elif repo == "workbook-a":
        with open("/home/huyiwen/lean-tactics/Lean-Workbook/a.jsonl") as f:
            lines = [json.loads(line) for line in f]
        return lines


lines = get_data("github")
analyze(lines, analyze_github, "github-" + time.strftime("%Y%m%d-%H%M%S"))
