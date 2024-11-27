import argparse
import itertools
import math
import os
import subprocess
import re
import time
import os
from settings import build_cmd, run_interruptable_cmd_on_js_h100, MODEL_SIZE_TO_N_NODES_BAISC, N_NODES_TO_BATCH_SIZE, CTX, PROMPT_LEN
from parse_log import _parselog



def log2_generator(limit, reversed=False):
    assert limit & (limit - 1) == 0
    num = 1 if not reversed else limit
    if not reversed:
        while num <= limit:
            yield num
            num *= 2
    else:
        while num >= 1:
            yield num
            num //= 2
def extract_node_integers(input_string):
    # Regular expression to match the pattern
    pattern = r'(\w+)\[(\d+)-(\d+)\]'
    
    # Try to match the pattern in the input string
    match = re.match(pattern, input_string)
    
    if match:
        # Extract the base name and the range
        base_name = match.group(1)
        start = int(match.group(2))
        end = int(match.group(3))
        
        return start, end
    else:
        raise ValueError('Invalid input string')

def main(args):
    ctx = CTX
    prompt_len = PROMPT_LEN

    for size in args.model_size:
        bs = N_NODES_TO_BATCH_SIZE[MODEL_SIZE_TO_N_NODES_BAISC[size]]
        for train_n_mbs, rollout_n_mbs in [(1,2),(2,2),(2,4),(4,1),(4,2),(4,4),(1,8),(2,8),(4,8),(8,8)]:
            xx = build_cmd(size, bs, ctx, prompt_len, args.scale_both, rollout_n_mbs, train_n_mbs)
            if xx is not None:
                cmd, logfile = xx
                parse_success, oom = _parselog(
                    actor_size=size,
                    critic_size=7 if not args.scale_both else 13,
                    bs=bs,
                    ctx=ctx,
                    prompt_len=prompt_len,
                    rollout_n_mbs=rollout_n_mbs,
                    train_n_mbs=train_n_mbs,
                )
                print(parse_success, oom, logfile)

                if parse_success:
                    if not oom:
                        print(f">>>>> Find existing success logfile: {logfile}")
                        break
                    else:
                        print(f">>>>> Find existing oom logfile, skipping this configuration, continue experiment with model size {size}")
                        continue

                s, e = extract_node_integers(args.nodelist)
                assert e -s + 1 == MODEL_SIZE_TO_N_NODES_BAISC[size]
                os.system(f"python3 /mnt/bs_fs/rayc.py stop -s {s} -e {e}")
                time.sleep(10)
                run_interruptable_cmd_on_js_h100(cmd, args.nodelist, logfile)
                os.system(f"python3 /mnt/bs_fs/rayc.py stop -s {s} -e {e}")
                time.sleep(10)

                parse_success, oom = _parselog(
                    actor_size=size,
                    critic_size=7 if not args.scale_both else 13,
                    bs=bs,
                    ctx=ctx,
                    prompt_len=prompt_len,
                    rollout_n_mbs=rollout_n_mbs,
                    train_n_mbs=train_n_mbs,
                )
                if parse_success and not oom:
                    break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", "-x", type=int, choices=[7, 13, 34, 70], required=True, nargs="+")
    parser.add_argument("--scale_both", "-s", action="store_true")
    parser.add_argument("--nodelist", type=str, default=None)
    args = parser.parse_args()
    main(args)
    # for x in log2_generator(256, reversed=True):
    #     print(x)
