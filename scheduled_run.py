import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor


def run_command(command):
    subprocess.run(command, shell=True)

def concurrent_run_commands(commands):
    with ThreadPoolExecutor(max_workers=len(commands)) as executor:
        executor.map(run_command, commands)

def main():
    os.system(f"python3 sosp_benchmark.py -c -a -x 70 -g 2")
    concurrent_run_commands([f"python3 sosp_benchmark.py -c -a -x {model_size} -g 2" for model_size in [7, 13, 34]])
    concurrent_run_commands([f"python3 sosp_benchmark.py -c -x {model_size}" for model_size in [7, 13, 34, 70]])
    concurrent_run_commands([f"python3 sosp_benchmark.py -a -x {model_size}" for model_size in [7, 13, 34, 70]])
    # while True:
    #     lines = subprocess.check_output("squeue -u fw", shell=True).decode("ascii").strip()
    #     if len(lines.split("\n")) <= 1:
    #         commands = [f"python3 sosp_benchmark.py -c -a -x {model_size} -g 2" for model_size in [7, 13, 34]]
    #         with ThreadPoolExecutor(max_workers=len(commands)) as executor:
    #             executor.map(run_command, commands)
    #     else:
    #         time.sleep(300)

if __name__ == "__main__":
    main()