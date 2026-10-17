"""Master script: reproduces every simulation result in the paper in one command.

    python run_all.py                  # one replication (about 30 seconds)
    python run_all.py --iterations 1000

Steps, in order (all paths are relative to this folder):
1. code/nkc_models.py      simulates the three task structures  -> output/results/*.csv
2. code/make_figures.py    builds Figs. 1, 3, 4, 5, 6, 8, 9, 11 -> output/figures/
3. code/worked_examples.py prints the Figure 2 worked example
4. tests/verify_output.py  checks the CSVs against the reference run (default settings only)
Console output of every step, and figures/summary.json, are saved in output/logs/.
"""
import argparse
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(ROOT, "code")
OUT = os.path.join(ROOT, "output")


def step(name, args, log):
    print(f"[{time.strftime('%H:%M:%S')}] {name}", flush=True)
    t0 = time.time()
    with open(os.path.join(OUT, "logs", log), "w") as f:
        f.write(f"$ python {' '.join(args)}\n")
        f.flush()
        res = subprocess.run([sys.executable, *args], cwd=CODE, stdout=f, stderr=subprocess.STDOUT)
        f.write(f"\n[exit code {res.returncode}, {time.time() - t0:.1f} s]\n")
    if res.returncode != 0:
        sys.exit(f"{name} failed; see output/logs/{log}")
    print(f"           done in {time.time() - t0:.1f} s", flush=True)


def main():
    p = argparse.ArgumentParser(description="Reproduce all simulation results.")
    p.add_argument("--iterations", type=int, default=1, help="replications per task structure (default 1)")
    p.add_argument("--seed", type=int, default=0, help="random seed (default 0)")
    a = p.parse_args()
    os.makedirs(os.path.join(OUT, "logs"), exist_ok=True)
    # paths passed to the scripts are relative to code/, where they run
    results, figures = os.path.join("..", "output", "results"), os.path.join("..", "output", "figures")
    sim = ["nkc_models.py", "--outdir", results, "--seed", str(a.seed), "--iterations", str(a.iterations)]
    if a.iterations > 1:
        sim.append("--no-sequences")
    step("Simulating the three task structures", sim, "01_simulation.log")
    step("Building the figures", ["make_figures.py", "--results", results, "--figdir", figures], "02_figures.log")
    step("Printing the worked examples", ["worked_examples.py"], "03_worked_examples.log")
    shutil.copy(os.path.join(OUT, "figures", "summary.json"), os.path.join(OUT, "logs", "summary.json"))
    if a.seed == 0 and a.iterations == 1:
        step("Checking the output against the reference run",
             [os.path.join("..", "tests", "verify_output.py")], "04_verify_output.log")
    print(f"All outputs are in {os.path.relpath(OUT, ROOT)}/")


if __name__ == "__main__":
    main()
