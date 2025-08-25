#!/usr/bin/env python3
"""
Execute the 2nd cell of playground/bench.ipynb in-process and call bench_type2
once to trigger StateLeaky.forward (decorated with profile) and generate
baseline.prof. Uses CPU if CUDA is unavailable.
"""
import json
import os
import runpy
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
NB_PATH = os.path.join(REPO_ROOT, "playground", "bench.ipynb")


def load_second_cell_source(nb_path: str) -> str:
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    cells = nb.get("cells", [])
    if len(cells) < 2:
        raise RuntimeError("Notebook has fewer than 2 cells")
    cell = cells[1]
    if cell.get("cell_type") != "code":
        raise RuntimeError("Second cell is not a code cell")
    source_list = cell.get("source", [])
    return "".join(source_list)


def main() -> int:
    # Ensure local repo is importable (snntorch package)
    sys.path.insert(0, REPO_ROOT)

    code = load_second_cell_source(NB_PATH)

    # Make device selection robust
    # Robustly patch any explicit CUDA-only assignment
    import re
    code = re.sub(
        r"device\s*=\s*['\"]cuda['\"]",
        'device = "cuda" if __import__("torch").cuda.is_available() else "cpu"',
        code,
    )
    # Replace any explicit .to('cuda') or .to('cuda:N') calls
    code = re.sub(
        r"\.to\(\s*['\"]cuda(?::\d+)?['\"]\s*\)",
        ".to('cuda' if __import__('torch').cuda.is_available() else 'cpu')",
        code,
    )

    # Execute the cell code in its own globals dict
    g = {"__name__": "__main__"}
    exec(compile(code, filename=f"{NB_PATH}#cell2", mode="exec"), g)

    # Ensure working directory at repo root so baseline.prof lands here
    os.chdir(REPO_ROOT)

    prof_path = os.path.join(REPO_ROOT, "baseline.prof")

    # If the cell already executed forward, the file should exist
    if not os.path.exists(prof_path):
        # Try calling bench_type2 (cell 5 style)
        if "bench_type2" in g:
            g["bench_type2"](10)
        # Or try calling layer.forward(input_) (cell 1 style)
        elif "layer" in g and "input_" in g:
            g["layer"].forward(g["input_"])
        else:
            # Fallback: minimal call to trigger profiling
            import torch
            from snntorch._neurons.stateleaky import StateLeaky
            device = (
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            layer = StateLeaky(beta=0.9, channels=1).to(device)
            x = torch.arange(1 * 1 * 1).float().view(1, 1, 1).to(device)
            layer.forward(x)

    # Confirm baseline.prof exists
    if not os.path.exists(prof_path):
        raise SystemExit("baseline.prof was not generated; check decorator and execution.")

    print(f"Generated: {prof_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
