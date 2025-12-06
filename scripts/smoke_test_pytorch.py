#!/usr/bin/env python3
"""
Smoke test for PyTorch availability in this environment.
Exit codes:
 0 - success (torch import + simple ops passed)
 2 - import error (torch not installed)
 3 - runtime error during operations
"""

import sys
import traceback


def main():
    try:
        import torch
    except Exception as e:
        print("ERROR: Failed to import torch:", e, file=sys.stderr)
        traceback.print_exc()
        return 2

    print("torch version:", getattr(torch, "__version__", "unknown"))

    try:
        cuda_available = torch.cuda.is_available()
    except Exception as e:
        print("WARNING: torch.cuda.is_available() raised:", e)
        cuda_available = False
    print("CUDA available:", cuda_available)
    print("torch.version.cuda:", getattr(torch.version, "cuda", None))

    # Simple tensor sanity check
    try:
        x = torch.tensor([1.0, 2.0])
        y = x * 2
        print("tensor op OK:", x.tolist(), "->", y.tolist())

        if cuda_available:
            try:
                d = torch.device("cuda")
                xg = x.to(d)
                print("Moved tensor to GPU device:", xg.device)
            except Exception as e:
                print("WARNING: failed to move tensor to GPU:", e)
        return 0
    except Exception as e:
        print("ERROR during tensor operations:", e, file=sys.stderr)
        traceback.print_exc()
        return 3


if __name__ == "__main__":
    sys.exit(main())

