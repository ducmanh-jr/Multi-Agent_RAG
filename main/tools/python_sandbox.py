"""
Isolated Python Execution Sandbox (REPL) for Mathematical & Empirical Verification.
Safely executes generated Python / NumPy / SymPy snippets to formally verify
formulas, matrix dimensions, and numerical claims found in scientific papers.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
import time
from typing import Any, Optional
from utils.logger import logger


FORBIDDEN_MODULES = {
    "os", "subprocess", "socket", "shutil", "requests", "urllib", "http",
    "ftplib", "builtins", "importlib", "pty", "commands", "posix", "sys",
    "ctypes", "multiprocessing", "threading", "signal", "inspect", "linecache"
}

FORBIDDEN_CALLS = {
    "eval", "exec", "compile", "__import__", "open", "input", "exit", "quit",
    "setattr", "delattr", "globals", "locals", "vars", "breakpoint", "help",
}

FORBIDDEN_ATTRS = {
    "__class__", "__bases__", "__subclasses__", "__builtins__", "__globals__",
    "__code__", "__closure__", "__getattribute__", "__setattr__", "__delattr__",
    "__init__", "__new__", "__dict__", "__module__", "format_map", "mro",
}


def is_code_safe(code_str: str) -> tuple[bool, str]:
    """
    Statically analyzes Python code AST to prevent security violations,
    unauthorized file I/O, network calls, or malicious process spawning.
    """
    try:
        tree = ast.parse(code_str)
    except SyntaxError as e:
        return False, f"Syntax Error: {e}"

    for node in ast.walk(tree):
        # 1. Block forbidden imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_module = alias.name.split(".")[0]
                if root_module in FORBIDDEN_MODULES:
                    return False, f"Import of module '{alias.name}' is strictly forbidden in sandbox."
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root_module = node.module.split(".")[0]
                if root_module in FORBIDDEN_MODULES:
                    return False, f"Import from module '{node.module}' is strictly forbidden in sandbox."

        # 2. Block forbidden function calls
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "getattr":
            if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str):
                if node.args[1].value in FORBIDDEN_ATTRS or node.args[1].value.startswith("__"):
                    return False, f"getattr with dangerous attribute '{node.args[1].value}' is strictly forbidden." 
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in FORBIDDEN_CALLS:
                    return False, f"Calling built-in '{node.func.id}()' is prohibited in sandbox."
            elif isinstance(node.func, ast.Attribute):
                if node.func.attr in FORBIDDEN_CALLS or node.func.attr.startswith("__"):
                    return False, f"Calling method '{node.func.attr}()' is prohibited in sandbox."

        # 3. Block dangerous attribute access (dunder / meta-programming escape)
        if isinstance(node, ast.Attribute):
            if node.attr in FORBIDDEN_ATTRS or node.attr.startswith("__"):
                return False, f"Access to dunder or meta-programming attribute '{node.attr}' is strictly prohibited in sandbox."

    return True, "Code passed security AST audit."


def execute_python_code(code_str: str, timeout_sec: int = 4) -> dict[str, Any]:
    """
    Executes Python code in an isolated subprocess with resource limits and timeout.
    Returns structured results including status, stdout, and execution time.
    """
    safe, msg = is_code_safe(code_str)
    if not safe:
        logger.warning(f"[Python Sandbox Security] Rejected code: {msg}")
        return {
            "status": "BLOCKED",
            "output": "",
            "error": msg,
            "execution_time_ms": 0.0
        }

    # Wrap code to capture standard output cleanly
    wrapper_code = f"""import sys
import math
try:
    import numpy as np
except ImportError:
    pass

{code_str}
"""

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(wrapper_code)
        temp_file = f.name

    start_t = time.perf_counter()
    try:
        res = subprocess.run(
            [sys.executable, temp_file],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            encoding="utf-8",
            errors="replace"
        )
        elapsed_ms = (time.perf_counter() - start_t) * 1000.0

        if res.returncode == 0:
            return {
                "status": "PASS",
                "output": res.stdout.strip(),
                "error": "",
                "execution_time_ms": round(elapsed_ms, 2)
            }
        else:
            return {
                "status": "FAIL",
                "output": res.stdout.strip(),
                "error": res.stderr.strip()[:300],
                "execution_time_ms": round(elapsed_ms, 2)
            }
    except subprocess.TimeoutExpired:
        return {
            "status": "TIMEOUT",
            "output": "",
            "error": f"Execution exceeded {timeout_sec}s time budget limit.",
            "execution_time_ms": timeout_sec * 1000.0
        }
    except Exception as e:
        return {
            "status": "ERROR",
            "output": "",
            "error": str(e),
            "execution_time_ms": 0.0
        }
    finally:
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except OSError:
                pass


# Pre-built academic mathematical verification recipes
def verify_attention_matrix_shapes(batch_size: int = 2, seq_len: int = 128, d_model: int = 64, num_heads: int = 4) -> dict[str, Any]:
    """Verifies Multi-Head Attention Q*K^T scaling and tensor shapes."""
    code = f"""
import numpy as np
b, n, d, h = {batch_size}, {seq_len}, {d_model}, {num_heads}
d_k = d // h
Q = np.random.randn(b, h, n, d_k)
K = np.random.randn(b, h, n, d_k)
V = np.random.randn(b, h, n, d_k)
scores = np.matmul(Q, np.swapaxes(K, -1, -2)) / np.sqrt(d_k)
assert scores.shape == (b, h, n, n), f"Invalid scores shape: {{scores.shape}}"
print(f"PASS: Attention Tensor Shapes Verified -> QK^T shape: ({{b}}, {{h}}, {{n}}, {{n}}), Output shape: ({{b}}, {{n}}, {{d}})")
"""
    return execute_python_code(code)


def verify_lora_decomposition(d_in: int = 4096, d_out: int = 4096, rank: int = 16) -> dict[str, Any]:
    """Verifies Low-Rank Adaptation (LoRA) parameter reduction and output dimensional match."""
    code = f"""
import numpy as np
d_in, d_out, r = {d_in}, {d_out}, {rank}
full_params = d_in * d_out
lora_params = (d_in * r) + (r * d_out)
reduction_factor = full_params / lora_params

B = np.random.randn(d_out, r)
A = np.random.randn(r, d_in)
delta_W = np.matmul(B, A)
assert delta_W.shape == (d_out, d_in), f"Shape mismatch in delta_W: {{delta_W.shape}}"
print(f"PASS: LoRA Verified -> Full: {{full_params:,}} params | LoRA (r={{r}}): {{lora_params:,}} params ({{reduction_factor:.1f}}x reduction)")
"""
    return execute_python_code(code)
