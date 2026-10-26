"""Exercise actual CUDA forward/backward kernels before loading PPO weights."""
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def main():
    result = {"status": "FAIL", "torch": torch.__version__, "cuda": torch.version.cuda,
              "compiled_arches": torch.cuda.get_arch_list()}
    try:
        assert torch.cuda.is_available(), "CUDA unavailable"
        result["device"] = torch.cuda.get_device_name(0)
        result["capability"] = list(torch.cuda.get_device_capability(0))
        torch.manual_seed(42)
        x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        y = x @ x.T
        y.float().square().mean().backward()
        assert torch.isfinite(y).all() and torch.isfinite(x.grad).all()
        q = torch.randn(1, 8, 64, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        attention = F.scaled_dot_product_attention(q, q, q)
        attention.float().square().mean().backward()
        torch.cuda.synchronize()
        assert torch.isfinite(attention).all() and torch.isfinite(q.grad).all()
        result.update(status="PASS", bf16_matmul_backward=True, sdpa_backward=True,
                      peak_memory_bytes=torch.cuda.max_memory_allocated())
    except Exception as error:
        result["exception"] = f"{type(error).__name__}: {error}"
    path = Path(__file__).resolve().parents[1] / "evidence/ppo-gpu-check.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
