import torch
print("torch version:", torch.__version__)
print("has torch.xpu:", hasattr(torch, "xpu"))
print("torch.xpu.is_available:", torch.xpu.is_available() if hasattr(torch, "xpu") else None)

try:
    import intel_extension_for_pytorch as ipex
    print("ipex version:", ipex.__version__)
    print("has ipex.xpu:", hasattr(ipex, "xpu"))
    print("ipex.xpu.is_available:", ipex.xpu.is_available() if hasattr(ipex, "xpu") else None)
except Exception as e:
    print("ipex import error:", e)

import platform, subprocess, sys
print("python:", sys.version)
print("platform:", platform.platform())