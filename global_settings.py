global_data_root = '/datadisk/yanshou/ssy/TrafficPPT/data'

import torch
# Auto-detect device: cuda, musa, mps, or cpu
global_device = 'cpu'
try:
    if hasattr(torch.backends, 'cuda') and torch.cuda.is_available():
        global_device = 'cuda'
except Exception as e:
    pass
try:
    if torch.musa.is_available():
        global_device = 'musa'
except Exception as e:
    pass
try:
    if torch.mps.is_available():
        global_device = 'mps'
except Exception as e:
    pass
print(f"Detected device: {global_device}")
def get_local_device(index: int = 0) -> str:
    global_device
    # sanitize index
    try:
        idx = int(index)
    except Exception:
        idx = 0
    if idx < 0:
        idx = 0
    if global_device == 'cuda':
        count = torch.cuda.device_count()
        if count > 0:
            if idx >= count:
                idx = 0
            return f'cuda:{idx}'
        return 'cpu'
    if global_device == 'musa':
        count = torch.musa.device_count()
        if count > 0:
            if idx >= count:
                idx = 0
            return f'musa:{idx}'
        return 'cpu'
    if global_device == 'mps':
        return 'mps'
    return 'cpu'

if global_device == 'cuda':
    gpu_name = torch.cuda.get_device_name(0)
    if "A30" in gpu_name or "A40" in gpu_name or "A100" in gpu_name:
        torch.set_float32_matmul_precision('high') # highest, high, medium
        print(f'device is {gpu_name}, set float32_matmul_precision to high')

if __name__ == "__main__":
    print(f"Using device: {global_device}")