import torch
from torch.profiler import profile, ProfilerActivity, schedule

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=0, warmup=1, active=2, repeat=1),
    record_shapes=True,
    with_stack=True,
) as prof:
    for step in range(5):
        x = torch.randn(256, 256).cuda()
        y = x @ x.T
        prof.step()

    prof.export_chrome_trace("trace.json")  # ✅ 导出可视化 trace
