import json
from pathlib import Path
import torch

cfg = json.loads(Path("configs/paths.json").read_text(encoding="utf-8"))
print("data_dir:", cfg["data_dir"])
print("outputs_dir:", cfg["outputs_dir"])

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))

# outputs_dir에 "hello.txt" 하나 써보기(권한/경로 체크)
out = Path(cfg["outputs_dir"])
out.mkdir(parents=True, exist_ok=True)
(out / "hello.txt").write_text("ok\n", encoding="utf-8")
print("wrote:", out / "hello.txt")
