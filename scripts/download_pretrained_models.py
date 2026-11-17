from pathlib import Path

from huggingface_hub import login, snapshot_download

login()

MODELS = {
    "TookaBERT-Base": "PartAI/TookaBERT-Base",
    "roberta-fa-zwnj-base": "HooshvareLab/roberta-fa-zwnj-base",
    "mdeberta-v3-base": "microsoft/mdeberta-v3-base",
}

base_dir = Path("models/pretrained")
base_dir.mkdir(parents=True, exist_ok=True)

for folder_name, repo_id in MODELS.items():
    target = base_dir / folder_name
    print(f"\n=== Downloading {repo_id} -> {target} ===")
    snapshot_download(
        repo_id=repo_id,
        local_dir=target,
        ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.onnx"],
    )
    print(f"Done: {folder_name}")