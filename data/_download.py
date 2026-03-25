from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="TianxingChen/RMBench",
    repo_type="dataset",
    local_dir=".",
    allow_patterns=["data/*/demo_clean/**"],
    resume_download=True,
)
