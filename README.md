<h1 align="center">RMBench: Memory-Dependent Manipulation Benchmark</h1>

RMBench: Memory-Dependent Robotic Manipulation Benchmark with Insights into Policy Design. <i>Under Review</i>, [PDF](https://arxiv.org/pdf/2603.01229) | [arXiv](https://arxiv.org/abs/2603.01229) | [Website](https://rmbench.github.io/) | [Join our Community 🔥](https://robotwin-platform.github.io/doc/community/index.html)

> Tianxing Chen*, Yuran Wang*, Mingleyang Li*, Yan Qin*, Hao Shi, Zixuan Li, Yifan Hu, Yingsheng Zhang, Kaixuan Wang, Yue Chen, Hongcheng Wang, Renjing Xu, Ruihai Wu, Yao Mu, Yaodong Yang, Hao Dong†, Ping Luo†

# 📰 Updates

**2026.09.05** — Switch policy serving, evaluation, and newly collected trajectories to the shared [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab) stack used by RoboTwin 2.0.

**2026.07.14** — Since the previously trained Mem-0 checkpoints were not backed up before our development machine was recycled, we have re-organized the training and now publicly release the retrained model weights:

- **M(1) tasks**: due to limited computational resources, all M1 tasks were trained jointly into a single multi-task `m1_mix` model. The complete model, the processed `m1_mix` dataset, training/inference configs, and all evaluation logs and videos are available at [qiuly/Mem-0-m1mix-RMBench](https://huggingface.co/qiuly/Mem-0-m1mix-RMBench) and [qiuly/Mem-0-m1mix-dataset-RMBench](https://huggingface.co/datasets/qiuly/Mem-0-m1mix-dataset-RMBench).
- **M(n) tasks**: per-task execution-module checkpoints for `battery_try`, `blocks_ranking_try`, `cover_blocks` and `press_button`, together with per-task normalization stats and evaluation results, are available at [qiuly/Mem-0-mn-RMBench](https://huggingface.co/qiuly/Mem-0-mn-RMBench).

Detailed evaluation results can be found in the Hugging Face model cards above.

# 🧑🏻‍💻 RMBench Usage

> This project is built upon [RoboTwin 2.0](https://github.com/robotwin-Platform/RoboTwin). Policy training and evaluation now go through [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab), the same serving stack as RoboTwin.

Existing clones need a one-time layout update: `script/` is now `scripts/`, and `task_config/` is now `env_cfg/task_config/`. Evaluation goes through the `XPolicyLab` submodule. Keep the environment variable names `ROBOTWIN_EVAL_ARGS_FILE` and `ROBOTWIN_SUPPRESS_EVAL_CONFIG` — XPolicyLab client scripts still read those names. The in-repo `policy/` tree is still present and will be removed in a follow-up.

## 1. Installation
First, prepare a conda environment.

```
conda create -n RMBench python=3.10 -y
conda activate RMBench
```

Clone recursively so the XPolicyLab submodule is present:

```
git clone --recurse-submodules https://github.com/RoboTwin-Platform/RMBench.git
cd RMBench
```

For an existing checkout:

```
git submodule update --init --recursive XPolicyLab
```

Then install the simulator environment, CuRobo, and the editable XPolicyLab package:

```
bash scripts/_install.sh
```

To refresh the XPolicyLab pin:

```
bash scripts/update_xpolicylab.sh
bash scripts/update_xpolicylab.sh --stage --install
```

## 2. Download Assets
To download the assets, run the following command. If you encounter any rate-limit issues, please log in to your Hugging Face account by running `huggingface-cli login`:

```
bash scripts/_download_assets.sh
```

## 3. Download Data

The currently published Hugging Face dump still uses the previous RoboTwin-style layout (`data/<task>/demo_clean/...`):

```
bash scripts/_download_data.sh
```

Newly collected demonstrations are written in XPolicyLab trajectory format:

```text
data/<task_config>/<task_name>/<embodiment>/data/episode_0000000.hdf5
```

`<embodiment>` follows the `embodiment` field of the task config (`aloha_agilex` for the default `aloha-agilex` setup).

> **Decode images only through `decode_image_bit`.** XPolicyLab-format cameras are encoded image bits, not a stable JPEG you can pass to `cv2.imdecode` / PIL. Prefer:
>
> ```python
> from XPolicyLab.utils.process_data import decode_image_bit
> rgb = decode_image_bit(image_bits)  # RGB
> ```
>
> A local copy lives in [`data/decode_image_bit.py`](data/decode_image_bit.py). Do **not** add `cv2.cvtColor(..., COLOR_BGR2RGB)` after it.

<details>
<summary>If you need to collect the data (we actually recommend downloading it directly)</summary>

> In RMBench, we always use `demo_clean` setting.

Running the following command will first search for a random seed for the target collection quantity, and then replay the seed to collect data.

```
bash collect_data.sh ${task_name} ${task_config} ${gpu_id}
# Example: bash collect_data.sh cover_blocks demo_clean 0
```
</details>

## 4. Convert to LeRobot (Optional)

Many XPolicyLab policies train on LeRobot datasets. After you have XPolicyLab-format HDF5 under `data/<task_config>/<task>/<embodiment>/data/`, convert with the shared scripts in `XPolicyLab/scripts/`. Those scripts already decode through `decode_image_bit`.

```bash
export HF_LEROBOT_HOME=/path/with/enough/space/lerobot

python XPolicyLab/scripts/transform_lerobot_v21_format.py \
  "demo_clean.cover_blocks.aloha_agilex" \
  --repo_id cover_blocks_demo_clean \
  --max_episode 50
```

Keep `--data_type` as the default `RoboDojo` — RMBench XPolicyLab trajectories share that HDF5 layout.

## 5. Evaluate Policies via XPolicyLab

All evaluation goes through `scripts/eval_policy.sh`. The policy adapter must exist under `XPolicyLab/policy/<policy_name>/` (see the [XPolicyLab policy catalog](https://github.com/XPolicyLab/XPolicyLab/tree/main/policy)). Mem-0 is `Mem_0`.

`--env-cfg-type` selects the XPolicyLab action profile (`arx_x5` matches RMBench's default aloha-agilex layout). Task settings live in `env_cfg/task_config/`.

**Local evaluation (multi-task, multi-GPU):**

```bash
bash scripts/eval_policy.sh multitask \
  --config env_cfg/eval/all_tasks.yml \
  --policy-name Mem_0 \
  --ckpt-name <checkpoint> \
  --env-cfg-type arx_x5 \
  --policy-conda-env <policy_env> \
  --eval-env-conda-env RMBench \
  --action-type joint
```

Add `--dry-run` to validate the schedule without launching anything.

XPolicyLab `Mem_0` classifies tasks with `XPolicyLab/policy/Mem_0/Mem_0/xpolicylab_adapter/task_config.json`. Names missing from that file default to **M1**. `battery_try`, `blocks_ranking_try`, and `press_button` are Mn-style RMBench tasks but are not listed under `Mn` in the current pin; patch that JSON (or pass planner GPUs / `VLLM_URL`) before expecting Mn planning.

**Split deployment (remote policy server + local simulator):**

```bash
# On the policy-server host:
bash scripts/eval_policy.sh serve --config env_cfg/eval/remote_server.yml

# On the simulator host:
bash scripts/eval_policy.sh multitask \
  --config env_cfg/eval/all_tasks.yml \
  --policy-name Mem_0 \
  --env-cfg-type arx_x5 \
  --eval-env-conda-env RMBench \
  --enable-remote \
  --policy-server-ip <server_ip> --policy-server-port <port>
```

# 👍 Citations

If you find our work useful, please consider citing:

```
@article{chen2026rmbench,
  title={RMBench: Memory-Dependent Robotic Manipulation Benchmark with Insights into Policy Design},
  author={Chen, Tianxing and Wang, Yuran and Li, Mingleyang and Qin, Yan and Shi, Hao and Li, Zixuan and Hu, Yifan and Zhang, Yingsheng and Wang, Kaixuan and Chen, Yue and others},
  journal={arXiv preprint arXiv:2603.01229},
  year={2026}
}
```

# 🏷️ License

This repository is released under the MIT license. See [LICENSE](./LICENSE) for additional details.
