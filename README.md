<h1 align="center">RMBench</h1>

RMBench: Memory-Dependent Robotic Manipulation Benchmark with Insights into Policy Design<br>
[PDF (pre-release)](https://github.com/RMBench/RMBench.github.io/blob/main/pre_release_version.pdf). The official release is coming Soon.

> This project is built upon [RoboTwin 2.0](https://github.com/robotwin-Platform/RoboTwin), and you can seamlessly transfer your policy code between the two projects.


# 🧑🏻‍💻 RMBench Usage 

## 1. Installation
First, prepare a conda environment.

```
conda create -n RMBench python=3.10 -y
conda activate RMBench
```

RMBench Repo: https://github.com/RoboTwin-Platform/RMBench

```
git clone https://github.com/RoboTwin-Platform/RMBench.git
```

Then, run `script/_install.sh` to install basic conda envs and CuRobo:

```
bash script/_install.sh
```

## 2. Download Assets
To download the assets, run the following command. If you encounter any rate-limit issues, please log in to your Hugging Face account by running `huggingface-cli login`:

```
bash script/_download_assets.sh
```

## 3. Download 

Please run the following command to download all data.

```
bash script/_download_data.sh
```

<details>
<summary>If you need to collect the data (we actually recommend downloading it directly)</summary>

> In RMBench, we always use `demo_clean` setting.

Running the following command will first search for a random seed for the target collection quantity, and then replay the seed to collect data.

Please strictly follow our tutorial in [RoboTwin 2.0 Doc - Collect Data](https://robotwin-platform.github.io/doc/usage/collect-data.html).

```
bash collect_data.sh ${task_name} ${task_config} ${gpu_id}
# Example: bash collect_data.sh cover_blocks demo_clean 0
```
</details>

## 4. Run Policies

1. Mem-0 (ours): [See Mem-0 Document](./policy/Mem-0/README.md)
2. DP: [See DP Document](https://robotwin-platform.github.io/doc/usage/DP.html)
3. ACT: [See ACT Document](https://robotwin-platform.github.io/doc/usage/ACT.html)
4. Pi 0.5: [See Pi 0.5 Document](https://robotwin-platform.github.io/doc/usage/Pi05.html)
5. X-VLA: [See X-VLA Document](./policy/X-VLA/README.md)
6. Other Policies (Pi0, RDT, etc): [See Document](https://robotwin-platform.github.io/doc/usage) and [See Folder](./policy/)
6. **Configure your policy:** [See Tutorial Here](https://robotwin-platform.github.io/doc/usage/deploy-your-policy.html)

# 👍 Citations

If you find our work useful, please consider citing:

```
```

# 🏷️ License

This repository is released under the MIT license. See [LICENSE](./LICENSE) for additional details.