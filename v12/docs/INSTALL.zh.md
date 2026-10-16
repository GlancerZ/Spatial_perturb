# 安装与从零运行 V12

本目录是可单独上传的 GitHub 源码包。以下命令从仓库的 v12/ 目录执行（先 cd v12）；原始数据、scGPT权重、
训练cache和本项目已训练checkpoint均不进入Git。新clone可从普通scGPT重新训练；
历史 step10234 权重当前没有公开下载链接，不能直接复现该权重的推理。

## 环境

历史已验证：Alliance Killarney，StdEnv/2023、python/3.11.5、arrow/23.0.1、
项目 state venv，计算节点报告 PyTorch 2.13.0；训练4×L40S，评测1×L40S。
本仓库不提供该HPC环境的完整freeze锁，不能将这些记录当作可移植pip lock。
`requirements.txt` 是按实际imports给出的依赖范围，尚未在新机器完整重装验证。

可移植建议为Linux、Python3.11、NVIDIA CUDA GPU。先按
[PyTorch官方安装选择器](https://pytorch.org/get-started/locally/)安装与驱动匹配的CUDA wheel，
再安装项目。`torch>=2.9`是所用Muon API的最低推荐范围；
[PyTorch2.9 Muon API](https://docs.pytorch.org/docs/2.9/generated/torch.optim.Muon.html)
已包含本实现依赖的 `adjust_lr_fn=match_rms_adamw`、`ns_coefficients`、`ns_steps`。
并非声明所有范围内组合都通过训练验证。无需安装scgpt包、torchtext或flash-attn：
本实现只加载官方源码的两个embedding类并使用标准PyTorch Transformer。

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# 此处先执行PyTorch官方选择器给出的CUDA安装命令。
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
```

下面的Python、cache构建、测试、训练和推理在Alliance必须进入经过核实的计算节点执行，
激活当前任务授权的环境；不能在登录节点执行。Slurm资源申请与环境激活遵循本机/项目规则，
本指南不自动申请、取消或续接allocation。普通独立GPU主机使用可移植入口；
Slurm专用的历史控制器仍保留其资源检查。原模型和脚本源码保留其hash。

## 下载普通scGPT初始资产

```bash
bash scripts/setup_scgpt_assets.sh resources/scgpt_official
```

脚本使用官方whole-human Drive文件ID，下载三个文件并核验预期SHA256，
下载固定commit的 `scgpt/model/model.py`，复制原始来源manifest。输出结构为：

```text
resources/scgpt_official/
  source_manifest.json
  code/scgpt/model/model.py
  scGPT_human/{args.json,vocab.json,best_model.pt}
```

来源见[官方model zoo](https://github.com/bowang-lab/scGPT#pretrained-scgpt-model-zoo)及
[官方Drive文件夹](https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y)。
若Google限流/要求登录，可以从该文件夹手动取得同名文件并对照脚本的SHA256，
不使用替代模型。脚本拒绝覆盖既有目录；失败后检查已下载文件，再选新目录完成。
来源manifest格式含 backbone_kind=scgpt、model_name=whole-human、repository、
code_commit、download_file_ids、weights_sha256、vocab_sha256；loader按文件内容核验来源。
LR派生表已在 `resources/lr_v6/`，配置固定其SHA256，勿替换为不同数据库后沿用旧实验名。

## 原始数据→核质counts→geometry→训练cache

[数据来源与下载清单](DATA.zh.md)列出公开URL、原始格式和适用范围。
先从清单取得一个Xenium原始输出archive；原生 `cells.csv.gz`、
`cell_feature_matrix.h5` 和 `transcripts.parquet` 必须来自同一样本。
以下以本地实际下载路径和样本ID为例；不要将 Visium HD 总量矩阵当作核质counts。

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python examples/prepare_xenium_local.py \
  --archive /absolute/path/to/xenium_output.zip \
  --sample-id MY_SAMPLE --output-dir work/xenium
python scripts/configure_local_v12.py \
  --h5ad-dir work/xenium/h5ad --geometry-root work/xenium/geometry \
  --work-dir work/v12 --assets-dir resources/scgpt_official
python scripts/portable_v12.py --task prepare-cache -- \
  --config work/v12/v12.yaml --workers 1
```

生成器输出绝对路径 `work/v12/sources.yaml`、`work/v12/v12.yaml`，
保留原配置data/model/loss/optimizer参数；每个H5AD映射到geometry下同名`_k255`目录。
原始reference与核质H5AD分开保存，核质counts来自transcript位置/核分配，
cache读取原始整数counts再log1p，不能用仅有整细胞counts的矩阵补造核/质。
示例单样本是新训练数据范围；只有原始未校正的完整48片、相同counts/geometry与协议才能对照历史一轮结果。

## 测试与训练

在已核实计算节点和正确环境中：

```bash
python -m unittest discover -s tests -p 'test_subcellular_v12*.py'
python scripts/portable_v12.py --task train -- \
  --config work/v12/v12.yaml --run-update-limit 2
# 检查日志、有效有限loss、checkpoint及恢复状态；同一输出目录续接：
python scripts/portable_v12.py --task train -- \
  --config work/v12/v12.yaml --resume
```

原配置global batch64/microbatch4，单GPU通过累积更新；显存与剩余时限必须先核实。
不要为绕过资源不足静默修改batch、优化器、分箱或恢复点。`--run-update-limit 2`
仅为smoke；不代表一个epoch。正式训练到完整epoch才称完成。
分布式训练可用 `torchrun --standalone --nproc-per-node=N scripts/portable_v12.py --task train -- --config work/v12/v12.yaml`，
其中N必须为已授权且实际可用GPU数量，Slurm环境同时保持原脚本的compute guard。

## 推理

先准备输入的显式推理schema（新文件，不覆盖训练输入）：

```bash
python examples/prepare_inference_h5ad.py \
  --input work/xenium/h5ad/MY_SAMPLE.h5ad \
  --output work/inference/MY_SAMPLE.h5ad
```

此转换验证 `X=nuclear_counts+extranuclear_counts`，保留整数counts，
新增 `cytoplasmic_counts`、`spatial_um` 别名和来源说明。
推理只接受一个physical section、唯一barcodes、实际测量panel；
`--input-mode compartments`输出完整V12读出。

普通GPU主机显式选择独立计算（禁止在Alliance登录节点使用此标志）：

```bash
python scripts/infer_v12_local.py --standalone-compute \
  --assets-dir resources/scgpt_official \
  --lr-path resources/lr_v6/curated_single_gene_pairs.csv \
  --h5ad work/inference/MY_SAMPLE.h5ad \
  --checkpoint /absolute/path/to/a/trusted/step_XXXXXX.pt \
  --expected-step ACTUAL_CHECKPOINT_STEP --input-mode compartments \
  --output-dir work/inference/result --centers 1024 --batch-size 8
```

`ACTUAL_CHECKPOINT_STEP`必须替换为检查点真实整数step，旧权重终点才是10234。
该入口对checkpoint引用的资产路径作显式重定位并验证内容，不能改checkpoint去绕过hash。
Slurm计算节点也可使用此入口而不加`--standalone-compute`。
原 `infer_visium_hd_v12.py` 仍要求真实Slurm job+step且checkpoint内原资产路径可访问。
只对可信来源checkpoint执行加载（原实现使用PyTorch checkpoint反序列化）。

推理写出H5AD读出、provenance与图；须检查真实输出和日志才能判断成功。
本次发布仅做静态审核，不包含新环境的安装、GPU smoke、cache/训练/推理实跑；
历史V12成功证据与HDmix待验证状态见[模型说明](MODEL.zh.md)。

## GitHub目录与独立GPU主机

在目标GitHub仓库克隆后，先执行 `cd v12`；本指南的所有路径和命令相对于该目录。
上面的预处理、cache和训练示例默认在真实Slurm计算节点运行。普通独立CUDA服务器使用：

```bash
python examples/prepare_xenium_local.py --standalone-compute \
  --archive /absolute/path/to/xenium_output.zip --sample-id MY_SAMPLE --output-dir work/xenium
python scripts/portable_v12.py --standalone-compute --task prepare-cache -- \
  --config work/v12/v12.yaml --workers 1
python scripts/portable_v12.py --standalone-compute --task train -- \
  --config work/v12/v12.yaml --run-update-limit 2
```

使用生成器生成配置后再构建cache。推理schema转换 `prepare_inference_h5ad.py` 也应在计算环境中运行。
`--standalone-compute`不可用来绕过HPC登录节点限制；历史HD扩展保持其Slurm专用入口。
