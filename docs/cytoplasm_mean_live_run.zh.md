# 胞质masked-mean模型：运行记录快照

> 历史运行记录，作业号、时间和进度为当时快照，不代表当前运行状态。Muon 四检查点的已完成验证见 [验证报告](cytoplasm_mean_muon_checkpoint_validation.zh.md)；新空间编码尚未完成 GPU 验证。

## 最新：2026-09-15 16:46 UTC，补测 AdamW step75

用户要求测试新检查点。16:45查到训练step80、最新恢复文件step75，自动评测仍等待step200。已在计算节点复制并验证step75，固定为 `outputs/cytoplasm_mean_full_adamw_v1/checkpoints/step_00000075.pt`，不修改训练中的resume.pt。独立卡933354于16:46:43 UTC启动相同5200中心、256细胞KNN、batch8的验证；socket `cytoplasm_933354_latest_adamw_eval`，工具会话57802，日志 `logs/adamw_latest_validation.log`。结果目标 `outputs/cytoplasm_mean_full_adamw_v1/downstream_validation/step_00000075/metrics.json`。此点可与共同step0比较；Muon没有step75归档，不能冒充同一步数优化器对照，正式对照仍等200/400/700。

## 最新：2026-09-15，AdamW 优化器对照

用户要求检验下降是否由 Muon 引起。保留 Muon step1000 完整恢复检查点（已在计算节点读回，包含优化器状态），停止其训练进程并复用双卡作业932549。AdamW输出为 `outputs/cytoplasm_mean_full_adamw_v1`，日志 `logs/cytoplasm_mean_full_adamw_v1.log`，tmux socket `cytoplasm_932549_full_adamw`。启动入口首次因缺少multiprocessing主函数保护而失败，修复后重新启动。

已核实两个rank都加载并逐项验证同一个Muon initial.pt；AdamW覆盖11,027,488参数、Muon参数0。保持48切片全量数据、每卡batch8、两卡总batch16、累积1、峰值LR2e-5、总日程3086更新、warmup154及原有数据/模型/掩码配置。仅优化器名称和输出路径改变；使用run_update_limit在700步暂停，不缩短学习率日程。恢复时只跑剩余步数至700。相同数值学习率的对照不能替代两种优化器各自调参比较。

独立单卡933354已启动等待脚本，在200/400/700归档出现后自动评测同一个5200中心验证子集，batch8；socket `cytoplasm_933354_adamw_validation`，日志 `logs/adamw_validation.log`。Muon初始模型指标可作共同起点，因为权重相同。当前双卡allocation剩余时间不足以完成700步，将按原机制在到时前保存恢复检查点；等待脚本不会自动申请续训资源。训练持有工具会话79538，评测持有会话85351。

## 最新：2026-09-15 15:01 UTC，评测已迁移至独立单卡

用户明确要求新增一张 GPU 专门评测。先核实原作业 932549 两卡利用率 99–100%，再申请独立作业 **933354**（rack10-10，L40S 46,068 MiB，16 CPU、120 GB、6小时）。首次自动审核因已有作业而拒绝；提供两卡均占用的检查证据和用户最新明确授权后，重新审核通过。

实际申请命令（沿用已验证可用的 Vulcan 账户）：

```bash
sbatch --account=aip-junding --partition=gpubase_bygpu_b2 --nodes=1 --gpus-per-node=1 --cpus-per-task=16 --mem=120G --time=06:00:00 --mail-type=ALL --mail-user=bowen.zhao@mail.mcgill.ca --job-name=cyto_eval_dedicated --wrap="sleep 21600"
```

新节点环境核实后，停止原 socket `cytoplasm_932549_validation` 的两个评测进程；确认训练 socket 仍在，训练已到 step790。新评测在 `cytoplasm_933354_dedicated_validation` 的 eval 会话运行，由 srun 持有，工具会话60401。启动于15:00:54 UTC，顺序0、700、200、400，batch从1改8，仅遇到CUDA OOM才自动减半。固定cohort及上下文不变，复用完整供体缓存；中断供体需要重算。

已观测step0复用前两个供体，并继续提取TILD111LA至200/400；batch8正常，瞬时显存12,069/46,068 MiB、GPU利用率77%。尚无新的F1结论或可靠完成时间。完整日志为 `logs/mean_dedicated_933354.log`，具体提取日志为 `logs/mean_dedicated_933354_step_00000000_batch8.log`；四点完成后自动调用汇总脚本。以下并行评测状态为迁移前历史。

## 最新：全量模型下游检查点验证已启动

用户要求先看到检查点的稳定改善。已在作业932549的计算节点独立tmux（socket `cytoplasm_932549_validation`）启动两个评测worker，与训练并行。worker0依次评step0、200；worker1依次评step700、400。已实测两worker都开始提取第一个供体，训练仍正常；并行运行时设备显存约33GiB/卡。

使用同一固定无标签抽样：10训练供体各400中心、3验证供体各400中心，共5200中心；每个中心独立256细胞KNN，最终仅胞质masked mean。完整数据预训练模型的快速验证子集，不是完整下游cohort。分类器为同一C网格及收敛检查，保存逐供体指标和最佳C的验证预测；输出cohort哈希核对。没有使用测试供体选方案。

远端日志 `logs/mean_full_validation_worker_0.log` 和 `logs/mean_full_validation_worker_1.log`，结果 `outputs/cytoplasm_mean_full_muon_v1/downstream_validation/step_*/metrics.json`。提取按供体保存，支持allocation结束后复用已完成供体。`scripts/summarize_mean_validation.py` 可在计算节点检查cohort/context/pooling一致性并生成趋势报告。

评价标准是总体方向和逐供体一致性；不要求每个点严格单调，也不把微小波动称为显著改善。若重建loss下降但下游不改善，优先诊断，不以重建loss作为继续扩大训练的唯一依据。目前评测尚未完成，没有新F1结论。

## 最新：已启动全量Muon训练

### 2026-09-15续训状态

首次全量运行在allocation926338结束前正常暂停于update572/3086（18.5%），保存 `epoch=0,micro_step=572,complete=False` 的resume.pt；最后独立归档是step500。旧作业allocation持有满4小时后正常COMPLETED，训练进程退出码0，不是模型训练崩溃。最后记录step570训练loss1.5352，step500预训练验证loss1.8974；尚无正式全量模型下游评测。

重新查询Vulcan无活动作业后，沿用授权申请两卡作业 **932549**（aip-junding / gpubase_bygpu_b2，原32CPU/230G/4小时资源），已RUNNING于 **rack12-06**。已启动计算节点tmux续训入口，复用原完整配置和resume.pt，不重做batch选择或重置初始化；模型、优化器、数据、batch和schedule均不改变。新的退出状态按作业ID记录，避免旧process.exit阻止续训。

按step200/300/400/500归档时间差约32–33分钟每100步，剩余2514步约需13–14小时计算时间，另加恢复及评测。不能声称本次4小时allocation就能完成整epoch。

用户要求扩大batch、切回Muon并启用全量训练后，已在原Vulcan作业926338上启动 `cytoplasm_mean_full_muon_v1`，未新增或取消allocation。旧pilot已完成300步，不从其AdamW权重续训；正式轮从相同GenePT初始化来源从头开始。

- 48个已准备切片完整缓存，`max_cells=0`、`max_train_steps=0`；恢复256细胞空间子图，保留既有低重叠中心采样及训练／验证划分。全量指不截断输入切片细胞，不表示每个细胞都选为中心。
- 容量探测使用每细胞512有效RNA token、完整256细胞上下文、BF16、checkpointing、双卡DDP及实际Muon更新。每卡8：peak allocated30.884GiB、reserved31.563GiB、6.163秒/更新（3次更新均值，含热身）；每卡16出现CUDA OOM，未继续测32/64。
- 正式每卡batch8、总batch16、累积1；每卡每次2048细胞位置。相比pilot每卡32位置扩大64倍。选择8是显存实测结果，不沿用pilot batch2。
- Muon参数9,437,184；辅助AdamW参数1,590,304。`match_rms_adamw`，两组峰值均2e-5，比例1:1；1epoch=3086更新，warmup154，cosine到2e-6。
- **正式update1已记录**：loss3.7595、LR1.298701e-7。实际训练已开始，不只是容量测试或初始化。
- 远端日志 `logs/cytoplasm_mean_full_muon_v1.log`，输出 `outputs/cytoplasm_mean_full_muon_v1/`，tmux socket `cytoplasm_926338_full_muon`。
- 调度、容量选择及实际Vulcan配置已复制到本地 `outputs/cytoplasm_mean_full_muon_v1/`。
- allocation截止前180秒自动保存完整resume；任务本身不会scancel。全epoch未保证能在剩余allocation内完成，必要时按原配置续训。正式轮下游分析尚未执行。

以下是已完成的早期pilot历史记录，不代表当前正式训练规模。

用户最终确认的方案是：胞质有效RNA token做masked mean pooling（排除PAD，除以有效数量），每个细胞一个向量参与邻域attention。它取代本任务早期拟议的逐RNA cross-attention方案。

## 资源与位置

- Vulcan作业：**926338**，节点 **rack15-08**，2张NVIDIA L40S（各46068MiB）。
- 本集群和Vulcan的作业查询均为空后提交。标准 `rrg-junding` 账户在Vulcan无效；查询有效关联及历史891454后使用 `aip-junding`。
- 实际提交：`sbatch --account=aip-junding --partition=gpubase_bygpu_b2 --nodes=1 --gpus-per-node=2 --cpus-per-task=32 --mem=230G --time=04:00:00 --mail-type=ALL --mail-user=bowen.zhao@mail.mcgill.ca --job-name=cyto_mean_pilot --wrap="sleep 14400"`。
- 独立远端目录：`/scratch/glancerz/spatial-pretrain-v3-cytoplasm-20260914`；data链接复用原工作区，代码、cache及outputs独立。
- 环境：`source scripts/vulcan_compartment_env.sh`，已在计算节点验证Python3.11.5及两张GPU。
- 计算节点tmux socket：`cytoplasm_926338_cytoplasm_mean_pilot_v1`，session `train`。srun保持该tmux所属step存活。控制脚本没有scancel。
- 首轮日志：远端 `logs/cytoplasm_mean_pilot_v1_first_round.log`。
- 输出：远端 `outputs/cytoplasm_mean_pilot_v1/`。

## 实现与验证

新增 `cytoplasm_mean` 模式：在现有pre-norm结构中对有效胞质token求均值，跨细胞QKV宽度256，不展平RNA槽位；排除自身及重复记忆细胞，无胞质或无邻居时旁路；交互结果广播到本细胞有效胞质位置，再接原tokenwise FFN。核内位置不直接更新，细胞内attention仍允许两区室混合。该模式最终embedding也只平均有效胞质token；历史模式保持原提取方式。

- GPU上25项compartment测试通过。
- 最终胞质embedding过滤变更后，4项mean模式测试再次通过：分母与路由、独立基因排列、空胞质及梯度、Flash与reference一致性。
- 双卡smoke：2次更新，训练loss 3.2267→2.9458；最终验证loss 2.9946；12.2秒，rank0峰值allocated 0.62GiB、reserved 0.79GiB；日志包含done。仅验证可运行性，不代表下游改善。
- 48个切片各最多8192行的pilot缓存已完成；这是有行截取偏倚的快速子集，不是完整48切片训练。

## 在执行的首个审查点

已启动 `scripts/hold_cytoplasm_iteration_tmux.sh cytoplasm_mean_pilot_v1`。

流程：测试→缓存→双卡smoke→正式训练100次更新→step0和step100冻结probe→写first_round.exit。正式配置总预算300更新、每卡batch2、累积4、16细胞上下文、AdamW峰值2e-5、5%warmup及cosine至10%。100步暂停不会取消allocation；完成首轮后检查验证结果再决定续训或修改。

probe固定每供体最多400个无标签随机抽取中心，只用训练和验证供体；每个中心在原切片独立取KNN，输出胞质均值。原始测试供体不用于pilot选方案。子集指标不得与历史全细胞benchmark直接比较。

正式试训已实际开始：两进程DDP、11,027,488参数；当前已观测update1 loss3.6849、update5 loss3.6682、update10 loss3.3455、update15 loss3.1804。初始checkpoint已保存，调度记录核实总300更新、15步warmup、总batch16。尚无下游结果；不能以这些训练loss宣称实验成功。

## 首轮检查与续训状态

首轮100步及两次probe完成，`first_round.exit=0`。step100训练loss1.6352、预训练验证loss1.8470。固定快速验证集的细胞类型macro-F1：step0=0.313351、step100=0.308957；区域：0.200541→0.206793。结果混合，不能宣布有一致改善；保持配置继续完成300步退火，而非立即更换方案。

已在同一个allocation启动续训tmux（socket `cytoplasm_926338_continue`），按200/300步保存并评测，最后完成最终验证。仍不取消allocation。最新采样时update105/300，loss1.6546，AdamW LR=1.597942e-5。

运行中nvidia-smi采样：GPU0=1507MiB / 46068MiB，GPU1=1511MiB / 46068MiB；这是当时device-used，不是全程PyTorch allocated峰值。CSV位于远端 `logs/cytoplasm_mean_pilot_v1_gpu.csv`，每5秒采一次。暂停时曾采到每卡3MiB，不应当作训练显存。

每卡batch2个子图、每子图16细胞、累积4；两卡有效batch16个子图（256个细胞位置，可能跨子图重复）。首100步从initial.pt到step100归档相隔约159.3秒，含启动尾部、验证及保存；单次probe约142–146秒。由此估计剩余195次更新约5分钟，连同200/300评测、恢复和最终验证约10–15分钟。该估计仅针对本轮300步pilot，不代表整个迭代任务完成时间。
