# V12 模型与复现边界

当前发布模型是 `SubcellularV12`，入口配置为
`configs/historical/subcellular_v12_compartment_20260930.yaml`。模型、数据编码、损失和优化器源码
保留原字节；本地路径由新增配置生成器处理。

## 输入和数据契约

- 配对原始核内/核外整数计数，基因名匹配官方 human scGPT vocabulary；未测量基因不当作测量零。
- 训练 H5AD 的 `layers/nuclear_counts`、`layers/extranuclear_counts` 是原始 counts，
  `X=N+C`；`obsm/spatial` 坐标单位为 µm。缓存保留原始 counts 的 float32 log1p 表示，
  V12 再恢复并核实整数 targets；不能拿归一化表达矩阵代替。
- 中心核、质分别按可见正表达基因做确定性 51-bin，ties 用 left 规则。
  同一个 query gene 在两者中同时遮盖，并先于分箱、可见计数统计和 LR 特征计算。
- 每个中心最多8个真实邻居、距离≤100 µm，排除中心自身和重复细胞。邻居输入是 N+C；
  每路最多512 tokens；每中心最多64个 query genes；gene mask=0.15，neighbor dropout=0.2。
- 缓存划分为空间8×8 quantile blocks，seed42、train fraction0.95、context64、center stride32。
  原 epoch 覆盖所有合格 cached centers，不等于每个原始细胞都做过中心。

## 模型与目标

共享普通 scGPT whole-human 编码核与质；从普通 scGPT 初始化，不从 V11 或 scGPT-spatial 初始化。
保留核、质两路局部信息；独立邻域编码生成 spatial，通过可控制残差得到 fused。
LR 为 receiver×sender 软先验，中心 LR 只使用可见 receptor counts；纯 spatial 不读取中心表达。
LR 表路径和 SHA256 写入配置，loader 会校验。

输出 `nucleus`、`cytoplasm`、`local`、`spatial`、`fused` 各512维，
`concat` 将核、质、邻域拼成1536维；`concat_pca512` 是额外评测变换。
共享 gene-conditioned decoder 预测总量 NB 分布及核分配 beta-binomial：
`p(N,C)=p(T=N+C) p(N|T)`，在五个模式 nucleus/cytoplasm/local/spatial/fused 上平均联合 NLL。
这是配对同时观测的计数重建目标，不是已识别的真实时间速度或命运预测。

## 优化与历史验证

仅预训练 backbone 最后2层可训练，用 AdamW 5e-6；新隐藏二维矩阵用
`torch.optim.Muon` 1e-4；decoder/embedding/norm/bias等辅助参数用 AdamW 1e-4。
Muon 固定 momentum=.95、nesterov=true、ns_steps=5、adjust_lr_fn=match_rms_adamw。
global batch64、microbatch4、共享warmup+cosine、一轮训练；每100步归档、每25步恢复状态。
DDP 的有效细胞分母覆盖整个global update，续接不重置学习率。

历史原始48片 Xenium 训练完成10234/10234步，104个检查点评测完成；历史计算节点
50项单测以及4卡两步/优化器恢复 smoke 通过。该结论来自项目运行记录，不能替代本发布包的新环境验证。
旧 step10234 checkpoint 尚未随本 Git 仓库发布，也没有已公开的下载URL；
仅下载scGPT初始权重不能复现该已训练模型的推理结果。

HDmix 是后续实现：Visium HD total-only cells 不得伪造核/质监督，只有分割来源明确且
count-conservation通过的cells可进入核质分支。该扩展仍待本发布对应计算验证，
不能把其代码存在解释为已完成训练或已发布权重。

固定5200验证队列的探针仅在训练部分拟合，统一row L2；concat终点Cell F1=0.580745、
Region F1=0.370085，维数为scGPT基线的3倍。512维fused终点0.537857/0.382652；
原始邻域counts基线Region F1=0.463435，因此现有结果不证明训练后的空间模块优于简单邻域统计。
