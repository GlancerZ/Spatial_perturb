# 数据来源、用途与准备边界

本发布包只保存公开来源清单，不包含原始数据、缓存、H5AD、大图、分割结果或模型权重。数据应从原发布方下载；各数据集沿用其自身许可、引用和访问条件。所有链接来自当前项目已经保存的来源文件及下载脚本，本次只做本地整理，**未逐一联网验证链接当前是否有效**。历史下载成功、HTTP 状态及校验记录不等于今天已重新验证。

## 当前模型实际使用什么

V12 的正式预训练来源是 `configs/historical/xenium48_compartment_sources.yaml` 中的 **48 个 Xenium 样本**，使用原始未校正的核／质计数缓存。精确样本及对应下载链接见 [v12_xenium48.tsv](../data_sources/v12_xenium48.tsv)。一个 epoch 遍历的是既有缓存中 eligible cached centers，不能解释为每个原始细胞均做过训练中心。

原始 49 样本 compartment inventory 中的 `Xenium_Prime_Human_Ovary_FF` 不在该 48 样本配置中。更广的 Vizgen、CosMx、MERFISH、seqFISH、STARmap、SEAAD、DBKERO、GEO 等收集不能统一声称已经进入 V12 训练。候选数据也可能是小鼠、细胞系、派生区域或同一组织的重复发布，不能直接把清单行数当供体数。

每个 Xenium 样本通常下载完整 `*_outs.zip`，保留细胞归属、RNA 分子表和真实核归属信息。只下载细胞表达矩阵与中心坐标不足以重建核／质。核／质输入必须基于真实分子分割或经核验的官方 bin 归属；不能用表达量猜测拆分。预处理应校验逐基因 `N+C=total`、基因轴、细胞 ID 与坐标，并在有足够资源的计算节点运行。

48 个样本的来源采用官方 10x page/file manifest；其中正常骨髓、正常骨两个样本未留存精确 landing page，其 CDN URL 依据 compartment inventory 的 `samples/xenium/<version>/<sample>/<sample>_outs.zip` 和已验证的官方 URL 格式重建，尚未在本轮联网验证。表内对这两行作了明确标记。

## Visium HD 五样本混合计划

[visium_hd5.tsv](../data_sources/visium_hd5.tsv) 包含肾、前列腺癌、脑癌、心脏及乳腺癌五个样本的 **10 个完整文件链接**：每样本 `feature_slice.h5` 与 `barcode_mappings.parquet`。URL 按项目已使用的官方下载脚本和版本化来源清单展开，非临时签名链接。2026-10-11 的项目记录为下载、传输及 SHA256 核对已完成；记录同时明确尚未转换、提交或训练，因此此处按“计划混合续训”发布。

这一路径使用官方细胞分割和 2 µm capture bin 的 `In_cell` / `In_nucleus` 归属汇总核／质整数 UMI，不应表述为逐 RNA 亚细胞定位。两文件必须来自同一样本及同次 Space Ranger 输出。五个物理 slide/area ID 随链接保留，但没有统一患者 ID，不能声称患者级排重。

CRC 物理样本 `H1-VM2JXXK/A1` 及其 Colon_Cancer / Colorectal_Cancer 别名明确排除于该训练计划。

## 广泛收集来源与缺失输入

[collected_sources.tsv](../data_sources/collected_sources.tsv) 收录官方 Xenium 文件映射、DeepSpaceDB 解析到的 10x/GEO/DBKERO 最小归档、SEAAD、SCP2510、皮肤图谱，以及 subcellular data agent 的来源与下载候选。DeepSpaceDB 是索引层，下载优先追溯发布方；不同索引可能重复指向同一样本。候选清单未逐项确认已下载，也未逐项确认可直接训练。

需要保留的限制：

- Vizgen FFPE 16 个人肿瘤样本已有处理后 H5AD，但官方原始文件领取页有注册要求，历史匿名访问为 403；不能把处理后表达当完整核／质输入。RNA2seg 是公开派生小区域集，不能把区域当独立样本。
- SEAAD Xenium 的 reduced mirror 只有 expression、centroid、可选 panel，这些文件本身缺核／质 RNA 归属；SEAAD MERFISH aggregate H5AD 也须另核实输入。
- CosMx NSCLC 八个 Flat 包是原型设备数据，已记录比例为 0.18 µm/px；不要套用其他商用版本比例。`CellComp` 等原始字段需按每个数据版本核实。
- Bento seqFISH 派生 H5AD 缺少低计数 RNA 的分子坐标，不能把缺失坐标解释为零。原始 seqFISH+ MAT/ROI/DAPI 与 STARmap 等来源虽已记录，但仍需完成配准、核界和 panel 检查。
- SODB Fang2022Conservation 的历史 API 为 502，尚无解析出的实验下载链接；清单保留来源入口和限制。

## Benchmark 单独下载

[benchmark_sources.tsv](../data_sources/benchmark_sources.tsv) 列出 GSE316387 黑色素瘤合并对象、预先指定区域的表达／中心坐标，LungPF 样本下载，以及 CRC、ENACT 专家注释、SpaceHack 区域注释和 HiVis liver M6 的来源。

CRC 的专家核标签与辅助区域标签有不同来源。官方 scGPT-spatial 的 SpatialHuman30M 来源表包含该 CRC 切片，其结果不能当成完全未见来源的泛化证明。HiVis liver M6 虽有原始 2 µm 表达，公开包中缺少本切片核／细胞归属表，目前不能直接用于核／质 benchmark。benchmark 条目仅表示评测用途或候选用途，不保证每项已完成评测。

普通 scGPT 预训练权重是初始化资产；官方 scGPT-spatial 权重是对照 baseline。它们不是本清单中的原始训练数据，安装和资产准备按环境文档处理。

## 清单追溯及验证

清单整理依据当前项目的以下文件，不把其本机绝对路径写入公开 TSV：

- `configs/historical/xenium48_compartment_sources.yaml`；`data/pretrain_data/manifests/xenium49_compartment_inputs.tsv`。
- `data/pretrain_data/10x_xenium/manifests/10x_xenium_page_file_links.tsv` 与 `scgpt_spatial_missing_10x_{file_urls.txt,pages.tsv}`。
- `data/pretrain_data/manifests/visium_hd_mix_20261011/sources.tsv`；`scripts/download_visium_hd_mix_sources.sh`。
- 各 source root 下的 `manifests/`；`data/subcellular_data_agent/*/downloads.tsv`；已保存的 data-agent `download_candidates.tsv`。
- `data/benchmarking_data/lungpf/download_manifest.json`；GSE316387 下载脚本；CRC benchmark 与 unified-corpus 文档。

本次核对了 TSV 列数、48 样本配置的逐项覆盖、HD5 每样本两个文件和 URL 字段格式，并扫描了本机路径及签名/token 参数。没有下载原始文件、运行项目 Python、预处理或启动训练。原始 acquisition 清单中的 checksum 应在实际下载后重新核对；本目录的来源清单不能替代下载完整性校验。

HD5 的十个历史 acquisition/transfer SHA256 保存在 [visium_hd5.sha256](../data_sources/visium_hd5.sha256)，使用提供方相对目录，无本机绝对路径。下载后可从包含 `samples/spatial-exp/` 的根目录执行 `sha256sum -c`；这些是项目内容指纹，不冒称发布方签发的 checksum。
