# 共享表达分箱与胞质相对位置编码

本文记录共享分箱与空间编码的实现状态。代码和本文已发布到 GitHub；编写时尚未部署至训练集群，也没有切换或重启现有 Muon/AdamW 作业。

## 表达分箱

新配置中的 `data.compartment_bin_edges` 是唯一边界来源。训练的数据集和评测均使用这些边界，处理顺序都是 log1p → float16 存储精度 → 正值分箱。0 留给真实零表达，正值映射至 1..num_bins-1；MASK 和 PAD ID 不变。相同表达值遇到重复分位数边界时会进入同一个类别，允许空类别，不按行随机拆散计数相同的值。

准备脚本只从每张预训练切片的训练 split 抽样，核内和胞质共用一套边界。每切片最多2048训练细胞、50000正表达值，汇总拟合分位数；这是每切片贡献有上限的训练样本分布，不声称是所有转录本的精确全局分布。边界写入 YAML 后随 config 存入 checkpoint，另存抽样来源 JSON。无需修改或重建表达缓存。

旧 checkpoint 没有共享边界，评测仍保留旧中位数近似并发出显式 warning、记录 `legacy_median_approximation`。不能把旧模型临时换成新分箱就称为训练/评测一致。旧配置关闭位置特性时仍可使用旧恢复签名。

## 相对位置仅进入胞质跨细胞 attention

对每个接收细胞 i 和发送细胞 j，计算 `delta = coords[j] - coords[i]`。不是统一相对中心细胞的方向。

- 角度：`atan2(dy, dx)` 映射至 [0, 2π)，分16类，每类22.5°；第0类从+x方向起。遵循输入坐标轴约定，图像+y向下时视觉方向也随之变化。
- 距离：默认边界10、20、40、80、160、320 µm，共7类；精确落在边界时进入右侧区间，320 µm及以上为最后一类。配置 `cell_coordinate_scale` 将坐标转成µm，已有µm坐标设1；像素坐标需要填入真实µm/像素比例。
- 两张独立 embedding 表输出每个 attention head 的标量偏置，打分为 `q_i·k_j/sqrt(d) + angle_embedding[a_ij] + distance_embedding[r_ij]`。
- 不改变初始 gene/value/compartment/role embedding。每层只有胞质跨细胞 attention 使用空间偏置。PAD、同一细胞、重复发送细胞仍按原掩码排除；坐标重合的不同细胞保留距离项，角度项为0。
- 空间表零初始化，使初始化时保持内容 attention；训练后学习空间偏好。平移不影响结果，但方向分箱不具备旋转不变性。尚未加旋转增强，也未改变KNN大小和邻域内全连接结构。
- 跨细胞有偏置的 attention 使用自动 SDPA 后端，不能保证继续走旧 Flash 内核；细胞内基因 attention 的 Flash 设置不变。重新训练前需测量显存和吞吐。

## 准备新的训练配置

在计算节点、原项目环境中执行。用独立输出目录，不覆盖现有实验。基础配置中的优化器、学习率和训练日程保持原样：

```bash
python scripts/prepare_spatial_mean_config.py \
  --base-config configs/xenium_compartment_smoke.yaml \
  --output-config configs/cytoplasm_mean_spatial_v2.yaml \
  --output-dir outputs/cytoplasm_mean_spatial_v2 \
  --coordinate-scale 1
```

上述命令使用仓库提供的可移植示例配置；执行前应配置真实数据路径并准备对应缓存。实际全量实验可替换为自己的全量 AdamW 配置。

随后使用原训练入口加载生成的配置。空间开关默认关闭以兼容旧实验；以上准备脚本会开启新实验的空间编码并写入共享边界。准备脚本本次尚未在真实缓存上运行，因此还没有生成声称已拟合的新配置。

评测仍使用 `scripts/evaluate_cytoplasm_pilot.py`，它从检查点构造空间模块，传入每个邻域的坐标并读取共享边界；新特性的供体缓存签名包含边界及空间配置，不复用旧提取规则的缓存。

## 验证状态

已完成修改文件的 Python AST 语法检查和 `git diff --check`。新增 `tests/test_spatial_bias.py` 覆盖方向、距离边界、单位缩放、平移不变性、重复细胞、PAD/核内排除、空间参数梯度、零偏置兼容、细胞置换和激活检查点反向传播，以及训练/评测分箱一致性。

当前沙箱的 Slurm socket 访问被拒绝，尚未执行张量单元测试、真实数据分箱拟合或 GPU 烟雾测试；不报告这些测试已通过。在可用计算节点执行：

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_spatial_bias.py'
PYTHONPATH=src python -m unittest discover -s tests -p 'test_compartment*.py'
```
