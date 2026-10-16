# 发布验证记录（2026-10-16）

已验证源码提交：`f45483948b6a6de085e914a38a009984fd59687a`。

[GitHub Actions 完整结果](https://github.com/GlancerZ/Spatial_perturb/actions/runs/38911159419)为 success；同一提交的 push 检查也通过。

验证环境为 GitHub 托管 Ubuntu 24.04、Python 3.11、PyTorch 2.9.0 CPU。

- 安装 `requirements.txt` 并以 editable 模式安装 V12 包：通过。
- 全部发布 Python 与两个可移植 shell 脚本的语法检查：通过。
- V12 模型、数据编码、计数损失及评测协议：50 项单测通过。
- 稀疏 HD 数据加载与固定混合采样：16 项单测通过。
- 可移植 provenance 与资产路径重定位边界：6 项单测通过。

合计 72 项。可移植 provenance 测试采用小型模拟资产，不能代替真实大检查点加载。没有在新环境重新下载完整数据、执行 GPU 训练或推断，不能将 CPU 单测通过解释为新的生物学效果。

首次 CI 发现发布目录的非根路径 `data/` 忽略规则误排除了 Python 数据模块。已改为 `/data/` 并补齐 21 个模块；现有 330 项源码快照全部在 Git 中且 SHA256 匹配。此次修复没有修改模型结构、训练参数或放宽测试。

后续验证说明仅更新 Markdown，沿用以上已通过的源码提交，不重复消耗 CI 运行。
