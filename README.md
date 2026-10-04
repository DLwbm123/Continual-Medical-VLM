# 临床概念与诊断证据的持续学习

基于用户提供的研究启动包，当前实施范围为 M0 数据审计与 M1 单任务可行性。模型是 BiomedCLIP + 并行诊断/概念头 + 固定文本词表对齐。不是 MedCBR 完整数值复现，也没有验证医学推理或持续学习收益。

## 当前交付

- `DATA_AUDIT.md`：11 个分类任务共 205,902 图；额外 BrEaST 锚点、病例分组和概念支持。
- `DATA_GAPS.md`、`ASSUMPTIONS.md`、`ENVIRONMENT.md`：未验证事项、冻结开发协议、实际依赖和来源 revision。
- `M1_REPORT.md`、`results/`：实际实验、聚合学习曲线、支持数与负结果。
- `tools/audit_classification.py`、`tools/audit_breast.py`：实际数据适配与私有 manifest。
- `core.py`、`concepts.py`、`run_single.py`：模型、掩码损失、评价和单任务训练。
- `schemas/breast_v1.json`：7 组真实概念编码。`*.example.json` 和 `pilot.template.yaml` 仍然仅为启动包示例，不是执行配置。
- `tests/`：27 项已运行测试，包括分组泄漏、CE/BCE、未知/不适用掩码、多正例/固定词表、教师、旧头、replay 和 CPU RNG/优化器恢复。没有声称验证了 GPU bitwise resume。

## 数据和模型

原始数据、临床表、病例 ID、逐病例预测与权重不在仓库。请通过 `SOURCES.md` 的官方链接在有权限的环境获取数据，遵守各原始许可。已下载数据无需重复下载。模型必须使用 `ENVIRONMENT.md` 的 revision 和匹配 tokenizer/config。

BrEaST 是锚点训练/验证/测试数据，不是纯外部评价集。252 个病灶病例按 seed 0 分成 150/49/53；另 4 个 normal 保留为 excluded。测试集未进行模型预测。margin 是多标签，posterior 是单选；未知、不适用、不可评价值绝不补阴性。

## 可复现入口

所有真实路径从环境变量传入；在已有兼容 CUDA 环境中使用 `requirements.txt` 的依赖。源码根目录和项目隔离依赖目录加入 `PYTHONPATH`。为了避免进程参数暴露任务信息，实际运行用 `python -` 从 stdin 载入入口；下面命令需先设置路径变量。

```bash
python -m unittest discover -s tests -v

# DATA_ROOT points to the official BrEaST download directory.
# AUDIT_OUTPUT must be new; CONCEPT_SCHEMA points to schemas/breast_v1.json.
python - < tools/audit_breast.py

# Required: CONFIG_TEMPLATE, RUN_OUTPUT, MANIFEST, SNAPSHOT_DIR,
# CONCEPT_SCHEMA, DATA_ROOT, CODE_ROOT, ENTRYPOINT, NEUTRAL_TMP,
# PYTHON_EXECUTABLE, PYTHONPATH, CUDA_VISIBLE_DEVICES.
# ENTRYPOINT points to run_single.py; RUN_OUTPUT ends in a new run's artifacts/.
# Set HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1,
# CUBLAS_WORKSPACE_CONFIG=:4096:8 before launch.
# CONFIG_TEMPLATE: configs/breast_smoke.json, then configs/breast_single.json.
python - < tools/launch.py

# After the single-task run completes, use its RUN_OUTPUT plus MANIFEST/schema.
python - < tools/summarize_run.py

# Optional plotting environment: matplotlib 3.10.6 plus numpy.
python - < tools/plot_run.py
```

启动器保存有效配置、源码快照、中性入口、PID 和日志，使用独立后台会话；拒绝覆盖已有 run。训练保存 step、optimizer、采样器和各 RNG 状态；运行配置的 `resume` 可指向本项目自己的 checkpoint。不要加载来源不明的 pickle checkpoint。

`RESEARCH_START_PLAN.md` 是用户提供的阶段研究计划。当前没有启动 M2/P1–P4 或 M3。未来概念遗忘评价要在 M1 的逐概念能力可接受后进行；并行头提升不构成诊断依赖概念的证据。
