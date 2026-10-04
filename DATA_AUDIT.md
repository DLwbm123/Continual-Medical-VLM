# 数据审计（M0）

审计日期：2026-10-04。实际数据与私有 manifest 位于用户指定训练服务器的数据盘；公开仓库只保留聚合信息，位置映射见本地私有运行记录。

## 原有 11 个分类任务

| 任务 | 类数 | train | val | test | 合计 | 能力 |
|---|---:|---:|---:|---:|---:|---|
| BTMRI | 4 | 2854 | 1141 | 1717 | 5712 | A |
| BUSI | 3 | 389 | 155 | 236 | 780 | A |
| CHMNIST | 8 | 2496 | 1000 | 1504 | 5000 | A |
| COVID_19 | 4 | 10582 | 4232 | 6351 | 21165 | A |
| CTKidney | 4 | 6221 | 2487 | 3738 | 12446 | A |
| DermaMNIST | 7 | 7006 | 1003 | 2005 | 10014 | A |
| KneeXray | 5 | 5778 | 826 | 1656 | 8260 | A |
| Kvasir | 8 | 2000 | 800 | 1200 | 4000 | A |
| LungColon | 5 | 12500 | 5000 | 7500 | 25000 | A |
| OCTMNIST | 4 | 97476 | 10832 | 1000 | 109308 | A |
| RETINA | 4 | 2108 | 841 | 1268 | 4217 | A |

共 205,902 张图像。只读盘点覆盖 205,913 个物理文件（含 11 个 split JSON）；实际适配器核对全部路径、类别、split 与图像头。没有缺失引用、重复路径跨 split、mask 混入分类或未引用图像。未做全量像素解码、内容哈希或近重复图像搜索，这些结果不能证明患者独立性。

下载的版本没有 patient/lesion/slide ID、mask 或图像级概念标签；manifest 保留这些未知值而不从文件名猜测。CT/MRI 切片、病理 patch 和增强副本仍可能跨划分相关。全部沿用上游划分，未按模型结果重划分。

DermaMNIST train 实测 7,006、OCTMNIST train 实测 97,476，分别比来源 README 数字少 1；保留实际上游文件及 split，没有补造样本。两者分发图像为 224×224，但本轮未证实是否由低分辨率重采样得到。

来源：数据 [TahaKoleilat/BiomedCoOp](https://huggingface.co/datasets/TahaKoleilat/BiomedCoOp)，revision `c50b6304ea49623188dc7f0f470c511b48538108`；split [HealthX-Lab/BiomedCoOp](https://github.com/HealthX-Lab/BiomedCoOp)，commit `23710b0b867e8cb4a5ea95b0ccbaa5e35203db0d`。分发许可不能替代各原始医学数据集许可，本仓库不重新分发数据。

## BrEaST 概念锚点

用户明确授权补充，角色固定为锚点训练/验证/测试，不称为纯外部测试。使用 TCIA v1（2024-01-08；文件日期 2023-12-15），CC BY 4.0。[数据 DOI](https://doi.org/10.7937/9WKK-Q141)，[说明论文](https://doi.org/10.1038/s41597-024-02984-z)。

252 个病灶病例做 benign/malignant 二分类；4 个 normal 留在私有 manifest 的 excluded 集。源文档说明 256 图来自 256 人，因此 source CaseID 用作患者分组代理，并未恢复真实患者 ID。随机种子 0，按诊断分层，每类 floor(60%) train、floor(20%) val，其余 test；不按概念覆盖或模型表现重选划分。

概念字典见 `schemas/breast_v1.json`：shape/echogenicity/posterior/calcifications 为类别型，margin 为多标签，halo/skin 为二元。源表的 margin 多选明确完整时才把未选值编码为阴性。`not applicable`、空白、`not available`、`indefinable` 均不产生负监督。calcifications 的 indefinable 作为不可评价而非病理阴性。

只使用专家视觉描述作概念；BIRADS、Interpretation、Diagnosis、Verification、症状和病史均不输入模型。诊断验证方式在私有 manifest 中保留；并非所有病例都有活检。主肿块与其他肿块 mask 都与原图尺寸核对，mask 仅保留作后续区域研究，首轮不输入模型、不当作概念定位真值。

能力为 C：有真实征象与病灶分割，但没有各征象独立定位，也没有本项目新增临床复核。多病灶图的质量限制：mass-oriented 标签针对受检查主病灶，首轮全图模型不能证明它关注了该病灶。

实际核对通过：256 图 + 266 masks = 522 PNG；主图/掩膜对应与尺寸均可读，无孤立 PNG。已锁定 train 150（92/58），val 49（30/19），test 53（32/21），括号为 benign/malignant。全部 source CaseID 跨 split 不交叉。

验证集各组正例支持如下，完整三划分支持见 `results/breast_audit.json`：

| 组 | 取值顺序 | 正例数 |
|---|---|---|
| shape | irregular, oval, round | 26, 22, 1 |
| margin | circumscribed, indistinct, angular, microlobulated, spiculated | 27, 17, 5, 9, 5 |
| echogenicity | hypoechoic, heterogeneous, anechoic, isoechoic, complex cystic/solid, hyperechoic | 24, 14, 1, 4, 3, 3 |
| posterior | no, shadowing, enhancement, combined | 34, 8, 6, 1 |
| halo | yes | 16 |
| calcifications | no, in a mass, intraductal | 46, 3, 0 |
| skin | yes | 2 |

训练/测试中的不可评价钙化保留为 masked；验证无 intraductal 正例，完整钙化字典 AUROC/F1 必须是 NA。
