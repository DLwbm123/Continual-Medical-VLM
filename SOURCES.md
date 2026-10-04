# 来源与边界

核对日期：2026-10-04。下列是用于确定方法/数据事实的原始来源；本启动方案的大部分实验设计为建议，不是这些论文已验证的结论。

## S0：用户上传论文
Qin 等，Diagnose Like a Doctor: Misdiagnosis-Aware Structured Memory for Continual Learning of Medical Vision-Language Models。
用户提供的 PDF：第 3 页 §2.1 为 task-incremental 定义；第 6 页 §3.1 为 11 个数据集与 BiomedCLIP 等实现条件。本方案不要求继续其外部检索流程，也不把本文任务顺序当真实临床时间轴。

## S1：MedCBR 方法稿
https://arxiv.org/html/2603.08921v1
§3.1、§3.3：并行诊断/概念头与图文联合目标；§4.1 与附录：数据及评价。不是持续学习论文。启动方案只借鉴概念建模，不直接转用其训练时长、结果或生成式解释性能。

## S2：MedCBR 作者代码与说明
https://github.com/mharmanani/MedCBR
作者标注 CVPR 2026 Findings；README 主训练流程使用 ViT-L/14，并包含报告生成/推理阶段。本方案首轮不运行这两个生成模块。实际复制源码前需核查许可证，研究前固定 commit。

## S3：MedCBR 模型实现
https://raw.githubusercontent.com/mharmanani/MedCBR/main/src/models/clip.py
检查到了 BiomedCLIP 模型类、概念分类器和 multitask/bottleneck/fusion 路径。仅做代码阅读，未在用户环境运行。默认 context length 和维数配置需在迁移时核验。

## S4：BrEaST 数据说明论文
https://www.nature.com/articles/s41597-024-02984-z
Curated benchmark dataset for ultrasound based breast lesion analysis，Scientific Data，2024。数据含多类图像级描述与病灶掩膜；应保留缺失/不适用状态和原始字段语义。不是所有病例都有病理活检确认；验证来源需逐例保留。

## S5：BrEaST 官方数据入口
https://www.cancerimagingarchive.net/collection/breast-lesions-usg/
图像/分割与临床 XLSX 标签为不同下载项。存在概念数据的公开来源，不代表用户已经下载或同意新增下载。使用需遵守数据引用与许可。

## S6：Microsoft BiomedCLIP 官方模型卡
https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
核查官方模型名、文本编码器/tokenizer/预处理与加载路径。以实际 revision 锁定为准，不将任意通用 CLIP 模型替代而不说明。

## 非来源结论
H1/H2/H3、三任务锚点设计、5–8 个概念组开发建议、回放数量、更新次数、保持门控、实验阶梯与验收点均是本项目的研究设计建议；没有完成真实数据验证或完整的新颖性检索。


## 本轮实际锁定

- MedCBR commit：`d556a0ff590fbcdc6fafd1c3b7d09a0f667b67ac`；未发现 LICENSE/COPYING，仅 README badge，未复制其源码进入公开仓库。
- BrEaST v1：Pawłowska, A., et al. (2024), Curated benchmark dataset for ultrasound based breast lesion analysis (Breast-Lesions-USG) [dataset], The Cancer Imaging Archive. https://doi.org/10.7937/9WKK-Q141 。TCIA v1 updated 2024-01-08，CC BY 4.0；图像和临床表文件日期 2023-12-15。
- 描述性论文：Pawłowska et al., Scientific Data 11 (2024), https://doi.org/10.1038/s41597-024-02984-z 。Table 3 决定 7 组字段类型；Margin 是多标签，Posterior features 是单选。
- BiomedCLIP 模型 revision：`9f341de24bfb00180f1b847274256e9b65a3a32e`，模型与数据下载不代表可重新分发，仓库不包含权重和样本。
