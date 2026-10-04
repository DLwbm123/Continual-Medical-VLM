# 环境与来源锁定

实测：Linux 训练服务器；Python 3.12.3；NVIDIA RTX PRO 5000 Blackwell（72 GB 级）；PyTorch 2.7.1+cu128、torchvision 0.22.1；运行参数使用 BF16。复用既有 CUDA 环境，只把 open_clip_torch 3.1.0、timm 1.0.19、ftfy 6.3.1、wcwidth 0.2.13 安装到项目隔离依赖目录，未升级共享环境。其余依赖见 requirements.txt。

模型：microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224，revision `9f341de24bfb00180f1b847274256e9b65a3a32e`。下载完整 checkpoint 及配置/tokenizer，权重 783,705,670 bytes。辅助文本架构 config 来自 microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract，revision `d673b8835373c6fa116d6d8006b33d48734e305d`。文本权重由完整 BiomedCLIP checkpoint 装载，不另外下载 BERT 权重。训练时离线加载，运行时核对 image/text 都是 512 维，context length 256。

MedCBR 参考：`mharmanani/MedCBR@d556a0ff590fbcdc6fafd1c3b7d09a0f667b67ac`。审计 README、依赖与 src/models/clip.py。原版主要使用 CLIP ViT-L/14，也提供 BiomedCLIP 类；有 multitask、bottleneck、fusion 接口及 LLM 模块。本实现只借鉴并行诊断/概念/图文监督机制；冻结文本、全图补边、缺失掩码、固定词表对齐和持续学习辅助函数均为本项目工程设定。原作者报告、训练时长与结果未作为本项目结果。

运行前完成数据盘挂载、余量与小型读写探针；数据、模型、日志、checkpoint 均存于规定数据盘。每次 GPU 启动前查询可用显存，使用中性临时入口和环境变量传入路径，启动后检查 ps 与 nvidia-smi。没有终止其他任务。

公开边界：源代码、可移植配置、聚合指标、聚合学习曲线、实验报告。私有保留真实路径、manifest、病例 ID、图像/掩膜/临床表、逐病例预测、权重和 checkpoint。没有计算文件哈希或声称逐字节一致。

图表为单独的聚合结果后处理；服务器训练环境未安装 matplotlib，图表使用本地应用附带的 Python 3.12.14 与项目私有隔离目录中的 matplotlib 3.10.6 / numpy 2.3.5 生成，不改变训练环境。
