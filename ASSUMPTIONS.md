# 已锁定的首轮工程设定

- 仅 M0/M1：原有 BUSI 用于分类工程检查，BrEaST 是唯一概念训练锚点；尚未运行 M2/P1–P4 或 M3。
- BiomedCLIP 视觉骨干和并行线性头可训练；文本编码器、logit scale 冻结。运行时验证表示维度 512、文本长度 256。
- 所有输入全图等比例补边到 224×224，再使用 checkpoint 的 evaluation transform；无随机增强、无 oracle ROI、无真实标签或文件名输入文本编码器。低分辨率可能损伤微小征象，后续应单独研究。
- 固定类别文本和真实概念值词表。每个分类概念值只保留一个模板；二元/多标签用固定 present/absent 模板对。相同概念值共享一个正类原型，避免重复描述互为负例。训练标签只用于选择监督目标，推理始终对全部固定候选评分。
- L = CE_diagnosis + mean_group(masked CE/BCE_concept) + CE_class_text + mean_group(masked CE/BCE_concept_text)。各项系数 1。没有合法概念的 batch 产生可微零。
- AdamW：视觉 lr 1e-5，头 lr 1e-4，weight decay 0.01，batch 32，clip norm 1；seed 0、BF16、TF32 关闭、deterministic algorithms。每步从训练集均匀有放回采样；首轮不做类权重或概念重采样。
- 小样本检查：每诊断类 12 个训练病例、100 updates；正式单任务从原始 checkpoint 重新初始化，1000 updates，每 200 updates 只读验证。最终报告预定的 step 1000，不用测试或验证曲线挑最佳 checkpoint。
- 原有 BUSI 小样本检查每类 8 图、100 updates，无真实概念；其结果不能与 BrEaST 二分类直接比较。
- 固定测试集仅做标注结构和支持数审计，未用于模型预测、阈值选择或调参；二元阈值固定 0.5。
- 全字典 AUROC 遇缺失正/负支持记 NA；类别型概念 F1 遇真值类别缺失也记 NA。逐属性 support/NA 保留；不删难例提高均分。并行概念性能不证明诊断依赖概念。
- source CaseID 一人一图的分组依据来自原作者，不是独立重新去重审计。小样本区间描述病例抽样不确定性，不代表训练种子方差。

`indefinable` 钙化的工程编码采取保守整组屏蔽，未利用其可能包含的部分阳性信息；原始表完整保留。类别型 accuracy 以病例计算，margin accuracy 为每病例五属性正确率的均值；后者受多数阴性影响，必须结合逐属性 F1/AUROC 阅读。
