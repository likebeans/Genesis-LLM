# Tokenizer 评估模块 - AGENTS.md

## 模块概览

`evaluate/` 模块负责验证扩充后 tokenizer 和模型的质量，包括分词对比、OOV 统计和 MLM Loss 评估。

---

## 目录结构

```
evaluate/
├── eval_tokenizer.py   # 分词对比评估
└── eval_mlm_loss.py    # MLM Loss 评估
```

---

## 脚本详解

### 1. `eval_tokenizer.py` - 分词对比评估

**用途**：对比基座 tokenizer 与扩充 tokenizer 的分词效果。

**运行示例**：
```bash
uv run python tokenizer/evaluate/eval_tokenizer.py \
  --base_tokenizer bert-base-chinese \
  --new_tokenizer tokenizer/merged_tokenizer \
  --input_file tokenizer/data/tokenizer_data/tokenizer_clean.txt \
  --num_samples 20 \
  --report tokenizer/logs/eval_report.md
```

**评估指标**：
| 指标 | 说明 | 期望 |
|-----|------|------|
| 平均 token 数 | 同一文本的 token 数量 | 扩充后更少 |
| OOV 率 | 未知词/`[UNK]` 比例 | 扩充后更低 |
| 分词样例 | 可视化对比 | 拆分更合理 |

**输出**：追加到 `--report` 指定的文件

---

### 2. `eval_mlm_loss.py` - MLM Loss 评估

**用途**：在小样本上计算 Masked Language Model 的 Loss，验证模型 sanity。

**运行示例**：
```bash
uv run python tokenizer/evaluate/eval_mlm_loss.py \
  --model tokenizer/merged_model_mlm \
  --tokenizer tokenizer/merged_model_mlm \
  --data_file tokenizer/data/tokenizer_data/tokenizer_clean.txt \
  --max_samples 2000 \
  --max_length 256 \
  --report tokenizer/logs/eval_report.md
```

**评估指标**：
| 指标 | 说明 | 期望 |
|-----|------|------|
| MLM Loss | 掩码语言模型损失 | 越低越好 |
| 困惑度 | exp(loss) | 越低越好 |

**说明**：
- 未跑 MLM 适配时，可用 `tokenizer/merged_model` 作为基线
- Loss 仅作对比参考，新增 token 未训练时 Loss 会偏高

---

## 评估流程

### 快速自检清单

1. **分词对比**：
   - 用典型样例测试：通用文本、领域术语、数字、符号、emoji、对话模板
   - 检查：拆分是否减少、有无 `[UNK]`

2. **长度与 OOV**：
   - 统计平均/中位 token 数
   - 对比新旧 byte fallback/OOV 比例

3. **模型 sanity**：
   - 在小样本上跑 MLM 验证
   - 检查 Loss 是否正常，无报错

---

## 评估报告

脚本会自动追加结果到 `tokenizer/logs/eval_report.md`：

```markdown
## Tokenizer 评估报告

### 分词对比 (2025-11-26)
- 样本数：20
- 基座平均 token 数：45.3
- 扩充平均 token 数：38.7
- 长度减少：14.6%

### MLM Loss (2025-11-26)
- 模型：tokenizer/merged_model_mlm
- 样本数：2000
- 平均 Loss：2.34
- 困惑度：10.4
```

---

## 指标解读

| 指标 | 好的表现 | 需注意 |
|-----|---------|--------|
| 平均 token 数 | 扩充后减少 10-30% | 减少过多可能过拟合 |
| OOV 率 | 接近 0% | >1% 需检查词表 |
| MLM Loss | 与基座差异 <0.5 | 差异过大需更多 MLM 训练 |
| 分词样例 | 领域词不被拆碎 | 常见词被拆碎需排查 |

---

## 注意事项

⚠️ **样本代表性**：评估样本应覆盖通用和领域文本

⚠️ **MLM 前提**：`eval_mlm_loss.py` 需要 MLM 模型，若只做 CLM 训练则跳过

⚠️ **基线对比**：始终与基座 tokenizer 对比，而非绝对值

⚠️ **定期评估**：每次扩充后都应运行评估，保留历史记录
