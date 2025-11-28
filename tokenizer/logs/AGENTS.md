# 日志目录 - AGENTS.md

## 模块概览

`logs/` 目录存放词表扩充各阶段的运行日志和配置快照，用于追溯、复现和审计。

---

## 目录结构

```
logs/
├── baseline_tokenizer_config.json   # 基座 tokenizer 配置快照
├── eval_report.md                   # 评估报告（自动追加）
├── train_spm.log                    # 辅助 tokenizer 训练日志（可选）
└── merge.log                        # tokenizer 合并日志（可选）
```

---

## 文件说明

### `baseline_tokenizer_config.json`

**用途**：记录基座 tokenizer 的关键配置，确保辅助 tokenizer 对齐。

**生成方式**：
```bash
uv run python tokenizer/aux_tokenizer/extract_tokenizer_config.py \
  --tokenizer bert-base-chinese \
  --output tokenizer/logs/baseline_tokenizer_config.json
```

**内容示例**：
```json
{
  "source_tokenizer": "bert-base-chinese",
  "tokenizer_class": "BertTokenizerFast",
  "is_fast": true,
  "vocab_size": 21128,
  "model_max_length": 512,
  "padding_side": "right",
  "special_tokens": {
    "unk_token": "[UNK]",
    "sep_token": "[SEP]",
    "pad_token": "[PAD]",
    "cls_token": "[CLS]",
    "mask_token": "[MASK]"
  }
}
```

---

### `eval_report.md`

**用途**：评估脚本自动追加的评估结果汇总。

**生成方式**：由 `eval_tokenizer.py` 和 `eval_mlm_loss.py` 自动写入。

**内容示例**：
```markdown
## Tokenizer 评估报告

### 分词对比 (2025-11-26)
- 样本数：20
- 基座平均 token 数：45.3
- 扩充平均 token 数：38.7

### MLM Loss (2025-11-26)
- 平均 Loss：2.34
```

---

### `train_spm.log`（可选）

**用途**：辅助 tokenizer 训练的详细日志。

**内容**：超参配置、输入文件列表、训练统计、baseline 快照引用。

---

### `merge.log`（可选）

**用途**：tokenizer 合并过程的日志。

**内容**：新增 token 数量、合并参数、输出路径。

---

## 注意事项

⚠️ **保留日志**：日志文件是复现实验的关键，勿随意删除

⚠️ **版本管理**：建议将日志纳入版本控制或定期备份

⚠️ **时间戳**：所有日志应包含 UTC 时间戳便于追溯
