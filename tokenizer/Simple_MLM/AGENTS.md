# 简单 MLM 训练模块 - AGENTS.md

## 模块概览

`Simple_MLM/` 提供简单的 Masked Language Model 微调脚本，用于让新增 token 获得语义表示。

---

## 目录结构

```
Simple_MLM/
└── train_mlm.py   # MLM 微调脚本
```

---

## 脚本详解

### `train_mlm.py` - MLM 微调

**用途**：在扩充词表后，通过短期 MLM 训练让新增 token 的 embedding 学到语义。

**背景**：
- 扩充词表后，新增 token 的 embedding 是随机初始化的
- 直接用于下游任务效果不佳
- 通过 MLM 训练可以快速赋予新 token 基本语义

**运行示例**：
```bash
uv run python tokenizer/Simple_MLM/train_mlm.py \
  --tokenizer tokenizer/merged_model \
  --model tokenizer/merged_model \
  --data_file tokenizer/data/tokenizer_data/tokenizer_clean.txt \
  --output_dir tokenizer/merged_model_mlm \
  --max_length 256 \
  --mlm_probability 0.15 \
  --batch_size 16 \
  --learning_rate 5e-5 \
  --max_steps 1000
```

**关键参数**：
| 参数 | 说明 | 默认值 |
|-----|------|--------|
| `--tokenizer` | 扩充后的 tokenizer 路径 | tokenizer/merged_model |
| `--model` | 扩充后的模型路径 | tokenizer/merged_model |
| `--data_file` | 训练语料 | tokenizer_clean.txt |
| `--output_dir` | 输出目录 | tokenizer/merged_model_mlm |
| `--max_length` | 最大序列长度 | 256 |
| `--mlm_probability` | 掩码比例 | 0.15 |
| `--batch_size` | 批次大小 | 16 |
| `--learning_rate` | 学习率 | 5e-5 |
| `--max_steps` | 训练步数 | 1000 |

---

## 训练流程

1. **加载模型**：加载扩展 embedding 后的模型
2. **加载数据**：读取清洗后的行文本
3. **Tokenize**：将文本转换为 token
4. **MLM 训练**：
   - 随机掩码 15% 的 token
   - 预测被掩码的 token
   - 更新 embedding 和模型参数
5. **保存模型**：输出到 `output_dir`

---

## 输出

训练完成后，`output_dir` 包含：
- `model.safetensors` - 模型权重
- `config.json` - 模型配置
- `tokenizer.json` - Tokenizer 配置
- `vocab.txt` - 词表

**使用方式**：
```python
from transformers import AutoModelForMaskedLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("tokenizer/merged_model_mlm")
model = AutoModelForMaskedLM.from_pretrained("tokenizer/merged_model_mlm")
```

---

## 训练建议

**步数选择**：
- 快速验证：500-1000 步
- 基本语义：2000-5000 步
- 充分训练：10000+ 步

**资源消耗**：
- 1000 步约需 10-30 分钟（取决于 GPU）
- 显存约 8-16GB（batch_size=16, max_length=256）

---

## 临时文件

训练过程会在 `tokenizer/mlm_tmp/` 生成临时文件和 checkpoint，训练完成后可清理。

---

## 注意事项

⚠️ **模型类型**：此脚本使用 `AutoModelForMaskedLM`，适用于 BERT 类模型

⚠️ **CausalLM**：如果基座是 GPT/LLaMA 类模型，应使用 CLM 而非 MLM

⚠️ **后续训练**：MLM 只是初步赋予语义，建议后续配合 CPT/SFT 进一步训练

⚠️ **评估验证**：训练后运行 `eval_mlm_loss.py` 验证效果
