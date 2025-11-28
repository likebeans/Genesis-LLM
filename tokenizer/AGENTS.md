# Tokenizer 词表扩充模块 - AGENTS.md

## 模块概览

`tokenizer/` 模块负责 **词表扩充（Vocabulary Expansion）** 工作流，目标是为基座模型追加中文/领域词表，减少专业术语被拆碎或 OOV，同时保持兼容性。

**核心流程**：
1. 记录基座 tokenizer 配置
2. 清洗语料（百科 + 领域语料）
3. 训练辅助 tokenizer（对齐基座配置）
4. 从分词对比筛选新增 token
5. 将新 token 追加到基座 tokenizer
6. 扩展模型 embedding
7. 评估与验证

---

## 目录结构

```
tokenizer/
├── aux_tokenizer/       # 🔧 辅助 tokenizer 训练（核心脚本）
├── data/                # 📊 训练语料数据
├── evaluate/            # 📈 评估脚本
├── logs/                # 📝 运行日志
├── Simple_MLM/          # 🧪 简单 MLM 实验代码
├── merged_tokenizer/    # 📦 [产物] 合并后的 tokenizer
├── merged_model/        # 📦 [产物] 扩展 embedding 后的模型
├── merged_model_mlm/    # 📦 [产物] MLM 预训练后的模型
├── mlm_tmp/             # 📦 [产物] MLM 训练临时文件
└── 扩充词表开发方案.md   # 📖 详细开发文档
```

---

## 核心目录详解

### `aux_tokenizer/` - 辅助 tokenizer 训练

存放辅助 tokenizer 的训练脚本和产物，是词表扩充的核心模块。

| 文件 | 用途 |
|-----|------|
| `train_aux_tokenizer.py` | 训练辅助 SentencePiece tokenizer（BPE/Unigram） |
| `extract_tokenizer_config.py` | 提取基座 tokenizer 配置，用于对齐 |
| `select_new_tokens.py` | 对比分词结果，筛选高频新增 token |
| `extend_base_tokenizer.py` | 将新 token 追加到基座 tokenizer |
| `resize_embeddings.py` | 扩展模型 embedding 层以适配新词表 |
| `chinese_spm_20000.model/.vocab` | 训练好的辅助 tokenizer（20k 词表） |
| `new_tokens_list.txt` | 筛选出的新增 token 列表 |
| `README.md` | 使用说明 |

**典型工作流**：
```bash
# 1. 提取基座配置
uv run python tokenizer/aux_tokenizer/extract_tokenizer_config.py

# 2. 训练辅助 tokenizer
uv run python tokenizer/aux_tokenizer/train_aux_tokenizer.py

# 3. 筛选新增 token
uv run python tokenizer/aux_tokenizer/select_new_tokens.py

# 4. 扩充基座 tokenizer
uv run python tokenizer/aux_tokenizer/extend_base_tokenizer.py

# 5. 扩展模型 embedding
uv run python tokenizer/aux_tokenizer/resize_embeddings.py
```

---

### `data/` - 训练语料

存放用于训练辅助 tokenizer 的语料数据。

| 文件/目录 | 用途 |
|----------|------|
| `fetch_and_clean.py` | 获取并清洗语料的脚本 |
| `tokenizer_data/` | 清洗后的训练语料（txt 格式） |
| `README.md` | 数据来源与清洗说明 |

**语料要求**：
- UTF-8 编码，行粒度文本
- 建议：中文百科 ~8GB + 领域语料
- 已清洗：去 HTML/Markdown、去重、过滤过短/过长

---

### `evaluate/` - 评估脚本

用于评估扩充后 tokenizer 的质量。

| 文件 | 用途 |
|-----|------|
| `eval_tokenizer.py` | 分词对比：OOV 率、token 数、覆盖率 |
| `eval_mlm_loss.py` | 评估 MLM 模型的困惑度/Loss |
| `README.md` | 评估指南 |

**评估指标**：
- **OOV 率**：未知词比例（越低越好）
- **平均 token 数**：同一文本的 token 数变化
- **困惑度**：语言模型评估（需要模型训练后）

---

### `logs/` - 运行日志

存放各阶段的运行日志，用于追溯和调试。

- `train_spm.log` - 辅助 tokenizer 训练日志
- `merge.log` - tokenizer 合并日志
- `eval_report.md` - 评估报告

---

### `Simple_MLM/` - MLM 实验代码

简单的 Masked Language Model 实验代码，用于验证扩充后词表的有效性。

---

## 产物目录（运行后生成）

以下目录是脚本运行后的产物，无需手动创建或修改：

| 目录 | 说明 |
|-----|------|
| `merged_tokenizer/` | 合并新 token 后的 tokenizer 文件（可直接用于训练） |
| `merged_model/` | 扩展 embedding 后的模型（可用于后续 CPT/SFT） |
| `merged_model_mlm/` | 经过 MLM 预训练的模型（新 token 已有语义） |
| `mlm_tmp/` | MLM 训练过程的临时文件和 checkpoint |

**使用方式**：
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# 加载扩充后的 tokenizer
tokenizer = AutoTokenizer.from_pretrained("tokenizer/merged_tokenizer")

# 加载扩展 embedding 后的模型
model = AutoModelForCausalLM.from_pretrained("tokenizer/merged_model")
```

---

## 配置文件

词表扩充的配置位于 `config/tokenizer_config/config.yaml`：

```yaml
# 基座 tokenizer
base_tokenizer: bert-base-chinese

# 辅助 tokenizer 训练参数
aux_tokenizer:
  vocab_size: 20000        # 新增词表大小
  model_type: bpe          # BPE 或 Unigram
  character_coverage: 0.9995
  split_digits: true
  byte_fallback: true

# 输出路径
output:
  tokenizer_path: tokenizer/merged_tokenizer
  model_path: tokenizer/merged_model
```

---

## 注意事项

⚠️ **只追加不删除**：新 token 只能追加到词表末尾，不能删除或重排基座 token

⚠️ **需要训练**：新增 token 的 embedding 是随机初始化的，需要 CPT/SFT 才能获得语义

⚠️ **特殊 token 顺序**：确保特殊 token（如 `<|im_start|>`）顺序与基座一致

⚠️ **备份**：保留旧 tokenizer/模型备份，便于回滚

---

## 相关文档

- [扩充词表开发方案](./扩充词表开发方案.md) - 完整开发流程与细节
- [aux_tokenizer/README.md](./aux_tokenizer/README.md) - 训练脚本使用说明
- [evaluate/README.md](./evaluate/README.md) - 评估方法说明
- [data/README.md](./data/README.md) - 数据准备说明
