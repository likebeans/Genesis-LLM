# 辅助 Tokenizer 训练模块 - AGENTS.md

## 模块概览

`aux_tokenizer/` 是词表扩充的**核心脚本模块**，负责训练辅助 tokenizer、筛选新增 token、扩充基座 tokenizer 和扩展模型 embedding。

---

## 目录结构

```
aux_tokenizer/
├── extract_tokenizer_config.py  # Step 1: 提取基座 tokenizer 配置
├── train_aux_tokenizer.py       # Step 2: 训练辅助 SentencePiece tokenizer
├── select_new_tokens.py         # Step 3: 筛选高频新增 token
├── extend_base_tokenizer.py     # Step 4: 将新 token 追加到基座
├── resize_embeddings.py         # Step 5: 扩展模型 embedding
├── chinese_spm_20000.model      # [产物] 辅助 tokenizer 模型
├── chinese_spm_20000.vocab      # [产物] 辅助 tokenizer 词表
├── new_tokens_list.txt          # [产物] 筛选出的新增 token 列表
└── train_spm.log                # [产物] 训练日志
```

---

## 脚本详解

### 1. `extract_tokenizer_config.py` - 提取基座配置

**用途**：记录基座 tokenizer 的关键配置，确保辅助 tokenizer 与基座分词规则一致。

**运行示例**：
```bash
uv run python tokenizer/aux_tokenizer/extract_tokenizer_config.py \
  --tokenizer bert-base-chinese \
  --output tokenizer/logs/baseline_tokenizer_config.json \
  --config config/model_config/config.yaml
```

**输出**：`baseline_tokenizer_config.json`，包含：
- `tokenizer_class`、`vocab_size`、`model_max_length`
- `special_tokens`、`padding_side`、`truncation_side`
- `sp_model_kwargs`（SentencePiece 配置）

---

### 2. `train_aux_tokenizer.py` - 训练辅助 Tokenizer

**用途**：使用 SentencePiece 在清洗语料上训练新的子词表（BPE/Unigram）。

**运行示例**：
```bash
uv run python tokenizer/aux_tokenizer/train_aux_tokenizer.py \
  --input_glob "tokenizer/data/tokenizer_data/tokenizer_clean.txt" \
  --output_dir tokenizer/aux_tokenizer \
  --prefix chinese_spm_20000 \
  --vocab_size 20000 \
  --model_type bpe \
  --character_coverage 0.9995 \
  --split_digits \
  --byte_fallback \
  --baseline_config tokenizer/logs/baseline_tokenizer_config.json
```

**关键参数**：
| 参数 | 说明 | 默认值 |
|-----|------|--------|
| `--vocab_size` | 新增词表大小 | 20000 |
| `--model_type` | `bpe` 或 `unigram` | bpe |
| `--character_coverage` | 字符覆盖率 | 0.9995 |
| `--split_digits` | 拆分数字 | True |
| `--byte_fallback` | 字节回退 | True |

**输出**：`chinese_spm_20000.model`、`.vocab`、`train_spm.log`

---

### 3. `select_new_tokens.py` - 筛选新增 Token

**用途**：对比辅助 tokenizer 与基座 tokenizer 的分词结果，筛选被基座拆碎的高频 token。

**运行示例**：
```bash
uv run python tokenizer/aux_tokenizer/select_new_tokens.py \
  --baseline_config tokenizer/logs/baseline_tokenizer_config.json \
  --aux_tokenizer tokenizer/aux_tokenizer \
  --aux_sp_model tokenizer/aux_tokenizer/chinese_spm_20000.model \
  --input_glob "data_process/final_data/pretrain/*train*.jsonl" \
  --max_samples 100000 \
  --min_freq 5 \
  --max_base_pieces 3 \
  --output tokenizer/aux_tokenizer/new_tokens_list.txt
```

**筛选逻辑**：
1. 用辅助 tokenizer 分词，得到候选 token
2. 用基座 tokenizer 分词同一文本
3. 若基座拆分过碎（>3 pieces）或出现 `[UNK]`，计入候选频次
4. 过滤低频（<5）、已存在于基座的 token
5. 按频率排序输出

**输出**：`new_tokens_list.txt`（每行一个 token）

---

### 4. `extend_base_tokenizer.py` - 扩充基座 Tokenizer

**用途**：将筛选出的新 token 追加到基座 tokenizer。

**运行示例**：
```bash
uv run python tokenizer/aux_tokenizer/extend_base_tokenizer.py \
  --token_config config/tokenizer_config/config.yaml \
  --baseline_config tokenizer/logs/baseline_tokenizer_config.json \
  --tokens_file tokenizer/aux_tokenizer/new_tokens_list.txt \
  --output_dir tokenizer/merged_tokenizer
```

**注意事项**：
- 只追加不删除/重排基座 token
- 特殊 token 顺序保持一致
- 自动更新 config 中的 `extended_tokenizer` 字段

**输出**：`tokenizer/merged_tokenizer/`（完整 tokenizer 文件）

---

### 5. `resize_embeddings.py` - 扩展模型 Embedding

**用途**：调整模型 embedding 层以适配扩充后的词表大小。

**运行示例**：
```bash
uv run python tokenizer/aux_tokenizer/resize_embeddings.py \
  --token_config config/tokenizer_config/config.yaml \
  --output_dir tokenizer/merged_model
```

**关键操作**：
- 调用 `model.resize_token_embeddings(new_vocab_size)`
- 新增 token 的 embedding 随机初始化
- 同时保存模型和 tokenizer

**输出**：`tokenizer/merged_model/`（完整模型 + tokenizer）

---

## 产物文件说明

| 文件 | 说明 |
|-----|------|
| `chinese_spm_20000.model` | SentencePiece 模型文件 |
| `chinese_spm_20000.vocab` | SentencePiece 词表（文本格式） |
| `new_tokens_list.txt` | 筛选出的新增 token（每行一个） |
| `train_spm.log` | 训练日志（超参、输入文件、统计） |

---

## 完整工作流

```bash
# Step 1: 提取基座配置
uv run python tokenizer/aux_tokenizer/extract_tokenizer_config.py \
  --tokenizer bert-base-chinese \
  --output tokenizer/logs/baseline_tokenizer_config.json

# Step 2: 训练辅助 tokenizer
uv run python tokenizer/aux_tokenizer/train_aux_tokenizer.py \
  --input_glob "tokenizer/data/tokenizer_data/*.txt" \
  --vocab_size 20000

# Step 3: 筛选新增 token
uv run python tokenizer/aux_tokenizer/select_new_tokens.py \
  --input_glob "data_process/final_data/pretrain/*.jsonl"

# Step 4: 扩充基座 tokenizer
uv run python tokenizer/aux_tokenizer/extend_base_tokenizer.py \
  --output_dir tokenizer/merged_tokenizer

# Step 5: 扩展模型 embedding
uv run python tokenizer/aux_tokenizer/resize_embeddings.py \
  --output_dir tokenizer/merged_model
```

---

## 注意事项

⚠️ **配置对齐**：辅助 tokenizer 的 `split_digits`、`byte_fallback` 等需与基座一致

⚠️ **语料质量**：输入语料需已清洗去重，避免噪声 token

⚠️ **频率阈值**：`min_freq` 过低会引入噪声，过高会遗漏有用 token

⚠️ **后续训练**：新增 token 的 embedding 是随机的，需 MLM/CPT/SFT 获得语义
