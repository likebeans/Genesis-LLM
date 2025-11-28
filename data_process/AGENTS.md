# Data Processing Module (data_process/) - AGENTS.md

## 模块概览

`data_process/` 模块负责 LLM 训练数据的采集和处理，采用**两阶段设计**：

1. **Fetch 阶段**：从 HuggingFace Hub 下载数据集，原样保存（"搬运工"原则）
2. **Process 阶段**：清洗、去重、质量筛选，输出训练就绪的数据

**设计理念**：
- 采集与处理分离，最大化灵活性
- 采集阶段不改字段、不丢列，保留原始数据完整性
- 处理阶段支持多种清洗和过滤策略
- 支持 JSONL 和 Parquet 双格式输出

## 目录结构

```
data_process/
├── fetch_dataset.py       # 数据采集脚本
├── process_data.py        # 数据处理脚本
├── fetch_data/            # 原始数据存放目录
│   ├── pretrain/         # 预训练数据
│   ├── finetune/         # 微调数据
│   └── reward/           # 奖励模型数据
└── final_data/            # 处理后数据存放目录
    ├── pretrain/         # 清洗后的预训练数据
    ├── finetune/         # 清洗后的微调数据
    └── reward/           # 清洗后的奖励数据
```

---

## 1. 数据采集（fetch_dataset.py）

### 功能说明

从 HuggingFace Hub 下载数据集，保存为 JSONL 格式。**核心原则："搬运工"** - 不修改任何字段，不改变数据结构。

### 使用方法

#### 基础用法

```bash
# 下载完整数据集
uv run python data_process/fetch_dataset.py \
  --dataset_name shibing624/medical \
  --output_dir data_process/fetch_data
```

#### 指定子目录（Parquet 数据集）

```bash
# 下载 pretrain 子目录
uv run python data_process/fetch_dataset.py \
  --dataset_name shibing624/medical \
  --data_subdir pretrain \
  --revision refs/convert/parquet \
  --output_dir data_process/fetch_data \
  --sample_limit 1000

# 下载 finetune 子目录
uv run python data_process/fetch_dataset.py \
  --dataset_name shibing624/medical \
  --data_subdir finetune \
  --revision refs/convert/parquet \
  --output_dir data_process/fetch_data \
  --sample_limit 1000

# 下载 reward 子目录
uv run python data_process/fetch_dataset.py \
  --dataset_name shibing624/medical \
  --data_subdir reward \
  --revision refs/convert/parquet \
  --output_dir data_process/fetch_data \
  --sample_limit 1000
```

### 命令行参数

| 参数 | 说明 | 默认值 | 示例 |
|------|------|--------|------|
| `--dataset_name` | HF 数据集名称（必需） | - | `shibing624/medical` |
| `--config_name` | 数据集配置名称（可选） | `None` | `default` |
| `--data_subdir` | Parquet 子目录（可选） | `None` | `pretrain` / `finetune` / `reward` |
| `--output_dir` | 输出目录 | `data_process/fetch_data` | 自定义路径 |
| `--sample_limit` | 每个拆分的采样数量 | `0`（全量） | `1000` |
| `--revision` | 数据集版本/分支 | `None` | `refs/convert/parquet` |
| `--config` | HF 镜像配置文件 | `config/model_config/config.yaml` | 自定义路径 |
| `--local_files_only` | 仅使用本地缓存 | `False` | - |
| `--force_download` | 强制重新下载 | `False` | - |

### 输出格式

**文件命名**：`{dataset_name}__{split}.jsonl`

**示例**：
```
data_process/fetch_data/pretrain/shibing624_medical__train.jsonl
data_process/fetch_data/finetune/shibing624_medical__train.jsonl
data_process/fetch_data/finetune/shibing624_medical__validation.jsonl
```

**JSONL 格式**：每行一个 JSON 对象，保留原始字段
```json
{"text": "糖尿病是一种代谢性疾病...", "source": "medical_wiki", "id": 12345}
{"input": "什么是高血压？", "output": "高血压是指..."}
```

### 特性

- **自动 HF 镜像**：读取 `config/model_config/config.yaml` 设置 `HF_ENDPOINT`
- **预览功能**：终端显示前 5 行数据，便于快速验证
- **灵活采样**：支持全量下载或按比例采样
- **缓存管理**：支持本地缓存或强制重新下载

---

## 2. 数据处理（process_data.py）

### 功能说明

对采集的原始数据进行清洗、去重、质量筛选，输出训练就绪的数据。

### 使用方法

#### 基础清洗

```bash
uv run python data_process/process_data.py \
  --input data_process/fetch_data/pretrain/shibing624_medical__train.jsonl \
  --output_dir data_process/final_data \
  --subdir pretrain
```

#### 完整处理流程

```bash
uv run python data_process/process_data.py \
  --input data_process/fetch_data/pretrain/shibing624_medical__train.jsonl \
  --output_dir data_process/final_data \
  --subdir pretrain \
  --min_chars 5 \
  --max_chars 2000 \
  --max_turns 1 \
  --dedup \
  --near_dedup \
  --jaccard_threshold 0.9 \
  --sentiment_filter negative \
  --sentiment_model uer/roberta-base-finetuned-jd-binary-chinese \
  --to_parquet
```

### 命令行参数

#### 输入输出

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input` | 输入 JSONL 文件路径（必需） | - |
| `--output_dir` | 输出目录 | `data_process/final_data` |
| `--subdir` | 输出子目录 | 自动推断或必需指定 |
| `--to_parquet` | 额外输出 Parquet 格式 | `False` |

#### 文本清洗

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--min_chars` | 最小字符数 | `5` |
| `--max_chars` | 最大字符数 | `10000` |
| `--max_turns` | 最大对话轮数（多轮对话） | `1` |

#### 去重

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--dedup` | 启用精确去重（哈希） | `False` |
| `--near_dedup` | 启用相似去重（Jaccard） | `False` |
| `--jaccard_threshold` | Jaccard 相似度阈值 | `0.9` |

#### 质量筛选

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--sentiment_filter` | 情感过滤 | `None` |
| `--sentiment_model` | 情感分类模型 | `uer/roberta-base-finetuned-jd-binary-chinese` |

### 处理流程

```
原始数据 (JSONL)
    ↓
1. 文本提取
   - 提取 text/input/output 字段
   - 合并多轮对话
    ↓
2. 文本清洗
   - 去除 HTML/Markdown 标签
   - 去除控制字符
   - 压缩多余空白
    ↓
3. 长度过滤
   - 过滤过短文本 (< min_chars)
   - 过滤过长文本 (> max_chars)
    ↓
4. 精确去重 (--dedup)
   - 基于 SHA256 哈希
    ↓
5. 相似去重 (--near_dedup)
   - 基于字符 3-gram Jaccard 相似度
    ↓
6. 质量筛选
   - 打印字符比例检查
   - 情感过滤（可选）
    ↓
清洗后数据 (JSONL/Parquet)
```

### 输出格式

**JSONL 格式**：每行一个 JSON 对象，包含 `text` 字段
```json
{"text": "糖尿病是一种代谢性疾病，主要特征是血糖水平持续升高..."}
{"text": "高血压是指动脉血压持续升高的慢性疾病..."}
```

**Parquet 格式**（可选）：同目录下生成 `.parquet` 文件，列式存储，更高效

**统计日志**：
```
总样本数: 10000
保留样本: 8523
精确重复: 892
相似重复: 456
过短文本: 89
过长文本: 40
```

---

## 数据流向

```
HuggingFace Hub
       ↓
  fetch_dataset.py
       ↓
data_process/fetch_data/
  ├── pretrain/
  ├── finetune/
  └── reward/
       ↓
  process_data.py
       ↓
data_process/final_data/
  ├── pretrain/
  ├── finetune/
  └── reward/
       ↓
训练脚本 (CPT/SFT/RLHF)
```

---

## 最佳实践

### 1. 采集阶段

✅ **推荐做法**：
- 使用 `--sample_limit` 先采样小批量数据验证
- 确认数据格式正确后再全量下载（`--sample_limit 0`）
- 使用 `--force_download` 确保获取最新数据
- 按子目录分别采集（pretrain/finetune/reward）

❌ **避免做法**：
- 不要在采集阶段修改字段或过滤数据
- 不要混合不同用途的数据到同一目录

### 2. 处理阶段

✅ **推荐做法**：
- 先用小数据集测试处理流程
- 根据任务需求调整 `min_chars` 和 `max_chars`
- 预训练数据启用 `--dedup` 和 `--near_dedup`
- 微调数据谨慎使用去重（避免误删有效样本）
- 使用 `--to_parquet` 输出双格式（兼容性更好）

❌ **避免做法**：
- 不要过度清洗导致数据损失
- 不要在小数据集上使用相似去重（容易误删）
- 情感过滤需谨慎（可能引入偏见）

### 3. 数据质量检查

**采集后检查**：
```bash
# 查看数据样本
head -n 5 data_process/fetch_data/pretrain/shibing624_medical__train.jsonl

# 统计行数
wc -l data_process/fetch_data/pretrain/*.jsonl
```

**处理后检查**：
```bash
# 查看清洗后样本
head -n 5 data_process/final_data/pretrain/shibing624_medical__train_clean.jsonl

# 对比处理前后行数
wc -l data_process/fetch_data/pretrain/*.jsonl
wc -l data_process/final_data/pretrain/*.jsonl
```

---

## 常见问题

### Q1: 如何处理多配置数据集？

使用 `--config_name` 参数：
```bash
uv run python data_process/fetch_dataset.py \
  --dataset_name wikitext \
  --config_name wikitext-103-v1
```

### Q2: 如何处理本地数据集？

`datasets` 库支持本地路径：
```bash
uv run python data_process/fetch_dataset.py \
  --dataset_name /path/to/local/dataset \
  --local_files_only
```

### Q3: 相似去重太慢怎么办？

对于大规模数据（>100万样本），建议：
- 使用专业库：MinHash、text-dedup
- 分批处理
- 或跳过相似去重，仅使用精确去重

### Q4: 如何自定义清洗规则？

修改 `process_data.py` 中的 `clean_text()` 函数，添加自定义规则。

### Q5: 输出文件命名规则？

**采集阶段**：`{dataset_name}__{split}.jsonl`
**处理阶段**：`{input_filename}_clean.jsonl`

---

## 依赖项

- `datasets>=4.4.1` - HuggingFace 数据集库
- `transformers>=4.57.3` - 情感分类模型（可选）
- `pyyaml>=6.0.3` - 配置文件解析

---

## 相关文档

- [项目根目录 AGENTS.md](../AGENTS.md) - 项目整体说明
- [配置目录 AGENTS.md](../config/AGENTS.md) - 配置文件详解
- [词表扩充文档](../tokenizer/扩充词表开发方案.md) - 词表扩充工作流

---

## 注意事项

⚠️ **网络依赖**：
- 数据采集需要访问 HuggingFace Hub
- 情感过滤需要下载分类模型
- 建议配置 HF 镜像加速（`config/model_config/config.yaml`）

⚠️ **存储空间**：
- 原始数据和处理后数据会同时存在
- 大规模数据集需要充足的磁盘空间
- 建议定期清理不需要的中间数据

⚠️ **数据隐私**：
- 确保下载的数据集符合使用许可
- 敏感数据需要额外的隐私保护措施
- 不要将包含敏感信息的数据提交到公开仓库
