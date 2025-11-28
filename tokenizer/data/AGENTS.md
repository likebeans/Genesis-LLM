# Tokenizer 训练数据模块 - AGENTS.md

## 模块概览

`data/` 模块负责获取和清洗用于训练辅助 tokenizer 的语料数据。

---

## 目录结构

```
data/
├── fetch_and_clean.py   # 数据获取与清洗脚本
└── tokenizer_data/      # [产物] 清洗后的训练语料
    ├── tokenizer_clean.txt   # 清洗后的行文本
    └── data_prep.log         # 清洗统计日志
```

---

## 脚本详解

### `fetch_and_clean.py` - 数据获取与清洗

**用途**：从 HuggingFace 或本地获取语料，清洗后输出为行文本格式。

**运行示例**：
```bash
uv run python tokenizer/data/fetch_and_clean.py \
  --output_dir tokenizer/data/tokenizer_data \
  --source hf_wikipedia \
  --dataset_name wikimedia/wikipedia \
  --hf_config 20231101.zh \
  --sample_limit 0 \
  --min_chars 10 \
  --max_chars 2000 \
  --config config/model_config/config.yaml
```

**关键参数**：
| 参数 | 说明 | 默认值 |
|-----|------|--------|
| `--source` | 数据来源：`hf_wikipedia` 或 `local` | hf_wikipedia |
| `--dataset_name` | HuggingFace 数据集名称 | wikimedia/wikipedia |
| `--hf_config` | 数据集配置（如 `20231101.zh`） | - |
| `--sample_limit` | 采样上限（0=全量） | 0 |
| `--min_chars` | 最小字符数 | 10 |
| `--max_chars` | 最大字符数 | 2000 |
| `--input_glob` | 本地模式下的文件匹配 | - |

---

## 清洗流程

1. **拉取数据**：从 HuggingFace Wikipedia 或本地文件获取
2. **文本清洗**：
   - 去除 HTML/Markdown 标记
   - 去除控制字符
   - 合并多余空白
   - 裁剪过长行
3. **去重**：基于 MD5 精确去重（归一化后）
4. **过滤**：
   - 过滤过短行（<`min_chars`）
   - 过滤过长行（>`max_chars`）
   - 丢弃空行、纯噪声
5. **记录日志**：输出统计信息

---

## 输出文件

### `tokenizer_clean.txt`
清洗后的行文本，每行一条文本，UTF-8 编码。

**格式示例**：
```
中华人民共和国是位于东亚的社会主义国家
北京是中国的首都，也是政治、文化中心
人工智能是计算机科学的一个分支
```

### `data_prep.log`
JSON 格式的清洗日志，包含：

```json
{
  "source": "hf_wikipedia",
  "dataset_name": "wikimedia/wikipedia",
  "hf_config": "20231101.zh",
  "sample_limit": 0,
  "min_chars": 10,
  "max_chars": 2000,
  "hf_endpoint": "https://hf-mirror.com",
  "timestamp": "2025-11-26T04:22:32.350165+00:00",
  "stats": {
    "total_lines": 28030648,
    "kept": 13242929,
    "too_short": 12904402,
    "duplicates": 1876075,
    "empty_after_clean": 7146,
    "too_long": 96
  }
}
```

**字段说明**：
- `total_lines`：原始总行数
- `kept`：最终保留行数
- `too_short`：过短被过滤
- `duplicates`：重复被去除
- `too_long`：过长被过滤

---

## 语料要求

**推荐语料**：
- 中文百科（Wikipedia）：约 8GB
- 领域语料（可选）：医疗、法律、金融等

**格式要求**：
- UTF-8 编码
- 行粒度文本（每行一条）
- 已清洗（无 HTML、无乱码）

---

## 本地数据模式

如使用本地文件：
```bash
uv run python tokenizer/data/fetch_and_clean.py \
  --source local \
  --input_glob "path/to/your/*.txt" \
  --output_dir tokenizer/data/tokenizer_data
```

支持的格式：
- `.txt`：直接读取每行
- `.tsv`：取第一列

---

## 注意事项

⚠️ **HF 镜像**：中国用户建议在 `config/model_config/config.yaml` 配置镜像加速

⚠️ **磁盘空间**：Wikipedia 全量约 8GB，清洗后约 3-5GB

⚠️ **采样测试**：建议先用 `--sample_limit 100000` 测试流程

⚠️ **日志追溯**：保留 `data_prep.log` 用于复现和审计
