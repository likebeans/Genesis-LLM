# 合并后 Tokenizer 产物目录 - AGENTS.md

## 说明

`merged_tokenizer/` 是 **脚本运行后的产物目录**，存放扩充后的 tokenizer 文件。

**生成方式**：
```bash
uv run python tokenizer/aux_tokenizer/extend_base_tokenizer.py \
  --output_dir tokenizer/merged_tokenizer
```

---

## 目录内容

| 文件 | 说明 |
|-----|------|
| `tokenizer.json` | Tokenizer 配置（Fast tokenizer） |
| `tokenizer_config.json` | Tokenizer 元数据 |
| `vocab.txt` | 词表文件 |
| `added_tokens.json` | 新增 token 映射 |
| `special_tokens_map.json` | 特殊 token 映射 |

---

## 使用方式

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("tokenizer/merged_tokenizer")
```

---

## 注意事项

⚠️ 此目录由脚本自动生成，请勿手动修改

⚠️ 修改词表后需重新运行 `extend_base_tokenizer.py`
