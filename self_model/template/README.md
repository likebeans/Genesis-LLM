# 对话模板存放说明

用途：存放不同模型/项目的对话模板（prompt 格式），用于训练/推理时适配。训练脚本会优先使用 tokenizer 自带的 `apply_chat_template`，若模型未内置则回退到这里的模板并将输入数据格式化成字符串。

建议：
- 按模型或用途区分文件名，如 `alpaca.json`, `chatml.json`, `mistral.json` 等。
- 每个模板文件包含 role/format 占位符，训练脚本读取后拼装指令。
- 在训练配置（`config/self_model_config/*.yaml`）中指明使用的模板路径。

可用模板（按厂商/系列）：
- `qwen2_chatml.json`：Qwen2/Qwen2.5 Instruct，ChatML 风格。
- `llama3_instruct.json`：Llama 3 / Llama 3.1 Instruct，Meta 官方格式。
- `chatglm3.json`：智谱 ChatGLM3，含 `[gMASK]sop` 前缀与 `<|user|>/<|assistant|>` 角色标记。
- `baichuan2_chat.json`：百川 Baichuan2-Chat，使用 `<reserved_106>/<reserved_107>` 角色标记。
- `deepseek_chat.json`：深度求索 DeepSeek-LLM-Chat，使用 `User:/Assistant:` 前缀及 `<｜begin▁of▁sentence｜>/<｜end▁of▁sentence｜>` 特殊 token（需保留这些特殊字符）。

示例 JSON（alpaca 风格）：
```json
{
  "system": "You are a helpful assistant.",
  "user": "{input}",
  "assistant": "{output}",
  "separator": "\n"
}
```

示例 JSON（chatml 风格）：
```json
{
  "messages": [
    {"role": "system", "template": "You are a helpful assistant."},
    {"role": "user", "template": "{input}"},
    {"role": "assistant", "template": "{output}"}
  ],
  "separator": "<|sep|>"
}
```

在 `config/self_model_config/finetune.yaml` 中添加字段（示例）：
```yaml
template_path: self_model/template/chatml.json
```
