

## Tokenizer Comparison
- base: bert-base-chinese
- new : tokenizer/merged_tokenizer
- samples: 20 from tokenizer/data/tokenizer_data/tokenizer_clean.txt
- avg_len_base: 109.70
- avg_len_new : 83.50
- delta_len   : -26.20 (negative = new splits shorter)


## MLM Loss
- model: tokenizer/merged_model_mlm
- tokenizer: tokenizer/merged_model_mlm
- data: tokenizer/data/tokenizer_data/tokenizer_clean.txt
- samples: 2000
- max_length: 256
- batch_size: 8
- avg_loss: 6.2671
