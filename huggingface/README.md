# Reproducing Embedding Generation with HuggingFace's Transformers

One just run reproduce_doc_embedding.py or reproduce_query_embedding.py to reproduce embedding generation for documents (passages) or queries. These scripts are self-included.

Some notes for understanding the key factors:
1. The tokenization in DPR's code seems to be wrongly writen. It always set the last token position **after many possible paddings** to `sep_token_id`. Please refere to [this issue](https://github.com/facebookresearch/DPR/issues/208).
2. One cannot use [`DPRContextEncoderTokenizer`](https://huggingface.co/docs/transformers/model_doc/dpr#transformers.DPRContextEncoderTokenizer) or [`DPRQuestionEncoderTokenizer`](https://huggingface.co/docs/transformers/model_doc/dpr#transformers.DPRQuestionEncoderTokenizer) direcly for processing all of `input_ids`, `token_type_ids` and `attention_mask`. One reason is due to the last point about the `sep_token_id`. Another reason is, the DPR's official code uses a very different `token_type_ids` (all zeros) from that of HF's.
3. It is found that both running device type (CPU/GPU) and batch size can influence the results!
4. Please pay attention to the data loading. It is recommended to use the CSV reader to load the data. This is because in the DPR's official collection ([`data.wikipedia_split.psgs_w100`](https://github.com/facebookresearch/DPR/blob/02e6454d3217db322c8d9f4401299684b9299723/dpr/data/download_data.py#L31)), the passages are all quoted, which will be extraced from quotes in the DPR official code.
