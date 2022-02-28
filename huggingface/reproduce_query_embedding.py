import base64
from io import BytesIO
import numpy as np
import torch
import requests
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer

# What we can get from running DPR's official code:
# Question: "who got the first nobel prize in physics" (the very first query in `data.retriever.qas.nq-test`)
# First 10 dimensions: tensor([-0.2951,  0.0967, -0.1200,  0.2249, -0.1496,  0.0503, -0.0231, -0.1769, -0.2681, -0.0883])

# ======================= DPR HF example ======================= 
tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
model = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
model.eval()

texts = ['who got the first nobel prize in physics']
inputs = tokenizer(
    texts,
    add_special_tokens=True,
    max_length=256,
    pad_to_max_length=True,
    truncation=True,
    return_tensors='pt'
)
input_ids = inputs['input_ids'][0].tolist()
input_ids[-1] = tokenizer.sep_token_id  # So this is the key to give it a full reproduction. It seems to be a bug in the official code: https://github.com/facebookresearch/DPR/issues/210
input_ids = torch.LongTensor([input_ids])
print(input_ids)
# tensor([[  101,  2040,  2288,  1996,  2034, 10501,  3396,  1999,  5584,   102,
#              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#              0,     0,     0,     0,     0,   102]])

with torch.no_grad():
    embeddings = model(input_ids).pooler_output.tolist()
print(embeddings[0][:10])
# [-0.2951129376888275, 0.09669505059719086, -0.11997669190168381, 0.22485391795635223, -0.1496342569589615, 
# 0.050305984914302826, -0.023147625848650932, -0.1769288182258606, -0.2681022584438324, -0.08832291513681412]
