---
license: apache-2.0
language:
- fa
pipeline_tag: fill-mask
mask_token: "<mask>"
widget:
- text: "توانا بود هر که <mask> بود ز دانش دل پیر برنا بود"
- text: "شهر برلین در کشور <mask> واقع شده است."
- text: "بهنام <mask> از خوانندگان مشهور کشور ما است."
- text: "رضا <mask> از بازیگران مشهور کشور ما است."
- text: "سید ابراهیم رییسی در سال <mask> رییس جمهور ایران شد."
- text: "دیگر امکان ادامه وجود ندارد. باید قرارداد را <mask> کنیم."
---
# Model Details

TookaBERT models are a family of encoder models trained on Persian in two sizes base and large. These Models pre-trained on over 500GB of Persian data including a variety of topics such as News, Blogs, Forums, Books, etc. They pre-trained with the MLM (WWM) objective using two context lengths.

For more information you can read our paper on [arXiv](https://arxiv.org/abs/2407.16382).

## How to use

You can use this model directly for Masked Language Modeling using the provided code below.

```Python
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("PartAI/TookaBERT-Base")
model = AutoModelForMaskedLM.from_pretrained("PartAI/TookaBERT-Base")

# prepare input
text = "شهر برلین در کشور <mask> واقع شده است."
encoded_input = tokenizer(text, return_tensors='pt')

# forward pass
output = model(**encoded_input)
```

It is also possible to use inference pipelines such as below.

```Python
from transformers import pipeline

inference_pipeline = pipeline('fill-mask', model="PartAI/TookaBERT-Base")
inference_pipeline("شهر برلین در کشور <mask> واقع شده است.")
```

You can use this model to fine-tune it over your dataset and prepare it for your task.

- DeepSentiPers (Sentiment Analysis) <a href="https://colab.research.google.com/drive/1Vn5QTYutdCo6iXVTmsPW9K4t8xVk14ji#scrollTo=1B1YrypZxajF"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab Code" width="87" height="15"/></a>
- ParsiNLU - Multiple-choice (Multiple-choice) <a href="https://colab.research.google.com/drive/1boXMnRIwqAYGU7oxJtRjgib7Fu-O--x5#scrollTo=7jVb9E4SDPNb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab Code" width="87" height="15"/></a>

## Evaluation

TookaBERT models are evaluated on a wide range of NLP downstream tasks, such as Sentiment Analysis (SA), Text Classification, Multiple-choice, Question Answering, and Named Entity Recognition (NER).
Here are some key performance results:

| Model name       | DeepSentiPers (f1/acc) | MultiCoNER-v2 (f1/acc) | PQuAD (best_exact/best_f1/HasAns_exact/HasAns_f1)   | FarsTail (f1/acc)  | ParsiNLU-Multiple-choice (f1/acc) | ParsiNLU-Reading-comprehension (exact/f1) | ParsiNLU-QQP (f1/acc) |
|------------------|------------------------|------------------------|-----------------------------------------------------|--------------------|-----------------------------------|-------------------------------------------|-----------------------|
| TookaBERT-large  | **85.66/85.78**        | **69.69/94.07**        | **75.56/88.06/70.24/87.83**                         | **89.71/89.72**    | **36.13/35.97**                   | **33.6/60.5**                             | **82.72/82.63**       |
| TookaBERT-base   | <u>83.93/83.93</u>     | <u>66.23/93.3</u>      | <u>73.18</u>/<u>85.71</u>/<u>68.29</u>/<u>85.94</u> | <u>83.26/83.41</u> | 33.6/<u>33.81</u>                 | 20.8/42.52                                | <u>81.33/81.29</u>    |
| Shiraz           | 81.17/81.08            | 59.1/92.83             | 65.96/81.25/59.63/81.31                             | 77.76/77.75        | <u>34.73/34.53</u>                | 17.6/39.61                                | 79.68/79.51           |
| ParsBERT         | 80.22/80.23            | 64.91/93.23            | 71.41/84.21/66.29/84.57                             | 80.89/80.94        | **35.34/35.25**                   | 20/39.58                                  | 80.15/80.07           |
| XLM-V-base       | <u>83.43/83.36</u>     | 58.83/92.23            | <u>73.26</u>/<u>85.69</u>/<u>68.21</u>/<u>85.56</u> | 81.1/81.2          | **35.28/35.25**                   | 8/26.66                                   | 80.1/79.96            |
| XLM-RoBERTa-base | <u>83.99/84.07</u>     | 60.38/92.49            | <u>73.72</u>/<u>86.24</u>/<u>68.16</u>/<u>85.8</u>  | 82.0/81.98         | 32.4/32.37                        | 20.0/40.43                                | 79.14/78.95           |
| FaBERT           | 82.68/82.65            | 63.89/93.01            | <u>72.57</u>/<u>85.39</u>/67.16/<u>85.31</u>        | <u>83.69/83.67</u> | 32.47/32.37                       | <u>27.2/48.42</u>                         | **82.34/82.29**       |
| mBERT            | 78.57/78.66            | 60.31/92.54            | 71.79/84.68/65.89/83.99                             | <u>82.69/82.82</u> | 33.41/33.09                       | <u>27.2</u>/42.18                         | 79.19/79.29           |
| AriaBERT         | 80.51/80.51            | 60.98/92.45            | 68.09/81.23/62.12/80.94                             | 74.47/74.43        | 30.75/30.94                       | 14.4/35.48                                | 79.09/78.84           |

\*Note because of the randomness in the fine-tuning process, results with less than 1% differences are considered together.

## Contact us

If you have any questions regarding this model, you can reach us via the [community](https://huggingface.co/PartAI/TookaBERT-Base/discussions) of the model in Hugging Face.