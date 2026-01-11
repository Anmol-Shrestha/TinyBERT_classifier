# TinyBERT Sentiment Analysis (Binary Classification)

This project demonstrates **fine-tuning TinyBERT for binary sentiment classification (Positive / Negative)** using the Hugging Face ecosystem. The goal is to build a **lightweight, efficient NLP model** suitable for real-world and production-oriented use cases.

---

## 📌 Project Overview

- **Task:** Binary sentiment classification  
- **Model:** TinyBERT  
- **Framework:** PyTorch + Hugging Face Transformers  
- **Evaluation Metric:** Accuracy  

---

## 🧰 Technologies Used

### Data Processing
- `pandas`
- `datasets`
- `torch`
- `transformers`

### Evaluation
- `numpy`
- `evaluate` (Hugging Face)


### Model Building
- `transformers`


> The `evaluate` library is used instead of `scikit-learn` for simplicity and seamless integration with the Trainer API.

---

## 🧠 Model Architecture

### Tokenizer
- **`AutoTokenizer`**
- Tokenizes the `review` text column

### Model
- **`AutoModelForSequenceClassification`**
- Pretrained TinyBERT backbone
- Binary classification head
- Configured with:
  - `num_labels = 2`
  - `label2id`
  - `id2label`

---

## ⚙️ Training Configuration

### TrainingArguments

| Parameter | Value |
|--------|------|
| Epochs | 3 |
| Learning Rate | 2e-5 |
| Train Batch Size | 32 |
| Eval Batch Size | 32 |
| Evaluation Strategy | epoch |
| Logging | Disabled (`report_to="none"`) |

---

## 🏋️ Training Setup

The model is trained using Hugging Face’s `Trainer` API:

```python
Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics,
    tokenizer=tokenizer
)
```

---

## ✂️ Data Tokenization

The `review` column is tokenized with the following configuration:

- `padding=True`  
- `truncation=True`  
- `max_length=300`  

### Tokenized Features

Each review is converted into:
- `input_ids`
- `token_type_ids`
- `attention_mask`

### Dataset Columns (After Tokenization)

```text
review | sentiment | label | input_ids | token_type_ids | attention_mask
```

---

## 📊 Evaluation

Evaluation is performed using Hugging Face’s `evaluate` library:

```python
import evaluate
accuracy = evaluate.load("accuracy")
```

### Results

| Metric | Value |
|------|------|
| Loss | 0.30 |
| Accuracy | 0.88 |
| Epochs | 3 |

---

## 🚀 Inference

```python
from transformers import pipeline
import torch

device = 0 if torch.cuda.is_available() else -1

classifier = pipeline(
    "text-classification",
    model="tinybert-sentiment-analysis",
    device=device
)
```

---

## 🤏 Why TinyBERT?

### Advantages
- Fast training and inference
- Lower memory footprint
- Easy to deploy in production

### Trade-offs
- Slightly lower accuracy compared to larger transformer models

---

## 🔍 Why Transformers?

- Handle unstructured and noisy text effectively
- Minimal manual feature engineering required
- Capture contextual semantics better than traditional NLP models

---

## ✅ Conclusion

This project showcases how **compact transformer models like TinyBERT** can deliver strong performance while remaining efficient and production-friendly.

---


