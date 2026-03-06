# Customer Complaint Classification using PyTorch

This project builds a **deep learning text classification model using PyTorch** to automatically categorize customer complaints into predefined categories.

The goal is to help organizations **automatically route customer support tickets** to the correct department using Natural Language Processing (NLP).

Example use cases include:

- Customer support automation
- Financial complaint classification
- AI-based document routing
- NLP-powered service analytics

---

# Model Overview

The model uses a **Convolutional Neural Network (CNN)** for text classification.

Pipeline:

Complaint Text  
↓  
Tokenization  
↓  
Word Index Encoding  
↓  
Embedding Layer  
↓  
1D Convolution Layer  
↓  
Global Average Pooling  
↓  
Fully Connected Layer  
↓  
Predicted Complaint Category

---

# Files Included

| File | Description |
|-----|-------------|
| train.py | Script used to train the neural network |
| model.py | Defines the CNN text classification model |
| evaluate.py | Evaluates the trained model using classification metrics |
| words.json | Vocabulary list used to encode text tokens |
| text.json | Tokenized customer complaint sentences |
| labels.npy | Numeric category labels for each complaint |

---

# Dataset

The dataset consists of **tokenized customer complaint records**.

Each complaint is represented as a list of tokens which are converted into numerical indices using the vocabulary.

Example complaint:

```
["i", "called", "because", "i", "have", "been", "receiving", "calls", "about", "a", "debt"]
```

Dataset components:

- Tokenized complaint text
- Vocabulary mapping
- Category labels

---

# Model Architecture

The neural network architecture consists of the following layers.

### Embedding Layer

Converts word indices into dense vector representations.

```
Embedding(vocab_size, 64)
```

---

### Convolution Layer

Extracts local patterns within sequences of words.

```
Conv1D(kernel_size=3)
```

---

### Pooling Layer

Global average pooling compresses the sequence into a fixed-length representation.

---

### Fully Connected Layer

Produces the final classification prediction.

```
Linear(embed_dim → num_classes)
```

---

# Training Configuration

| Parameter | Value |
|-----------|------|
| Optimizer | Adam |
| Learning Rate | 0.05 |
| Loss Function | CrossEntropyLoss |
| Epochs | 3 |
| Embedding Dimension | 64 |
| Sequence Length | 50 |

---

# How to Run

### Train the Model

```
python train.py
```

After training, the script will generate:

```
model.pth
```

---

### Evaluate the Model

```
python evaluate.py
```

Evaluation metrics include:

- Accuracy
- Precision
- Recall

---


# Python Libraries Used

- PyTorch
- NumPy
- Scikit-learn

---
