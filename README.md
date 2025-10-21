# MM-811-Programming-Assignment
MM 811 (AI in Multimedia) Programming Assignment

This assignment focuses on creating autoregression models that memorize training samples, to understand that learning the global distribution leads to memorization.


## Project Directory Structure

```
Project/
│
├── Q1/
│   ├── model.py
│   ├── train_eval.py
│   └── main.py
│
├── Q2/
│   ├── autoencoder_model.py
│   ├── train_autoencoder.py
│   └── main.py
│
├── Q3/
│   ├── Sequence_to_Token/
│   │   ├── autoencoder_best.pth
│   │   └── q3_seq2token.py
│   │
│   └── Sequence_to_Sequence/
│       ├── autoencoder_best.pth
│       └── q3_seq2seq.py
│
└── Q4/
    └── q4.py
```

---

## Folder Overview

### **Q1/**
Contains the implementation for **Question 1**, including:
- `model.py` → Defines the neural network model.  
- `train_eval.py` → Handles model training and evaluation.  
- `main.py` → Main script to run the Question-1 task.  

---

### **Q2/**
Includes the **autoencoder-based solution** for Question 2:
- `autoencoder_model.py` → Defines the autoencoder architecture.  
- `train_autoencoder.py` → Script for training and validation.  
- `main.py` → Main script to run the Question-2 task.  

---

### **Q3/**
This includes two folders:
- **Sequence_to_Token/**
  - `autoencoder_best.pth` → Pretrained autoencoder weights.  
  - `q3_seq2token.py` → Implementation for the Sequence-to-Token model.  
- **Sequence_to_Sequence/**
  - `autoencoder_best.pth` → Pretrained autoencoder weights.  
  - `q3_seq2seq.py` → Implementation for the Sequence-to-Sequence model.  

---

### **Q4/**
Contains the implementation for **Question 4**:
- `q4.py` → Python file for the Question-4 task.  
