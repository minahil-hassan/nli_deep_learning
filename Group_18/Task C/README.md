# DeBERTa + BiLSTM for Natural Language Inference (NLI)

This repository contains code and resources for training and evaluating a hybrid **DeBERTa + BiLSTM** model on a **Natural Language Inference (NLI)** task.

Given a **premise** and a **hypothesis**, the model determines whether the hypothesis is **entailed** by the premise (binary classification).

---

## 📁 Project Structure

The project is organized into three main Jupyter notebooks:

| Notebook | Purpose |
|----------|---------|
| `1 hyperparameter_tuning.ipynb` | Runs **Optuna** to find optimal hyperparameters for the model. |
| `2 model_training.ipynb`        | Trains the **DeBERTa + BiLSTM** model using the selected hyperparameters and saves the trained weights to disk. |
| `3 demo_and_evaluation.ipynb`  | Loads the trained model for inference on the test set and outputs predictions and evaluation metrics. |

> **Note:** You must run notebook `2 model_training.ipynb` before running `3_demo_and_evaluation.ipynb` to generate the required model file. Otherwise, download model from: 
[Pretrained Model on Google Drive](https://drive.google.com/uc?id=10BkUfTR1drMR7fZ1bOmyTEEBUuLxIMqY)


---

## 📊 Datasets

The model is trained and evaluated using three CSV files:

- `train.csv`: Contains ~24,000 premise–hypothesis pairs for training.
- `dev.csv`: Contains ~6,000 validation examples for tuning and evaluation.
- `test.csv`: Contains the final test set used for generating predictions.

Each file includes:
- `premise`: The original statement.
- `hypothesis`: The statement to evaluate.
- `label`: 0 or 1 (except for the test set, which is unlabeled).

---

## 🧠 Model Overview

This model architecture is of a transformer-based model with a layer of sequential model, combining the strengths of both.

### 🔧 Architecture

- **Transformer**: `microsoft/deberta-v3-base`
- **Sequence model**: Bidirectional LSTM
  - `hidden_dim = 384`
  - `num_layers = 2`
- **Dropout**: 0.3892
- **Classifier**: `Linear(768, 2)`

### 🧪 Final Hyperparameters (from Optuna)

```yaml
learning_rate: 8.663e-06
weight_decay: 0.000437
dropout: 0.3892
hidden_dim: 384
batch_size: 32
num_epochs: 6
```

---

## 🖥️ Requirements

- Python 3.8+
- PyTorch 1.11.0+
- Transformers 4.18.0+
- Optuna
- scikit-learn
- pandas, numpy, matplotlib

---

## 📈 Results

The final model achieved:

- **Accuracy**: 92%
- **F1-score**: 92%
- **Evaluation set**: 6,000 samples from `dev.csv`

---

## 📝 Model Card

See [`my_model_card.md`](./my_model_card.md) for a complete model description including architecture, training details, limitations, and metrics.

---

## ⚠️ Notes

- Inputs longer than 512 tokens will be truncated by the tokenizer.
- The model may inherit biases from its DeBERTa-v3 pretrained components.

---

## 📂 How to Use

1. Run `deberta_hyperparameter_tuning.ipynb` (optional).
2. Run `deberta_model_training.ipynb` to train and save the model file.
3. Run `deberta_demo.ipynb` to generate test predictions.

Note: The following code block in `deberta_demo.ipynb` downloads the model stored in GDRive and stored it in the runtime env to load the model for testing:
```yaml
# 1) Install gdown if necessary 
!pip install gdown 
# 2) Download the trained model file from Google Drive 
!gdown "https://drive.google.com/uc?id=1RuBncwHrCyTw1ct686zWZWc3e7w6NYX7" -O best_deberta_bilstm_model.pt 
```

---

## 👩‍💻 Authors

Minahil Tariq and Qian Ning Phang  
Track: Natural Language Inference (NLI)

---

## 📜 License

This project is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
