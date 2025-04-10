---
{}
---
language: en
license: cc-by-4.0
tags:
- text-classification
- natural-language-inference
- transformers
- bilstm
repo: https://github.com/minahil-hassan/nli_deep_learning

---

# Model Card for j45485mt-h36441qp-nli

<!-- Provide a quick summary of what the model is/does. -->

This model performs Natural Language Inference (NLI), determining whether a given hypothesis logically follows from a premise.


## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

This model combines a pretrained DeBERTa-v3-base transformer with a bidirectional LSTM classifier. It was trained to determine the logical relationship between a premise and a hypothesis (binary classification: entailment or not). The model leverages contextual token embeddings from DeBERTa and sequential reasoning capabilities of an LSTM to capture nuanced linguistic dependencies with a custom head using drop out and fully connected classifier layer.

- **Developed by:** Minahil Tariq and Qian Ning Phang
- **Language(s):** English
- **Model type:** Supervised
- **Model architecture:** DeBERTa-v3-base encoder followed by a bidirectional LSTM layer (hidden_dim=384), dropout (p=0.3892), and a linear classifier for binary prediction.
- **Finetuned from model [optional]:** microsoft/deberta-v3-base

### Model Resources

<!-- Provide links where applicable. -->

- **Repository:** https://huggingface.co/microsoft/deberta-v3-base
- **Paper or documentation:** https://arxiv.org/abs/2111.09543
- **Our Trained model:** [trained and fine tuned Model on Google Drive](https://drive.google.com/uc?id=10BkUfTR1drMR7fZ1bOmyTEEBUuLxIMqY)


## Training Details

### Training Data

<!-- This is a short stub of information on the training data that was used, and documentation related to data pre-processing or additional filtering (if applicable). -->

24,000+ premise–hypothesis pairs for training sourced from the NLI domain.

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Training Hyperparameters

<!-- This is a summary of the values of hyperparameters used in training the model. -->


      - learning_rate: 8.663394579529044e-06
      - weight_decay: 0.00043725068136265345
      - dropout: 0.38920213235632034
      - hidden_dim: 384
      - batch_size: 32
      - num_epochs: 6

#### Speeds, Sizes, Times

<!-- This section provides information about how roughly how long it takes to train the model and the size of the resulting model. -->


      - total training time: ~8 hours (20 Optuna trials)
      - model size: ~730MB
      - typical duration per epoch: ~20–30 minutes

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data & Metrics

#### Testing Data

<!-- This should describe any evaluation data used (e.g., the development/validation set provided). -->

A 6,000+ pair validation set held out from the same dataset as training.

#### Metrics

<!-- These are the evaluation metrics being used. -->


      - Accuracy
      - Precision
      - Recall
      - F1-score

### Results

The model obtained an F1-score of 92% and an accuracy of 92%.

## Technical Specifications

### Hardware


      - RAM: at least 16 GB
      - Storage: at least 2GB
      - GPU: T4 or V100

### Software


      - Transformers 4.18.0
      - Optuna
      - Pytorch 1.11.0

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

Inputs longer than 512 tokens are truncated by the tokenizer.
      Like most language models, this model may inherit biases from its pretrained components (DeBERTa).

## Additional Information

<!-- Any other information that would be useful for other people to know. -->

Hyperparameters were optimized using Optuna with MedianPruner.
      Final model weights and evaluation scripts are available on the linked repository.
