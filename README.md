# Fine-Grained Multi-Label Emotion Detection

This project compares classical machine learning and transformer-based deep learning approaches for fine-grained multi-label emotion classification using the **GoEmotions** dataset.

The project evaluates TF-IDF-based Logistic Regression and Naive Bayes baselines, followed by fine-tuning and hyperparameter experiments with **DistilBERT**.

## Dataset

The project uses the **GoEmotions** dataset, which contains Reddit comments annotated with fine-grained emotion labels.

The dataset used in this project contains **28 emotion classes** and supports multi-label classification, meaning that a single comment can be associated with more than one emotion.

- Training samples: 43,410
- Validation samples: 5,426
- Test samples: 5,427
- Dataset source: https://github.com/google-research/google-research/tree/master/goemotions/data

The notebook downloads the required dataset files automatically from the official GoEmotions repository.

## Project Workflow

1. Dataset loading and exploratory data analysis
2. Text cleaning and multi-label encoding
3. TF-IDF feature extraction
4. Classical baseline models: Logistic Regression and Naive Bayes
5. Logistic Regression hyperparameter tuning
6. DistilBERT fine-tuning
7. DistilBERT hyperparameter experiments
8. Final model evaluation

## Models and Results

| Model | Micro-F1 | Macro-F1 |
|---|---:|---:|
| Logistic Regression | 0.489 | 0.439 |
| Tuned Logistic Regression | 0.500 | 0.448 |
| Naive Bayes | 0.140 | 0.041 |
| Final DistilBERT | **0.567** | **0.467** |

The final DistilBERT model achieves the strongest overall performance. Its final evaluation also gives:

- Precision: 0.650
- Recall: 0.502
- ROC-AUC: 0.922
- PR-AUC: 0.509

## Key Insights

- DistilBERT outperforms the classical baseline models in overall multi-label classification performance.
- Logistic Regression provides a substantially stronger classical baseline than Naive Bayes.
- Performance varies across emotion classes, reflecting the highly imbalanced label distribution.
- Some low-frequency emotion classes remain difficult to predict.

## Limitations

- The dataset is highly imbalanced across emotion classes.
- Some emotion classes contain very few examples, making reliable prediction difficult.

## Future Work

Future improvements could include tuning classification thresholds for individual emotion classes, extending the DistilBERT hyperparameter search, and exploring additional methods for handling class imbalance.

## Technologies

Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, PyTorch, Hugging Face Transformers, and DistilBERT.

## Repository Structure

```text
.
├── Fine_Grained_Multi_Label_Emotion_Detection.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

The dataset and trained model checkpoint are not stored in the repository. Dataset files are downloaded automatically when the notebook is run, while model checkpoints are created locally during training.
