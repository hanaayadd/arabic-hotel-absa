# Arabic Aspect-Based Sentiment Analysis of Hotel Reviews

**MAAI 902 – NLP & LLMs · Assessment 2 (NLP Solution Design) · Hana Ayad · University of Wollongong in Dubai**

A star rating doesn't tell a hotel *what* guests liked or disliked. This project takes an Arabic hotel review,
finds the aspects it talks about (rooms, service, location, food and so on) and predicts the sentiment towards each one.
It compares three kinds of NLP model on the same data, tests them on review sources they were not trained on, and turns
the results into a manager dashboard and a summary that cites real guest sentences. The work is linked to the tourism
priority of the UAE National Strategy for Artificial Intelligence 2031.

**Outputs (data, figures, results, trained models):** [Google Drive folder](https://drive.google.com/drive/folders/1GxARmgG_TP3I1EyNAk20TbezwYEvTsHn?usp=sharing)

## How it works

```
Arabic review → cleaning → aspect detection → sentiment per aspect → dashboard + cited LLM summary
                            (TF-IDF, multi-label)   1. TF-IDF + logistic regression / SVM
                                                     2. AraBERT v0.2 / CAMeLBERT-MSA, fine-tuned
                                                     3. Qwen2.5-7B-Instruct, zero- and few-shot prompting
```

Sentiment is treated as sentence-pair classification: `[CLS] sentence [SEP] aspect [SEP]` → positive / negative / neutral.

## Notebooks

Run them in order in Google Colab. Each notebook saves its outputs to `MyDrive/MAAI902_ABSA/`, so the later ones can pick them up.

| # | Notebook | What it does | Runtime |
|---|---|---|---|
| 01 | `01_data_and_preprocessing.ipynb` | Data exploration, two cleaning pipelines, validation split grouped by sentence | CPU, ~2 min |
| 02 | `02_classical_baseline.ipynb` | TF-IDF baseline, class-weight and preprocessing comparisons, top n-grams, aspect detection | CPU, ~15 min |
| 03 | `03_transformer_finetuning.ipynb` | AraBERT vs CAMeLBERT, with and without class weights, 3 seeds each; error analysis; Integrated Gradients | T4 GPU, ~70 min |
| 04 | `04_llm_crossdomain_demo.ipynb` | LLM prompting, cross-domain test on M-ABSA, manager summary, Gradio demo | T4 GPU, ~25 min |

To run: open a notebook in Colab, set **Runtime → Change runtime type → T4 GPU** for 03 and 04, then **Runtime → Run all**.
Colab already includes most packages, and the notebooks install the few that are missing; `requirements.txt` lists
everything needed to run the project outside Colab.

The interactive demo is the last cell of notebook 04: it launches a Gradio page where you paste an Arabic review and see
each aspect with its sentiment and confidence.

## Results (SemEval-2016 Arabic hotels test set, 2,604 opinions; BERT models are the mean of 3 seeds)

| Model | Accuracy | Macro-F1 | Neutral F1 |
|---|---|---|---|
| TF-IDF + logistic regression | 82.0 | 67.1 | 29.1 |
| AraBERT v0.2 | 88.4 | 73.6 | 38.4 |
| CAMeLBERT-MSA | 87.7 | 73.5 | 39.3 |
| AraBERT v0.2, class-weighted (used in the demo) | 83.0 | 71.6 | 38.0 |

The class-weighted AraBERT is used in the demo because it had the best macro-F1 on the validation set, which is where
models were chosen. It also raises neutral recall from 32% to 60%, at a cost of about two macro-F1 points on the test set.

On the same 300-row subset, the prompted LLM has the highest accuracy (87.3%) but the lowest macro-F1 (66.7% zero-shot,
68.5% few-shot) and a neutral F1 of only 15.8%. On unseen M-ABSA reviews the fine-tuned model's macro-F1 drops by about
10 points (hotel 58.1, restaurant 65.1, laptop 62.2).

## Repository contents

- `01`–`04` `.ipynb`: the four notebooks above, with their outputs
- `requirements.txt`: Python packages
- `README.md`: this file

## Data

- **SemEval-2016 Task 5, Arabic hotels** (Pontiki et al., 2016): Hugging Face `srinivasbilla/semeval-2016-absa-reviews-arabic`
- **M-ABSA** Arabic hotel, restaurant and laptop test sets (Wu et al., 2025): `Multilingual-NLP/M-ABSA`, used only for the cross-domain test

## Reproducibility

Fixed seeds (42, 13, 7), model selection on the validation set only, test set used once, LLM decoding at temperature 0.

## Ethics

The reviews are public and anonymous, and no personal data is stored. Results are reported per aspect, never per staff
member. Every point in the LLM summary must cite real review sentences, and this is checked automatically. Label noise,
dialect coverage, the weak neutral class and the drop on new domains are discussed in the report.
