# Human Language Technology

This repository contains the coursework completed for the **Introduction to Human Language Technologies** subject. The project studies semantic textual similarity using interpretable linguistic features and traditional machine-learning regression models.

## Project

**Semantic Textual Similarity:** Predicts how semantically similar two sentences are by combining match-based, lexical, and syntactic features. The methodology includes text preprocessing, feature extraction, baseline evaluation, backward feature elimination, and cross-dataset testing with Ridge Regression, Random Forest Regression, and Support Vector Regression. See the [`project presentation`](IHLT%20Presentation.pdf).

## Methodology

The engineered features include word and lemma n-gram overlap, number and date similarity, WordNet-based semantic similarity, part-of-speech and dependency-label distances, sequence matching, named-entity overlap, and sentence-length similarity. Feature selection removes redundant or weak predictors before the final models are evaluated on the MSRpar, MSRvid, SMTeuroparl, OnWN, and SMTnews datasets.

## Repository Structure

```text
.
└── IHLT Presentation.pdf       # Methodology, experiments, results, and conclusions
```

The results show that carefully designed linguistic features can capture meaningful semantic structure while remaining interpretable, although they are less effective at modeling context than embedding-based approaches.
