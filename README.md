# Steam Game Review Classification Using Machine Learning

Predicting whether a Steam user **recommends** a game (1) or **does not recommend** it (0) from the text of their review alone. The project compares five classical machine learning classifiers trained on TF-IDF features.

## Highlights

- 17,494 English Steam reviews across 44 games (2011–2018)
- TF-IDF features over a 54,261-term vocabulary
- Five classifiers compared on the same 80/20 split
- **Best model: Support Vector Machine, at 85.97% accuracy and 0.855 macro F1**
- Logistic Regression is nearly as accurate (85.57%) and trains about 850× faster

## Dataset

`train.csv` (the same file is also included as `archive (1).zip`) contains one review per row:

| Column | Description | Used as |
|---|---|---|
| `review_id` | Unique review identifier | Dropped |
| `title` | Game name (44 titles) | Not used for training |
| `year` | Year posted (2011–2018, 178 missing) | Not used for training |
| `user_review` | Review text | **Input** |
| `user_suggestion` | 1 = Recommended, 0 = Not Recommended | **Target** |

Class balance: 9,968 Recommended (57.0%) and 7,526 Not Recommended (43.0%). Always predicting the majority class would give about 57.9% accuracy on the test set.

## Method

```
train.csv ──► clean (drop missing reviews, drop review_id)
          ──► TF-IDF (lowercase, English stop words, min_df=1)
          ──► 80/20 train/test split (random_state=42)
          ──► train 5 classifiers (scikit-learn defaults)
          ──► evaluate: accuracy, precision/recall/F1, confusion matrix
```

Classifiers: Random Forest, Gradient Boosting, Multinomial Naive Bayes, Logistic Regression, and Support Vector Machine (RBF kernel).

## Results

Test set: 3,499 reviews (2,026 Recommended, 1,473 Not Recommended).

| Classifier | Accuracy | Macro F1 | Recall (Not Rec.) | Recall (Rec.) |
|---|---|---|---|---|
| Random Forest | 83.31% | 0.825 | 0.726 | 0.911 |
| Gradient Boosting | 76.76% | 0.750 | 0.599 | 0.890 |
| Multinomial Naive Bayes | 79.08% | 0.768 | 0.568 | 0.953 |
| Logistic Regression | 85.57% | 0.851 | 0.798 | 0.897 |
| **Support Vector Machine** | **85.97%** | **0.855** | **0.803** | **0.901** |

The following figures come from a reproduction run with the same pipeline:

| Classifier | Train acc. | Test acc. | ROC AUC | Train time |
|---|---|---|---|---|
| Random Forest | 99.98% | 83.17% | 0.909 | ~43 s |
| Gradient Boosting | 80.11% | 76.74% | 0.863 | ~46 s |
| Multinomial Naive Bayes | 86.03% | 79.08% | 0.904 | ~0.01 s |
| Logistic Regression | 91.14% | 85.54% | 0.932 | ~0.2 s |
| Support Vector Machine | 98.81% | 85.97% | 0.934 | ~171 s |

### Key findings

- **The linear and margin-based models win.** The SVM and Logistic Regression clearly outperform the tree ensembles on sparse TF-IDF features.
- **Logistic Regression is the most practical choice.** It comes within 0.4 percentage points of the SVM, trains almost instantly, and its coefficients are interpretable.
- **Random Forest over-fits.** It scores 99.98% on the training data but only 83.2% on the test set.
- **Gradient Boosting under-fits.** With 100 shallow trees, it reaches the lowest accuracy of the five models.
- **Naive Bayes is biased toward "Recommended".** It misses 43% of negative reviews at the default threshold, although its AUC (0.904) shows that it ranks reviews reasonably well.
- **The influential terms make intuitive sense.** Positive terms include *great, best, love, amazing, fun* and *free*. Negative terms include *worst, money, terrible, waste, pay, update* and *unplayable*, which reflects common complaints about monetisation and updates in free-to-play games.

## Repository structure

```
├── py_code.ipynb      # Colab notebook with outputs (metrics + plots)
├── py_code.py         # Same code as a Python script
├── train.csv          # Dataset
└── archive (1).zip    # Zipped copy of the dataset
```

## How to run

```bash
pip install pandas scikit-learn matplotlib seaborn
```

1. Open `py_code.ipynb` in Jupyter or Google Colab, or run `py_code.py`.
2. Update the dataset path in the script (it currently points to `/content/sample_data/train.csv`) so that it matches your setup, for example `train.csv`.
3. Run all cells. The script prints a classification report and a confusion matrix for each model, followed by the best classifier and the class-distribution plots.

Note: the SVM is the slowest step and takes a few minutes on a typical CPU.

## Known issues and possible improvements

- The "Training and Testing Accuracy" chart in the script plots **test** accuracy for both bars. To show real training accuracy, record `clf.score(X_train, y_train)` for each model.
- The confusion-matrix tick labels are placed at cell edges rather than centres. Passing `xticklabels`/`yticklabels` to `sns.heatmap` fixes this.
- The TF-IDF vectorizer is fitted on the full dataset before the split. Fitting it on the training data only, for example inside a `Pipeline`, avoids using test-set statistics.
- `RandomForestClassifier` and `GradientBoostingClassifier` have no `random_state`, so their results vary slightly between runs.
- Possible extensions include cross-validation, hyperparameter tuning, n-grams, class weighting or threshold tuning to improve recall on negative reviews, removing the "Early Access Review" prefix (present in 32.8% of reviews), and comparisons with transformer models such as BERT.

## Requirements

Python 3.8+, pandas, scikit-learn, matplotlib and seaborn.
