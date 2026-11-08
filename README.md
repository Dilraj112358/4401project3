# DATA 4401 Project 3 — Team Bordeaux

Predict red versus white wine from review text and explain the differences and similarities in review language.

## Run order

1. `minh_eda.ipynb`: prepare the combined, deduplicated data and save `df_combined.parquet`.
2. `adding_colors.ipynb`: assign categories and save `df_with_color.parquet`.
3. `data_split.ipynb`: save the stratified train, validation, and test partitions.
4. `modeling_1.ipynb`, `modeling_2.ipynb`, `modeling_4.ipynb`: compare Naive Bayes, logistic regression, linear SVM, and DistilBERT on nested 10-, 100-, and 1,000-review California Cabernet Sauvignon/Chardonnay training samples. Each uses all eligible reviews in the saved validation partition.
5. `modeling_5.ipynb`: keep 1,000 balanced training reviews and expand to all labeled California red/white wines, validating on every eligible California red/white review in the saved validation partition.
6. `modeling_6.ipynb`: grow the same California training set to 10,000 balanced reviews, retaining Model 5's original 1,000 reviews and the same validation set. Compare the three TF-IDF classifiers locally; train DistilBERT in Colab, download `modeling_6_distilbert_results.zip`, and place it beside the notebook to add its verified results to the same comparison. No dummy baseline is included.
7. `modeling_7_1.ipynb` and `modeling_7_2.ipynb`: compare 20,000 California-only versus 20,000 all-region red/white training reviews. Both use 13,000 red and 7,000 white training reviews, all 31,614 red/white reviews in the saved validation partition, three local TF-IDF models, and a Colab DistilBERT results ZIP.
8. `full_size_models.ipynb`: compare the three TF-IDF classifiers on all red/white training and validation reviews. Its earlier exploratory models with review-length features are retained.

The revised small-sample modeling notebooks run independently from `train.parquet` and `validation.parquet`; they never load `test.parquet`. DistilBERT is optional to run locally and is best run with a Colab GPU. `modeling_3.ipynb` is a separate historical five-category DistilBERT experiment; the current sequence focuses on red versus white.

Keep the Parquet files in the notebook directory. The TF-IDF sections require pandas, numpy, pyarrow, and scikit-learn. The DistilBERT sections additionally require PyTorch, Transformers, Accelerate, and a compatible Hugging Face Datasets installation because Transformers may import it internally.

## Evaluation decisions

- Begin narrowly with California Cabernet Sauvignon (red) and Chardonnay (white). Label auditing is still needed before broader modeling; varietal-derived labels are assumptions about finished wine style.
- Use only `description` as model input. Metadata selects the subset and supplies labels.
- Normalize and deduplicate review text before sampling; exclude conflicting color labels.
- Use every eligible California Cabernet Sauvignon/Chardonnay review in the saved validation partition for Models 1, 2, and 4. Keep training samples nested and disjoint from validation. The full-size notebook uses every red/white review in the saved validation partition.
- Fit TF-IDF only on training reviews through a pipeline. Use accuracy, macro F1, per-class metrics, and confusion matrices.
- Validation supports development, not final evaluation. The original Models 1–2 sampled before the saved split and exposed some saved-test rows. Exclude all previously exposed rows from the final holdout before evaluating the chosen model once. Distinct review text does not guarantee independence between related wines.

## Model decision log

The entries below document the original pre-split experiments and are retained as historical context. Their 78% and 93% validation results are not directly comparable with the revised notebooks, which use the saved split and full in-scope validation data.

### Dummy baseline

- **Purpose:** establish a reference that ignores review text.
- **Method:** `DummyClassifier(strategy="most_frequent")`, fitted on the original ten training observations.
- **Observed accuracy:** 50% training and 50% validation on the balanced samples.
- **Interpretation:** always predicting one color performs at 50% here. This is a deterministic constant prediction, not random guessing.
- **Decision:** train a classifier using review text to assess whether language improves prediction.

### Model 1 — 10 training reviews

- **Training sample:** five red and five white reviews; random seed 42.
- **Features:** TF-IDF unigrams, at most 100 features, `min_df=1`.
- **Classifier:** logistic regression, L2 regularization, `C=1.0`, `max_iter=1000`.
- **Observed accuracy:** 100% training, 78% validation; a 22-percentage-point gap. Results were obtained in the initial execution and subsequently reported by the team.
- **Interpretation:** the gap suggests overfitting with a very small training sample. Validation performance exceeds the 50% dummy baseline but is limited to two varietals in California.
- **Decision:** increase training data while holding the scope, model settings, and validation reviews fixed. More data may improve generalization; improvement is not guaranteed.

### Model 2 — 100 training reviews

- **Historical result:** the original notebook run reported 98% training accuracy and 93% validation accuracy. The current `modeling_2.ipynb` uses the saved split and therefore produces different, directly comparable results.
- **Training sample:** 50 red and 50 white reviews. Retain the original ten and add 45 unused reviews per color with seed 42.
- **Change from Model 1:** training size increases from 10 to 100.
- **Controls:** same subset, validation observations, TF-IDF settings, and logistic regression settings. TF-IDF vocabulary and weights are relearned using the enlarged training set.
- **Reason:** test whether adding data improves validation performance and reduces the apparent overfitting.

| Measure | Model 1 | Model 2 |
|---|---:|---:|
| Training observations | 10 | 100 |
| Validation observations | 100 | 100 |
| Training accuracy | 100% | 98% |
| Validation accuracy | 78% | 93% |
| Training minus validation accuracy | 22 percentage points | 5 percentage points |
| Validation macro F1 | 0.777 | 0.930 |

**Historical outcome:** the original 100-review model improved validation accuracy by 15 percentage points and reduced the gap. The revised 10-, 100-, and 1,000-review notebooks now provide the comparable learning curve on the saved split.

## Log future experiments

For each model, record the training subset and size, what changed, why it changed, training and validation metrics, interpretation, and the next decision. Preserve earlier results rather than overwriting them. Broader geographic or varietal scope is a separate experiment from simply adding observations.


