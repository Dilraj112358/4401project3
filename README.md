# DATA 4401 Project 3 — Team Bordeaux

Predict red versus white wine from review text and explain the differences and similarities in review language.

## Run order

1. `minh_eda.ipynb`: prepare the combined, deduplicated data and save `df_combined.parquet`.
2. `adding_colors.ipynb`: assign categories and save `df_with_color.parquet`.
3. `modeling_1.ipynb`: dummy baseline and Model 1, trained on ten reviews.
4. `modeling_2.ipynb`: Model 2, trained on 100 reviews, with a reproduced Model 1 comparison.

Each modeling notebook runs independently from top to bottom using `df_with_color.parquet`. New experiments belong in separate notebooks named `modeling_3.ipynb`, `modeling_4.ipynb`, and so on.

Keep the Parquet files in the notebook directory. Required packages: pandas, numpy, pyarrow, and scikit-learn.

## Evaluation decisions

- Begin narrowly with California Cabernet Sauvignon (red) and Chardonnay (white). Label auditing is still needed before broader modeling; varietal-derived labels are assumptions about finished wine style.
- Use only `description` as model input. Metadata selects the subset and supplies labels.
- Normalize and deduplicate review text before sampling; exclude conflicting color labels.
- Use a fixed, balanced validation set of 100 reviews: 50 per color. Exclude these from every training sample in this comparison.
- Fit TF-IDF only on training reviews through a pipeline. Use accuracy, macro F1, per-class metrics, and confusion matrices.
- This validation set supports development, not final evaluation. Reserve a separate untouched final test set before broader model selection. Distinct review text does not guarantee independence between related wines.

## Model decision log

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

- **Status:** implemented; run `modeling_2.ipynb` and record the results below.
- **Training sample:** 50 red and 50 white reviews. Retain the original ten and add 45 unused reviews per color with seed 42.
- **Change from Model 1:** training size increases from 10 to 100.
- **Controls:** same subset, validation observations, TF-IDF settings, and logistic regression settings. TF-IDF vocabulary and weights are relearned using the enlarged training set.
- **Reason:** test whether adding data improves validation performance and reduces the apparent overfitting.

| Measure | Model 1 | Model 2 |
|---|---:|---:|
| Training observations | 10 | 100 |
| Validation observations | 100 | 100 |
| Training accuracy | 100% | Pending team run |
| Validation accuracy | 78% | Pending team run |
| Training minus validation accuracy | 22 percentage points | Pending team run |
| Validation macro F1 | Record from notebook | Pending team run |

**Outcome and next decision:** pending results. Check validation accuracy and macro F1 alongside the training–validation gap. A smaller gap alone is not evidence of improvement if validation performance also falls. If overfitting persists, consider more data or stronger regularization; if both scores are low, investigate features or model flexibility. Change one factor at a time and document the reason.

## Log future experiments

For each model, record the training subset and size, what changed, why it changed, training and validation metrics, interpretation, and the next decision. Preserve earlier results rather than overwriting them. Broader geographic or varietal scope is a separate experiment from simply adding observations.


