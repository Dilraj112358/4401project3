"""Build the read-only results notebook and analysis handoff from saved artifacts."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "analysis_results"


def table(frame):
    # Keep the analysis reproducible without the optional tabulate dependency.
    frame = frame.copy().fillna("")
    return "\n".join([
        "| " + " | ".join(map(str, frame.columns)) + " |",
        "| " + " | ".join("---" for _ in frame.columns) + " |",
        *["| " + " | ".join(map(str, row)) + " |" for row in frame.itertuples(index=False, name=None)],
    ])


def main():
    audit = json.loads((OUT / "audit_summary.json").read_text())
    lock = json.loads((OUT / "selection_lock.json").read_text())
    final = json.loads((OUT / "final_metrics.json").read_text())
    masking = json.loads((OUT / "masking_experiment.json").read_text())
    comparison = pd.read_csv(OUT / "audited_model_comparison.csv")
    regions = pd.read_csv(OUT / "regional_validation.csv")
    full = pd.read_csv(OUT / "original_full_size_comparison.csv")
    words = pd.read_csv(OUT / "top_language_features.csv")
    shared = pd.read_csv(OUT / "shared_language_training.csv")
    prevalence = pd.read_csv(OUT / "sensory_prevalence_training.csv")
    candidates = comparison.loc[comparison.stage.isin(["allregion_20000", "allregion_full"]) & comparison.split.eq("validation")]
    geo = regions.loc[regions.model.eq("Linear SVM")]
    training = comparison.loc[comparison.stage.eq(lock["selected_stage"]) & comparison.model.eq(lock["selected_model"]) & comparison.split.eq("training")].iloc[0]
    metrics = final["metrics"]
    ci = final["bootstrap_95pct_intervals"]
    red = words.loc[words.associated_color.eq("red"), "term"].head(10).tolist()
    white = words.loc[words.associated_color.eq("white"), "term"].head(10).tolist()
    sensory_shared = prevalence.loc[(prevalence.minimum_class_pct.ge(10)) & (prevalence.absolute_gap_pp.le(10))]
    readme = f"""# Team Bordeaux: completed analysis, steps 1–5

This handoff completes auditing, language interpretation, model selection, and a
single locked final evaluation. It is supporting analysis for the team's final
presentation/report, rather than the final course report or slide deck.

## Selected model and evaluation

The selected model is **{lock['selected_model']}**, trained on
**{lock['training_rows']:,}** audited all-region reviews (`{lock['selected_stage']}`).
It uses description-only unigram/bigram TF-IDF, `min_df=2`, 10,000 maximum features,
sublinear term frequency, and `C=1`. Features are fitted on training rows only.
Selection used the highest validation macro F1 among six all-region candidates
(three classifiers at 20,000 and full training size), with accuracy as tie-breaker.
The majority dummy is a reference, not a selection candidate.

| Population | Reviews | Accuracy | Macro F1 |
| --- | ---: | ---: | ---: |
| Selected model training | {int(training.rows):,} | {training.accuracy:.4%} | {training.macro_f1:.4f} |
| Development validation | {lock['validation_scores']['rows']:,} | {lock['validation_scores']['accuracy']:.4%} | {lock['validation_scores']['macro_f1']:.4f} |
| Recovered final holdout | {metrics['rows']:,} | {metrics['accuracy']:.4%} | {metrics['macro_f1']:.4f} |

Final accuracy's conditional 95% bootstrap interval is
**{ci['accuracy']['low']:.2%}–{ci['accuracy']['high']:.2%}**; macro F1's is
**{ci['macro_f1']['low']:.4f}–{ci['macro_f1']['high']:.4f}**.
Holdout class counts: {final['class_counts']}. Confidence intervals resample review
outcomes with the trained model fixed; they do not include uncertainty in label
rules, model selection, training, or correlation among related wines.

![Final confusion matrix](final_confusion_matrix.png)

The frozen pipeline is `selected_pipeline.joblib`; the exact selection, training
IDs, validation IDs, dependency versions, policy, and SHA-256 checks are in
`selection_lock.json`. The evaluation function writes a start marker and refuses
to evaluate again. Reading saved predictions or this notebook does not rerun it.

## 1. Historical exposure and holdout recovery

The audit inventories **{audit['history_versions_reviewed']} distinct historical
notebook code versions** from Git history. It reconstructs the original Models
1–2 samples and the executed EDA notebook's separate 80/10/10 split, in addition
to the current saved split. The EDA experiment used 20,243 current saved-test rows
for training and 5,683 for validation. The original Models 1–2 additionally
intersect current test rows; overlapping counts must not be added as if disjoint.

Of the saved test's 31,615 red/white rows, **5,679** have no recovered predictive
training/validation text exposure. Excluding wine families appearing in any
reconstructed exposed population leaves **2,517** candidates. The family proxy is
the tuple (winery, variety, province, region_1, designation), with missing fields
normalized to empty strings. It intentionally groups vintages and can also merge
different wines; it is not a definitive bottle identifier.

The frozen label rules exclude **{final['excluded_by_label_rules']:,}** of these
candidates. The remaining **{metrics['rows']:,}** were evaluated once, after model
selection. Every original row has flags in `exposure_manifest.csv`; final label
exclusions are in `final_holdout_label_flags.csv`.

**This is a recovered predictive holdout, not a pristine external test set.**
Whole-data EDA and the earlier descriptive word heatmap included aggregate
information about it. Uncommitted/deleted experiments or runs on other computers
cannot be reconstructed. Family and label exclusions also change the population.
Future independent reviews would provide stronger confirmation.

## 2. Label audit

The fixed mapping covers 45 common varieties or explicitly color-named blends.
The retained development subset contains 108,264 red/white rows before normalized
duplicate removal. Explicit rosé, sparkling, dessert/fortified, and orange styles
are excluded; original data and labels are preserved. No test-row label changes
were made. The same policy is applied to each partition independently.

Read [LABEL_AUDIT.md](LABEL_AUDIT.md) for confirmed conflicts, inspected examples,
known false positives, and remaining uncertainty. Counts by country and variety
are in `country_coverage.csv` and `variety_coverage.csv`. Coverage is broad but the
subset is not representative of all varieties or the world wine market.

## 3. Differences and similarities in language

The selected linear classifier's strongest associations include:

- Red: {', '.join(red)}.
- White: {', '.join(white)}.

Negative coefficients favor red and positive coefficients favor white; class
order was checked explicitly. Coefficients describe conditional predictive
associations, not causal sensory effects. Correlated words and varietal names can
share or substitute for one another. Magnitudes depend on the TF-IDF transform.

![Language coefficients](language_coefficients.png)

Training-document frequencies supply a separate descriptive check. Words are
counted once per review. Among the predefined sensory vocabulary, words present
in at least 10% of both classes and differing by at most 10 percentage points are:

{table(sensory_shared[['term','red_pct','white_pct']].round(2))}

Corpus-wide shared vocabulary also includes
{', '.join(shared.term.head(10))}. `shared_language_training.csv` requires at least
10% prevalence in each color and a prevalence ratio from 0.5 to 2; shared does
not imply identical usage. These summaries use selected-model training data,
not final holdout descriptions.

![Sensory vocabulary prevalence](sensory_language.png)

The masking sensitivity experiment retrained the same selected classifier after
removing mapped grape/color names, aliases, and compound-name components from
training and validation text. Validation accuracy changed from
**{masking['unmasked']['accuracy']:.4%}** to **{masking['masked']['accuracy']:.4%}**,
and macro F1 from **{masking['unmasked']['macro_f1']:.4f}** to
**{masking['masked']['macro_f1']:.4f}**. This measures robustness to explicit clues;
the broad mask can remove ordinary words and cannot isolate a causal contribution.
The mask was not used to select or modify the final model.

`validation_errors.csv` and `validation_error_examples.csv` preserve development
errors with original text, metadata, predictions, and signed decision margins.
High-confidence mistakes are useful for finding residual proxy-label problems;
low-confidence cases can contain overlapping sensory language. An error alone
does not establish which label is correct.

The [validation error review](VALIDATION_ERROR_REVIEW.md) documents apparent
source-label errors, a white Pinot Noir, a missed Vin Gris style, and genuine
sensory overlap. The rules do not perfectly establish finished bottle colors:
**all reported scores measure agreement with the retained proxy labels**. These
findings were documented without changing the frozen model or final holdout.

## 4. Model selection and geographic generalization

All audited all-region selection candidates use the same 27,121 validation rows:

{table(candidates[['stage','model','accuracy','balanced_accuracy','macro_f1']].round(4))}

The matched 20,000-review stages both use 13,000 red and 7,000 white training
reviews and the same validation rows. Their separate geographic scores are:

{table(geo[['stage','group','rows','accuracy','macro_f1']].round(4))}

"Outside California" includes records whose province is not recorded as
California; it is a complement of that metadata field, not an independent
geographic verification. Changes in geographic/varietal coverage can improve
transfer beyond California. This comparison alone does not isolate geography
from reviewer, varietal, or other composition changes and does not establish
causation.

The narrow 10/100/1,000 training samples are nested; the California 1,000/10,000/
20,000 stages are also nested. Within those sequences, shrinking training–
validation gaps are consistent with reduced variance. A positive gap alone is
not proof of problematic overfitting. Low training and validation performance
would instead suggest underfitting. Scope changes are reported separately from
training-size changes, and all stage metrics are in `audited_model_comparison.csv`.

![Narrow-stage learning curve](learning_curve.png)

The final section of the original full-size notebook was also reproduced on its
original red/white population, with the same TF-IDF and classifier settings.
Those results remain separate because they use different label eligibility:

{table(full.loc[full.split.eq('validation'),['model','rows','accuracy','balanced_accuracy','macro_f1']].round(4))}

Existing DistilBERT archives were checked against their row-level predictions,
source labels, and training/validation fingerprints. Scores are in
`verified_original_distilbert_scores.csv`. They use the original population and
are contextual comparisons, not contestants in the audited model selection.
The ZIPs do not include model weights, so a checkpoint cannot be recovered from
the predictions alone. The final selected model is a fully saved TF-IDF pipeline.

## 5. Final evaluation and reproducibility

`final_metrics.json`, `final_classification_report.csv`,
`final_confusion_matrix.csv`, `final_regional_metrics.csv`, and
`final_holdout_predictions.csv` contain the final evaluation outputs. There was
no refitting, tuning, or model switching after opening this holdout for scoring.

Use Python 3.12 with `requirements-analysis.txt`. The run used
`/opt/anaconda3/bin/python` because the default Homebrew Python lacks the required
packages. From the project directory, the first-run sequence is:

```sh
python -m unittest test_bordeaux_analysis.py
python bordeaux_analysis.py audit
python bordeaux_analysis.py develop
python bordeaux_analysis.py evaluate
python summarize_bordeaux_analysis.py
```

The saved run is intentionally locked. Do not delete its lock/start marker to
rerun or improve the reported score. To reproduce development experiments, copy
the source and original input Parquet files to a separate directory without the
saved output folder. Any repeated holdout scoring is reproduction, not a new
independent test. The results notebook reads saved artifacts only.

Original notebooks, original Parquet files, and historical summary CSV are
preserved. The pre-existing user edit in `modeling_2.ipynb` was not changed.
Final slides, the course report, and personal reflections remain separate work.
"""
    (OUT / "README.md").write_text(readme)
    import nbformat
    notebook = nbformat.v4.new_notebook()
    notebook.metadata = {"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}, "language_info":{"name":"python","version":"3.12"}}
    markdown = nbformat.v4.new_markdown_cell
    code = nbformat.v4.new_code_cell
    notebook.cells = [
        markdown("# Team Bordeaux: audited results\n\nThis notebook reads saved results. It never fits a model or opens test descriptions. See [the full analysis handoff](analysis_results/README.md) and [label audit](analysis_results/LABEL_AUDIT.md)."),
        code("from pathlib import Path\nimport json\nimport pandas as pd\nfrom IPython.display import display, Image, Markdown\nRESULTS = Path('analysis_results')\nlock = json.loads((RESULTS / 'selection_lock.json').read_text())\nfinal = json.loads((RESULTS / 'final_metrics.json').read_text())\ndisplay(Markdown(f\"**Selected:** {lock['selected_model']}, {lock['training_rows']:,} training reviews. **Recovered holdout:** {final['metrics']['rows']:,} reviews.\"))\ndisplay(pd.DataFrame([lock['validation_scores'], final['metrics']], index=['validation', 'recovered holdout']))"),
        markdown("## Population and evaluation limitations\n\nThe audit reconstructs prior predictive exposure, excludes exposed wine families, and applies frozen label rules. Whole-data aggregate EDA was previously seen, so the final population is a recovered predictive holdout rather than a pristine external test. Confidence intervals condition on the retained population and fixed trained model."),
        code("audit = json.loads((RESULTS / 'audit_summary.json').read_text())\ndisplay(pd.Series(audit['saved_test_exposure_counts_nonexclusive'], name='rows; flags overlap').to_frame())\ndisplay(pd.read_csv(RESULTS / 'label_coverage.csv'))"),
        markdown("## Model progression and geographic transfer\n\nCompare stages with matching validation populations. The geographic experiment fixes training size and class mix; it does not isolate geography from every other population difference."),
        code("comparison = pd.read_csv(RESULTS / 'audited_model_comparison.csv')\ndisplay(comparison.round(4))\ndisplay(Image(filename=str(RESULTS / 'learning_curve.png')))\nregions = pd.read_csv(RESULTS / 'regional_validation.csv')\ndisplay(regions[regions['model'].eq('Linear SVM')].round(4))\ndisplay(pd.read_csv(RESULTS / 'original_full_size_comparison.csv').round(4))"),
        markdown("## Distinctive and shared review language\n\nCoefficients are associations, not causal effects. Shared words occur frequently in both classes. All word-frequency summaries use training data."),
        code("display(Image(filename=str(RESULTS / 'language_coefficients.png')))\ndisplay(Image(filename=str(RESULTS / 'sensory_language.png')))\ndisplay(pd.read_csv(RESULTS / 'shared_language_training.csv').head(20).round(2))\nmasking = json.loads((RESULTS / 'masking_experiment.json').read_text())\ndisplay(pd.DataFrame([masking['unmasked'], masking['masked']], index=['unmasked', 'masked']).round(4))"),
        markdown("## Validation error inspection\n\nThese are development errors, not final test examples. A model error can reflect overlapping language or an imperfect proxy label; it is not sufficient evidence to change a label."),
        code("examples = pd.read_csv(RESULTS / 'validation_error_examples.csv')\nwith pd.option_context('display.max_colwidth', 160):\n    display(examples[['row_id','color','predicted_color','variety','description','decision_margin_for_white']])"),
        markdown("## Locked final evaluation\n\nThe selected saved pipeline was evaluated once. Reading this table does not call it again. The bootstrap intervals do not cover label uncertainty or all possible wine-family dependence."),
        code("display(pd.DataFrame(final['classification_report']).T.round(4))\ndisplay(pd.DataFrame(final['bootstrap_95pct_intervals']).T.round(4))\ndisplay(Image(filename=str(RESULTS / 'final_confusion_matrix.png')))\ndisplay(pd.read_csv(RESULTS / 'final_regional_metrics.csv').round(4))"),
    ]
    nbformat.write(notebook, ROOT / "bordeaux_results.ipynb")
    print("Wrote analysis_results/README.md and bordeaux_results.ipynb")


if __name__ == "__main__":
    main()
