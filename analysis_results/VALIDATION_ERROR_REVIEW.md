# Validation error review

The frozen SVM makes 255 errors against the retained validation proxy labels
(27,121 reviews). The error tables rank examples by signed decision margin;
negative favors red, positive favors white. Margins are not probabilities.

Inspection of the most confident errors reveals two distinct issues:

| Row | Proxy label → prediction | Review evidence | Interpretation |
| --- | --- | --- | --- |
| 20818 | red → white | `Red Blend` metadata, but prose describes a Viognier/Chardonnay/Sauvignon Blanc/Marsanne blend with pineapple, peach, apricot | Likely original metadata/label error; not simply a failed sensory classifier |
| 24016 | red → white | `Red Blend` metadata, but description says mainly Chardonnay with Sauvignon Blanc, honeysuckle and pear | Another apparent source-label inconsistency |
| 79844 | red → white | Pinot Noir metadata, but review explicitly calls it a "truly white Pinot Noir" | Varietal color is an imperfect proxy for finished wine color |
| 116474 | red → white | Pinot Noir with `Vin Gris` designation; review describes pale pink wine and apple/citrus | A style that the fixed exclusion patterns did not catch |
| 161341 | white → red | Pink Pinot Grigio; raspberry/strawberry jam language | Mixed color/style cues and a residual pink style |
| 96252 | white → red | Sauvignon Blanc with unripe berry and herbaceous language | Sensory overlap can favor the other class even when the proxy label is plausible |
| 41826 | white → red | Chardonnay described through oak, spice, alcohol, acids and tannins | White wine review uses vocabulary associated with red wines |

These examples were inspected as interpretation, without changing the frozen
model or selectively repairing labels. No test errors were used for rule or model
changes. They demonstrate that final accuracy measures agreement with the
audited **proxy labels**, not independently verified bottle colors. A future
version should verify these styles/labels before collecting a fresh independent
holdout. Retrospectively changing exclusions or labels to improve this run's
reported final score would undermine its evaluation.

The complete development errors and original descriptions remain in
`validation_errors.csv`; `validation_error_examples.csv` holds ten high-margin
examples per actual class. Their occurrence is evidence of residual limitations,
not a measured estimate of the overall label error rate.
