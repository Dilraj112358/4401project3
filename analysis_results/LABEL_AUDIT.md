# Label audit and population definition

This is an exclusion audit, not an externally verified relabeling of every bottle.
The original `color` values and all original Parquet files are preserved.

The new analysis retains 45 common varietals or explicitly red/white blend
designations from the team's original mapping. See `confident_variety_map.csv`.
It excludes unmapped/less familiar varieties, conflicting normalized descriptions,
missing text, and indicators of rosé, sparkling, dessert, fortified, or explicitly
orange wine. Sweet fruit flavors and unaccented floral "rose" in description text
do not by themselves trigger exclusion. The rules were set using development
data before final evaluation. They are frozen in `selection_lock.json` and applied
unchanged to the recovered holdout.

The procedure retains 108,264 development rows before normalized duplicate and
cross-partition removal: 81,143 training and 27,121 validation rows. This is about
85.6% of the original 126,455 red/white development rows. Country and varietal
coverage are recorded separately; this is a large multi-country subset, not a
probability sample of worldwide wine. Review selection and the reduced variety
list affect representativeness.

## Inspection findings

The deterministic samples in `development_style_sample.csv` and
`retained_label_sample.csv` make the audit reviewable. Inspection found clear
counterexamples to assigning color solely from the grape:

| Source row | Existing label | Evidence in development data | Disposition |
|---|---|---|---|
| 49376 | red, Cabernet Sauvignon | "This dry, crisp rosé is 100% Cabernet Sauvignon." | Exclude |
| 113383 | red, Pinot Noir | Glenora 2011 Rosé Pinot Noir | Exclude |
| 133978 | red, Zinfandel | Montevina 1999 White Zinfandel | Exclude |
| 91852 | red, Sangiovese | Description explicitly says sparkling wine, made using méthode Champenoise | Exclude |
| 125986 | white, Chardonnay | Pommery Summertime Blanc de Blancs Brut, Champagne | Exclude |
| 130248 | white, Bordeaux-style White Blend | Title identifies Sauternes | Exclude |
| 168921 | white, Riesling | Title identifies Icewine | Exclude |
| 63555 | white, Gewürztraminer | Description explicitly calls it an “orange” wine with six weeks on skins | Exclude |

The rules are conservative and are **not** a perfect style classifier. Inspection
also found valid red/white wines they exclude:

| Source row | Why it is flagged | Limitation |
|---|---|---|
| 156651 | `Methode a l'Ancienne` designation | The broad metadata rule also catches a traditional Pinot Noir name |
| 122287 | Thorny Rose producer name | Plain `Rose` in metadata can be a brand, not a color |
| 168760 | "Almost as pale in color as a rosé" | A comparison to rosé does not establish that the bottle is rosé |
| 118446 | "almost like a dessert wine" | A stylistic comparison does not establish dessert classification |

These observations are documented rather than presented as corrected ground
truth. Broad exclusions deliberately trade coverage for a more conservative
binary task. Plain floral descriptions, such as row 124903's "rose petal" Pinot
Noir, remain eligible. The code also preserves idiomatic "at first blush" and
producer names such as "Château du Port".

Remaining uncertainty includes unmentioned sparkling/dessert styles, incomplete
metadata, incorrect original varietal labels, sweet table wines, and mixed-color
blends. No manual test-row corrections or performance-driven test exclusions
were made. Some exclusions use review text, which changes the target population;
results should always be described as applying to this retained subset.
