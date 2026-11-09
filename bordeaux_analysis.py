"""Reproducible Team Bordeaux audit, development, and locked final evaluation.

Run from this directory: python bordeaux_analysis.py audit
                        python bordeaux_analysis.py develop
                        python bordeaux_analysis.py evaluate
Development never loads eligible holdout descriptions. Evaluation refuses to run
twice and checks that the frozen pipeline, inputs, and analysis code are unchanged.
"""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import zipfile

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/bordeaux-matplotlib")
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "analysis_results"
SEED = 42
LABELS = ["red", "white"]

# Restricted to common, explicitly color-designated blends and familiar varietals
# in the team's existing taxonomy. No review-text relabeling is performed.
RED = {
    "Cabernet Sauvignon", "Cabernet Franc", "Pinot Noir", "Merlot", "Syrah",
    "Shiraz", "Zinfandel", "Malbec", "Sangiovese", "Sangiovese Grosso",
    "Nebbiolo", "Tempranillo", "Gamay", "Petite Sirah", "Barbera", "Carmenère",
    "Grenache", "Garnacha", "Nero d'Avola", "Aglianico", "Montepulciano",
    "Red Blend", "Bordeaux-style Red Blend", "Rhône-style Red Blend", "Portuguese Red",
}
WHITE = {
    "Chardonnay", "Sauvignon Blanc", "Riesling", "Pinot Gris", "Pinot Grigio",
    "Grüner Veltliner", "Viognier", "Gewürztraminer", "Chenin Blanc", "Albariño",
    "Pinot Blanc", "Verdejo", "Garganega", "Vermentino", "Torrontés", "Sémillon",
    "White Blend", "Bordeaux-style White Blend", "Rhône-style White Blend", "Portuguese White",
}
COLOR_MAP = {**dict.fromkeys(RED, "red"), **dict.fromkeys(WHITE, "white")}

# Rules intentionally err toward exclusion, not automatic correction. Plain
# "rose" in review prose is NOT sufficient: many valid red reviews describe flowers.
STYLE_METADATA = r"\b(?:ros[eé]|rosado|rosato|blush|white\s+zinfandel|blanc\s+de\s+noirs?|sparkling|brut|extra\s+dry|cava|cr[eé]mant|champagne|prosecco|p[eé]tillant|frizzante|spumante|sekt|m[eé]thode|ice\s*wine|eiswein|late\s+harvest|botryti[sz]ed|beerenauslese|trockenbeerenauslese|dessert|fortified|port[ -](?:style|wine)|sherry|sauternes|vin\s+santo|passito|orange\s+wine)\b"
STYLE_DESCRIPTION = r"\b(?:rosé|rosado|rosato|white\s+zinfandel|blanc\s+de\s+noirs?|sparkling|sparkler|bubbly|brut|cr[eé]mant|prosecco|frizzante|spumante|ice\s*wine|eiswein|late[ -]harvest|botryti[sz]ed|beerenauslese|trockenbeerenauslese|dessert\s+wine|fortified|sweet\s+wine|(?:orange|amber)[\u201d\"']?\s+wines?)\b"
SENSORY = ["tannins", "cherry", "blackberry", "plum", "apple", "pear", "citrus", "lemon", "peach", "acidity", "fruit", "finish", "aromas", "oak", "spice", "fresh", "balanced", "texture"]


def save_json(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, default=str) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def norm(s):
    return s.fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip().str.casefold()


def load(name):
    return pd.read_parquet(ROOT / name)


def flags(frame):
    """Apply fixed rules separately in each partition; never learn from test text."""
    meta = frame[["title", "designation"]].fillna("").agg(" ".join, axis=1)
    out = pd.DataFrame(index=frame.index)
    out["outside_confident_varieties"] = ~frame.variety.isin(COLOR_MAP)
    out["mapping_disagrees"] = frame.variety.map(COLOR_MAP).notna() & frame.color.ne(frame.variety.map(COLOR_MAP))
    out["style_in_metadata"] = meta.str.contains(STYLE_METADATA, case=False, regex=True)
    out["style_in_description"] = frame.description.fillna("").str.contains(STYLE_DESCRIPTION, case=False, regex=True)
    out["empty_description"] = norm(frame.description).eq("")
    out["not_red_white"] = ~frame.color.isin(LABELS)
    return out


def group_keys(frame):
    # A conservative wine-family proxy, including missing designation. This can
    # merge distinct bottlings, but reduces leakage across vintages of a producer.
    cols = ["winery", "variety", "province", "region_1", "designation"]
    values = pd.DataFrame({c: norm(frame[c]) for c in cols}, index=frame.index)
    return values.apply(lambda row: json.dumps(row.tolist(), ensure_ascii=False), axis=1)


def pool(frame):
    f = flags(frame)
    x = frame.loc[~f.any(axis=1)].copy()
    x["description_key"] = norm(x.description)
    conflicts = x.groupby("description_key").color.nunique()
    x = x.loc[~x.description_key.isin(conflicts[conflicts.gt(1)].index)]
    return x.drop_duplicates("description_key")


def assert_separate(train, val):
    assert train.index.is_unique and val.index.is_unique
    assert set(train.index).isdisjoint(val.index)
    assert set(norm(train.description)).isdisjoint(norm(val.description))


def scores(actual, predicted):
    return {"rows": len(actual), "accuracy": accuracy_score(actual, predicted),
            "balanced_accuracy": balanced_accuracy_score(actual, predicted),
            "macro_f1": f1_score(actual, predicted, labels=LABELS, average="macro", zero_division=0)}


def historical_inventory():
    """Inventory every distinct historical notebook code version for review."""
    versions, seen = [], set()
    commits = subprocess.check_output(["git", "rev-list", "--all"], cwd=ROOT, text=True).split()
    for rev in reversed(commits):
        names = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", rev], cwd=ROOT, text=True).splitlines()
        for name in names:
            if not name.endswith(".ipynb"):
                continue
            raw = subprocess.check_output(["git", "show", f"{rev}:{name}"], cwd=ROOT)
            notebook = json.loads(raw)
            code = "\n".join("".join(c.get("source", [])) for c in notebook["cells"] if c["cell_type"] == "code")
            digest = hashlib.sha256(code.encode()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            versions.append({"first_commit": rev, "notebook": name, "code_sha256": digest,
                             "uses_split": "train_test_split(" in code,
                             "uses_full_colored_data": "df_with_color.parquet" in code,
                             "has_saved_execution": any(c.get("execution_count") is not None for c in notebook["cells"]),
                             "code": code})
    save_json("historical_notebook_inventory.json", versions)
    return versions


def audit():
    OUT.mkdir(exist_ok=True)
    if (OUT / "selection_lock.json").exists():
        raise RuntimeError("Development is already locked; preserve the completed audit.")
    w, train, val, test = [load(n) for n in ["df_with_color.parquet", "train.parquet", "validation.parquet", "test.parquet"]]
    assert w.index.is_unique
    for x in [train, val, test]:
        assert x.index.is_unique
        assert norm(x.description).equals(norm(w.loc[x.index, "description"]))
    assert set(train.index).isdisjoint(val.index) and set(train.index).isdisjoint(test.index) and set(val.index).isdisjoint(test.index)
    inventory = historical_inventory()
    # Reproduce the executed minh_eda 80/10/10 experiment exactly, BEFORE filtering.
    rw = w.loc[w.color.isin(LABELS)]
    eda_train, temp = train_test_split(rw, test_size=.2, stratify=rw.color, random_state=SEED)
    eda_val, _ = train_test_split(temp, test_size=.5, stratify=temp.color, random_state=SEED)
    # Original Models 1–2 sampled the unsplit dataset, with fixed 100-review val.
    narrow = rw.loc[rw.province.eq("California") & rw.variety.isin(["Cabernet Sauvignon", "Chardonnay"])].copy()
    narrow["key"] = norm(narrow.description)
    conflict = narrow.groupby("key").color.nunique()
    narrow = narrow.loc[narrow.key.ne("") & ~narrow.key.isin(conflict[conflict.gt(1)].index)].drop_duplicates("key")
    first, small_val = [], []
    for color in LABELS:
        selected = narrow.loc[narrow.color.eq(color)].sample(n=55, random_state=SEED)
        first.append(selected.iloc[:5]); small_val.append(selected.iloc[5:])
    used = set(pd.concat(first + small_val).key)
    extra = [narrow.loc[narrow.color.eq(c) & ~narrow.key.isin(used)].sample(n=45, random_state=SEED) for c in LABELS]
    old_train = pd.concat(first + extra)
    old_val = pd.concat(small_val)
    # All-data EDA/word heatmaps exposed aggregate descriptions, so this is a
    # recovered predictive holdout, not a pristine never-seen external test set.
    exposure = pd.DataFrame(index=w.index)
    exposure["saved_train"] = w.index.isin(train.index)
    exposure["saved_validation"] = w.index.isin(val.index)
    exposure["eda_train"] = w.index.isin(eda_train.index)
    exposure["eda_validation"] = w.index.isin(eda_val.index)
    exposure["original_model12_train"] = w.index.isin(old_train.index)
    exposure["original_model12_validation"] = w.index.isin(old_val.index)
    exposed = w.loc[exposure.any(axis=1)]
    exposure["normalized_text_exposed"] = norm(w.description).isin(set(norm(exposed.description)))
    exposure["wine_family_exposed"] = group_keys(w).isin(set(group_keys(exposed)))
    exposure["saved_test"] = w.index.isin(test.index)
    exposure["candidate_predictive_holdout"] = exposure.saved_test & w.color.isin(LABELS) & ~exposure[["saved_train", "saved_validation", "eda_train", "eda_validation", "original_model12_train", "original_model12_validation", "normalized_text_exposed"]].any(axis=1)
    exposure["eligible_holdout"] = exposure.candidate_predictive_holdout & ~exposure.wine_family_exposed & w.color.isin(LABELS)
    exposure.index.name = "row_id"
    exposure.to_csv(OUT / "exposure_manifest.csv")
    # For audit and sampling, inspect DEVELOPMENT DATA ONLY.
    dev = pd.concat([train.assign(partition="train"), val.assign(partition="validation")])
    f = flags(dev)
    f.index.name = "row_id"
    f.to_csv(OUT / "development_label_flags.csv")
    status = dev[["partition", "color", "variety", "country", "province"]].copy()
    status["retained"] = ~f.any(axis=1)
    status.groupby(["partition", "color", "retained"]).size().rename("rows").to_csv(OUT / "label_coverage.csv")
    status.groupby(["partition", "country", "retained"], dropna=False).size().rename("rows").to_csv(OUT / "country_coverage.csv")
    status.groupby(["partition", "variety", "retained"]).size().rename("rows").to_csv(OUT / "variety_coverage.csv")
    candidates = dev.loc[(f.style_in_metadata | f.style_in_description) & dev.color.isin(LABELS) & dev.variety.isin(COLOR_MAP), ["partition", "color", "variety", "title", "designation", "description"]].copy()
    candidates.to_csv(OUT / "development_style_candidates.csv", index_label="row_id")
    candidates.sample(n=min(60, len(candidates)), random_state=SEED).to_csv(OUT / "development_style_sample.csv", index_label="row_id")
    sampled = []
    clean = dev.loc[~f.any(axis=1)]
    for (part, color), g in clean.groupby(["partition", "color"]):
        sampled.append(g.sample(n=min(15, len(g)), random_state=SEED))
    pd.concat(sampled)[["partition", "color", "variety", "title", "description"]].to_csv(OUT / "retained_label_sample.csv", index_label="row_id")
    mapping = pd.DataFrame(sorted(COLOR_MAP.items()), columns=["variety", "color"])
    mapping.to_csv(OUT / "confident_variety_map.csv", index=False)
    summary = {"history_versions_reviewed": len(inventory), "total_rows": len(w),
               "saved_split_rows": {"train": len(train), "validation": len(val), "test": len(test)},
               "saved_test_exposure_counts_nonexclusive": exposure.loc[test.index].sum().astype(int).to_dict(),
               "development_exclusion_reasons_nonexclusive": f.sum().astype(int).to_dict(),
               "development_retained": int((~f.any(axis=1)).sum()),
               "label_policy": "restricted existing variety mapping; exclude explicit alternative-style indicators; never auto-relabel",
               "holdout_status": "recovered predictive holdout; whole-data aggregate EDA was previously seen",
               "unrecoverable_execution": "Uncommitted/deleted experiments and other computers cannot be audited. modelattempt has no saved executed split/training and refers to absent label column.",
               "inputs": {n: sha(ROOT / n) for n in ["df_with_color.parquet", "train.parquet", "validation.parquet", "test.parquet"]}}
    save_json("audit_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def vectorizer():
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=10_000, sublinear_tf=True)


def models():
    return {"Naive Bayes": make_pipeline(vectorizer(), MultinomialNB(alpha=1.0)),
            "Logistic regression": make_pipeline(vectorizer(), LogisticRegression(C=1., max_iter=2000, random_state=SEED)),
            "Linear SVM": make_pipeline(vectorizer(), LinearSVC(C=1., max_iter=5000, random_state=SEED))}


def sample(frame, counts):
    return pd.concat([frame.loc[frame.color.eq(c)].sample(n=n, random_state=SEED) for c, n in counts.items()]).sample(frac=1, random_state=SEED)


def mask_pattern():
    # Names are from the frozen confident mapping, not test data. Remove individual
    # components of compound grape names so "cabernet" cannot survive masking.
    # Do not mask non-grape categories such as the original taxonomy's "Apple".
    varieties = set(COLOR_MAP)
    names = varieties | {"red", "white", "rose", "rosé", "rosado", "rosato", "chard", "cab", "gewurztraminer"}
    for v in varieties:
        names.update(t for t in re.findall(r"[^\W\d_]+", v, re.UNICODE) if len(t) >= 3)
    return re.compile(r"(?<!\w)(?:" + "|".join(re.escape(x) for x in sorted(names, key=len, reverse=True)) + r")(?!\w)", re.I)


def regional_rows(stage, name, frame, pred):
    regions = pd.Series(np.where(frame.province.eq("California"), "California", "Outside California"), index=frame.index)
    return [{"stage": stage, "model": name, "group": g, **scores(frame.loc[regions.eq(g), "color"], pred[regions.eq(g).to_numpy()])} for g in regions.unique()]


def interpret(model, train, val, pred):
    vect, classifier = list(model.named_steps.values())
    assert classifier.classes_.tolist() == LABELS
    coeff = pd.DataFrame({"term": vect.get_feature_names_out(), "coefficient_for_white": classifier.coef_[0]})
    coeff.sort_values("coefficient_for_white").to_csv(OUT / "all_coefficients.csv", index=False)
    top = pd.concat([coeff.nsmallest(20, "coefficient_for_white").assign(associated_color="red"), coeff.nlargest(20, "coefficient_for_white").assign(associated_color="white")])
    top.to_csv(OUT / "top_language_features.csv", index=False)
    # Frequencies are document prevalence rather than repeated token counts.
    rows = []
    for word in SENSORY:
        hit = train.description.str.contains(r"\b" + re.escape(word) + r"\b", case=False)
        p = hit.groupby(train.color).mean().reindex(LABELS)
        rows.append({"term": word, "red_pct": 100*p.red, "white_pct": 100*p.white,
                     "absolute_gap_pp": 100*abs(p.red-p.white), "minimum_class_pct": 100*min(p.red,p.white)})
    prevalence = pd.DataFrame(rows)
    prevalence.to_csv(OUT / "sensory_prevalence_training.csv", index=False)
    # Corpus-wide shared words: appear in >=10% of both colors and prevalence
    # ratio between 1/2 and 2. Common does not mean non-discriminative.
    cv = CountVectorizer(binary=True, stop_words="english", min_df=5, max_features=10000)
    X = cv.fit_transform(train.description)
    p0 = np.asarray(X[train.color.eq("red").to_numpy()].mean(axis=0)).ravel()
    p1 = np.asarray(X[train.color.eq("white").to_numpy()].mean(axis=0)).ravel()
    common = pd.DataFrame({"term": cv.get_feature_names_out(), "red_pct": p0*100, "white_pct": p1*100})
    common["minimum_class_pct"] = np.minimum(p0,p1)*100
    common["prevalence_ratio_white_red"] = (p1+1e-10)/(p0+1e-10)
    common = common.loc[(common.minimum_class_pct >= 10) & common.prevalence_ratio_white_red.between(.5,2)].sort_values("minimum_class_pct", ascending=False)
    common.to_csv(OUT / "shared_language_training.csv", index=False)
    errors = val.copy()
    errors["predicted_color"] = pred
    errors["decision_margin_for_white"] = classifier.decision_function(vect.transform(val.description))
    errors = errors.loc[errors.color.ne(errors.predicted_color)]
    errors.to_csv(OUT / "validation_errors.csv", index_label="row_id")
    errors.assign(abs_margin=errors.decision_margin_for_white.abs()).sort_values("abs_margin", ascending=False).groupby("color").head(10).to_csv(OUT / "validation_error_examples.csv", index_label="row_id")
    fig, axes = plt.subplots(1, 2, figsize=(12, 7))
    for ax, color in zip(axes, LABELS):
        t = top.loc[top.associated_color.eq(color)].head(15).iloc[::-1]
        ax.barh(t.term, t.coefficient_for_white, color="#9f2d3f" if color == "red" else "#bd9c38")
        ax.set_title(f"Terms associated with {color}")
        ax.set_xlabel("Coefficient toward white (negative favors red)")
    fig.tight_layout(); fig.savefig(OUT / "language_coefficients.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5))
    ys = np.arange(len(prevalence))
    ax.bar(ys-.2, prevalence.red_pct, .4, label="Red", color="#9f2d3f")
    ax.bar(ys+.2, prevalence.white_pct, .4, label="White", color="#bd9c38")
    ax.set_xticks(ys, prevalence.term, rotation=55, ha="right")
    ax.set_ylabel("Training reviews mentioning word (%)"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "sensory_language.png", dpi=160); plt.close(fig)


def develop():
    if (OUT / "selection_lock.json").exists():
        raise RuntimeError("Model selection is already frozen. Do not tune after evaluation.")
    audit_info = json.loads((OUT / "audit_summary.json").read_text())
    for n, digest in audit_info["inputs"].items():
        assert sha(ROOT / n) == digest, f"Input changed: {n}"
    raw_train, raw_val = load("train.parquet"), load("validation.parquet")
    val = pool(raw_val)
    train = pool(raw_train)
    train = train.loc[~train.description_key.isin(val.description_key)].copy()
    assert_separate(train, val)
    # Fixed-stage comparisons on the audited population. All stages use the
    # same hyperparameters; only intended population or training size changes.
    ca = train.loc[train.province.eq("California")]
    narrow = ca.loc[ca.variety.isin(["Cabernet Sauvignon", "Chardonnay"])]
    narrow_val = val.loc[val.province.eq("California") & val.variety.isin(["Cabernet Sauvignon", "Chardonnay"])]
    ca_val = val.loc[val.province.eq("California")]
    order = {c: narrow.loc[narrow.color.eq(c)].sample(n=500, random_state=SEED) for c in LABELS}
    stages = []
    for n in [10,100,1000]:
        x = pd.concat([order[c].iloc[:n//2] for c in LABELS]).sample(frac=1, random_state=SEED)
        stages.append((f"narrow_{n}", x, narrow_val))
    ca1000 = sample(ca, {"red":500,"white":500})
    extra = sample(ca.loc[~ca.index.isin(ca1000.index)], {"red":4500,"white":4500})
    ca10000 = pd.concat([ca1000,extra]).sample(frac=1,random_state=SEED)
    extra20 = sample(ca.loc[~ca.index.isin(ca10000.index)], {"red":8000,"white":2000})
    ca20000 = pd.concat([ca10000,extra20]).sample(frac=1,random_state=SEED)
    all20000 = sample(train,{"red":13000,"white":7000})
    stages += [("california_1000",ca1000,ca_val), ("california_10000",ca10000,ca_val),
               ("california_20000_global",ca20000,val), ("allregion_20000",all20000,val), ("allregion_full",train,val)]
    records, regional, fitted = [], [], {}
    memberships = []
    for stage, x, v in stages:
        assert_separate(x,v)
        memberships.extend({"stage":stage,"partition":part,"row_id":int(i)} for part,frame in [("train",x),("validation",v)] for i in frame.index)
        for name, model in {"Majority baseline": DummyClassifier(strategy="most_frequent"), **models()}.items():
            model.fit(x.description,x.color)
            for split, f in [("training",x),("validation",v)]:
                pred=model.predict(f.description)
                records.append({"stage":stage,"model":name,"split":split,**scores(f.color,pred)})
                if split == "validation" and stage in ["california_20000_global","allregion_20000","allregion_full"]:
                    regional.extend(regional_rows(stage,name,v,pred))
            if stage in ["allregion_20000","allregion_full"] and name != "Majority baseline":
                fitted[(stage,name)] = model
            print("Finished",stage,name,flush=True)
        pd.DataFrame(records).to_csv(OUT / "audited_model_comparison.csv",index=False)
    pd.DataFrame(memberships).to_csv(OUT / "development_memberships.csv",index=False)
    pd.DataFrame(regional).to_csv(OUT / "regional_validation.csv",index=False)
    # Complete the original full_size_models final comparison on its ORIGINAL
    # population. Keep these separate from audited-model selection.
    legacy_v = raw_val.loc[raw_val.color.isin(LABELS)].copy()
    legacy_t = raw_train.loc[raw_train.color.isin(LABELS)].copy()
    legacy_t = legacy_t.loc[~norm(legacy_t.description).isin(set(norm(legacy_v.description)))]
    legacy_t = legacy_t.loc[~norm(legacy_t.description).duplicated()]
    assert_separate(legacy_t, legacy_v)
    legacy = []
    for name, model in models().items():
        model.fit(legacy_t.description,legacy_t.color)
        for split,frame in [("training",legacy_t),("validation",legacy_v)]:
            legacy.append({"stage":"legacy_full_size","model":name,"split":split,**scores(frame.color,model.predict(frame.description))})
        print("Finished original full-size",name,flush=True)
    pd.DataFrame(legacy).to_csv(OUT / "original_full_size_comparison.csv",index=False)
    # Selection considers only audited, all-region candidates evaluated on the
    # identical audited validation rows. No DistilBERT checkpoint is available.
    table = pd.DataFrame(records)
    candidates=table.loc[table.stage.isin(["allregion_20000","allregion_full"]) & table.split.eq("validation") & table.model.ne("Majority baseline")]
    best=candidates.sort_values(["macro_f1","accuracy","rows"],ascending=[False,False,False],kind="stable").iloc[0]
    stage,name=best.stage,best.model
    model=fitted[(stage,name)]
    selected_train=all20000 if stage == "allregion_20000" else train
    joblib.dump(model,OUT / "selected_pipeline.joblib",compress=3)
    pred=model.predict(val.description)
    prediction_table=val[["color","country","province","variety"]].rename(columns={"color":"actual_color"}).copy()
    prediction_table["predicted_color"]=pred
    prediction_table.to_csv(OUT / "selected_validation_predictions.csv",index_label="row_id")
    # Interpretation uses the current selected linear model; if NB won, fit an
    # SVM companion on the same training rows, explicitly labeled as such.
    companion = model if name in ["Linear SVM","Logistic regression"] else models()["Linear SVM"].fit(selected_train.description,selected_train.color)
    interpret(companion,selected_train,val,companion.predict(val.description))
    masking=mask_pattern()
    masked_train=selected_train.description.map(lambda t: masking.sub(" ",t))
    masked_val=val.description.map(lambda t: masking.sub(" ",t))
    masked=models()[name].fit(masked_train,selected_train.color)
    save_json("masking_experiment.json",{"model":name,"training_stage":stage,"unmasked":scores(val.color,pred),"masked":scores(val.color,masked.predict(masked_val)),"pattern":masking.pattern,"notes":"Retrained on masked training text and evaluated on masked validation text. Broad component masking can remove ordinary sensory words; this is a sensitivity analysis, not causal attribution."})
    # Verify existing BERT results directly from prediction CSVs. These retain
    # original labels/populations and are not candidates in the audited contest.
    bert=[]
    for archive in sorted(ROOT.glob("modeling_*_distilbert_results.zip")):
        with zipfile.ZipFile(archive) as z:
            metadata=json.loads(z.read("metadata.json"))
            for split,filename in [("training","training_predictions.csv"),("validation","validation_predictions.csv")]:
                p=pd.read_csv(io.BytesIO(z.read(filename)))
                assert p.row_id.is_unique
                source=raw_train if split == "training" else raw_val
                assert set(p.row_id).issubset(source.index)
                assert p.set_index("row_id").actual_color.eq(source.loc[p.row_id,"color"]).all()
                rows=[[int(i),str(source.loc[i,"color"]),str(source.loc[i,"description"])] for i in sorted(p.row_id)]
                fingerprint=hashlib.sha256(json.dumps(rows,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
                assert metadata[f"{split}_fingerprint"] == fingerprint
                bert.append({"archive":archive.name,"split":split,**scores(p.actual_color,p.predicted_color)})
    pd.DataFrame(bert).to_csv(OUT / "verified_original_distilbert_scores.csv",index=False)
    # Save plots and decision record before final test labels/predictions.
    fig,ax=plt.subplots(figsize=(8,5))
    for name_ in ["Naive Bayes","Logistic regression","Linear SVM"]:
        for split,style in [("training","--"),("validation","-")]:
            t=table.loc[table.stage.str.startswith("narrow_") & table.model.eq(name_) & table.split.eq(split)].sort_values("rows")
            # Validation row count is fixed; use stage size for x-axis.
            ax.plot([int(s.rsplit('_',1)[1]) for s in t.stage],t.macro_f1,style,marker='o',label=f"{name_}, {split}")
    ax.set_xscale("log");ax.set_xlabel("Narrow training reviews");ax.set_ylabel("Macro F1");ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(OUT / "learning_curve.png",dpi=160);plt.close(fig)
    lock={"selected_stage":stage,"selected_model":name,"selection_rule":"Highest audited all-region validation macro F1; accuracy breaks ties. Fixed C=1, no test-based tuning.","validation_scores":scores(val.color,pred),"training_rows":len(selected_train),"training_row_ids":selected_train.index.astype(int).tolist(),"validation_row_ids":val.index.astype(int).tolist(),"pipeline_sha256":sha(OUT / "selected_pipeline.joblib"),"code_sha256":sha(__file__),"exposure_manifest_sha256":sha(OUT / "exposure_manifest.csv"),"input_sha256":audit_info["inputs"],"interpretation_model":name if companion is model else "Linear SVM companion","versions":{"python":platform.python_version(),"pandas":pd.__version__,"sklearn":sklearn.__version__,"numpy":np.__version__,"joblib":joblib.__version__},"label_policy":{"mapping":COLOR_MAP,"metadata_exclusion_pattern":STYLE_METADATA,"description_exclusion_pattern":STYLE_DESCRIPTION},"evaluation_policy":"Recovered predictive holdout only; exclude exposed review text and wine families, apply frozen style rules, then evaluate once. Do not claim a pristine external holdout."}
    save_json("selection_lock.json",lock)
    print("FROZEN",stage,name,scores(val.color,pred),flush=True)


def bootstrap(actual,pred,repeats=2000):
    # Conditional on this fixed model and selected population; not retraining uncertainty.
    a=np.asarray(actual);p=np.asarray(pred);rng=np.random.default_rng(SEED)
    cms=confusion_matrix(a,p,labels=LABELS)
    draws=rng.multinomial(len(a),cms.ravel()/len(a),size=repeats).reshape(-1,2,2)
    tp=draws[:,[0,1],[0,1]].astype(float)
    recall=np.divide(tp,draws.sum(axis=2),out=np.zeros_like(tp),where=draws.sum(axis=2)!=0)
    f1=np.divide(2*tp,draws.sum(axis=1)+draws.sum(axis=2),out=np.zeros_like(tp),where=(draws.sum(axis=1)+draws.sum(axis=2))!=0)
    values={"accuracy":tp.sum(axis=1)/len(a),"balanced_accuracy":recall.mean(axis=1),"macro_f1":f1.mean(axis=1)}
    return {k:{"low":float(np.quantile(v,.025)),"high":float(np.quantile(v,.975))} for k,v in values.items()}


def evaluate():
    if (OUT / "final_evaluation_started.json").exists():
        raise RuntimeError("Final evaluation already started. Read saved results; do not evaluate again.")
    lock=json.loads((OUT / "selection_lock.json").read_text())
    assert sha(__file__) == lock["code_sha256"],"Analysis changed after freeze"
    assert sha(OUT / "selected_pipeline.joblib") == lock["pipeline_sha256"]
    assert sha(OUT / "exposure_manifest.csv") == lock["exposure_manifest_sha256"]
    for name,digest in lock["input_sha256"].items():
        assert sha(ROOT / name) == digest,f"Input changed: {name}"
    save_json("final_evaluation_started.json",{"pipeline_sha256":lock["pipeline_sha256"],"policy":lock["evaluation_policy"]})
    manifest=pd.read_csv(OUT / "exposure_manifest.csv",index_col="row_id")
    eligible=manifest.index[manifest.eligible_holdout]
    test=load("test.parquet").loc[eligible].copy()
    test_flags=flags(test)
    test_flags.to_csv(OUT / "final_holdout_label_flags.csv",index_label="row_id")
    test=pool(test)
    assert len(test)>0 and set(test.color)==set(LABELS)
    dev=pd.concat([load("train.parquet"),load("validation.parquet")])
    assert_separate(dev,test)
    assert set(group_keys(test)).isdisjoint(group_keys(dev))
    assert not manifest.loc[test.index,["eda_train","eda_validation","original_model12_train","original_model12_validation","normalized_text_exposed","wine_family_exposed"]].any().any()
    model=joblib.load(OUT / "selected_pipeline.joblib")
    pred=model.predict(test.description)
    final={"model":lock["selected_model"],"stage":lock["selected_stage"],"holdout_description":lock["evaluation_policy"],"eligible_before_label_rules":len(eligible),"excluded_by_label_rules":int(test_flags.any(axis=1).sum()),"class_counts":test.color.value_counts().to_dict(),"metrics":scores(test.color,pred),"bootstrap_95pct_intervals":bootstrap(test.color,pred),"interval_method":"2000 multinomial bootstrap replicates of the empirical confusion matrix; equivalent to resampling independent review-level (actual,predicted) pairs. Conditional on frozen model and retained population.","classification_report":classification_report(test.color,pred,labels=LABELS,output_dict=True,zero_division=0),"confusion_matrix_labels":LABELS,"confusion_matrix":confusion_matrix(test.color,pred,labels=LABELS).tolist()}
    save_json("final_metrics.json",final)
    predictions=test[["color","variety","country","province"]].rename(columns={"color":"actual_color"}).copy()
    predictions["predicted_color"]=pred
    predictions.to_csv(OUT / "final_holdout_predictions.csv",index_label="row_id")
    pd.DataFrame(final["classification_report"]).T.to_csv(OUT / "final_classification_report.csv")
    pd.DataFrame(final["confusion_matrix"],index=["actual red","actual white"],columns=["predicted red","predicted white"]).to_csv(OUT / "final_confusion_matrix.csv")
    pd.DataFrame(regional_rows(lock["selected_stage"],lock["selected_model"],test,pred)).to_csv(OUT / "final_regional_metrics.csv",index=False)
    fig,ax=plt.subplots(figsize=(5,4));cm=np.array(final["confusion_matrix"])
    ax.imshow(cm,cmap="Blues")
    for i in range(2):
        for j in range(2):ax.text(j,i,str(cm[i,j]),ha="center",va="center",color="white" if cm[i,j]>cm.max()/2 else "black")
    ax.set_xticks([0,1],LABELS);ax.set_yticks([0,1],LABELS);ax.set_xlabel("Predicted");ax.set_ylabel("Actual");ax.set_title("Recovered holdout: frozen model")
    fig.tight_layout();fig.savefig(OUT / "final_confusion_matrix.png",dpi=160);plt.close(fig)
    print(json.dumps(final,indent=2),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step",choices=["audit","develop","evaluate"])
    args=parser.parse_args()
    os.chdir(ROOT)
    {"audit":audit,"develop":develop,"evaluate":evaluate}[args.step]()
