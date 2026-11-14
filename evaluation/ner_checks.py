"""Robustness checks for the few-shot-prompted NER output (NER_Extracted.json).

1. Confidence interval for a perfect score on a 5-abstract gold set.
2. Source grounding: is each extracted numeric value present in the source abstract?
   NER entries carry no reliable link to their abstract, so the source is identified as
   the abstract(s) that contain the extracted alloy name.
3. DOI validity: are extracted DOIs real DOIs of the corpus (taken from the Scopus link)?
"""
import csv
import json
import random
import re
from urllib.parse import unquote

import pandas as pd

from common import DATA, RESULTS, SEED, wilson_ci

NUM = re.compile(r"\d+(?:\.\d+)?")


def squash(t):
    return re.sub(r"\s+", "", str(t)).lower()


def main():
    ner = json.load(open(DATA / "NER_Extracted.json", encoding="utf-8"))
    scopus = pd.read_csv(DATA / "Scopus_Abstracts_Dataset.csv")
    abstracts = [squash(a) for a in scopus["Abstract"]]
    real_dois = {unquote(m.group(1)).lower() for m in
                 (re.search(r"doi=([^&]+)", str(l)) for l in scopus["Link"]) if m}

    out = {"gold_set": {}}
    for k in (5,):
        lo, hi = wilson_ci(k, 5)
        out["gold_set"]["5_of_5_abstracts_correct_wilson95"] = [lo, hi]
    out["gold_set"]["abstracts_needed_for_lower_bound>=0.95_if_all_correct"] = next(
        n for n in range(5, 1000) if wilson_ci(n, n)[0] >= 0.95)

    alloys = [a for e in ner for a in e["alloys"]]
    dois = [str(a.get("doi", "")).rstrip(",").strip().lower() for a in alloys]
    out["doi"] = {
        "alloy_records": len(alloys),
        "placeholder_like(xxxx/example/1234)": sum(bool(re.search(r"x{3,}|example|1234|123456", d)) for d in dois),
        "matches_a_real_doi_in_corpus": sum(d in real_dois for d in dois),
        "corpus_abstracts": len(scopus),
        "corpus_unique_abstracts": int(scopus["Abstract"].nunique()),
    }

    matched_entries, checked, found = 0, 0, 0
    per_record, audit_pool = [], []
    for e in ner:
        for a in e["alloys"]:
            name = squash(a.get("name", ""))
            src_idx = [i for i, ab in enumerate(abstracts) if name and name in ab]
            src = [abstracts[i] for i in src_idx]
            if len(src_idx) == 1:
                audit_pool.append((scopus["Abstract"][src_idx[0]], a))
            if not src:
                continue
            values = []
            for prop in (a.get("performance_properties") or {}).values():
                for m in (prop or {}).get("measurements", []) or []:
                    nums = NUM.findall(str(m.get("value", "")))
                    if nums:
                        values.append(nums[0])
            if not values:
                continue
            matched_entries += 1
            hits = [any(v in ab for ab in src) for v in values]
            checked += len(hits)
            found += sum(hits)
            per_record.append(sum(hits) / len(hits))
    out["value_grounding"] = {
        "alloy_records_with_identifiable_source_abstract": matched_entries,
        "numeric_values_checked": checked,
        "numeric_values_found_verbatim_in_source": found,
        "fraction_grounded": found / checked if checked else None,
        "fraction_grounded_wilson95": wilson_ci(found, checked),
        "records_fully_grounded": sum(p == 1 for p in per_record) / len(per_record),
    }
    json.dump(out, open(RESULTS / "ner_checks.json", "w"), indent=2)
    sample = random.Random(SEED).sample(audit_pool, min(30, len(audit_pool)))
    with open(RESULTS / "ner_manual_audit_sheet.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["abstract", "extracted_record_json", "composition_correct(0/1)", "phase_correct(0/1)",
                    "TP_property_values", "FP_property_values", "FN_property_values", "rater_notes"])
        for ab, rec in sample:
            w.writerow([ab, json.dumps(rec, ensure_ascii=False), "", "", "", "", "", ""])
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
