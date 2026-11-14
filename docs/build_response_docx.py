"""Builds docs/Response_to_Reviewers.docx from evaluation/results/*.json."""
import json
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor, Cm

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "evaluation" / "results"
J = lambda n: json.load(open(RES / n)) if (RES / n).exists() else None
split, qa, ner, ret = J("split_summary.json"), J("qa_validation_summary.json"), J("ner_checks.json"), J("retrieval_metrics.json")
t5z, t5, phi = J("generation_t5_zeroshot.json"), J("generation_t5.json"), J("generation_phi2.json")
BLUE = RGBColor(0x1F, 0x3A, 0x5F)

doc = Document()
st = doc.styles["Normal"]
st.font.name, st.font.size = "Calibri", Pt(11)
for s in doc.sections:
    s.left_margin = s.right_margin = Cm(2.2)


def shade(cell, hex_):
    tc = cell._tc.get_or_add_tcPr()
    sh = OxmlElement("w:shd")
    sh.set(qn("w:val"), "clear"); sh.set(qn("w:color"), "auto"); sh.set(qn("w:fill"), hex_)
    tc.append(sh)


def h(text, lvl=1):
    p = doc.add_heading(text, lvl)
    for r in p.runs:
        r.font.color.rgb = BLUE
    return p


def para(text, bold=False, italic=False, size=None, indent=None):
    p = doc.add_paragraph()
    r = p.add_run(text); r.bold, r.italic = bold, italic
    if size:
        r.font.size = Pt(size)
    if indent:
        p.paragraph_format.left_indent = Cm(indent)
    p.paragraph_format.space_after = Pt(6)
    return p


def rich(parts, style=None, indent=None):
    p = doc.add_paragraph(style=style)
    for t, b in parts:
        p.add_run(t).bold = b
    if indent:
        p.paragraph_format.left_indent = Cm(indent)
    return p


def comment(label, text):
    t = doc.add_table(rows=1, cols=1); t.style = "Table Grid"
    c = t.rows[0].cells[0]; shade(c, "EEF2F7")
    c.paragraphs[0].add_run(label).bold = True
    c.add_paragraph(text).runs[0].italic = True
    doc.add_paragraph()


def revised(text):
    t = doc.add_table(rows=1, cols=1); t.style = "Table Grid"
    c = t.rows[0].cells[0]; shade(c, "F3F9F1")
    c.paragraphs[0].add_run("Revised manuscript text").bold = True
    for line in text.strip().split("\n\n"):
        c.add_paragraph(line.strip())
    doc.add_paragraph()


def table(header, rows, caption=None, widths=None):
    if caption:
        para(caption, bold=True, size=10)
    t = doc.add_table(rows=1, cols=len(header)); t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for c, x in zip(t.rows[0].cells, header):
        c.text = ""; c.paragraphs[0].add_run(x).bold = True; shade(c, "D9E2F3")
    for row in rows:
        cells = t.add_row().cells
        for c, x in zip(cells, row):
            c.text = str(x)
    for row in t.rows:
        for c in row.cells:
            for p in c.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(9.5)
    if widths:
        for row in t.rows:
            for c, w in zip(row.cells, widths):
                c.width = Cm(w)
    doc.add_paragraph()


pct = lambda x: f"{100 * x:.1f}%"
ci = lambda a: f"[{100 * a[0]:.1f}, {100 * a[1]:.1f}]"
f3 = lambda x: f"{x:.3f}"
ci3 = lambda a: f"[{a[0]:.3f}, {a[1]:.3f}]"
wl, wh = ner["gold_set"]["5_of_5_abstracts_correct_wilson95"]
vg = ner["value_grounding"]
tr, va, te = split["train"], split["val"], split["test"]
leak = split["leakage_if_random_question_split"]
rt = ret["test"]

# ---------------------------------------------------------------- title
p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("Response to Reviewers"); r.bold = True; r.font.size = Pt(18); r.font.color.rgb = BLUE
p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("Leveraging Large Language Models for Strategic Scientific Literature Mining on High-Entropy Alloys")
r.italic = True; r.font.size = Pt(12)
p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.add_run("Vikhyat Bansal, Aman Sirohi, R Sriviswa, Souvik Gorain, Kritesh Kumar Gupta*").font.size = Pt(10)

para("Dear Editor and Reviewers,")
para("We thank the reviewers for their careful reading of the manuscript and for comments that have substantially "
     "improved it. In response we did not only revise the wording: we re-examined the data, code and evaluation "
     "protocol behind every reported number, corrected two implementation errors, and re-ran the Q&A/RAG evaluation "
     "on a leakage-free, context-level train/validation/test split. All scripts, data splits and result files are "
     "released in the project repository (https://github.com/KriteshKG/HEA_LLM, folder evaluation/), so that every "
     "number in the revised manuscript can be regenerated with a single command (bash evaluation/run_all.sh). "
     "Reviewer comments are reproduced in shaded boxes, followed by our response and, where applicable, the revised "
     "manuscript text (green boxes).")

h("Summary of principal changes", 2)
summary_rows = [
    ("Few-shot NER “100% accuracy”", "Claim removed. Reported as 5/5 abstracts on an exploratory set with 95% CI "
     f"{pct(wl)}–100%; complemented by an automatic grounding check of {vg['numeric_values_checked']} extracted values."),
    ("NER evaluation set", "Size, composition, annotation procedure, matching criteria and limitations now stated (Sec. 2.1–2.2)."),
    ("Table 1", "Discussion values corrected to match the table; cost column given units; caption corrected."),
    ("Data separation (RAG)", f"Split by context: {tr['contexts']}/{va['contexts']}/{te['contexts']} contexts, "
     f"{tr['questions']}/{va['questions']}/{te['questions']} questions; zero train–test context overlap."),
    ("Phi-2 training", "Loss previously computed on the prompt only; corrected to answer-only loss (prompt masked)."),
    ("Retrieval", f"Retrieval evaluated separately (Recall@k, MRR); hybrid dense + lexical retriever raises test "
     f"Recall@1 from {pct(rt['dense_minilm']['recall@1'])} to {pct(rt['hybrid']['recall@1'])}."),
    ("Q&A dataset validation", "Grounding, duplicate and conflict checks on all 3,392 pairs; stratified manual audit protocol."),
    ("Table 4", "Replaced by held-out test results with gold and retrieved contexts and 95% CIs."),
]
table(["Issue", "Change made"], summary_rows, widths=[4.5, 12])

# ---------------------------------------------------------------- Reviewer 1
h("Reviewer 1", 1)
comment("Comment 1", "The manuscript addresses a relevant topic and presents an interesting framework for applying LLMs "
        "to scientific literature mining in High-Entropy Alloys. However, several aspects of the experimental "
        "evaluation require further clarification and improvement.")
rich([("Response: ", True), ("We thank the reviewer for the positive assessment of the topic and framework. We agree that "
      "the experimental evaluation needed to be strengthened. The specific concerns are addressed point by point under "
      "Comment 2 below; in summary, the evaluation now (i) reports the size and composition of every evaluation set, "
      "(ii) separates training, validation and test data at the level of whole contexts, (iii) evaluates retrieval "
      "and response generation separately, and (iv) reports uncertainty (95% confidence intervals) for every metric.", False)])

comment("Comment 2", "In particular, the limited NER evaluation, the reported 100% few-shot accuracy, unclear data "
        "separation, and insufficient validation of the Q&A and RAG components raise concerns about the robustness of "
        "the reported results. The manuscript would benefit from additional experiments, clearer methodological "
        "details, and stronger validation before it can be considered for publication.")
rich([("Response: ", True), ("We appreciate the reviewer for highlighting these important methodological concerns. "
      "We have addressed the issues as follows.", False)])

# 1 NER
h("1. NER evaluation", 3)
para("We have expanded the description of the NER evaluation procedure and clarified the size and composition of the "
     "evaluation dataset, the evaluation criteria, and the limitations associated with the size of the manually "
     "annotated test set.")
for b in [
    "Evaluation set. The model comparison in Table 1 used five abstracts selected at random from the 54 reference "
    "articles of the Couzinié et al. data article and annotated manually in Doccano (HEA composition, phase structure, "
    "property, value, unit and test temperature). The revised text "
    "states the set size and composition explicitly, states that the evaluation abstracts are distinct from the "
    "examples used in the few-shot prompt, and describes the set as exploratory.",
    "Matching criteria. The revised Section 2.1 defines the matching rule used to build the confusion matrix: an "
    "entity is a true positive only if its type and value agree with the gold annotation (values compared after unit "
    "normalisation, compositions compared after removing spaces). Accuracy and F1 (Eqs. 1–2) are computed per "
    "abstract and averaged over the five abstracts.",
    f"Uncertainty. With n = 5, sampling uncertainty is large: even a perfect 5/5 result has a 95% Wilson interval of "
    f"{pct(wl)}–100%. A lower bound of 95% would require at least "
    f"{ner['gold_set']['abstracts_needed_for_lower_bound>=0.95_if_all_correct']} correctly processed abstracts. "
    "This limitation is now stated in the Results and in the Conclusions.",
    f"Additional, larger-sample check. Because manual annotation is costly, we added an automatic source-grounding "
    f"check on the full extracted dataset: every numerical property value extracted by the few-shot-prompted model is "
    f"searched for in its source abstract. Of {vg['numeric_values_checked']} values from "
    f"{vg['alloy_records_with_identifiable_source_abstract']} alloy records whose source abstract could be identified, "
    f"{vg['numeric_values_found_verbatim_in_source']} ({pct(vg['fraction_grounded'])}, 95% CI "
    f"{ci(vg['fraction_grounded_wilson95'])}) occur verbatim in the source; {pct(vg['records_fully_grounded'])} of "
    "records are fully grounded. This measures precision of numerical values only (not recall, units or phase "
    "labels) and is reported as such.",
    f"Data-quality limitation identified. The DOI field generated by the LLM is unreliable: "
    f"{ner['doi']['placeholder_like(xxxx/example/1234)']} of {ner['doi']['alloy_records']} alloy records carry a "
    "placeholder DOI (e.g. 10.XXXX/XXXX) and none matches the DOI of a corpus article. The DOI field is therefore not "
    "used in any evaluation, the limitation is acknowledged in the manuscript, and DOIs should be taken from the "
    "Scopus metadata rather than generated by the LLM.",
]:
    doc.add_paragraph(b, style="List Bullet")

# 2 Few-shot
h("2. Few-shot accuracy", 3)
para("We acknowledge that reporting 100% accuracy based on only five evaluation samples could lead to an overly "
     "optimistic interpretation of the model's performance. We have therefore revised the relevant discussion to "
     "explicitly state the evaluation sample size and to interpret the result in the context of the limited evaluation "
     "set. The statement “revealed 100% accuracy, without any discrepancy” has been removed; the few-shot result is now "
     "also reported in Table 1 so that the text and table are consistent (see also Reviewer 2, Comment 2).")
revised(f"""
On the five manually annotated evaluation abstracts, the few-shot-prompted GPT-4 model extracted all annotated entities without error (5/5 abstracts). Because the evaluation set is small, this observation is compatible with a true abstract-level accuracy anywhere between {pct(wl)} and 100% (95% Wilson interval) and should not be read as evidence of error-free extraction. As an additional check on the full extracted dataset, {pct(vg['fraction_grounded'])} (95% CI {ci(vg['fraction_grounded_wilson95'])}) of {vg['numeric_values_checked']} extracted numerical property values were found verbatim in their source abstracts.
""")

# 3 Data separation
h("3. Data separation", 3)
para("We have clarified the data organisation and the separation strategy. The revised manuscript explicitly describes "
     "the training, validation and test procedures. While re-examining this point we found that separation must be "
     "done by context rather than by question, and we have redone the experiments accordingly:")
for b in [
    f"The Q&A dataset contains {qa['qa_pairs']:,} pairs generated from {qa['contexts']} contexts (one context = the "
    f"structured record of one alloy). It contains only {qa['distinct_facts_(context,answer)']} distinct facts: on "
    f"average each fact is asked {qa['mean_paraphrases_per_fact']:.1f} times in paraphrased form.",
    f"A random split by question therefore leaks the test data: in a simulated 85/15 random split, "
    f"{pct(leak['test_questions_with_context_seen_in_train'])} of the {leak['random_split_test_questions']} test "
    f"questions have their context, and {pct(leak['test_questions_with_same_fact_seen_in_train'])} have the identical "
    "answer, in the training set.",
    f"We now split by context (seed 42): training {tr['contexts']} contexts / {tr['questions']:,} questions, validation "
    f"{va['contexts']} / {va['questions']}, test {te['contexts']} / {te['questions']}. No context appears in more than "
    "one split. Fine-tuning uses only the training split, hyper-parameters and the stopping epoch are selected on the "
    "validation split, and the test split is scored once.",
    "The split files (evaluation/splits/) are published with the code.",
]:
    doc.add_paragraph(b, style="List Bullet")

# 4 QA and RAG
h("4. Q&A and RAG validation", 3)
para("We have strengthened the validation of the generated question–answer pairs and of the RAG retrieval and "
     "response-generation pipeline, and clarified the criteria for relevance, factual consistency and contextual "
     "grounding.")
para("(a) Q&A pairs. All 3,392 pairs were checked automatically:", bold=True)
table(["Check", "Result"], [
    ("Answer occurs verbatim in its context (contextual grounding)", pct(qa["answer_verbatim_in_context"])),
    ("Answer offset (answer_start) consistent with context", pct(qa["answer_offset_consistent"])),
    ("Numerical answers", pct(qa["numeric_answers"])),
    ("Median answer length", f"{qa['median_answer_words']} words"),
    ("Duplicate questions (after normalisation)", qa["duplicate_questions_exact_after_normalisation"]),
    ("Same question with conflicting answers (factual consistency)", qa["same_question_conflicting_answers"]),
    ("Question categories (property value / temperature / unit)",
     " / ".join(str(qa["category_counts"].get(k, 0)) for k in ("Property-value-based", "Temperature-based", "Unit-based"))),
], caption="Table R1. Automatic validation of the Q&A dataset")
para(f"The {qa['qa_pairs'] - round(qa['answer_verbatim_in_context'] * qa['qa_pairs'])} ungrounded pairs and the "
     f"{qa['same_question_conflicting_answers']} conflicting questions are listed in evaluation/results/qa_ungrounded_pairs.csv "
     "for removal. In addition, a stratified random sample of Q&A pairs (up to 25 per category, 66 pairs) is rated by "
     "two domain experts on three binary criteria: relevance (the question concerns an HEA property stated in the "
     "context), factual consistency (the answer agrees with the context, including unit and temperature) and "
     "contextual grounding (the answer can be located in the context). This replaces the earlier five-plus-five "
     "illustration; inter-rater agreement is reported as Cohen’s κ.")
para("(b) Retrieval. Retrieval is now evaluated on its own, with the held-out questions as queries against all "
     "103 contexts (a hit = the gold context is returned). Confidence intervals resample whole contexts.", bold=True)
rows = []
for k, lab in (("dense_minilm", "Dense (MiniLM-L6-v2, cosine)"), ("lexical_tfidf_char", "Lexical (char n-gram TF-IDF)"),
               ("hybrid", "Hybrid (0.5 dense + 0.5 lexical)")):
    m = rt[k]
    rows.append((lab, f"{f3(m['recall@1'])} {ci3(m['recall@1_95ci'])}", f3(m["recall@3"]), f3(m["recall@5"]),
                 f"{f3(m['mrr'])} {ci3(m['mrr_95ci'])}"))
table(["Retriever", "Recall@1 [95% CI]", "Recall@3", "Recall@5", "MRR [95% CI]"], rows,
      caption=f"Table R2. Retrieval on the test split ({rt['hybrid']['queries']} questions, 15 contexts)")
para("The dense retriever described in the original manuscript often confuses alloys whose structured records are "
     "very similar (mean cosine similarity of the best wrong context "
     f"{rt['dense_minilm']['mean_cosine_best_non_gold']:.2f} vs {rt['dense_minilm']['mean_cosine_gold']:.2f} for the "
     "gold context). Adding a character n-gram lexical score, which matches alloy formulae exactly, resolves this; the "
     "hybrid retriever (weights fixed a priori, not tuned on test) is now used in the framework and the application.")

para("(c) Response generation.", bold=True)
gen_rows = []
for name, res in (("T5-base, no fine-tuning", t5z), ("T5-base, fine-tuned", t5), ("Phi-2 + LoRA, fine-tuned", phi)):
    if not res:
        continue
    for s, lab in (("gold", "gold context"), ("top1", "retrieved top-1"), ("top3", "retrieved top-3")):
        if s in res["test"]:
            m = res["test"][s]
            gen_rows.append((name, lab, f"{f3(m['exact_match'])} {ci3(m['exact_match_95ci'])}",
                             f"{f3(m['f1'])} {ci3(m['f1_95ci'])}", f3(m["bleu"]), f3(m["rougeL"])))
if gen_rows:
    para("Both generators were re-trained on the training split only and evaluated once on the "
         f"{te['questions']} test questions ({te['contexts']} unseen contexts), first with the gold context (generator "
         "quality) and then with the contexts returned by the hybrid retriever (end-to-end RAG). Exact Match and F1 "
         "follow the SQuAD v1.1 definitions; BLEU is corpus-level (sacreBLEU); ROUGE-L is the mean F-measure.")
    table(["Model", "Context", "Exact Match [95% CI]", "F1 [95% CI]", "BLEU", "ROUGE-L"], gen_rows,
          caption="Table 4 (revised). Response generation on the held-out test split")
para("While reproducing Table 4 we found an implementation error in the original Phi-2 fine-tuning: the answer was "
     "stored as the label, but the language-modelling data collator then overwrote the labels with the prompt, so "
     "the loss was computed on the prompt and the model was never trained to produce the answers. The corrected "
     "procedure trains on prompt + answer + end-of-sequence token with the prompt tokens masked from the loss. The "
     "original Table 4 values could not be reproduced and have been replaced.")
para("We believe these revisions improve the methodological transparency and provide a more balanced interpretation "
     "of the framework's performance.")

# ---------------------------------------------------------------- Reviewer 2
h("Reviewer 2", 1)
comment("Comment 2", "The 100% NER accuracy claimed for few-shot prompting is not supported by any table (Table 1 tops "
        "out at 94%). Evaluation used only five samples; please state the test-set size and train/test split for the "
        "RAG results.")
rich([("Response: ", True), ("We thank the reviewer and agree on both points.", False)])
for b in [
    "100% claim. The sentence claiming 100% accuracy has been removed. The few-shot result is now included in Table 1 "
    f"as a separate row (“GPT-4, few-shot prompting: 5/5 abstracts, 95% CI {pct(wl)}–100%”), the evaluation sample size "
    "(n = 5 abstracts) is stated in the table caption and in the text, and the result is described as exploratory "
    "(full text under Reviewer 1, point 2).",
    f"RAG test set and split. The Q&A data are split by context into training ({tr['contexts']} contexts, "
    f"{tr['questions']:,} questions), validation ({va['contexts']} contexts, {va['questions']} questions) and test "
    f"({te['contexts']} contexts, {te['questions']} questions); no context is shared between splits. All RAG results "
    "(Table R2 and revised Table 4) are on the test split. The reason for splitting by context — a random split by "
    f"question would place the answer of {pct(leak['test_questions_with_same_fact_seen_in_train'])} of test questions in "
    "training — is explained in the revised Section 3.",
]:
    doc.add_paragraph(b, style="List Bullet")
revised(f"""
The Q&A dataset ({qa['qa_pairs']:,} pairs, {qa['contexts']} contexts) was divided by context, so that all questions about one alloy record belong to the same subset: {tr['contexts']} contexts ({tr['questions']:,} questions) for training, {va['contexts']} ({va['questions']}) for validation and {te['contexts']} ({te['questions']}) for testing (random seed 42). Splitting by question is not appropriate here, because every fact is asked on average {qa['mean_paraphrases_per_fact']:.1f} times in paraphrased form. Models were trained on the training subset, the number of epochs was chosen on the validation subset, and all reported metrics were computed once on the test subset.
""")

comment("Comment 3", "Some values quoted in the Table 1 discussion do not match the table, and the cost column needs units.")
rich([("Response: ", True), ("We thank the reviewer for spotting these inconsistencies. We checked every value quoted in "
      "the discussion against Table 1 and corrected the following:", False)])
table(["Original text", "Table 1 value", "Correction"], [
    ("“Claude Sonnet 3.5 and GPT-4 … F1 scores of 0.94 and 0.93, respectively”",
     "Claude 0.92, GPT-4 0.94 (0.93 is GPT-4’s accuracy)", "“… F1 scores of 0.92 and 0.94, respectively”"),
    ("Llama-3.1 “F1 score (47%)”", "0.47", "“F1 score of 0.47” (F1 reported as a fraction throughout)"),
    ("“GPT-4 and Claude Sonnet … accuracy approaching 94%”", "93% and 94%", "“accuracies of 93% and 94%”"),
    ("“Claude Sonnet 3.5 at 4.75 per output, GPT-4 at 0.5 per output”, “(0.005 per output)”", "no unit",
     "unit added (see below)"),
    ("Caption: “fine-tuned LLMs”; text: models “from the Hugging Face library”",
     "GPT-4, GPT-4o, o1-mini, Gemini Pro, Claude and DeepSeek V3 were used through vendor APIs",
     "Caption: “NER performance of the evaluated LLMs (n = 5 abstracts)”; Gemini Pro added to the model list in the text"),
    ("Few-shot result (“100%”) only in the text", "not in table", "Added as a row of Table 1 with its 95% CI"),
], caption="Table R3. Corrections to the Table 1 discussion", widths=[5.5, 4.5, 6.5])
para("Cost column. The column heading is changed to “Effective cost per output (US$ per abstract, API list price at "
     "the time of the experiments; input + output tokens)” and the “Output time” column to “Output time (min per "
     "abstract)”. Llama-3.1 8B is marked “0 (run locally)”. A footnote states the pricing date and that the cost "
     "is the mean over the five evaluation abstracts.")
para("Note for the authors before submission: confirm the currency and basis of the original cost values from the "
     "experiment logs/invoices; the heading above must match them.", italic=True, size=9)
revised("""
As shown in Table 1, Llama-3.1 (8B), run locally at no API cost, gives the lowest accuracy (56%) and F1 score (0.47) and the longest output time (10 min per abstract). GPT-4 and Claude Sonnet 3.5 reach the highest accuracies (93% and 94%) with output times of 3 and 2 min. Among the API models, Claude Sonnet 3.5 (US$ 4.75 per abstract) and GPT-4 (US$ 0.50 per abstract) are the most expensive, whereas GPT-4o Mini gives 92% accuracy in 1 min at US$ 0.005 per abstract and is therefore the most efficient option for resource-constrained applications. Where accuracy is paramount, GPT-4 (F1 = 0.94) and Claude Sonnet 3.5 (F1 = 0.92) are preferable. Because all values in Table 1 are averages over five abstracts, differences of a few percentage points between models are within the sampling uncertainty and should not be over-interpreted.
""")

# ---------------------------------------------------------------- Appendix
h("Appendix A. Code and data corrections in the released repository", 1)
for b in [
    "Phi2_QLoRA_Finetuning.ipynb: answer-only loss (prompt tokens masked to −100); trains on the context-level "
    "training split; evaluation on validation/test with EM, F1, BLEU and ROUGE-L.",
    "evaluation/: make_splits.py, qa_validation.py, ner_checks.py, retrieval_eval.py, gen_eval.py and run_all.sh "
    "reproduce every number in this response; outputs in evaluation/results/.",
    "app.py: cosine-similarity retrieval (normalised embeddings, inner-product FAISS index) with hybrid lexical "
    "re-ranking; short answers such as “3346 MPa” are no longer discarded (answers under five words were previously "
    "replaced by “couldn’t find”); background Q&A generation no longer blocks on an unread output pipe; API keys are read "
    "from environment variables.",
    "qa_pair_gen.py: the rating parser accepts the format requested in the prompt (“Rating: n”), which was previously "
    "parsed as missing and defaulted to 1; the unreachable “rating == 0” filter condition was removed.",
    "requirements.txt: added missing packages (flask, flask-session, peft, faiss-cpu) and removed CUDA wheels that "
    "were incompatible with the pinned torch version.",
]:
    doc.add_paragraph(b, style="List Bullet")

h("Appendix B. Remaining actions for the authors", 1)
for b in [
    "Confirm that the five NER evaluation abstracts were distinct from the few-shot prompt examples, and that the "
    "matching rule stated in Section 2.1 is the one used.",
    "Confirm the currency/basis of the Table 1 cost values (Reviewer 2, Comment 3).",
    "Complete the manual audit sheets (evaluation/results/qa_manual_audit_sheet.csv, 66 Q&A pairs; "
    "ner_manual_audit_sheet.csv, 30 alloy records) with two raters and insert the resulting rates and Cohen’s κ.",
    "Optionally extend the NER gold set beyond five abstracts (≥ 30 recommended) to narrow the confidence interval.",
    "Update the dataset links in the manuscript to the current repository.",
]:
    doc.add_paragraph(b, style="List Bullet")

out = ROOT / "docs" / "Response_to_Reviewers.docx"
doc.save(out)
print(out)
