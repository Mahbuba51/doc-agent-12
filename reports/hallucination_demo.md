# §6 No-hallucination — the grounding gate, case by case

Cases: `tests/fixtures/hallucination_cases.jsonl`. Live assertions: `tests/test_hallucination.py`.

Queries and gold values are **real entries from `grading_kit/tasks.jsonl`** (t2, t10), not
invented scenarios. Evidence is a trimmed excerpt of the human transcription of `dolil_13`
in `grading_kit/labels.jsonl`.

**t10 is a standing hallucination trap already in the grading kit**: it asks for the annual
government khajna, which does not appear on the page, and its gold answer is
`INSUFFICIENT EVIDENCE`. Any number returned for it is fabricated by construction.

Why this matters on this corpus: `vision/ocr.py` D2 MEASURED records the reader inventing
universities and deed boilerplate rather than reporting that it could not read. On land
deeds that same failure produces an invented plot number or amount — the precision-first
NFR's worst outcome, and why abstention is treated here as a correct answer rather than a
failure to answer.

## Results

| case | task | grounded | confidence | citations | hallucination |
|---|---|---|---|---|---|
| `t10_invented_field_uncited` | t10 | `False` | 0.00 | 0 | **caught** |
| `t2_partial_hallucination` | t2 | `False` | 0.50 | 1 | **caught** |
| `t2_hallucinated_citation` | t2 | `False` | 0.00 | 0 | **caught** |
| `t2_correct_cited` | t2 | `True` | 1.00 | 1 | — |
| `t10_correct_abstention` | t10 | `False` | 0.00 | 0 | — |
| `KNOWN_GAP_invented_field_cited_to_a_real_chunk` | t10 | `True` | 1.00 | 1 | **NOT caught** |

## Cases

### `t10_invented_field_uncited`  (t10, gold `INSUFFICIENT EVIDENCE`)

Model output:
```
এই দলিলে বর্ণিত জমির বাৎসরিক সরকারি খাজনার পরিমাণ ৫০ টাকা।
```
Verdict: `grounded=False`, `confidence=0.00`, `citations=0`

THE CORE CASE. The page carries no khajna figure at all -- grading_kit gold for t10 is INSUFFICIENT EVIDENCE. A fluent invented number is exactly the failure vision/ocr.py D2 MEASURED recorded on this corpus, where the reader invented deed boilerplate rather than admitting it could not read. Caught: uncited.

### `t2_partial_hallucination`  (t2, gold `১ ১/২`)

Model output:
```
জমির পরিমাণ ১ ১/২ [dolil_13#p]। দলিলটি ১৯৭২ সালে সম্পাদিত হয়।
```
Verdict: `grounded=False`, `confidence=0.50`, `citations=1`

THE REALISTIC FAILURE: a true cited sentence carrying an invented one beside it. The date is nowhere on the page (labels.jsonl records deed_date as EMPTY BECAUSE ABSENT). Caught only because grounding requires EVERY sentence cited -- a one-citation-is-enough rule would have published this.

### `t2_hallucinated_citation`  (t2, gold `১ ১/২`)

Model output:
```
জমির পরিমাণ ১ ১/২ [dolil_999#p]।
```
Verdict: `grounded=False`, `confidence=0.00`, `citations=0`

Right answer, invented source. A plausible-looking chunk id the agent never retrieved. Caught: the citation is validated against the evidence and dropped, leaving the sentence uncited.

### `t2_correct_cited`  (t2, gold `১ ১/২`)

Model output:
```
জমির পরিমাণ ১ ১/২ [dolil_13#p]।
```
Verdict: `grounded=True`, `confidence=1.00`, `citations=1`

CONTROL. A correct, cited answer must pass. A gate that refuses this has made the system useless rather than safe.

### `t10_correct_abstention`  (t10, gold `INSUFFICIENT EVIDENCE`)

Model output:
```
INSUFFICIENT EVIDENCE
```
Verdict: `grounded=False`, `confidence=0.00`, `citations=0`

CONTROL. Abstention is the CORRECT answer here and matches grading_kit gold. Ungrounded is the right verdict -- there is nothing under it to be grounded in.

### `KNOWN_GAP_invented_field_cited_to_a_real_chunk`  (t10, gold `INSUFFICIENT EVIDENCE`)

Model output:
```
বাৎসরিক খাজনার পরিমাণ ৫০ টাকা [dolil_13#p]।
```
Verdict: `grounded=True`, `confidence=1.00`, `citations=1`

NOT CAUGHT -- asserted so the limit is tracked, not forgotten. The claim is false (no khajna on the page) but it is CITED to a chunk that was really retrieved. format_answer checks citation COVERAGE, not entailment: it proves an answer came from the evidence, never that the evidence says it. Closing this needs an entailment check (eval/judge.py JUDGE prompt), not a parser.

## What is NOT covered

`format_answer` checks citation **coverage**, not **entailment**. It proves an answer came
from the retrieved evidence; it cannot prove the evidence says it. The final case above is
that gap, asserted as a passing test so it stays tracked. Closing it needs an entailment
check — the `JUDGE` prompt in `llm/prompts.py`, run by `eval/judge.py` — not a stricter
parser.
