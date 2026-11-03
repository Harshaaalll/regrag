# Research notes: evidence gate and abstention

Written 2026-11-04. Topic: how to detect unsupported answers in RAG and when to abstain, for regulatory documents.

**How much to trust this file**
- The list comes from one WisPaper Deep Search (wispaper.ai, 96 results, query saved in WisPaper history). The "why it matches" lines are WisPaper's own summaries of each paper's abstract, shortened by me. **Unverified** unless marked "checked".
- "Checked" means I confirmed the paper exists and its main idea from the arXiv listing or search snippets. I have **not** read any full paper yet. No paper's numbers are quoted here.
- WisPaper's AI Survey and mind map are locked behind a Pro plan, so I did not use them. The survey below is my own short synthesis of the results, not a WisPaper output.
- Most results are 2026 preprints with 0 citations. Treat them as leads, not established methods.

## Reading list (ranked for RegRAG)

| # | Paper | Why it matches RegRAG | Status |
|---|---|---|---|
| 1 | [Citation-Enforced RAG for Fiscal Document Intelligence](https://arxiv.org/abs/2603.14170) (Shanivendra, 2026) | Closest to our setup: official tax documents, every paragraph must cite document, page and chunk, plus abstention driven by retrieval confidence. | checked (abstract only) |
| 2 | [VeriCite: reliable citations via rigorous verification](https://arxiv.org/abs/2510.11394) (Qian et al., SIGIR-AP 2025) | Three steps: draft, check each claim against passages with an NLI model, then write with citations. Same idea as our NLI gate. Authors report that removing the NLI step hurts citation quality; I have not verified their numbers. Code is public. | checked (abstract/summaries) |
| 3 | [Fine-grained Claim-level RAG Benchmark for Law (ClaimRAG-LAW)](https://arxiv.org/abs/2605.21071) (Das et al., 2026) | Evaluates legal RAG at the level of single claims, which is how our gate works. Synthetic dataset (GDPR and a French civil code), so limited transfer. | checked (abstract only) |
| 4 | Evidence Reliability for Selective Abstention in RAG (Zenodo, 2026) | WisPaper says: abstain using evidence reliability and confidence scores, pipeline includes NLI faithfulness checking. Not regulatory. Preprint on Zenodo, low confidence. | unverified |
| 5 | A Comparative Evaluation of Hallucination Detection Methods for RAG Systems (Lund Univ. student paper, 2026) | WisPaper says: compares LLM-as-judge with specialised classifiers, on regulatory documents. This is almost our gate ablation. Student thesis, low confidence. | unverified |
| 6 | Retromorphic Testing with Hierarchical Verification for Hallucination Detection in RAG (arXiv, 2026) | Breaks an answer into claims and checks faithfulness to context, like our `split_claims`. | unverified |
| 7 | FAIR-RAG: Faithful Adaptive Iterative Refinement (arXiv, 2025) | Structured "is the evidence enough?" step that finds gaps and retries. Matches our agentic evaluate and refine loop. | unverified |
| 8 | Hallucination-free? Reliability of leading AI legal research tools (Magesh et al., J. Empirical Legal Studies 2025, 850+ citations) | Shows RAG tools in law still hallucinate, which is the argument for a gate. Does not cover abstention. | unverified (well known) |
| 9 | How Much Do Legal RAG Systems Still Hallucinate? (Das et al., 2026) | Companion to #3, detailed evaluation of legal RAG hallucination. | unverified |
| 10 | Evidence-Aligned Entity Verification for Hallucination Detection in RAG (ACL Findings 2026) | A cheap check: do entities in the answer (numbers, names) appear in the evidence? Fits RBI answers full of amounts and dates. | unverified |
| 11 | Legal RAG Error Decomposition (2026) | Separates retrieval failure from reasoning failure and hallucination risk. Useful for explaining why a question was abstained. | unverified |
| 12 | Retrieval Quality, Evidence, and Appropriate Abstention in LLMs (2026) | Links retrieval quality to when abstention is right. Low confidence preprint. | unverified |

Left out on purpose: papers about model internals (ReDeEP, LLM-Check, Steer LLM Latents). They need access to hidden states, which Ollama does not expose, and they target other domains.

## Short survey (my synthesis, unverified)

Source links are the papers above.

1. **Retrieval-side abstention (cheap).** Refuse before generating when retrieval looks weak, for example a low top reranker score (#1, #4, #12). Pro: no extra model call. Con: a good-looking passage can still fail to contain the answer. Maps to our unbuilt `reranker_threshold` mode.
2. **Claim-level entailment (our current gate).** Split the answer into claims and test each against the cited text with an NLI model (#2, #6). Pro: shows which claim is unsupported. Con: NLI models struggle with long passages and numbers.
3. **LLM-as-judge fallback.** Used where NLI is unsure, or alone, compared with classifiers (#5). Pro: handles reasoning. Con: slow on a CPU, and the judge can be wrong.
4. **Citation checking.** Check that each cited chunk really supports the sentence that cites it (#2, #1). This is a different failure from "claim is false".
5. **Iterative evidence checking.** Decide if the evidence is enough, retry retrieval if not (#7). We already have this in the agentic retriever.
6. **Evaluation at claim level.** Score answers by supported and unsupported claims instead of one pass/fail (#3, #11). Our current benchmark only counts abstain accuracy and cited-page hits.

Gap in the literature I saw (from titles and summaries only): few papers combine NLI plus judge fallback with abstention on **regulatory** documents, and none I found use Indian banking rules. That is the room for RegRAG, but I have not searched deeply enough to say nobody has done it.

```
Evidence gate / abstention
├─ Before answering ── retrieval confidence (#1 #4 #12)
├─ After answering
│  ├─ claim entailment (NLI) (#2 #6)
│  ├─ LLM judge fallback (#5)
│  ├─ entity/number match (#10)
│  └─ citation correctness (#1 #2)
├─ Retry loop ── evidence sufficiency (#7)
└─ Measuring it ── claim-level metrics (#3 #9 #11)
```

## Three candidate ideas (pick one to rebuild small)

**A. Retrieval-confidence pre-gate (`reranker_threshold` mode).** Abstain when the best reranker score is below a threshold tuned on the golden set. Source: #1, #4. Small: one class and a config value. It completes the planned ablation (none / reranker-threshold / NLI+judge / LLM-only). Needs the reranker in memory, so it must run in a separate stage.

**B. NLI claim-to-citation linking.** Instead of trusting the model's `[n]` markers, use NLI to find which retrieved chunk entails each claim, then fix or drop wrong citations. Source: #2. Medium size. Targets citation correctness, which matters for the "citation or abstain" guardrail.

**C. Claim-level scoring and number/entity check.** Add per-answer metrics (share of supported claims) and a cheap check that every number, date and amount in the answer appears in the cited text. Source: #3, #10, #11. Small, needs no extra model, and runs offline in the unit tests. It also gives the gate ablation a better metric than abstain accuracy.

Suggested order if all three are wanted: C, then A, then B. Before building, read the full paper behind the chosen idea and check the authors' code where it exists (#2 has public code).
