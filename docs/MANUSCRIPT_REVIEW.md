# Manuscript Review Snapshot

This tracked note mirrors the local review checklist written into
`paper/FINAL_DRAFT_TODO.md`. It exists so the repository retains the manuscript
audit even though `paper/` itself is gitignored.

Reviewed sources:

- `paper/final_draft.pdf`
- `paper/2026-03-25-paper-ACM-draft.pdf`
- `paper/At least 50 references.docx`

## Highest-Priority Remaining Fixes

- Expand the bibliography to at least 50 references, including more ACM MM and
  adjacent multimedia retrieval citations.
- Replace the generic `Anonymous Author(s)` placeholder if the submission mode
  requires blinded names rather than the anonymous default.
- Remove the empty appendix heading (`A Research Methods`) unless the real
  appendix is inserted immediately afterward.
- Fix the known abstract spacing defects and the broken Figure 4 caption text.
- Correct the visible conceptual-figure artwork typos:
  `Learned Paritioning`, `T raining`, and `A verage`.

## Narrative / Conceptual Checks

- Keep the production framing explicit: recall is unavailable during serving, so
  entropy is the runtime proxy for ambiguity.
- Keep the single-backbone contrast explicit against BLISS-style repeated
  structures and LIRA's heavier local-index design.
- Avoid over-claiming the default low-cap AdaptIVF setting; the strongest
  competitiveness story in the finished run is carried by the `m80` variants.
- If ACM MM page pressure becomes acute, trim methodology detail before trimming
  results or systems evidence.

## Export Alignment Already Completed In-Repo

- Main-paper assets now use `main_*` prefixes only.
- Supplementary figures now use `appendix_*` prefixes only.
- Main tables now live under:
  - `tables/main_datasets_table.csv`
  - `tables/main_baselines_table.csv`
  - `tables/main_ablation_summary_table.csv`
- Supporting numeric tables now live under:
  - `tables/appendix_initialization_summary_table.csv`
  - `tables/appendix_competitiveness_summary_table.csv`
- The maintained supplementary-material source now lives at `docs/appendix.tex`.
