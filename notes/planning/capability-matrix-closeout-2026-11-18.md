# Capability publication closeout — 2026-11-18

**Existing item 34.1 is complete.** This publishes an evidence-based support map;
it does not implement unfinished features or close their parent requirements.

The [public matrix](../../docs/api/capability-matrix.md) and its
[curated records](../../docs/api/capabilities.json) cover:

- **50 feature families:** 26 verified within explicitly bounded contracts,
  19 partial, one native-only, two not implemented for their target and two
  deferred families. These are feature classifications, not completion percentages.
- **All 105 declared Python exports**, explicitly mapped to their capability row.
  A discovered method/class is not considered implemented merely because it exists.
- **All 268 stable checklist rows**, with no unclassified IDs or new checklist rows.
- Separate Rust/Python surfaces and implementation sources, applicable persistence,
  native/Python tests, contract documents, limits and remaining original-vision work.

The runtime/evidence pin is **cfc5628c53fee47b582418d3e4a2bdd395434b37**: 1,633 native
passed / 54 ignored; separate serial recovery and strict Clippy; fresh-wheel
3,385 maintained Python passed / one skipped / 14 established warnings and 290
executable documentation examples. Source existence is never the correctness proof.
The matrix identifies incomplete service/provenance/advanced execution, general
complex/Conv2D/accelerator domains, two-level Infomap, native-only layouts and
production certification rather than implying those are done.

## Reproducibility and drift guard

`python notes/scripts/build_capability_matrix.py --check` validates curated
statuses and API/test/contract references, full public-export/checklist coverage,
and exact generated Markdown. It uses only the standard library and does not
import Groggy or require a built extension. `--verify-runtime` additionally checks
production source bytes against the pinned qualification receipt. Paths are
repository-confined and Unicode files are read as UTF-8.

The validation command is added to the existing test and release native gates.
It does not install dependencies or add a separate scheduled workflow. Later
runtime changes are not automatically credited with the pinned evidence; broader
release qualification and documentation updates remain their existing gates.

Positive generation, exact-output checking and production-source verification
pass. Six negative trials reject duplicate feature IDs, unknown public exports,
missing evidence, nonexistent checklist prefixes, unsupported targets marked
verified and stale generated output. These exercise the validator rather than
claim graph functionality through documentation tests.

## Corrected stale claims and discoverability

The README/docs index now link the matrix and describe all 13 classic registered
algorithms. Unsupported universal pipeline/FFI speedups are removed from those
entry points. The old 2025 method/document completion report is explicitly
historical; the roadmap has a current checkpoint and marks earlier count/manual-
default/TODO observations as historical. No older evidence or owner answers are
deleted. Complete library-wide prose/example cleanup stays open under **34.2**.

## Progress

This slice closes **34.1**, following the independently completed **AR7.6**.
The fixed inventory is **217 checked / 51 unchecked / 268 total**;
**42 unchecked expanded-release rows and nine deferred** remain. No new rows,
policy decisions, original-feature retirements or production-readiness claims.
All 497 production source files remain byte-identical to the qualified runtime;
this slice changes documentation, its validator and validation wiring only.
