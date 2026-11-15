"""Verify the maintained root Python surface and curated executable API examples.

Uses the active in-tree extension. Historical nested surveys and the separately
maintained algorithm comparison benchmark are outside this compatibility gate.
Run from any directory: python notes/scripts/verify_immutable_view_migration.py
"""
from pathlib import Path
import os
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
DOCS = ['docs/immutable-read-views.md',
 'docs/api/python-owner-lifetimes.md',
 'docs/api/terminal-topology-pipelines.md',
 'docs/api/staged-graph-captures.md',
 'docs/api/staged-topology-additions.md',
 'docs/api/schema-migration.md',
 'docs/api/pipeline-endpoint-revisions.md',
 'docs/api/captured-pipeline-degrees.md',
 'docs/api/display-foundation.md',
 'docs/api/typed-pipeline-values.md',
 'docs/api/float64-properties.md',
 'docs/api/property-numeric-predicates.md',
 'docs/api/float64-numerical-results.md',
 'docs/api/table-profiles.md',
 'docs/api/table-numeric-dispatch.md',
 'docs/api/table-interchange.md',
 'docs/api/table-outliers.md',
 'docs/api/graph-query-literals.md',
 'docs/api/float-consumer-migration.md',
 'docs/api/float64-defaults.md',
 'docs/api/temporal-properties.md',
 'docs/api/explicit-checkpoints.md',
 'docs/api/automatic-publication.md',
 'docs/api/publishing-writes.md',
 'docs/api/property-values.md',
 'docs/api/durable-storage.md',
 'docs/api/meta-composition.md',
 'docs/api/table-writeback.md',
 'docs/api/subgraph-writes.md',
 'docs/api/table-materialization.md',
 'docs/api/transient-builder-outputs.md',
 'docs/api/atomic-builder-properties.md',
 'docs/api/builder-aggregations.md',
 'docs/api/builder-transformations.md',
 'docs/api/builder-sampling.md',
 'docs/api/builder-community.md',
 'docs/api/builder-subgraph-sampling.md',
 'docs/api/builder-foreach.md',
 'docs/api/builder-paths.md',
 'docs/api/builder-temporal.md',
 'docs/api/builder-temporal-diffs.md',
 'docs/api/builder-temporal-windows.md',
 'docs/api/composer-options.md',
 'docs/api/graph-copy.md',
 'docs/api/pinned-view-writes.md',
 'docs/api/table-source-pins.md',
 'docs/api/table-joins.md',
 'docs/api/table-row-operations.md',
 'docs/api/table-collection-reductions.md',
 'docs/api/graph-table-merge.md',
 'docs/api/table-collection-ownership.md',
 'docs/api/table-reshaping.md',
 'docs/api/graph-table-bundles.md',
 'docs/api/entity-table-adapters.md',
 'docs/api/legacy-table-exports.md',
 'docs/api/table-schema-invariants.md',
 'docs/api/graph-table-components.md',
 'docs/api/table-visualization.md',
 'docs/api/checked-table-conversions.md',
 'docs/api/effective-time-reads.md',
 'docs/api/property-history.md',
 'docs/api/owned-snapshots.md',
 'docs/api/atomic-pipelines.md',
 'docs/api/branch-state-reads.md',
 'docs/api/history-retention.md',
 'docs/api/graph-archives.md',
 'docs/api/storage-checkpoints.md',
 'docs/api/native-mutation-defaults.md',
 'docs/api/retained-integrity.md',
 'docs/api/pool-compaction.md',
 'docs/api/storage-verification.md',
 'docs/api/retained-purge.md',
 'docs/api/endpoint-differences.md',
 'docs/api/endpoint-revisions.md',
 'docs/api/state-restoration.md',
 'docs/api/graph-merges.md',
 'docs/api/change-records.md',
 'docs/api/write-coordinates.md',
 'docs/api/query-indexes.md',
 'docs/api/ordered-pinned-queries.md',
 'docs/api/graph-patterns.md',
 'docs/api/builder-optimization-order.md',
 'docs/api/builder-power.md',
 'docs/api/builder-elementary-math.md',
 'docs/api/effective-time-indexes.md',
 'docs/api/property-intervals.md',
 'docs/api/group-membership-indexes.md',
 'docs/api/query-capture.md',
 'docs/api/selected-property-scope.md',
 'docs/api/current-read-sharing.md',
 'docs/api/captured-community.md',
 'docs/api/captured-view-algebra.md',
 'docs/api/captured-view-traversal.md',
 'docs/api/structural-summaries.md',
 'docs/api/captured-view-entities.md',
 'docs/api/connected-column-alignment.md',
 'docs/api/connected-degree-reads.md',
 'docs/api/specialized-selections.md',
 'docs/api/current-full-view-cache.md',
 'docs/api/view-node-sampling.md',
 'docs/api/view-adjacency-export.md',
 'docs/api/connected-matrix-scope.md',
 'docs/api/selected-graph-copy.md',
 'docs/api/view-visualization.md',
 'docs/api/default-immutable-views.md']


def main():
    files = [str(path) for path in sorted((ROOT / "tests").glob("test_*.py"))
             if path.name != "test_algorithm_benchmark.py"]
    subprocess.run([sys.executable, "-m", "pytest", "-q", *files], cwd=ROOT, check=True)
    count = 0
    previous = Path.cwd()
    try:
        with tempfile.TemporaryDirectory(prefix="groggy-immutable-view-docs-") as work:
            os.chdir(work)
            for name in DOCS:
                scope = {"__name__": "__main__"}
                blocks = re.findall(r"```python\n(.*?)```", (ROOT / name).read_text(), re.S)
                for block in blocks:
                    exec(compile(block, name, "exec"), scope)
                    count += 1
                print(f"{name}: {len(blocks)} examples")
    finally:
        os.chdir(previous)
    print(f"PASSED {count} documentation examples")


if __name__ == "__main__":
    main()
