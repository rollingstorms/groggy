# Activation release receipts — 2026-10-22

Locked offline release build; idle runner, one warmup/seven samples, input setup
excluded and output creation/drop included. **55 cases**; prior 52 checksums match
within 1e-13. Heap figures are System requests/additional live bytes, not RSS.

| Operation | Logical shape | Median | Additional live heap | Allocation calls |
| --- | --- | ---: | ---: | ---: |
| GELU | 100,000 × 1 dense | 1.888 ms | 1,600,533 bytes | 14 |
| ReLU | 100,000 × 100,000 diagonal CSR | 12.251 ms | 10,354,922 bytes | 290,054 |
| Seeded dropout | 100,000 × 100,000 diagonal CSR | 12.568 ms | 9,306,346 bytes | 290,052 |

CSR cases assert sparse output and never allocate the ten-billion-cell logical
grid. Receipts include preserving 100,000 column-name strings, which account for
many small allocations; this is not a claim of an allocation-free sparse map.
Further metadata optimization and deployment budgets remain separate. This is a
current bounded receipt, not an old/new timing comparison or a production SLA.

Receipt: `benches/numerical_activations_2026-10-22.json`.
