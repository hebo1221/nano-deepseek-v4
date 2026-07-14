# P1 protected-pin causal diagnostic — S151

Date: 2026-07-14  
Decision: **S151 confirms the protected-retention mechanism**

On the exact 20 S151 instruction-persistence conversations, hierarchical 1x
without pins was prediction-identical to fixed 1x and scored 30/80. Enabling
pins scored 63/80: 34 query recoveries and one regression, for a net +33.

| Context | Fixed 1x | No pins | With pins | Queries |
|---:|---:|---:|---:|---:|
| 80 | 10 | 10 | 16 | 16 |
| 128 | 9 | 9 | 14 | 16 |
| 256 | 8 | 8 | 16 | 16 |
| 512 | 0 | 0 | 12 | 16 |
| 1024 | 3 | 3 | 5 | 16 |

Together with the S55 diagnostic, this establishes the causal path on both
pilot scales: protected pins preserve early instruction evidence; calibrated
1x quotas do not explain the gain. Multi-seed estimation is still required.

The raw SHA-256 is
`caa522e55b3c2e52c1ae21d51bed35619720bb9ffbe4a59a5a8ffda118546d1d`.
The checked result is `results/p1-protected-pin-diagnostic-s151.summary.json`.
