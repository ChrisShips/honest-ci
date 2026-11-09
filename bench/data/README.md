# Published labels

Each directory is one benchmark run's labels, as `python -m bench.label
export` writes them. The columns, and what each label means, are in
[bench/README.md](../README.md#labelling-hits).

## 20260701-20260827_n300_stars10_cap3_seed1

The run the top-level README quotes: pull requests opened from 1 July to 26
August 2026 in repositories with at least 10 stars, up to 300 per coding agent
and 3 per repository in an agent's group (seed 1), with pull requests people
opened in the same repositories at the same time as controls. 2,717 pull
requests: 1,598 from agents, 1,119 from people. It ended on 27 September 2026,
with the detector as it was before the fixes in v0.1.0 (source digest
`dff3f19c90a59bda6c6a32ecb15097574cef6d30faab93c59c60122b9018dce4`).

One person decided all 345 labels, one flag at a time, the first 30 without
seeing a suggestion. Codex drafted a label and a reason for every flag first.
Where the person's labels matched Codex's, its reason is kept (`reason_by` is
`suggestion`); where they did not, the person wrote the reason (`labeller`).
