# Gemma impact verification audit log

Append-only. This log is for impact evidence; it does not record prediction verdicts.

| Audit ID | Date | Author | Target / Claim | Gates | Oracle Class | Novelty | Verdict | Evidence |
|---|---|---|---|---|---|---|---|---|
| AUDIT-001 | 2026-11-23 | Impact analysis | Gemma 270M descriptive replay and deployment scenarios | G4, G8 | exact, sampled | N0, N1 | PARTIAL/FRAGILE | Arithmetic replays from the rounded transcription. Raw claims lack retrieved files, sample uncertainty is limited, and the impact plan is explicitly post-data. No realized deployment savings are measured. |
| AUDIT-002 | 2026-11-23 | Impact analysis | Mutation coverage of the synthetic controls (not Gemma measurements) | R5 | exact | N0 | PARTIAL/FRAGILE | Aletheia mutate found 51 sites, killed 12 and had 39 survivors; for example the decimal range guard is not directly exercised by the synthetic controls. R4 parity does not imply R5 mutation coverage. |
| AUDIT-003 | 2026-11-23 | Impact analysis | Bind impact artifacts, not the missing runner files | G8 | exact | N0 | VERIFIED | Artifact hashes bind to ratchet head `sha256:2a9df64bb059f0fc193eb0d3e7fa7f4a18895a17006f4f4e1fafd96a139d9c7b`. |
