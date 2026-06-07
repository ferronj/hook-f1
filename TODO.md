# TODO

Running list of follow-ups for the F1 Markov model project.

## Open

### Roster JSON regeneration when 2026 lineup changes

`data/roster_2026.json` was generated from `config_2026.DRIVERS_2026` and is currently used only for R1 (where `build_roster`'s prev-race fallback can't see 2026-only changes like Cadillac/Lindblad). If a mid-season swap changes `config_2026.DRIVERS_2026`, the JSON needs to be regenerated. One-liner:

```
uv run python -c "import json,sys; sys.path.insert(0,'.'); from config_2026 import DRIVERS_2026 as D, CONSTRUCTOR_NAMES as C; json.dump({str(d):{'constructor_id':c,'name':n,'abbreviation':a,'constructor_name':C[c]} for d,(c,n,a) in D.items()}, open('data/roster_2026.json','w'), indent=2)"
```

Worth folding into `generate_2026_data.py` if it gets touched again.

## Done

- [x] Remove Stage 8 from active prediction pipeline (kept as Stage 9 dep)
- [x] Add agent forecast scaffolding (`add_agent_prediction.py` + dashboard rendering)
- [x] Renumber 2026 calendar after Bahrain/Saudi cancellations
- [x] Dashboard: actuals vs. prediction comparison (per-race scoring of every model + agent on top-3 hits / top-10 overlap / MAE / Spearman ρ)
- [x] Backfill R1–R4 sim JSONs without Stage 8 (added `--roster-replace`, `data/roster_2026.json`, season-aware name normalization in `build_roster`)
- [x] Season summary view (sidebar switcher: per-race scorecard with metric selector + line chart + table + season totals; next-race preview card with composite podium and agent forecast)
