"""
Merge an agent-authored prediction into an existing simulation JSON file.

The simulation script (simulate_race.py) produces model predictions only.
This helper layers an "agent_prediction" entry on top, populated externally
(typically by Claude after web-searching current form). Running the simulation
directly never invokes this script — model predictions stand on their own.

Schema for the prediction payload (JSON):
{
  "model": "claude-opus-4-7",
  "rationale": "Markdown text explaining the call",
  "context": ["bullet", "facts", "from web search"],
  "podium": [
    {"rank": 1, "driver_id": 857, "name": "Oscar Piastri", "team": "McLaren", "abbreviation": "PIA"},
    {"rank": 2, ...},
    {"rank": 3, ...}
  ],
  "top10": [...]   // optional; same shape as podium entries, ranks 1-10
}

Usage:
    uv run python add_agent_prediction.py \\
        --sim data/sim_2026_miami.json \\
        --prediction agent_pred_miami.json

The merged JSON gains a top-level "agent_prediction" key. The dashboard
checks for its presence and renders accordingly. Re-running overwrites
the previous agent prediction (use --no-overwrite to refuse).
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


REQUIRED_PODIUM_FIELDS = {"rank", "name", "team"}


def validate_payload(payload):
    if "podium" not in payload or not isinstance(payload["podium"], list):
        raise ValueError("payload must contain a 'podium' list")
    if len(payload["podium"]) != 3:
        raise ValueError(f"podium must have 3 entries, got {len(payload['podium'])}")
    for i, entry in enumerate(payload["podium"]):
        missing = REQUIRED_PODIUM_FIELDS - set(entry.keys())
        if missing:
            raise ValueError(f"podium[{i}] missing fields: {missing}")
    if "top10" in payload:
        if not isinstance(payload["top10"], list):
            raise ValueError("top10 must be a list")
        if len(payload["top10"]) > 10:
            raise ValueError(f"top10 has {len(payload['top10'])} entries, max 10")


def merge(sim_path, prediction_payload, overwrite=True):
    sim_path = Path(sim_path)
    if not sim_path.exists():
        raise FileNotFoundError(sim_path)

    with open(sim_path) as f:
        sim = json.load(f)

    if "agent_prediction" in sim and not overwrite:
        raise RuntimeError(
            f"{sim_path.name} already has an agent_prediction; "
            f"pass overwrite=True or use --force"
        )

    validate_payload(prediction_payload)

    sim["agent_prediction"] = {
        "name": prediction_payload.get("name", "Agent Forecast (Claude)"),
        "description": prediction_payload.get(
            "description",
            "Predicted by Claude using web-searched current form, "
            "prior-season results, and qualitative judgment.",
        ),
        "model": prediction_payload.get("model", "claude-opus-4-7"),
        "generated_at": prediction_payload.get(
            "generated_at",
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
        ),
        "rationale": prediction_payload.get("rationale", ""),
        "context": prediction_payload.get("context", []),
        "podium": prediction_payload["podium"],
    }
    if "top10" in prediction_payload:
        sim["agent_prediction"]["top10"] = prediction_payload["top10"]

    with open(sim_path, "w") as f:
        json.dump(sim, f, indent=2)

    return sim_path


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sim", required=True, help="Path to sim_*.json")
    parser.add_argument("--prediction", required=True, help="Path to agent prediction JSON")
    parser.add_argument("--force", action="store_true", help="Overwrite existing agent_prediction")
    args = parser.parse_args()

    with open(args.prediction) as f:
        payload = json.load(f)

    out = merge(args.sim, payload, overwrite=True if args.force else True)
    print(f"Agent prediction merged into {out}")


if __name__ == "__main__":
    main()
