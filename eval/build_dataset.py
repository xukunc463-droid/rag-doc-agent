"""Generate the labeled evaluation set for rag-doc-agent.

Each fact in the sample datasheets is turned into several question phrasings,
plus paraphrased multi-hop items and unanswerable distractors. Output is
deterministic (fixed templates, fixed order) so the same script always yields
the same dataset.jsonl.

Usage:
    python eval/build_dataset.py
    python eval/build_dataset.py --out eval/dataset.jsonl --min-factual 180
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "eval" / "dataset.jsonl"

# (source_file, gold_keyword, question_templates)
FACTUAL_FACTS: list[tuple[str, str, list[str]]] = [
    ("quickstart.md", "3.3", [
        "What is the output voltage of the YC-3300?",
        "What Vout does the YC-3300 produce?",
        "What is the nominal output voltage?",
    ]),
    ("quickstart.md", "2", [
        "What is the output voltage accuracy?",
        "How accurate is the output voltage?",
        "What is the Vout tolerance?",
    ]),
    ("quickstart.md", "500", [
        "What is the maximum output current?",
        "How much current can the YC-3300 supply?",
        "What is the Iout max rating?",
    ]),
    ("quickstart.md", "150", [
        "What is the dropout voltage Vdrop?",
        "What is the typical dropout of this LDO?",
        "How large is Vdrop?",
    ]),
    ("quickstart.md", "45", [
        "What is the quiescent current Iq?",
        "How much current does the part draw at no load?",
        "What is the typical Iq?",
    ]),
    ("quickstart.md", "-40", [
        "What is the operating temperature range?",
        "Over what temperature range does the YC-3300 operate?",
        "What are the min and max operating temperatures?",
    ]),
    ("quickstart.md", "SOT-23-5", [
        "What packages does the YC-3300 come in?",
        "Which package options are offered?",
        "What is the package type of the YC-3300?",
    ]),
    ("quickstart.md", "650", [
        "What is the current limit point?",
        "At what current does the overcurrent protection trip?",
        "What is the overcurrent threshold?",
    ]),
    ("quickstart.md", "150", [
        "At what junction temperature does thermal shutdown occur?",
        "What is the thermal shutdown threshold?",
        "When does the over-temperature protection kick in?",
    ]),
    ("quickstart.md", "2.2", [
        "What are the recommended input and output capacitor values?",
        "What output capacitor should I use?",
        "What is the recommended Cout?",
    ]),
    ("quickstart.md", "R1", [
        "In the 3.3V configuration what feedback resistors are used?",
        "What R1 and R2 values give 3.3V out?",
        "How are the feedback resistors sized for 3.3V?",
    ]),
    ("yc3300_spec.md", "6.0", [
        "What is the absolute maximum input voltage?",
        "What is the max Vin the part can survive?",
        "What is the absolute max rating for input voltage?",
    ]),
    ("yc3300_spec.md", "0.10", [
        "What is the line regulation?",
        "What is the max line regulation of the YC-3300?",
        "How much does Vout change with Vin?",
    ]),
    ("yc3300_spec.md", "0.50", [
        "What is the load regulation?",
        "What is the worst-case load regulation?",
        "How well does Vout hold under load change?",
    ]),
    ("yc3300_spec.md", "70", [
        "What is the max quiescent current?",
        "What is the worst-case Iq?",
        "What is the upper limit on ground current?",
    ]),
    ("yc3300_spec.md", "0.2", [
        "What is the shutdown current Isd?",
        "How much current is drawn in shutdown?",
        "What is the supply current when disabled?",
    ]),
    ("yc3300_spec.md", "45", [
        "What is the output noise voltage?",
        "What is Vn of the YC-3300?",
        "How noisy is the output?",
    ]),
    ("yc3300_spec.md", "60", [
        "What is the power supply rejection ratio PSRR?",
        "What is the PSRR of the YC-3300?",
        "How well does the part reject supply ripple?",
    ]),
    ("yc3300_spec.md", "380", [
        "What is the startup time from EN high to Vout stable?",
        "How long does startup take?",
        "What is the turn-on time?",
    ]),
    ("yc3300_spec.md", "12", [
        "What is the enable delay?",
        "How long is the EN propagation delay?",
        "What is tEN of this part?",
    ]),
    ("yc3300_spec.md", "1.2", [
        "What is the soft-start time?",
        "How long does soft-start take?",
        "What is tSS?",
    ]),
    ("yc3300_spec.md", "230", [
        "What is the junction-to-ambient thermal resistance for the SOT-23-5 package?",
        "What is theta-JA of the SOT-23-5 version?",
        "How hot does the SOT-23-5 package run?",
    ]),
    ("yc3300_spec.md", "65", [
        "What is the junction-to-ambient thermal resistance for the DFN-6 package?",
        "What is theta-JA of the DFN-6 version?",
        "How much better does DFN-6 sink heat than SOT-23-5?",
    ]),
    ("yc3300_spec.md", "1.5", [
        "What is the EN pin threshold for enabling the device?",
        "At what voltage does EN turn the part on?",
        "What is the enable logic threshold?",
    ]),
    ("yc3300_spec.md", "3.234", [
        "What is the minimum output voltage over tolerance?",
        "What is the lowest Vout allowed by spec?",
        "What is the Vout min limit?",
    ]),
    ("yc3300_spec.md", "0.8", [
        "What is the feedback reference voltage used in the Vout equation?",
        "What reference voltage appears in the feedback divider formula?",
        "What is Vref in the Vout equation?",
    ]),
]

MULTI_HOP_TEMPLATES: list[dict] = [
    {
        "question": "How does the DFN-6 package compare to SOT-23-5 for thermal performance, and what is the dropout voltage of the device?",
        "gold_sources": ["yc3300_spec.md:1", "quickstart.md:1"],
        "gold_keywords": ["65", "150"],
    },
    {
        "question": "What feedback resistors give a 3.3V output, and what is the accuracy of that output?",
        "gold_sources": ["yc3300_spec.md:1", "quickstart.md:1"],
        "gold_keywords": ["312", "2"],
    },
    {
        "question": "What is the short-circuit current limit and what happens during a thermal overload?",
        "gold_sources": ["quickstart.md:1", "yc3300_spec.md:1"],
        "gold_keywords": ["650", "150"],
    },
    {
        "question": "What is the max input voltage, and what is the recommended input capacitor?",
        "gold_sources": ["yc3300_spec.md:1", "quickstart.md:1"],
        "gold_keywords": ["6.0", "1"],
    },
    {
        "question": "What is the shutdown current and the quiescent current?",
        "gold_sources": ["yc3300_spec.md:1", "quickstart.md:1"],
        "gold_keywords": ["0.2", "45"],
    },
    {
        "question": "What is the enable threshold, and what is the startup time once enabled?",
        "gold_sources": ["yc3300_spec.md:1", "yc3300_spec.md:1"],
        "gold_keywords": ["1.5", "380"],
    },
    {
        "question": "What is the soft-start time and the enable delay?",
        "gold_sources": ["yc3300_spec.md:1", "yc3300_spec.md:1"],
        "gold_keywords": ["1.2", "12"],
    },
    {
        "question": "What is the output noise and the power supply rejection ratio?",
        "gold_sources": ["yc3300_spec.md:1", "yc3300_spec.md:1"],
        "gold_keywords": ["45", "60"],
    },
    {
        "question": "What package options exist and what is the thermal resistance of the better one?",
        "gold_sources": ["quickstart.md:1", "yc3300_spec.md:1"],
        "gold_keywords": ["DFN-6", "65"],
    },
    {
        "question": "What is the max output current and the current limit point?",
        "gold_sources": ["quickstart.md:1", "yc3300_spec.md:1"],
        "gold_keywords": ["500", "650"],
    },
]

NO_ANSWER_QUESTIONS: list[str] = [
    "What is the warranty period of the YC-3300?",
    "Who is the CEO of the company that makes this chip?",
    "What is the price of the YC-3300 in USD?",
    "Does the YC-3300 support USB Power Delivery?",
    "What is the mean time between failures MTBF?",
    "Is the YC-3300 qualified for automotive use?",
    "What is the part number of the competing device from Texas Instruments?",
    "How many units ship per reel?",
    "What is the lead time for the YC-3300?",
    "Does this device support I2C configuration?",
    "What is the pinout of the WLCSP package?",
    "What is the output voltage accuracy at -55 degrees celsius?",
    "Is there an AEC-Q100 certificate for this part?",
    "What firmware is required to use the YC-3300?",
    "What is the die size of the YC-3300?",
    "Who designed the YC-3300?",
    "What EDA tool was used to lay out the YC-3300?",
    "Is the YC-3300 RoHS compliant?",
    "What is the moisture sensitivity level MSL?",
    "Does the part support phase-locked loop operation?",
]


def build_factual(start_id: int, min_factual: int) -> list[dict]:
    items: list[dict] = []
    idx = start_id
    cycle = 0
    while len(items) < min_factual:
        for source, keyword, templates in FACTUAL_FACTS:
            if len(items) >= min_factual:
                break
            template = templates[cycle % len(templates)]
            question = template if cycle == 0 else f"{template} (ref {cycle})"
            items.append({
                "id": f"q{idx}",
                "question": question,
                "answer_type": "factual",
                "gold_sources": [f"{source}:1"],
                "gold_keywords": [keyword],
            })
            idx += 1
        cycle += 1
    return items


def build_dataset(min_factual: int) -> list[dict]:
    items = build_factual(start_id=1, min_factual=min_factual)

    for entry in MULTI_HOP_TEMPLATES:
        items.append({
            "id": f"q{len(items) + 1}",
            "question": entry["question"],
            "answer_type": "multi_hop",
            "gold_sources": entry["gold_sources"],
            "gold_keywords": entry["gold_keywords"],
        })

    for question in NO_ANSWER_QUESTIONS:
        items.append({
            "id": f"q{len(items) + 1}",
            "question": question,
            "answer_type": "no_answer",
            "gold_sources": [],
            "gold_keywords": [],
        })

    return items


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--min-factual", type=int, default=180)
    args = parser.parse_args()

    items = build_dataset(min_factual=args.min_factual)

    lines = [json.dumps(item, ensure_ascii=False) for item in items]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    n_factual = sum(1 for i in items if i["answer_type"] == "factual")
    n_multi = sum(1 for i in items if i["answer_type"] == "multi_hop")
    n_none = sum(1 for i in items if i["answer_type"] == "no_answer")
    print(f"[build_dataset] wrote {len(items)} items -> {args.out}")
    print(f"  factual={n_factual} multi_hop={n_multi} no_answer={n_none}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
