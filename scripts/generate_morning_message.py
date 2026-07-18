#!/usr/bin/env python3
"""
Genererar Coach Murphys korta morgonmeddelande baserat på latest.json
(samma fil som redan syncas från Intervals.icu i det här repot).

Läser: latest.json (lokal fil, samma repo)
Skriver: morning_message.json (lokal fil, committas av GitHub Actions-steget)

Kräver miljövariabeln ANTHROPIC_API_KEY (satt som GitHub Secret i workflowen).
"""

import json
import os
from datetime import datetime, timezone

from anthropic import Anthropic

MODEL = "claude-sonnet-5"

SYSTEM_PROMPT = """Du är "Coach Murphy" - en simulerad huvudcoach i Klas eget
AI-coachingsystem. Din uppgift just nu är EN sak: skriv en kort morgon-
hälsning (2-3 meningar, på svenska) som kompletterar en befintlig Home
Assistant-morgonhälsning om väder och kalender. Du ska INTE upprepa väder
eller kalenderinfo - bara ge din egen korta observation.

PRIORITETSORDNING (första matchande vinner - välj EN kategori, blanda inte):
1. VARNING - om något faktiskt är värt att vara försiktig med idag (t.ex.
   readiness-läge "modify" eller "skip", HRV tydligt under baseline, RI lågt).
   Kort, sakligt, inte alarmistiskt. Om readiness är "go" och allt ser
   normalt ut, använd INTE denna kategori - hoppa till nästa.
2. VINST/FRAMGÅNG FRÅN IGÅR - något som faktiskt gick bra (genomfört pass,
   bra sömn, HRV över baseline, ett tydligt mönster av bra vanor). Konkret,
   inte generiskt bra-jobbat.
3. FRAMÅTBLICKANDE PROMPT FÖR DAGEN - en enkel idé (implementation intention,
   en sak att bestämma i förväg) kopplad till dagens planerade pass om det
   finns ett, eller en påminnelse om något sensoriskt att söka upp om det är
   en löprunda.

TON: rak och sakligt varm, som en observant vän - inte tvingat kall, inte
peppig på ett tomt sätt. Inga klyschor ("du kan göra det!", "toppenjobbat!").
Variera formulering och vinkel från dag till dag - detta körs dagligen och
ska inte kännas som samma mall varje gång.

Svara ENDAST med själva meddelandet, ingen inledning, inga rubriker, inga
citattecken runt texten."""


def load_latest_json():
    with open("latest.json", "r", encoding="utf-8") as f:
        return json.load(f)


def build_context(data: dict) -> str:
    """Plockar ut det morgonmeddelandet faktiskt behöver, inte hela filen."""
    current = data.get("current_status", {}).get("current_metrics", {})
    derived = data.get("derived_metrics", {})
    readiness = data.get("readiness_decision", {})
    recent = data.get("recent_activities", [])[:3]
    planned = data.get("planned_workouts", [])

    today = datetime.now(timezone.utc).date().isoformat()
    todays_planned = [w for w in planned if w.get("date") == today]

    context = {
        "hrv_idag": current.get("hrv"),
        "hrv_baseline_7d": derived.get("hrv_baseline_7d"),
        "sömn_timmar": current.get("sleep_hours"),
        "readiness_rekommendation": readiness.get("recommendation"),
        "readiness_anledning": readiness.get("reason"),
        "readiness_signaler": readiness.get("signals"),
        "senaste_pass": [
            {
                "typ": a.get("type"),
                "datum": a.get("date"),
                "varaktighet_h": a.get("duration_hours"),
                "distans_km": a.get("distance_km"),
                "feel": a.get("feel"),
                "rpe": a.get("rpe"),
            }
            for a in recent
        ],
        "dagens_planerade_pass": [
            {"namn": w.get("name"), "typ": w.get("type"), "beskrivning": w.get("description")}
            for w in todays_planned
        ],
    }
    return json.dumps(context, ensure_ascii=False, indent=2)


def generate_message(context_json: str) -> str:
    client = Anthropic()  # Läser ANTHROPIC_API_KEY från miljövariabel
    response = client.messages.create(
        model=MODEL,
        max_tokens=300,
        temperature=1.0,  # Högre temperatur för variation dag till dag
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Dagens data:\n{context_json}\n\nSkriv dagens morgonmeddelande.",
            }
        ],
    )
    return response.content[0].text.strip()


def main():
    data = load_latest_json()
    context_json = build_context(data)
    message = generate_message(context_json)

    output = {
        "date": datetime.now(timezone.utc).date().isoformat(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "message": message,
    }

    with open("morning_message.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Genererat meddelande: {message}")


if __name__ == "__main__":
    main()
