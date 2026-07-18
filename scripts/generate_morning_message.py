#!/usr/bin/env python3
"""
Genererar Coach Murphys korta morgonmeddelande baserat på latest.json
(samma fil som redan syncas från Intervals.icu i det här repot).

v3: Bygger vidare på v2:s dagar_sedan-fix (se den för bakgrund om buggen
med "igår" vs "idag"). Lägger dessutom till variation så meddelandena inte
blir strukturellt lika dag efter dag:
  - Läser gårdagens morning_message.json (om den finns) och instruerar
    modellen att INTE upprepa samma öppning/struktur
  - Väljer en av flera "vinklar" (siffra/känsla/detalj) baserat på dagens
    datum, som en extra knuff utöver temperature=1.0

Läser: latest.json, morning_message.json (om den finns sedan tidigare)
Skriver: morning_message.json (skrivs över, committas av GitHub Actions-steget)

Kräver miljövariabeln ANTHROPIC_API_KEY (satt som GitHub Secret i workflowen).
"""

import json
import os
from datetime import datetime, date
from zoneinfo import ZoneInfo

from anthropic import Anthropic

MODEL = "claude-sonnet-5"
LOCAL_TZ = ZoneInfo("Europe/Stockholm")

VINKLAR = [
    "Öppna meddelandet med en konkret siffra (HRV, sömntimmar, distans) innan du sätter den i sammanhang.",
    "Öppna meddelandet med hur kroppen/känslan verkar ligga till, INTE med en siffra först - väv in siffran senare i meningen.",
    "Öppna meddelandet med en konkret detalj från senaste passet (var det kördes, hur det kändes) snarare än ett mätvärde.",
]

SYSTEM_PROMPT_TEMPLATE = """Du är "Coach Murphy" - en simulerad huvudcoach i Klas eget
AI-coachingsystem. Din uppgift just nu är EN sak: skriv en kort morgon-
hälsning (2-3 meningar, på svenska) som kompletterar en befintlig Home
Assistant-morgonhälsning om väder och kalender. Du ska INTE upprepa väder
eller kalenderinfo - bara ge din egen korta observation.

VIKTIGT OM DATUM: varje aktivitet i datan har redan ett förberäknat fält
"dagar_sedan" (0 = idag, 1 = igår, 3 = för tre dagar sedan osv). ANVÄND
ALLTID det fältet för att avgöra om något hände "idag", "igår" eller
"för X dagar sedan" - räkna eller gissa ALDRIG själv utifrån råa datum.
Om det senaste passet har dagar_sedan=0, säg "dagens pass" eller liknande,
INTE "igår". Om dagar_sedan=3, säg "för tre dagar sedan", INTE "igår".
Detta är den vanligaste källan till fel - dubbelkolla alltid mot fältet
innan du skriver något om när ett pass ägde rum.

PRIORITETSORDNING (första matchande vinner - välj EN kategori, blanda inte):
1. VARNING - om något faktiskt är värt att vara försiktig med idag (t.ex.
   readiness-läge "modify" eller "skip", HRV tydligt under baseline, RI lågt).
   Kort, sakligt, inte alarmistiskt. Om readiness är "go" och allt ser
   normalt ut, använd INTE denna kategori - hoppa till nästa.
2. VINST/FRAMGÅNG - något som faktiskt gick bra nyligen (genomfört pass,
   bra sömn, HRV över baseline, ett tydligt mönster av bra vanor). Var
   EXAKT med tidsangivelsen (idag/igår/för X dagar sedan) enligt
   dagar_sedan-fältet. Konkret, inte generiskt bra-jobbat.
3. FRAMÅTBLICKANDE PROMPT FÖR DAGEN - en enkel idé (implementation intention,
   en sak att bestämma i förväg) kopplad till dagens planerade pass om det
   finns ett, eller en påminnelse om något sensoriskt att söka upp om det är
   en löprunda.

Om det inte finns något planerat pass idag ("dagens_planerade_pass" är tom),
säg det bara om det faktiskt stämmer enligt datan - hitta inte på att det
är en vilodag om du är osäker, kolla "dagens_planerade_pass"-fältet.

VARIATION (viktigt - detta körs dagligen och får inte kännas som samma mall):
{vinkel}

Om gårdagens meddelande visas nedan i användarens data: skriv INTE på samma
sätt. Byt öppningsmening, meningsbyggnad och vinkel jämfört med det.

TON: rak och sakligt varm, som en observant vän - inte tvingat kall, inte
peppig på ett tomt sätt. Inga klyschor ("du kan göra det!", "toppenjobbat!").

Svara ENDAST med själva meddelandet, ingen inledning, inga rubriker, inga
citattecken runt texten."""


def load_json_if_exists(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def days_ago(date_str: str, today: date) -> int:
    """Räknar ut antal dagar sedan ett datum (hanterar både rena datum och
    datum+tid-strängar som '2026-07-18T09:26:00')."""
    d = datetime.fromisoformat(date_str).date()
    return (today - d).days


def build_context(data: dict, today: date) -> str:
    """Plockar ut det morgonmeddelandet faktiskt behöver, inte hela filen."""
    current = data.get("current_status", {}).get("current_metrics", {})
    derived = data.get("derived_metrics", {})
    readiness = data.get("readiness_decision", {})
    recent = data.get("recent_activities", [])[:3]
    planned = data.get("planned_workouts", [])

    today_str = today.isoformat()

    # Bara riktiga träningspass idag, inte "Weekly"-målsättningsposter
    todays_planned = [
        w for w in planned
        if w.get("date") == today_str and w.get("type") not in ("TARGET", "NOTE")
    ]

    context = {
        "dagens_datum": today_str,
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
                "dagar_sedan": days_ago(a["date"], today) if a.get("date") else None,
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


def pick_vinkel(today: date) -> str:
    """Väljer en av VINKLAR-varianterna baserat på dagens datum, så samma
    dag alltid ger samma vinkel (deterministiskt) men olika dagar varierar."""
    idx = today.toordinal() % len(VINKLAR)
    return VINKLAR[idx]


def generate_message(context_json: str, vinkel: str, previous_message: str | None) -> str:
    client = Anthropic()  # Läser ANTHROPIC_API_KEY från miljövariabel
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(vinkel=vinkel)

    user_content = f"Dagens data:\n{context_json}\n\n"
    if previous_message:
        user_content += f"Gårdagens meddelande (skriv INTE likadant):\n{previous_message}\n\n"
    user_content += "Skriv dagens morgonmeddelande."

    response = client.messages.create(
        model=MODEL,
        max_tokens=300,
        temperature=1.0,
        system=system_prompt,
        messages=[{"role": "user", "content": user_content}],
    )
    return response.content[0].text.strip()


def main():
    with open("latest.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    today = datetime.now(LOCAL_TZ).date()

    previous = load_json_if_exists("morning_message.json")
    previous_message = previous.get("message") if previous else None

    context_json = build_context(data, today)
    vinkel = pick_vinkel(today)
    message = generate_message(context_json, vinkel, previous_message)

    output = {
        "date": today.isoformat(),
        "generated_at_utc": datetime.now().astimezone().isoformat(),
        "message": message,
    }

    with open("morning_message.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Genererat meddelande: {message}")


if __name__ == "__main__":
    main()
