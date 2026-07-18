#!/usr/bin/env python3
"""
Genererar Coach Murphys korta morgonmeddelande baserat på latest.json
(samma fil som redan syncas från Intervals.icu i det här repot).

v6: Tar bort v5:s fasta veckodags-tema-rotation. Klas vill hellre att
modellen själv väljer vad som är mest intressant just den dagen - t.ex.
prata om ett riktigt bra pass dagen efter, inte HRV "bara för att det är
den dagen på rotationen".

HRV får en hård spärr: den beräknas i Python (inte modellen) till en av
fyra trendflaggor (sustained_uptrend/sustained_downtrend/elevated_vs_baseline/
depressed_vs_baseline/no_clear_trend/insufficient_data). Modellen får bara
prata om HRV om flaggan INTE är "no_clear_trend" eller "insufficient_data" -
annars är det inte ett tillåtet ämne den dagen. Ingen väntetid inbyggd (som
en fast "en gång i veckan"-regel) - så fort en riktig trend syns i datan
kan den tas upp, hur få eller många dagar det än varit sedan sist.

Bygger vidare på v4/v5 (dagar_sedan-fix, kan_paverka_dagens_hrv-spärr,
bred kontext, HRV som trend inte punktvärde).

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

TON_VINKLAR = [
    "Öppna meddelandet med en konkret siffra eller detalj innan du sätter den i sammanhang.",
    "Öppna meddelandet med hur kroppen/känslan verkar ligga till, INTE med en siffra först.",
    "Öppna meddelandet med en konkret detalj (var, hur, vad) snarare än ett mätvärde.",
]

SYSTEM_PROMPT_TEMPLATE = """Du är "Coach Murphy" - en simulerad huvudcoach i Klas eget
AI-coachingsystem. Din uppgift just nu är EN sak: skriv en kort morgon-
hälsning (2-3 meningar, på svenska) som kompletterar en befintlig Home
Assistant-morgonhälsning om väder och kalender. Du ska INTE upprepa väder
eller kalenderinfo - bara ge din egen korta observation.

SÄKERHETSÖVERSTYRNING (kollas alltid FÖRST): om readiness_rekommendation är
"modify" eller "skip", eller om det finns en alert med severity "alarm" -
skriv en VARNING istället för något annat ämne. Kort, sakligt, inte
alarmistiskt. Annars: välj fritt ämne enligt nedan.

FRITT ÄMNESVAL - VIKTIGT: det finns INGEN fast dagordning för vad du ska
prata om. Titta på HELA datan och välj det som faktiskt är mest intressant
eller värt att nämna just idag - t.ex. ett ovanligt bra eller tufft pass
igår, en fin sträcka av konsekvent träning, hur nära nästa tävling det är,
en bra träningsfördelning den senaste veckan, eller en framåtblickande idé
för dagens planerade pass. Om Klas gjorde något imponerande igår är det
troligen mer intressant än ett rutinmässigt datapunkt-omnämnande.

HÅRD SPÄRR PÅ HRV: fältet "hrv_trend_signal" talar om ifall HRV:t faktiskt
har något intressant att säga just nu. Om värdet är "no_clear_trend" eller
"insufficient_data" - HRV FÅR INTE tas upp alls idag, välj ett annat ämne.
Om värdet är "sustained_uptrend", "sustained_downtrend",
"elevated_vs_baseline" eller "depressed_vs_baseline" - då är det ett giltigt
(men inte obligatoriskt) ämne, och ska i så fall beskrivas som en flerdagars-
trend, ALDRIG som "dagens värde" (mätningen hinner inte synka innan detta
meddelande skapas - se hrv_not).

VIKTIGT OM DATUM: varje aktivitet har ett fält "dagar_sedan" (0 = idag,
1 = igår osv). Använd ALLTID det för tidsangivelser - gissa aldrig.

VIKTIGT OM ORSAKSSAMBAND: koppla aldrig ett pass med dagar_sedan == 0 (hände
idag) till HRV som "bevis på återhämtning" - det är kronologiskt omöjligt.

TON: rak och sakligt varm, som en observant vän - inte tvingat kall, inte
peppig på ett tomt sätt. Inga klyschor ("du kan göra det!", "toppenjobbat!").

{ton_vinkel}

Om gårdagens meddelande visas nedan: välj annat ämne och annan struktur om
det går, så det inte känns repetitivt.

Svara ENDAST med själva meddelandet, ingen inledning, inga rubriker, inga
citattecken runt texten."""


def load_json_if_exists(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def days_ago(date_str: str, today: date) -> int:
    d = datetime.fromisoformat(date_str).date()
    return (today - d).days


def compute_hrv_trend(wellness: list, baseline_7d: float | None) -> tuple[str, dict]:
    """Räknar fram en trendflagga i Python istället för att låta modellen
    tolka råa, brusiga dagsvärden. Returnerar (flagga, detaljer)."""
    vals = [
        (w.get("date"), w.get("hrv_rmssd"))
        for w in wellness
        if w.get("hrv_rmssd") is not None
    ]
    vals = vals[-6:]  # senaste upp till 6 avläsningarna

    if len(vals) < 4:
        return "insufficient_data", {"antal_matningar": len(vals)}

    numbers = [v for _, v in vals]
    diffs = [numbers[i + 1] - numbers[i] for i in range(len(numbers) - 1)]

    # Sammanhängande uppgång/nedgång tre dagar i rad (den tydligaste signalen)
    if len(diffs) >= 3 and all(d > 0 for d in diffs[-3:]):
        return "sustained_uptrend", {
            "forsta": numbers[-4], "senaste": numbers[-1],
            "forandring": round(numbers[-1] - numbers[-4], 1)
        }
    if len(diffs) >= 3 and all(d < 0 for d in diffs[-3:]):
        return "sustained_downtrend", {
            "forsta": numbers[-4], "senaste": numbers[-1],
            "forandring": round(numbers[-1] - numbers[-4], 1)
        }

    # Annars: ligger senaste snittet tydligt över/under baseline?
    recent_avg = sum(numbers[-3:]) / len(numbers[-3:])
    if baseline_7d:
        pct_diff = (recent_avg - baseline_7d) / baseline_7d * 100
        if pct_diff >= 15:
            return "elevated_vs_baseline", {"procent_over_baseline": round(pct_diff, 1)}
        if pct_diff <= -15:
            return "depressed_vs_baseline", {"procent_under_baseline": round(pct_diff, 1)}

    return "no_clear_trend", {"senaste_varden": numbers[-3:]}


def build_context(data: dict, today: date) -> str:
    current = data.get("current_status", {}).get("current_metrics", {})
    derived = data.get("derived_metrics", {})
    readiness = data.get("readiness_decision", {})
    alerts = data.get("alerts", [])
    recent = data.get("recent_activities", [])[:3]
    planned = data.get("planned_workouts", [])
    wellness = data.get("wellness_data", [])
    weekly = data.get("weekly_summary", {})
    race_cal = data.get("race_calendar", {})
    phase = derived.get("phase_detection", {})
    seiler = derived.get("seiler_tid_7d_primary", {})
    consistency = derived.get("consistency_details", {})

    today_str = today.isoformat()

    todays_planned = [
        w for w in planned
        if w.get("date") == today_str and w.get("type") not in ("TARGET", "NOTE")
    ]

    senaste_pass = []
    for a in recent:
        da = days_ago(a["date"], today) if a.get("date") else None
        senaste_pass.append({
            "typ": a.get("type"),
            "dagar_sedan": da,
            "kan_paverka_dagens_hrv": (da is not None and da >= 1),
            "varaktighet_h": a.get("duration_hours"),
            "distans_km": a.get("distance_km"),
            "hojdmeter": a.get("elevation_m"),
            "temperatur_c": a.get("avg_temp"),
            "feel": a.get("feel"),
            "rpe": a.get("rpe"),
        })

    hrv_baseline_7d = derived.get("hrv_baseline_7d")
    trend_flag, trend_detaljer = compute_hrv_trend(wellness, hrv_baseline_7d)

    context = {
        "dagens_datum": today_str,
        "hrv_trend_signal": trend_flag,
        "hrv_trend_detaljer": trend_detaljer,
        "hrv_baseline_7d": hrv_baseline_7d,
        "hrv_not": "hrv_trend_signal ar forberaknad i Python - lita pa den, "
                   "gissa inte sjalv utifran rada varden. Om signalen ar "
                   "'no_clear_trend' eller 'insufficient_data': HRV ar INTE "
                   "ett tillatet amne idag.",
        "alerts": alerts,
        "readiness_rekommendation": readiness.get("recommendation"),
        "readiness_anledning": readiness.get("reason"),
        "veckans_sammanfattning": {
            "genomforda_pass": weekly.get("activities_count"),
            "total_traningstid": weekly.get("total_training_formatted"),
            "planerade_dagar": consistency.get("planned_days"),
            "matchade_dagar": consistency.get("matched_days"),
        },
        "tavling_och_fas": {
            "fas": phase.get("phase"),
            "fas_vecka": phase.get("phase_duration_weeks"),
            "nasta_tavling": race_cal.get("next_race", {}).get("name"),
            "dagar_till_tavling": race_cal.get("next_race", {}).get("days_until"),
        },
        "traningsfordelning": {
            "klassificering": seiler.get("classification"),
            "andel_latt_procent": seiler.get("z1_pct"),
            "andel_troskel_procent": seiler.get("z2_pct"),
            "andel_hart_procent": seiler.get("z3_pct"),
        },
        "senaste_pass": senaste_pass,
        "dagens_planerade_pass": [
            {"namn": w.get("name"), "beskrivning": w.get("description")}
            for w in todays_planned
        ],
    }
    return json.dumps(context, ensure_ascii=False, indent=2)


def pick_ton_vinkel(today: date) -> str:
    return TON_VINKLAR[today.toordinal() % len(TON_VINKLAR)]


def generate_message(context_json: str, ton_vinkel: str, previous_message: str | None) -> str:
    client = Anthropic()
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(ton_vinkel=ton_vinkel)

    user_content = f"Dagens data:\n{context_json}\n\n"
    if previous_message:
        user_content += f"Gårdagens meddelande (välj annat ämne/struktur om möjligt):\n{previous_message}\n\n"
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
    ton_vinkel = pick_ton_vinkel(today)
    message = generate_message(context_json, ton_vinkel, previous_message)

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
