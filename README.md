# Care Transition Efficiency & Placement Outcome Analytics — Dashboard

Interactive Streamlit dashboard for the HHS Unaccompanied Alien Children (UAC) Program dataset.

## Contents
- `app.py` — the dashboard application
- `uac_data.csv` — cleaned dataset, 720 dated reporting records spanning Jan 12, 2023 – Dec 21, 2025, used automatically by `app.py`
- `requirements.txt` — Python dependencies

## A note on data coverage
Reporting is **not on a strict daily cadence**: only 720 of the 1,075 calendar days in the covered span have a record. The gaps are concentrated on Fridays and Saturdays (only 2 Friday and 0 Saturday records out of 720), so the feed is effectively Sunday-through-Thursday. This means:
- Weekday-vs-weekend comparisons can't be made reliably from this data alone.
- "Consecutive days" figures in the Bottleneck Detection tab (e.g. backlog streaks) count consecutive *reporting* records, not calendar days — the dashboard labels and shows both so this isn't mistaken for an error.

See Section 5.1 and the Limitations section of the Research Paper for full detail.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
The app opens at `http://localhost:8501`.

## What's inside
- **Care Pipeline Flow** — daily flow across apprehension, CBP custody, transfer, HHS care, and discharge stages
- **Transfer & Discharge Efficiency** — Transfer Efficiency Ratio, Discharge Effectiveness Index, Pipeline Throughput Rate, with adjustable daily/7-day/30-day smoothing
- **Bottleneck Detection** — net daily flow into HHS care, Backlog Accumulation Rate, and a table of sustained backlog-building streaks
- **Outcome Trend Analysis** — Outcome Stability Score, monthly placement trend, year-over-year KPI comparison

## Sidebar controls
- Date range picker
- Rolling-window smoothing (daily / 7-day / 30-day)
- Metric visibility toggles
- Adjustable alert thresholds for Transfer Efficiency Ratio, Discharge Effectiveness, backlog-streak length, and Outcome Stability Score — breaches surface as visual banners at the top of the page

## Notes on KPI definitions
Three KPIs (Transfer Efficiency Ratio, Discharge Effectiveness, Pipeline Throughput) use the formulas specified in the project brief. Backlog Accumulation Rate and Outcome Stability Score were not given explicit formulas in the brief and were operationalized as:
- **Backlog Accumulation Rate** = (Transferred out − Discharged) ÷ HHS care (daily)
- **Outcome Stability Score** = 1 − (30-day rolling standard deviation ÷ 30-day rolling mean) of Discharge Effectiveness

Pipeline Throughput Rate and Outcome Stability Score use a trailing 30-*reporting*-day window (not 30 calendar days — see data coverage note above) and require at least 15 reporting days of history before showing a value; this matches the figures in the Research Paper and Executive Summary KPI tables.

See the accompanying research paper for full methodology and findings.
