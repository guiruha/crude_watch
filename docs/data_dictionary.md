# CrudeWatch Data Dictionary — `data/raw_files.xlsx`

## Dataset Overview

- **Sheet name:** `in`
- **Rows:** 758,843 daily bars (+1 header)
- **Date range:** 2010-06-06 → 2026-07-02 (UTC)
- **Source:** Databento **OHLCV-1d** (daily bar) data for **NYMEX/CME crude-oil
futures**, delivered via the CME Globex feed.
- **Distinct symbols:** 3,596

## Column Dictionary


| Column          | Meaning                             | Notes                                                                 |
| --------------- | ----------------------------------- | --------------------------------------------------------------------- |
| `ts_event`      | Bar timestamp (event time), UTC     | ISO-8601 nanosecond format; `T00:00:00Z` because these are daily bars |
| `rtype`         | Databento record type               | Constant `35` = OHLCV-1d (daily aggregate)                            |
| `publisher_id`  | Databento publisher/venue ID        | Constant `1` = GLBX.MDP3 (CME Globex MDP 3.0)                         |
| `instrument_id` | Databento numeric instrument ID     | Unique per contract per publisher; stable machine key for a symbol    |
| `open`          | Opening price of the bar            | USD per barrel (spreads can be negative)                              |
| `high`          | High price of the bar               |                                                                       |
| `low`           | Low price of the bar                |                                                                       |
| `close`         | Closing/settlement price of the bar |                                                                       |
| `volume`        | Contracts traded during the bar     | Integer                                                               |
| `symbol`        | Human-readable instrument symbol    | See symbology below (3,596 distinct values)                           |




## Symbol Construction

Every symbol builds from a **product root + month code + year digit**.

### CME Month Codes


| Code | Month | Code | Month | Code | Month |
| ---- | ----- | ---- | ----- | ---- | ----- |
| F    | Jan   | K    | May   | U    | Sep   |
| G    | Feb   | M    | Jun   | V    | Oct   |
| H    | Mar   | N    | Jul   | X    | Nov   |
| J    | Apr   | Q    | Aug   | Z    | Dec   |


**Year digit:** single digit (`0`–`9`), decade-ambiguous on its own
(e.g. `0` = 2010 *or* 2020); resolve using `ts_event` / `instrument_id`.

### Product Roots


| Root  | Product                                                                             |
| ----- | ----------------------------------------------------------------------------------- |
| `CL`  | WTI Light Sweet Crude Oil futures (NYMEX, 1,000 bbl) — core product                 |
| `MCL` | Micro WTI Crude Oil futures (100 bbl)                                               |
| `BZ`  | Brent Crude Oil (Last Day Financial) futures on NYMEX (1,000 bbl)                   |
| `HO`  | NY Harbor ULSD / heating oil (appears as a spread leg)                              |
| `WS`  | Related crude leg paired against `CL` (rare, e.g. `CLF8-WSF8`); low-confidence root |




## Symbol Families



### 1. Outrights (120 symbols)

A single contract: `CL` + month + year.

- `CLM1` = WTI June 2011
- `CLZ0` = WTI Dec 2010



### 2. Dash Spreads `LEG1-LEG2` (~2,631 symbols)

Exchange-listed two-leg spreads (buy leg 1 / sell leg 2).

- **Calendar** (same root, different months): `CLF0-CLG0` = long Jan-10 / short Feb-10 WTI
- **Inter-commodity** (different roots):
  - `CLF0-BZF0` = WTI vs Brent (the "arb")
  - `CLF2-MCLF2` = WTI vs Micro WTI
  - `CLF8-WSF8`



### 3. CME Colon / UDS Notation `CL:XX ...` (827 symbols)


| Code    | Type                                           | Legs | Example          | Reading                               |
| ------- | ---------------------------------------------- | ---- | ---------------- | ------------------------------------- |
| `CL:BF` | Butterfly                                      | 3    | `CL:BF F0-G0-H0` | +1 Jan / −2 Feb / +1 Mar              |
| `CL:BZ` | 2-leg calendar-style spread (Globex "BZ" type) | 2    | `CL:BZ F0-G0`    | Jan-10 vs Feb-10                      |
| `CL:C1` | Crack spread (product vs crude)                | 2    | `CL:C1 HO-CL F0` | Heating oil vs WTI, Jan-10            |
| `CL:FS` | Forward strip                                  | N    | `CL:FS 07M M6`   | 7-month strip starting Jun-16         |
| `CL:SA` | Strip average                                  | N    | `CL:SA 03M F8`   | 3-month average strip starting Jan-18 |




### 4. User-Defined Strategies `UD:CL: GN ...` (2 symbols)

Ad-hoc / user-defined Globex spreads keyed by an ID
(e.g. `UD:CL: GN 2518365`). Opaque exchange-generated combos.

## Distribution Summary


| Family                     | Count                                                |
| -------------------------- | ---------------------------------------------------- |
| Outrights                  | 120                                                  |
| Dash spreads               | ~2,631 (216 calendar + ~2,415 inter-commodity/other) |
| Colon UDS spreads          | 827 (`BF` 307, `BZ` 273, `C1` 240, `SA` 4, `FS` 3)   |
| User-defined               | 2                                                    |
| **Total distinct symbols** | **3,596**                                            |


