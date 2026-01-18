Question asked to the tradesurge agent:


@SeasonalPatterns.tsx can you design what we'll need from the artifacts server to serve this data? I want to apply the pattern we've established for test cases where I initially generate the json files for each instrument and the backend server does the filtering on those json files as parameters.

Discuss a plan of what the json would look like, how the filters could work and other UI components work to display the data accordingly



---

I’ll first skim `SeasonalPatterns.tsx` so the plan aligns with how the current UI works, then I’ll propose:

- JSON artifact shape per instrument/TF.
- Backend artifact endpoints and filter semantics.
- How the UI should request and display filtered seasonal data (and what extra UI pieces we might want).

I’ll also set a small TODO plan so we can iterate on this design cleanly.

I’ve reviewed the SeasonalPatterns component and will keep this answer at the design level: JSON schema, artifact filenames, backend endpoints and filters, and how the UI should wire into them (without touching code yet). I’ll also update the small TODO plan I started so it reflects this design work.

### 1. High‑level shape (mirroring testcase artifacts)

You already have the testcase pattern:

- Precompute JSON files into `RESULTS_DIR`.
- Artifacts server exposes simple HTTP endpoints that:
  - Load JSON from disk.
  - Apply **filtering/selection** on the server.
  - Return a JSON payload that the Next.js UI consumes.

For seasonality we can copy that pattern:

1. **Compute job** (offline):
   - For each `(instrument, timeframe)` generate a **single JSON** describing:
     - Instrument metadata.
     - Global seasonal curve (full year).
     - A list of **precomputed patterns** (windows) with full stats + per‑year trades.

2. **Artifacts server**:
   - Reads those instrument JSONs.
   - Applies filters (min win rate, min years, etc.) over the patterns list.
   - Returns a **filtered list + detail for each pattern**.

3. **SeasonalPatterns.tsx**:
   - Fetches a single payload per instrument/timeframe.
   - Uses it to populate:
     - Header (instrument, years).
     - Pattern dropdown + Prev/Next.
     - Seasonal curve chart.
     - Cumulative profit chart.
     - Pattern returns histogram.
     - Year‑by‑year table.
     - Stats panel.

---

### 2. JSON artifact per instrument / timeframe

**File naming & location**

- Directory under `RESULTS_DIR`, e.g.:

  - `RESULTS_DIR/seasonality/seasonality_RL_D.json`
  - `RESULTS_DIR/seasonality/seasonality_ES_D.json`

- Pattern (analogous to testcase files):

  - `seasonality_{instrument_symbol}_{timeframe}.json`

**Top‑level structure**

```jsonc
{
  "instrument": {
    "id": "RL",
    "symbol": "RL",
    "name": "Ralph Lauren Corporation",
    "asset_class": "equity",
    "sector": "Consumer Cyclical",
    "exchange": "NYSE"
  },
  "timeframe": "D",
  "coverage": {
    "first_date": "2005-10-28",
    "last_date": "2020-10-27",
    "years": 14,
    "timezone": "America/New_York"
  },
  // full-year seasonal curve, same for all patterns of this instrument/tf
  "seasonal_curve": [
    {
      "doy": 1,
      "date_label": "01 Jan",
      "time": "2020-01-01",
      "value": 100.0
    },
    {
      "doy": 2,
      "date_label": "02 Jan",
      "time": "2020-01-02",
      "value": 100.4
    }
    // ...
  ],
  "patterns": [
    {
      "pattern_id": "oct28_nov07",
      "label": "28 Oct – 7 Nov",
      "start_month": 10,
      "start_day": 28,
      "end_month": 11,
      "end_day": 7,
      "window_calendar_length_days": 11,

      "stats": {
        "years": 14,
        "pattern_annualized_return_pct": 504.6,
        "rest_of_year_return_pct": 1.97,
        "win_rate_pct": 66.67,
        "avg_return_pct": 4.19,
        "median_return_pct": 1.41,
        "total_profit_pts": 66.42,
        "avg_profit_pts_per_trade": 4.43,
        "winning_trades": 10,
        "losing_trades": 5,
        "avg_gain_pct": 9.15,
        "max_gain_pct": 19.49,
        "avg_loss_pct": -3.08,
        "max_loss_pct": -7.42,
        "current_winning_streak": 1,
        "max_winning_streak": 10,
        "std_dev_pct": 7.95
      },

      // one trade per year in that window
      "trades_by_year": [
        {
          "year": 2006,
          "start_date": "2005-10-28",
          "end_date": "2005-11-07",
          "start_price": 42.52,
          "end_price": 48.08,
          "profit": 5.56,
          "profit_pct": 13.08,
          "max_rise_pct": 13.08,
          "max_drop_pct": -3.55
        }
        // ...
      ],

      // for histogram
      "returns_by_year": [
        { "year": 2006, "return_pct": 5.6 },
        { "year": 2007, "return_pct": -0.1 }
        // ...
      ],

      // for cumulative profit chart
      "cumulative_profit_curve": [
        { "trade_index": 1, "value": 0.0 },
        { "trade_index": 2, "value": 12.0 },
        { "trade_index": 3, "value": 18.0 }
        // ...
      ]
    }

    // more patterns for this instrument/timeframe
  ]
}
```

Notes:

- `seasonal_curve` is **shared per instrument/timeframe**; the UI just overlays/highlights the current window.
- Each pattern contains everything needed for:
  - Stats panel (`stats`).
  - Pattern returns histogram (`returns_by_year`).
  - Cumulative chart (`cumulative_profit_curve`).
  - Trade history table (`trades_by_year`).

---

### 3. Backend endpoints & filters

Reuse the testcase style you shared (`/testcases/<id>/...`) but for seasonality.

#### 3.1. List instruments with seasonality

- `GET /seasonality/instruments`

Returns (for search/autocomplete):

```jsonc
[
  {
    "id": "RL",
    "symbol": "RL",
    "name": "Ralph Lauren Corporation",
    "asset_class": "equity",
    "sector": "Consumer Cyclical",
    "available_timeframes": ["D"],
    "coverage_years": 14
  },
  {
    "id": "ES",
    "symbol": "ES",
    "name": "E-mini S&P 500",
    "asset_class": "future",
    "sector": "Index",
    "available_timeframes": ["D"]
  }
]
```

Implementation: scan `RESULTS_DIR/seasonality/seasonality_*.json` and build this list once (optionally cached).

#### 3.2. Get patterns for one instrument/timeframe (with filters)

- `GET /seasonality/<instrument_id>/patterns`

Query params (server‑side filtering only, no recomputation):

- `timeframe` (required): e.g. `D`.
- `min_years` (optional): `stats.years >= min_years`.
- `min_win_rate` (optional): `stats.win_rate_pct >= min_win_rate`.
- `min_avg_return_pct` (optional).
- `direction` (optional now; future extension).
- `sort_by` (optional): one of:
  - `pattern_annualized_return_pct`
  - `avg_return_pct`
  - `win_rate_pct`
  - `std_dev_pct`
- `sort_dir`: `asc` | `desc` (default `desc`).
- `limit`: max patterns to return.

Response:

```jsonc
{
  "instrument": { ... },
  "timeframe": "D",
  "coverage": { ... },
  "seasonal_curve": [ ... ],      // always included
  "filters_applied": {
    "min_years": 10,
    "min_win_rate": 60,
    "sort_by": "pattern_annualized_return_pct",
    "sort_dir": "desc",
    "limit": 50
  },
  "patterns": [
    // subset of patterns from the artifact file after filtering & sorting
  ]
}
```

Implementation:

- Load `RESULTS_DIR/seasonality/seasonality_{instrument_id}_{timeframe}.json`.
- Work entirely on its `patterns` array:
  - Apply min_* filters.
  - Sort.
  - Slice to `limit`.
- Return the original `instrument`, `timeframe`, `coverage`, `seasonal_curve`, and the filtered `patterns`.

You could keep a helper similar to `serve_result` but this route **post‑processes** JSON before returning.

---

### 4. Mapping to [SeasonalPatterns.tsx](cci:7://file:///home/eloah/apps/tradesurge/tradesurge_web_nextjs/components/seasonality/SeasonalPatterns.tsx:0:0-0:0) UI

#### 4.1. Data flow in the component

State to add:

- `instrument` / `selectedInstrumentId`.
- `timeframe` (likely just `"D"` initially).
- `filters` (`minYears`, `minWinRate`, etc.).
- `patterns` (array from backend).
- `selectedPatternId` (string).
- Derived: `selectedPattern = patterns.find(p => p.pattern_id === selectedPatternId)`.

Fetches:

1. **Instrument selection / search**
   - Use `/seasonality/instruments` to power:
     - The TextInput search + dropdown (or reuse `SelectAsset`).
   - When instrument changes:
     - Fetch `/seasonality/<instrument_id>/patterns?timeframe=D&min_years=...`.

2. **Patterns payload**
   - Once fetched, set:
     - `instrumentMeta`, `coverage`, `seasonalCurve`, `patterns`.
     - Default `selectedPatternId = patterns[0]?.pattern_id`.

No per‑pattern HTTP calls needed; everything is in that payload.

#### 4.2. Filling existing UI with real data

- **Header section**:
  - `Title`: `instrument.name`.
  - Badge: `instrument.symbol`.
  - Sector line: `instrument.sector`.
  - Years label: from `coverage.years` and `coverage.first_date` / `last_date`.

- **Pattern dropdown & nav**:
  - `Select` options: `patterns.map(p => ({ value: p.pattern_id, label: p.label }))`.
  - `selectedPattern` drives all downstream charts.
  - Prev/Next buttons move index within `patterns`.

- **Main seasonal curve chart**:
  - Use `seasonal_curve` array:
    - Convert each object into `LineData` (`{ time: entry.time, value: entry.value }`).
  - Optionally:
    - Use `selectedPattern`’s `start_month/day` & `end_month/day` to visually highlight the window (e.g. overlay band).

- **Cumulative profit chart**:
  - Map `selectedPattern.cumulative_profit_curve` into [SeasonalPoint[]](cci:2://file:///home/eloah/apps/tradesurge/tradesurge_web_nextjs/components/seasonality/SeasonalPatterns.tsx:40:0-40:30):
    - `time`: trade index (1..N) or directly numeric as in `mockCumulativeProfit`.
    - `value`: `cum_profit`.

- **Pattern returns histogram**:
  - Map `selectedPattern.returns_by_year`:
    - `time`: `year`.
    - `value`: `return_pct`.
    - `color`: computed on the client (`>=0` → blue; `<0` → red).

- **Stats panel**:
  - All fields come from `selectedPattern.stats`:
    - `patternAnnualizedReturnPct` = `stats.pattern_annualized_return_pct`, etc.

- **Trade history table**:
  - Use `selectedPattern.trades_by_year` to replace `mockYearRows`.

#### 4.3. Where filters live in the UI

- **Basic filters** (likely in a modal or drawer triggered by “More”):
  - `min_years` slider / input.
  - `min_win_rate` slider.
  - `min_avg_return_pct`.
  - `sort_by` selector + ascending/descending toggle.
- On “Apply filters”:
  - Re‑fetch `/seasonality/<instrument_id>/patterns` with new query params.
  - Replace `patterns` and reset `selectedPatternId` (e.g. to first hit).

This keeps all ranking/threshold logic server‑side and the UI dumb.

---

### 5. Summary / next steps

- **Artifacts**:
  - One JSON per `(instrument, timeframe)` under `RESULTS_DIR/seasonality/`.
  - Root: `instrument`, `timeframe`, `coverage`, `seasonal_curve`, `patterns[]`.
  - Each pattern has `window`, `stats`, `trades_by_year`, `returns_by_year`, `cumulative_profit_curve`.

- **Backend**:
  - `GET /seasonality/instruments` – instrument list for search.
  - `GET /seasonality/<instrument_id>/patterns` – loads that JSON, filters `patterns[]` based on query params, returns filtered list + global curve.

- **UI**:
  - SeasonalPatterns.tsx pulls from these two endpoints.
  - Replaces all mocks with data from one `patterns` payload.
  - Header, charts, stats panel, and table all bind directly to `instrument`, `coverage`, `seasonal_curve`, and `selectedPattern`.

If you’d like, the next step can be to sketch a concrete TypeScript type for this JSON shape and a minimal API client function in the Next.js app.