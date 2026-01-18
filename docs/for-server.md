I’ll wire these missing Seasonax-style helpers directly into `seasonal.py` so the module becomes a full logical backend: cumulative profit series, per‑year % returns, KPI summariser, and pattern‑vs‑rest logic, all built on top of `per_year_results` and `WindowStats`. 

First I’ll scan `seasonal.py` to see the existing structures and function signatures, then design and implement the new helpers to fit cleanly into the current API. I’ll also keep parameters explicit (no hidden defaults) per your rules.

I'll add the missing Seasonax-style helpers directly into [seasonal.py](cci:7://file:///home/matrillo/apps/seasonal-analysis/seasonal/seasonal.py:0:0-0:0): an enrichment helper for per-year results, a cumulative profit series builder, a KPI summariser for a selected window, and a pattern-vs-rest decomposition built on df and per_year_results. I'll keep signatures explicit (no new defaults), avoid touching existing behavior unless it's clearly safe, and then summarize how to call each helper. Let me first confirm whether [plot_per_year_pnl](cci:1://file:///home/matrillo/apps/seasonal-analysis/seasonal/seasonal.py:567:0-588:14) is used anywhere else before changing it.



I'll now update [seasonal.py](cci:7://file:///home/matrillo/apps/seasonal-analysis/seasonal/seasonal.py:0:0-0:0) to add the Seasonax-style derived helpers: per-year return enrichment, cumulative profit series, KPI summary, and pattern-vs-rest decomposition, inserted just after [run_seasonal_analysis](cci:1://file:///home/matrillo/apps/seasonal-analysis/seasonal/seasonal.py:438:0-505:5). I won't modify existing behavior or plotting functions.

I’ll try the file edit again but with more compact helper implementations and minimal comments/docstrings, to avoid timeouts. I’ll insert the new helpers right after [run_seasonal_analysis](cci:1://file:///home/matrillo/apps/seasonal-analysis/seasonal/seasonal.py:438:0-505:5) and before the plotting functions, keeping all existing behavior unchanged.



Here’s what I’ve implemented in [seasonal.py](cci:7://file:///home/matrillo/apps/seasonal-analysis/seasonal/seasonal.py:0:0-0:0) to cover the missing Seasonax-style logic. All existing behavior is untouched; everything below is additive and hangs off [run_seasonal_analysis](cci:1://file:///home/matrillo/apps/seasonal-analysis/seasonal/seasonal.py:438:0-505:5)’s `per_year_results`.

### 1. Per-year percent and annualised returns

New helpers right after [run_seasonal_analysis](cci:1://file:///home/matrillo/apps/seasonal-analysis/seasonal/seasonal.py:438:0-505:5):

- **[_compute_pattern_return_frac(direction, entry_px, exit_px) -> float](cci:1://file:///home/matrillo/apps/seasonal-analysis/seasonal/seasonal.py:510:1-517:62)**  
  - Long: `(exit_px - entry_px) / entry_px`  
  - Short: `(entry_px - exit_px) / entry_px`  
  - Raises `ValueError` for unsupported directions or `entry_px == 0`.

- **[enrich_per_year_results_with_returns(df, per_year_results, trading_days_per_year) -> pd.DataFrame](cci:1://file:///home/matrillo/apps/seasonal-analysis/seasonal/seasonal.py:520:1-565:20)**  
  - Takes the original price `df` and the raw `per_year_results` table.  
  - For each row, it:
    - Counts trading bars between `entry_dt` and `exit_dt` in `df` ⇒ `pattern_trading_days`.
    - Computes `return_frac` via [_compute_pattern_return_frac](cci:1://file:///home/matrillo/apps/seasonal-analysis/seasonal/seasonal.py:510:1-517:62).
    - Computes `return_pct = return_frac * 100`.
    - Annualises using pattern length:
      - `annualised_return_frac = (1 + return_frac) ** (trading_days_per_year / pattern_trading_days) - 1`
      - `annualised_return_pct = annualised_return_frac * 100`.
  - Returns a copy of `per_year_results` with extra columns:
    - `pattern_trading_days`
    - `return_frac`, `return_pct`
    - `annualised_return_frac`, `annualised_return_pct`

This gives you exactly the per-trade % and annualised returns needed for the table and right-hand KPIs.

---

### 2. Cumulative profit curve for a window

- **[build_cumulative_profit_series(per_year_results, entry_mmdd, exit_mmdd, as_percent) -> pd.Series](cci:1://file:///home/matrillo/apps/seasonal-analysis/seasonal/seasonal.py:568:1-597:75)**

For a given window `(entry_mmdd, exit_mmdd)`:

- Filters `per_year_results` for that window, sorted by `year`.
- If `as_percent`:
  - Recomputes per-trade return (%) using [_compute_pattern_return_frac(...) * 100](cci:1://file:///home/matrillo/apps/seasonal-analysis/seasonal/seasonal.py:510:1-517:62).
  - Cumulative sum ⇒ series named `"cumulative_return_pct"`.
- Else:
  - Uses `pnl_points`, cumulative sum ⇒ series named `"cumulative_pnl_points"`.
- Index is `year` (int).

This is the “cumulative profit” line under the bar chart.

---

### 3. Seasonax-style KPI summary for a window

- **[summarize_window_kpis(df, per_year_results, entry_mmdd, exit_mmdd, trading_days_per_year) -> Dict[str, Dict[str, float]]](cci:1://file:///home/matrillo/apps/seasonal-analysis/seasonal/seasonal.py:600:1-696:6)**

For a specific window:

1. Filters `per_year_results` to that window.
2. Calls [enrich_per_year_results_with_returns](cci:1://file:///home/matrillo/apps/seasonal-analysis/seasonal/seasonal.py:520:1-565:20) to ensure all return columns exist.
3. Computes statistics:

   - **Basic:**
     - `num_trades`
     - `win_rate` (fraction of years with positive `return_pct`)

   - **Returns in % (`returns_pct`):**
     - `avg`, `median`, `min`, `max` of `return_pct`
     - `std` of `return_pct` (sample, `ddof=1`)
     - `avg_annualised`, `median_annualised` from `annualised_return_pct`

   - **Profit in points (`profit_points`):**
     - `avg`, `median`, `min`, `max`, `total` from `pnl_points`

   - **Gains (only positive years):**
     - `num_gains`
     - `avg_gain_pct`, `median_gain_pct`, `best_gain_pct`

   - **Losses (only negative years):**
     - `num_losses`
     - `avg_loss_pct`, `median_loss_pct`, `worst_loss_pct`

   - **Cumulative / risk:**
     - `max_drawdown_points` on cumulative `pnl_points`
     - `end_cumulative_pnl_points` (final cumulative PnL)
     - `sharpe_like`: mean of `annualised_return_pct` divided by stdev of `return_pct` (if defined)

Return structure:

```python
{
  "basic": {...},
  "returns_pct": {...},
  "profit_points": {...},
  "gains": {...},
  "losses": {...},
  "cumulative": {...},
  "risk": {...},
}
```

This is essentially the right-hand Seasonax KPI panel (annualised stats, gain/loss breakdown, dispersion, drawdown).

---

### 4. Pattern vs rest vs buy-and-hold

- **[compute_pattern_vs_rest_vs_buy_and_hold(df, per_year_results, entry_mmdd, exit_mmdd) -> Dict[str, float]](cci:1://file:///home/matrillo/apps/seasonal-analysis/seasonal/seasonal.py:699:1-753:6)**

Using all years in which the pattern has trades:

1. Restricts `df` to those years and computes per-bar log returns of `close`.
2. Builds a boolean mask over log-return index:
   - `True` inside any pattern window `[entry_dt, exit_dt]` (using `per_year_results` dates).
3. Splits log returns into:
   - `pattern_log` (inside pattern),
   - `rest_log` (outside pattern).

4. Sums logs and exponentiates to get total multiplicative returns:

   - `buyhold_return_frac` over all selected years.
   - `pattern_return_frac` over pattern windows only.
   - `rest_return_frac` over the complement.

5. Computes share-of-total:

   - `pattern_share_of_buyhold = pattern_return_frac / buyhold_return_frac` (if base≠0).
   - `rest_share_of_buyhold` similarly.

Return dict:

```python
{
  "buyhold_return_frac": ...,
  "pattern_return_frac": ...,
  "rest_return_frac": ...,
  "pattern_share_of_buyhold": ...,
  "rest_share_of_buyhold": ...,
}
```

This directly underpins the donut / “Pattern vs Rest” style visual.

---

### 5. How this maps to your Seasonax-style UI

Given:

```python
res = run_seasonal_analysis(...)
df  # original price DataFrame you passed / loaded
```

you now have:

- **Per-year table & % returns:**

  ```python
  enriched = enrich_per_year_results_with_returns(
      df=df,
      per_year_results=res.per_year_results,
      trading_days_per_year=252,  # or your chosen value
  )
  # Filter enriched by entry_mmdd/exit_mmdd for the per-year table.
  ```

- **Pattern bar chart and cumulative curve:**

  ```python
  # bar heights in %:
  per_window = enriched[
      (enriched["entry_mmdd"] == e) &
      (enriched["exit_mmdd"] == x)
  ].sort_values("year")

  # use per_window["return_pct"] for bar values

  cum_series = build_cumulative_profit_series(
      per_year_results=res.per_year_results,
      entry_mmdd=e,
      exit_mmdd=x,
      as_percent=True,
  )
  ```

- **Right-hand KPI panel:**

  ```python
  kpis = summarize_window_kpis(
      df=df,
      per_year_results=res.per_year_results,
      entry_mmdd=e,
      exit_mmdd=x,
      trading_days_per_year=252,
  )
  ```

- **Pattern vs rest:**

  ```python
  pr = compute_pattern_vs_rest_vs_buy_and_hold(
      df=df,
      per_year_results=res.per_year_results,
      entry_mmdd=e,
      exit_mmdd=x,
  )
  ```

At this point the module has all the logical pieces you outlined:
- Seasonal curves and multi-lookback overlays.
- Window grid + per-year trades.
- Per-year % returns + annualised returns.
- Cumulative profit curves.
- KPI breakdown (win rate, gains/losses, dispersion, drawdown).
- Pattern vs rest vs buy-and-hold attribution.

If you’d like, I can next:
- Switch [plot_per_year_pnl](cci:1://file:///home/matrillo/apps/seasonal-analysis/seasonal/seasonal.py:815:0-836:14) (and the [main.py](cci:7://file:///home/matrillo/apps/seasonal-analysis/main.py:0:0-0:0) export plot) to use `return_pct` instead of `pnl_points`, or
- Add a new plotting helper specifically for the cumulative profit series.