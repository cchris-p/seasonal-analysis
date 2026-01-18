- **Process Drafting**
    
    **What list of methods does it take to produce a table like this? Yearly averages solely?**
    
    Not yearly averages. Per-year trades computed from fixed entry/exit rules, then summary stats averaged across years.
    
    ## Methods (minimal set)
    
    1. **Instrument mapping**
    - Use the September CBOT Corn contract for each year (e.g., ZCU13, ZCU14, …).
    - Price units in cents/bushel. Contract size = 5,000 bu. Tick = 0.25¢ = $12.50.
    1. **Anchor-date resolution**
    - Entry target = July 18. Exit target = July 25.
    - Snap rule: first trading day on or after the target date (or define “closest trading day”; pick one rule and apply consistently).
    - Use official settlement prices for both entry and exit.
    1. **Per-year row construction**
    - `sell_date`, `exit_date`: resolved dates per rule.
    - `sell_price`, `exit_price`: settlement on those dates.
    - **Realized P/L (short):**
        
        `profit_cents = sell_price − exit_price`
        
        `profit_usd = profit_cents × 50`  (since 1.00¢ × 5,000 bu = $50)
        
        Round prices to the 0.25¢ tick before calculations.
        
    1. **Path extremes within the hold**
    - Inspect daily highs/lows from the session after entry through exit.
    - For a short:
        
        `best_eq_usd = max_over_window( (sell_price − low) × 50 )` with its date.
        
        `worst_eq_usd = min_over_window( (sell_price − high) × 50 )` with its date.
        
    - If equal to realized P/L on exit day, dates may coincide.
    1. **Stats block**
    - `winners = count(profit_usd > 0)`
    - `losers = count(profit_usd ≤ 0)`
    - `pct_correct = winners / total_trades`
    - `avg_win_usd = mean(profit_usd where > 0)`
    - `avg_loss_usd = mean(profit_usd where ≤ 0)`
    - `avg_net_usd = mean(profit_usd across all years)` and same in cents.
    
    ## Practical details to fix upfront
    
    - Holiday/weekend handling per exchange calendar.
    - Missing data policy (skip year vs. impute).
    - Limit-up/down days if highs/lows are constrained.
    - Continuous vs. specific contract: use the specific September contract for each year to avoid back-adjustment artifacts.
    - Commissions/slippage: exclude or apply a fixed deduction.
    
    Result: table rows are **per-year realized and path metrics**, not yearly averages. Only the **Stats** section uses averages and counts across years.
    
    **So what is the process of creating such a table ?**
    
    Build a seasonal trade-window backtest pipeline.
    
    ## Pipeline
    
    1. **Spec**
    - Instrument: CBOT Corn Sep (ZC U yy).
    - Contract size: 5,000 bu; tick 0.25¢ = $12.50; 1.00¢ = $50.
    - Entry target: Jul 18. Exit target: Jul 25.
    - Snap rule: first exchange trading day on/after target date.
    - Prices: settlements for entry/exit; daily high/low for path stats.
    1. **Calendars**
    - Generate CBOT trading days for each year. Apply weekend/holiday rules. Fix time zone to exchange local time.
    1. **Contract mapping**
    - For each year `y`, select that year’s Sep contract (no continuous/back-adjusted data).
    1. **Data pull**
    - Daily OHLC + settlement from Jul 1–Jul 31 for each mapped contract.
    1. **Date resolution**
    - `sell_date = next_trading_day_on_or_after(Jul 18)`
    - `exit_date = next_trading_day_on_or_after(Jul 25)`
    - Verify both exist in the dataset; otherwise mark year missing.
    1. **Per-year row**
    - Prices rounded to 0.25¢: `sell_px`, `exit_px`.
    - Realized P/L (short):
        
        `profit_cents = sell_px − exit_px`
        
        `profit_usd = profit_cents × 50`.
        
    1. **Path extremes within hold**
    - Window: sessions after `sell_date` through `exit_date` inclusive.
    - Best equity (short):
        
        `best_eq_usd = max((sell_px − low_d) × 50)` with `best_date`.
        
    - Worst equity (short):
        
        `worst_eq_usd = min((sell_px − high_d) × 50)` with `worst_date`.
        
    1. **Stats**
    - `winners = count(profit_usd > 0)`
    - `losers = count(profit_usd ≤ 0)`
    - `pct_correct = winners / (winners + losers)`
    - `avg_win_usd = mean(profit_usd | >0)`
    - `avg_loss_usd = mean(profit_usd | ≤0)`
    - `avg_net_usd = mean(profit_usd)`
    - Convert cents via `/50` as needed.
    1. **Adjustments (optional)**
    - Deduct commissions/slippage per trade.
    - Exclude limit-move days from path stats if using constrained highs/lows policy.
    1. **QC**
    - Cross-check settlements against a second source for a random sample.
    - Assert tick alignment and date snapping.
    - Log missing years; leave cells as “—” if unresolved.
    
    ## Output schema
    
    `year, sell_date, sell_price, exit_date, exit_price, profit_cents, profit_usd, best_eq_date, best_eq_usd, worst_eq_date, worst_eq_usd`.
    
    Result: table rows are per-year seasonal window outcomes; the Stats block aggregates those rows.
    

A phased, deterministic seasonal backtest methodology.

# Phase 0 — Spec & Governance

- Instrument: CBOT Corn September (ZC U yy), contract size 5,000 bu, tick 0.25¢ = $12.50.
- Window: sell on target **Jul 18**, exit on target **Jul 25** each year.
- Snap rule: first **trading day on or after** target date. Single rule across all years.
- Pricing: **settlements** for entry/exit; **daily high/low** for path stats.
- Units: cents/bu; 1.00¢ × 5,000 = **$50**.

# Phase 1 — Data Prep

- **Calendars:** Build exchange trading-day calendar for Jul 1–31 per year. Include holiday/early-close flags. Time zone = exchange local.
- **Contract mapping:** Map each year `y` → specific Sep contract for that year (no continuous/back-adjusted series).
- **Data pull:** Daily OHLC + settlement for Jul 1–31 for each mapped contract.
- **Sanity checks:** monotonic dates, no dup dates, prices align to 0.25¢ tick, no negative or zero prices.
- **Rounding policy:** round all reported prices to nearest 0.25¢ **before** P/L math.

# Phase 2 — Window Resolution

For each year:

- `sell_date = next_trading_day_on_or_after(Jul 18)`
- `exit_date = next_trading_day_on_or_after(Jul 25)`
- Deterministic fallbacks:
    - If either date absent → mark year **missing**; do not impute.
    - If `exit_date < sell_date` due to anomalies → drop year and log.

# Phase 3 — Per-Year Computation

Given `sell_px = settle[sell_date]`, `exit_px = settle[exit_date]` (cents/bu):

- **Realized P/L (short):**
    - `profit_cents = sell_px − exit_px`
    - `profit_usd = profit_cents × 50`
- **Path extremes** over window `[day after sell_date … exit_date]`:
    - `best_eq_usd = max((sell_px − low_d) × 50)`; record date.
    - `worst_eq_usd = min((sell_px − high_d) × 50)`; record date.
- If exit day sets best/worst, dates may coincide.

# Phase 4 — Table Assembly

Columns:

`year, sell_date, sell_price, exit_date, exit_price, profit_cents, profit_usd, best_eq_date, best_eq_usd, worst_eq_date, worst_eq_usd`.

- Use “—” for missing cells (e.g., unreadable source or missing day).
- Keep one row per year. No within-year averaging.

# Phase 5 — Stats Block

From realized `profit_usd`:

- `winners = count(profit_usd > 0)`
- `losers = count(profit_usd ≤ 0)`
- `pct_correct = winners / (winners + losers)`
- `avg_win_usd = mean(profit_usd | > 0)`
- `avg_loss_usd = mean(profit_usd | ≤ 0)`
- `avg_net_usd = mean(profit_usd)`
- Convert to cents via `/50` where needed.

# Phase 6 — QC & Audit

- **Cross-source price audit:** spot-check ≥10% of years against a second vendor.
- **Tick alignment:** all prices mod 0.25¢ = 0.
- **Calendar audit:** entry/exit dates are trading days; weekend/holiday snaps correct.
- **Limit moves:** flag sessions with exchange limit hits; document effect on path stats.
- **Idempotence:** reruns with identical inputs produce identical outputs (hash CSV).
- **Invariants:** `best_eq_usd ≥ realized_usd ≥ worst_eq_usd` for shorts should hold when using high/low correctly.
- **Worked example test:** e.g., 2013 row must yield `45.00¢ → \$2,250` if data matches.

# Phase 7 — Reproducibility

- Parameter file (YAML/JSON): window dates, snap rule, rounding policy, contract mapping rule, vendor name and data timestamp.
- Record code commit hash and data extract timestamp.
- Emit artifacts: per-year CSV, final table CSV, stats JSON, and a run log.

# Phase 8 — Options (document if used)

- Fixed per-trade commission and slippage.
- Alternative snap policy (“closest trading day” with deterministic tie-break).
- Exclude path stats on known constrained sessions.
- Robust missing-data policy (skip vs. partial compute).

# Answer to the core question

Not yearly averages. Rows are **per-year trades** from fixed seasonal rules. Only the **Stats** section averages across years.