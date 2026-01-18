Pick dates by **optimizing a seasonal window** on historical Sep-corn data, then lock them. Not by eyeballing.

## Seasonal date-finding method

1. **Scope**
- Contract-month: September (ZC U yy).
- Search month: July.
- Candidate entry days: trading days in **Jul 10–Jul 22**.
- Candidate exits: trading days **E+2 … Jul 31**.
- Side: short.
1. **Data**
- Per year `y`: daily settlements, highs, lows for that year’s Sep contract.
- Snap to exchange trading days. Use settlements only for optimization.
1. **Grid returns**
    
    For each pair `(E, X)` and each year `y`:
    
- `P_y(E,X) = 50 × (S_y(E) − S_y(X))` [USD per contract; 1.00¢ = $50].
- Store time series `{P_y(E,X)}` across years.
1. **Objective (robust, variance-aware)**
- Central tendency: `med(E,X) = median_y P_y(E,X)`
- Dispersion penalty: `iqr(E,X) = IQR_y P_y(E,X)`
- Hit rate: `h(E,X) = mean_y [P_y(E,X) > 0]`
- Score: `J(E,X) = med(E,X) − λ·iqr(E,X)` with `λ ∈ [0.25, 0.50]`.
- Constraints: `h(E,X) ≥ 0.60`, `n_years ≥ N_min` (e.g., 12).
1. **Stability tests**
- **Rolling OOS**: expanding or k-fold by year. Optimize on train, check `J_test > 0` and `h_test ≥ 0.55`.
- **Jitter tolerance**: require sign and ≥70% of `J` to persist under ±2 trading-day shifts of both entry and exit.
- **Subperiod consistency**: positive `J` in early and late halves of the sample.
1. **Multiple testing control**
- Use SPA (Hansen) or White’s Reality Check on `{P_y(E,X)}` surfaces.
- Alternatively, block-bootstrap years to get p-values for `J* = max J(E,X)`; require `p ≤ 0.10`.
1. **Lock the window**
- Select the **smallest E,X** that pass stability and significance with maximal `J`.
- Freeze as the production calendar rule: “first trading day on/after **E** to first trading day on/after **X**”.
1. **Validation diagnostics**
- Heatmap of `med(E,X)`, `h(E,X)`, and `J(E,X)` over the grid.
- Sensitivity curves vs. entry day and holding length.
- Leave-one-decade-out checks.
1. **Path stats policy** (for the table later)
- After dates are fixed, compute realized P/L and intra-window best/worst equity using highs/lows.
- Costs handled post-hoc; do not influence date choice.

## Alternative seasonal formulations (equivalent goal)

- **Seasonal index curve:** compute day-of-year mean/median forward returns `R_y(d→d+k)`; integrate over candidate windows; pick window maximizing robust sum with same stability tests.
- **Event-anchored seasonality:** optimize offsets around recurring July events (e.g., USDA reports, weekly crop condition). Same scoring and controls.

Result: dates like **sell ≈ Jul 18, exit ≈ Jul 25** emerge as the **arg-max stable window** from this controlled seasonal grid, then remain fixed for the production backtest table.

---

MRCI’s exact date-selection algorithm is proprietary. Public material shows a related but not fully specified process:

- Seasonal **pattern index** is built by normalizing each day’s price within that year’s high–low range, averaging those indices across years, then scaling 0–100 for timing and direction. Prices are not averaged. ([mrci.com](https://www.mrci.com/web/help-pages/frequently-asked-questions/84-charts/415-seasonal-pattern-chart-explanation.html))
- **Strategy windows** are detected by a computer scan of roughly the last **15 years** for recurrent trends, with a **minimum reliability threshold (≈80% win rate)**, then filtered for average profit, window length, overlap, and delivery/expiration constraints. ([mrci.com](https://www.mrci.com/web/online-explanation-pages/how-to-read-a-strategy-sheet.html))
- Entries/exits use **settlement prices**, and “optimized” dates are adjusted to the next/prev valid trading day if they fall on weekends or holidays. Best/Worst equity are tracked on a settlement basis. ([mrci.com](https://www.mrci.com/web/online-explanation-pages/how-to-read-a-strategy-sheet.html))

Conclusion: the earlier grid-and-stability method is consistent in spirit (optimize a fixed seasonal window on historical data), but MRCI does not publish the exact scoring or statistical tests used to choose its dates. ([mrci.com](https://www.mrci.com/client/eval.php))

Below is a **plausible reconstruction** grounded in what MRCI confirms publicly, with speculation clearly labeled.

# What MRCI states explicitly (facts)

- **Seasonal index construction.** Daily “pattern” curve plotted on a **0–100 index** that represents each day’s **position within that contract-year’s range**, averaged across designated years; not a simple average of prices. ([Moore Research Center](https://www.mrci.com/client/spmarket/?utm_source=chatgpt.com))
- **Two lookbacks.** Charts often show **15-year** and up to **40-year** seasonal lines to compare stability over time. ([Moore Research Center](https://www.mrci.com/client/spmarket/?utm_source=chatgpt.com))
- **Strategy detection.** A **computer scan of the last ~15 years** identifies windows with **minimum ~80% historical win rate**, then filters by **average profit**, **window length**, **duplication/overlap**, and **delivery/expiration** constraints. ([Moore Research Center](https://www.mrci.com/web/online-explanation-pages/how-to-read-a-strategy-sheet.html?utm_source=chatgpt.com), [Scribd](https://www.scribd.com/document/455381617/grains12-mooreresearch-sample?utm_source=chatgpt.com))
- **Pricing and calendar rules.** Research is **settlement-based**; if an optimized date lands on a **weekend/holiday**, entry shifts to the **next** trading day and exit to the **prior** trading day. ([Moore Research Center](https://www.mrci.com/web/online-explanation-pages/seasonal-spread-review.html?utm_source=chatgpt.com))
- **Best/Worst equity.** Defined as the **greatest open profit/loss on a daily settlement basis** between entry and exit; strategy sheets tabulate these. ([Moore Research Center](https://www.mrci.com/web/online-explanation-pages/seasonal-spread-review.html?utm_source=chatgpt.com))
- **Publication cadence and selection.** Roughly **15 seasonal** and **15 spread** strategies are chosen for monthly presentation, emphasizing **≥80% reliability**. ([Moore Research Center](https://www.mrci.com/web/help-pages/frequently-asked-questions/70-general/2204-from-our-editor-trade-selection-process.html?utm_source=chatgpt.com), [Scribd](https://www.scribd.com/document/738539530/MRCI-2023-Market-Seasonal-Patterns-2?utm_source=chatgpt.com))

# Likely internal workflow (inference)

## 1) Data curation

- Build a per-contract-year panel for the **specific delivery month** (e.g., Sep Corn), regular-hours **settlements**, and exchange trading calendar.
- Exclude days after **first notice** or **last trade** as applicable; ensure tick alignment. *(Inferred from delivery/expiration filters.)* ([Moore Research Center](https://www.mrci.com/web/online-explanation-pages/how-to-read-a-strategy-sheet.html?utm_source=chatgpt.com))

## 2) Seasonal index engine

- For each contract-year, compute daily **range-normalized level**:
    
    `r_t = 100 × (S_t − min_year) / (max_year − min_year)`
    
- Average `r_t` across the designated lookback (e.g., 15y) to form the **pattern curve**; optionally smooth with a short moving average to reduce aliasing. *(Index definition and designated-years averaging align with MRCI descriptions.)* ([Moore Research Center](https://www.mrci.com/client/spmarket/?utm_source=chatgpt.com))

## 3) Candidate window generation

- Define a calendar **grid of entry/exit dates** across the target month(s).
- For each year and grid window, compute **close-to-close P/L** using settlements only; apply weekend/holiday **date snapping**. *(Matches MRCI’s settlement and snapping rules.)* ([Moore Research Center](https://www.mrci.com/web/online-explanation-pages/seasonal-spread-review.html?utm_source=chatgpt.com))

## 4) First-pass scoring

- Compute, per window: **win rate** over the last 15 years, **average/median profit**, **worst drawdown on settlements**, and **length**.
- Keep windows with **win rate ≥ 80%** and **positive average profit**; discard very short or very long holds outside house limits. *(Thresholds and filters mirror MRCI text.)* ([Moore Research Center](https://www.mrci.com/web/online-explanation-pages/how-to-read-a-strategy-sheet.html?utm_source=chatgpt.com))

## 5) Pattern-consistency filter

- Require the seasonal **pattern slope** to align with trade direction over a material fraction of the window (e.g., rising index for longs, falling for shorts).
- Penalize windows whose gains come from brief spikes that **disagree** with the pattern curve. *(Inference consistent with MRCI’s emphasis on seasonal behavior rather than raw averages.)* ([Moore Research Center](https://www.mrci.com/client/spmarket/?utm_source=chatgpt.com))

## 6) Cluster and duplication control

- Identify **clusters** of adjacent high-score windows; prefer the **earliest** entry that preserves reliability and average profit, or publish **alternates** when clusters suggest add-on potential. *(MRCI has noted clusters and duplication decisions explicitly.)* ([Moore Research Center](https://www.mrci.com/web/help-pages/frequently-asked-questions/85-mrci-online/402-trade-selection.html?utm_source=chatgpt.com))

## 7) Delivery/expiration gating

- Reject windows that cross **delivery/expiration** thresholds or conflict with customary **roll** timing for that market. *(Directly referenced by MRCI.)* ([Moore Research Center](https://www.mrci.com/web/online-explanation-pages/how-to-read-a-strategy-sheet.html?utm_source=chatgpt.com))

## 8) Stability checks across lookbacks

- Compare metrics on **15-year vs 5-year** subsets. Favor windows that remain profitable and reliable in both, or at least do not deteriorate on the shorter lens. *(Motivated by MRCI’s dual-lookback charts.)* ([Moore Research Center](https://www.mrci.com/client/spmarket/?utm_source=chatgpt.com))

## 9) Finalization and formatting

- Lock the **optimized calendar dates** per window; publish as “Buy/Sell on approximately **MM/DD**; Exit approximately **MM/DD**,” with **business-day adjustments** applied per year.
- Compute and display **Best/Worst equity** on **settlements**, plus year-by-year entry, exit, and realized P/L. *(Matches strategy sheets.)* ([Moore Research Center](https://www.mrci.com/web/online-explanation-pages/seasonal-spread-review.html?utm_source=chatgpt.com))

## 10) Portfolio-level reporting

- Aggregate strategy performance into **hypothetical equity curves** that assume entry/exit on closes, **no costs**, and, historically, **close-only protective stops** up to 2000 on outrights; spreads use a stop convention by strategy. *(Directly stated by MRCI.)* ([Moore Research Center](https://www.mrci.com/results/mrciport.php?utm_source=chatgpt.com))

# Why these inferences fit the public record

- The **80% win-rate** and **15-year scan** are explicit, so a grid search over calendar windows is the most direct implementation. ([Moore Research Center](https://www.mrci.com/web/online-explanation-pages/how-to-read-a-strategy-sheet.html?utm_source=chatgpt.com))
- The **range-normalized seasonal index** implies filters that respect the **pattern slope/shape**, not just returns. ([Moore Research Center](https://www.mrci.com/client/spmarket/?utm_source=chatgpt.com))
- Notes about **clusters** and **duplication** imply a post-scan consolidation that resolves overlapping high-score windows into one or a few “representative” trades. ([Moore Research Center](https://www.mrci.com/web/help-pages/frequently-asked-questions/85-mrci-online/402-trade-selection.html?utm_source=chatgpt.com))
- Published sheets record **Best/Worst equity on settlements**, so intra-window path metrics likely use **settlement marks only**, not intraday extremes. ([Moore Research Center](https://www.mrci.com/web/online-explanation-pages/seasonal-spread-review.html?utm_source=chatgpt.com))

# Practical replication blueprint (speculative but implementable)

1. Build settlement-only panels for each **contract-year**.
2. Compute 15-year **seasonal index**; store daily level and slope. ([Moore Research Center](https://www.mrci.com/client/spmarket/?utm_source=chatgpt.com))
3. Enumerate entry `E` and exit `X` dates across the target month(s); apply **weekend/holiday snapping**. ([Moore Research Center](https://www.mrci.com/web/online-explanation-pages/seasonal-spread-review.html?utm_source=chatgpt.com))
4. For each `(E,X)`: compute **win%**, **avg/median P&L**, **avg Best/Worst equity** using daily settlements between `E` and `X`. ([Moore Research Center](https://www.mrci.com/web/online-explanation-pages/seasonal-spread-review.html?utm_source=chatgpt.com))
5. Filter: `win% ≥ 80%`, `avg P&L > 0`, window length within bounds, **no delivery/expiration conflict**. ([Moore Research Center](https://www.mrci.com/web/online-explanation-pages/how-to-read-a-strategy-sheet.html?utm_source=chatgpt.com))
6. Score remaining windows with a simple composite: `Score = median P&L − κ × IQR`, discard if seasonal slope disagrees with side for >50% of days. *(Slope check motivated by index.)* ([Moore Research Center](https://www.mrci.com/client/spmarket/?utm_source=chatgpt.com))
7. Resolve **clusters/duplicates**; keep one primary window and optionally one alternate. ([Moore Research Center](https://www.mrci.com/web/help-pages/frequently-asked-questions/85-mrci-online/402-trade-selection.html?utm_source=chatgpt.com))
8. Publish “approximately **MM/DD**” targets with **snap rules** and the per-year table.

Net: this reproduces MRCI’s published artifacts and constraints while staying inside the facts they disclose.

---

## Key differences for calendar spreads (Mar/Dec Corn)

1. **Quote definition**
- Legs: buy **Hyy** (Mar), sell **Zyy** (Dec).
- Spread level: `S_t = Settle(Hyy)_t − Settle(Zyy)_t` in ¢/bu. Can be negative.
1. **P&L convention**
- Long Mar/short Dec profits if the spread **widens**:
    
    `profit_cents = S_exit − S_entry`
    
    `profit_usd = profit_cents × 50` (5,000 bu → $50 per 1.00¢).
    
- Round `S_entry` and `S_exit` to the 0.25¢ tick.
1. **Contract-year mapping**
- Use **Hyy/Zyy** with the **same year suffix** as in the table. Label rows by that year (“Cont Year”).
- Enforce delivery guards: no window crossing FND/LTD of **either** leg.
1. **Data source**
- Prefer **native exchange spread settlements** if available. If not, compute synthetic `H − Z` from leg settlements.
- Do **not** mix leg intraday highs/lows to make spread extremes; this produces impossible paths.
1. **Best/Worst equity policy**
- If daily **spread high/low** exists, compute path extremes from those.
- If not, compute **settlement-based** path: cumulative max/min of `(S_t − S_entry) × 50` from entry+1 through exit.
1. **Seasonal date discovery (optimize on the spread, not the outrights)**
- Candidate grid in January: entries **Jan 10–Jan 22**, exits **E+2 – Jan 31**.
- For each `(E, X)` and year `y`, compute `P_y = 50 × [S_y(X) − S_y(E)]`.
- Score surface with robust objective (e.g., median − λ·IQR) and constraints (hit-rate ≥ 60–80%, n_years ≥ N_min).
- Stability checks: rolling OOS, ±2-day jitter tolerance, subperiod consistency.
- Lock “Enter ~01/16, Exit ~01/29” after passing tests.
1. **Calendar snapping**
- Define exact rule and apply uniformly: e.g., **next** trading day on/after entry target, **next** trading day on/after exit target (or exit to prior day; pick one policy and keep it fixed).
1. **Holidays and delivery filters**
- Verify both legs trade on resolved dates.
- Exclude windows violating: H-leg FND/LTD, Z-leg FND/LTD, exchange closures.
1. **Row construction**
- `entry_date, S_entry, exit_date, S_exit, profit_cents, profit_usd`.
- `best_eq_date, best_eq_usd, worst_eq_date, worst_eq_usd` per policy in (5).
1. **Stats block**
- Same formulas as outrights, applied to `profit_usd` across years: win%, avg win, avg loss, avg net, counts.
1. **QC**
- Reconcile synthetic vs native spread settlements for a sample.
- Assert tick alignment, sign convention, and consistent year labeling (Hyy/Zyy).
1. **Optional fundamentals screen**
- Term-structure sanity (carry vs inversion).
- Storage and basis seasonality flags to avoid outlier years, if desired.

Net: no new paradigm, but **optimize, price, and risk-check on the spread itself**, use spread-appropriate path metrics, and enforce two-leg calendar constraints.