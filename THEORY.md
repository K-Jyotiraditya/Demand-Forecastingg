# Demand Forecasting + Inventory Optimization

Read alongside the code; each section maps to a function in `forecasting.py`,
`inventory.py`, or `allocation.py`.

---

## 1. Forecasting: additive Holt-Winters

Retail demand has three moving parts — a **level**, a **trend**, and a repeating
**weekly** pattern. Triple exponential smoothing tracks all three
(`forecasting.holt_winters_additive`):

```
level_t   = α (y_t − S_{t−m}) + (1−α)(level_{t−1} + trend_{t−1})
trend_t   = β (level_t − level_{t−1}) + (1−β) trend_{t−1}
S_t       = γ (y_t − level_t)        + (1−γ) S_{t−m}
ŷ_{t+h}   = level_t + h·trend_t + S_{t−m+(h mod m)}
```

with `m = 7`. The three smoothing weights are fit by minimising in-sample
one-step SSE (`fit_holt_winters`, L-BFGS-B on `[0,1]³`). Forecasts are floored at
zero — demand cannot be negative.

**Baseline.** Every forecast is judged against **seasonal-naive** ("same weekday
last week"). If the smoother cannot beat that, the extra machinery is not earning
its place. On this data HW wins on MAE/RMSE.

**Why WAPE.** Daily SKU demand has many zero days, so MAPE (`|e|/y`) explodes or
divides by zero. WAPE `= Σ|e| / Σy` is the standard demand-planning accuracy
metric and is well-defined with zeros.

---

## 2. Lot sizing: EOQ

The Economic Order Quantity balances ordering cost against holding cost
(`inventory.eoq`):

```
EOQ = sqrt( 2 · D · S / H )
```

`D` = annual demand, `S` = fixed cost per order, `H` = annual holding cost per
unit. `D` is taken from the **long-run history**, because EOQ is a steady-state
quantity — using a short, possibly-zero forecast level would give degenerate
lot sizes.

---

## 3. Buffering uncertainty: safety stock, reorder point, newsvendor

* **Safety stock** (`inventory.safety_stock`): `z · σ · √L`, where `z = Φ⁻¹(SL)`
  for service level `SL` and `L` is the lead time. Crucially **σ is the forecast
  error** (RMSE), not raw demand volatility — you hold stock against what you
  *failed to predict*, which is the proper link from the forecaster to the policy.
* **Reorder point** (`reorder_point`): `μ·L + SS` — cover expected lead-time
  demand plus the buffer.
* **Newsvendor** (`newsvendor_quantity`): for a single perishable period, order to
  the **critical fractile** `Q* = μ + Φ⁻¹(Cu/(Cu+Co))·σ`, balancing underage cost
  `Cu` against overage cost `Co`.

---

## 4. Measuring the policy: the (Q,R) simulation

Closed forms assume normal demand; reality does not. `inventory.simulate_qr` runs
a **discrete-event** simulation over the realised hold-out demand: each day it
receives due orders, fills demand (unmet = lost sale), accrues holding cost on the
closing on-hand, and places a fixed order `Q` whenever the **inventory position**
(on-hand + on-order) falls to `R`. It returns the realised fill rate and the
holding/ordering/stockout cost breakdown.

Sweeping the target service level traces the **cost-of-service curve**
(`service_cost_frontier`): cost rises with the target while realised fill
saturates — the quantitative case against blindly chasing 99%.

---

## 5. Allocation: linear programming

Given a forecast, a budget, and finite capacity, choose order quantities `x_i` to
maximise profit (`allocation.allocate`):

```
max  Σ margin_i · x_i
s.t. Σ cost_i · x_i ≤ budget
     Σ x_i          ≤ capacity
     0 ≤ x_i ≤ forecast_demand_i
```

Linear objective, linear constraints, box bounds → a linear program, solved with
HiGHS via `scipy.optimize.linprog`. When margins are heterogeneous the LP loads
the highest-margin-per-dollar SKUs first and beats a proportional split — the
whole point of optimising rather than spreading evenly.

---

## 6. Limitations

* **Intermittent demand.** SKUs that are mostly zero with rare bulk orders break
  both Holt-Winters and the normal safety-stock approximation; the right tools are
  **Croston's method** / compound-Poisson models. We sidestep this by ranking SKUs
  on training-window volume, but a production system must handle the long tail.
* **Deterministic LP.** The allocation treats forecast demand as a hard cap; a
  stochastic / chance-constrained program would price in demand uncertainty.
* **Single echelon, constant lead time.** No multi-warehouse network, no
  lead-time variability, no supplier MOQs.
