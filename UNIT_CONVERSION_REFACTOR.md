# Unit Conversion Refactor: Cost Parameters from $/W to $/ton

**Date:** April 2026  
**Author:** ABSiCE development team  
**Status:** Implemented

---

## Summary

All per-unit cost parameters in the ABM-CE-PV model have been converted from **$/W** (dollars per Watt of PV capacity) to **$/ton** (dollars per metric ton of waste). This change was made to restore dimensional consistency after waste tracking was changed from Watts to metric tons.

The core arithmetic invariant after this refactor:

```
consumer_costs [$] = managed_waste [metric ton] × perceived_behavioral_control [$/ton]
```

No scaling factor is required at any cost multiplication site.

---

## Why It Was Needed

### Root Cause

The model originally tracked all waste volumes in **Watts of PV capacity** (functional unit). In that system, costs in **$/W** made sense: `waste [W] × cost [$/W] = $`.

A prior update changed waste input to metric tons by replacing `Yearly_Sum_Power_atEOL [W]` with `Yearly_Waste_EOL_Ton [metric ton]` from the consolidated PV ICE mass output file. This affected all waste tracking variables:

- `number_product_EoL` — now metric tons
- `number_used_product_EoL` — now metric tons
- `recycling_volume` — now metric tons
- `tot_prod_EoL` — now metric tons

However, all cost parameters (`recycling_cost`, `repairing_cost`, `landfill_cost`, `hoarding_cost`, etc.) remained in **$/W**, causing a dimensional mismatch at every cost multiplication site. The broken arithmetic produced costs in the nonsensical unit `metric tons × $/W = ton·$/W`, not dollars.

### Pre-existing Workarounds

Before this refactor, several workarounds had accumulated:

1. **Hardcoded `0.0077` proxy** in `update_perceived_behavioral_control`: transport costs (in $/kg) were multiplied by `0.0077` (the approximate kg/W ratio at year 2020) to convert them to $/W. This factor was a static approximation of the time-varying `dynamic_product_average_wght`.

2. **Mixed-unit bug** in `get_landfill_cost()` hazardous branch: `hazardous_landfill_cost [$/ton]` was added directly to `hazardous_waste_management_cost / 1E3 * dynamic_product_average_wght [$/W]`. These two terms had different units and could not be meaningfully summed.

3. **Implicit conversion** in landfill cost initialisation: `raw_cost [$/ton] / 1E3 * dynamic_product_average_wght [kg/W]` — stored as $/W to match the other cost parameters. Once waste moved to metric tons this pipeline became a source of confusion.

---

## Why $/ton Was Preferred

Three candidate unit systems were evaluated:

| Option | Arithmetic at multiplication | Scaling factor needed | Notes |
|---|---|---|---|
| Keep $/W, convert volumes to W at each site | `tons × 1000/wght × $/W = $` | Yes — `×1000 / dynamic_product_average_wght` at every site | `dynamic_product_average_wght` is time-varying; introduces temporal complexity into cost formulas |
| $/kg | `tons × 1000 × $/kg = $` | Yes — `× 1000` at every multiplication site | Cleaner than $/W but still requires explicit scaling |
| **$/ton (chosen)** | `tons × $/ton = $` | **None** | Directly matches the waste tracking unit; RTN landfill CSV already outputs $/ton |

**Key reasons $/ton was chosen:**

- **No scaling factor at cost multiplication sites.** With $/ton, `managed_waste [ton] × cost [$/ton] = $` is exact. No `* 1000` or `/ dynamic_product_average_wght` anywhere in the cost arithmetic.
- **RTN data files are naturally in $/ton.** `generate_landfill_costs.py` already outputs `TotalCost_$ / Shipped_kg * 1000 = $/ton`. Only `generate_recycling_costs.py` needed updating (was using `* 0.0077` to give $/W).
- **Transport cost formula simplifies.** `dist [km] × transportation_cost [$/ton/km] = $/ton` directly. The previous formula needed `/ 1E3 * dynamic_product_average_wght`; both factors are now removed.
- **TPB decisions are unaffected.** `tpb_perceived_behavioral_control` normalises all costs by `max(|cost|)` before weighting. Since all 5 `perceived_behavioral_control` entries scale by the same factor, the normalization cancels it identically and pathway rankings are preserved.
- **Eliminates `dynamic_product_average_wght` from financial formulas.** This time-varying ratio (kg/W, computed from PV ICE data) should not appear in cost arithmetic — its role is mass conversion, not cost scaling.

---

## Conversion Formula

All cost defaults are originally from literature in **$/W**. The factor `dynamic_product_average_wght ≈ 0.0077 kg/W` is computed at model startup from PV ICE data at year 2020:

```python
# ABM_CE_PV_Model.py ~line 1260
product_average_wght = total_massperm2 [kg/m²]
                     × (Yearly_Sum_Area_atEOL / Yearly_Sum_Power_atEOL) [m²/W]
                     ≈ 0.0077 kg/W
```

The conversion formula applied to all default parameters:

```
cost [$/ton] = cost [$/W] × 1000 [kg/ton] / 0.0077 [kg/W]
             ≈ cost [$/W] × 129,870
```

**Sanity check:** `0.0077 kg/W × 1000 W/kW = 7.7 kg/kW`, so `1 ton / 7.7 kg/kW ≈ 130 kW/ton`. At `$0.45/W` repair cost: `$0.45 × 130,000 W = $58,500/ton`. ✓

Note: these default values serve as **fallbacks only**. At runtime, most costs are replaced by data-driven RTN values or updated via learning curves.

---

## Changes Made

### `generate_recycling_costs.py`

| Location | Before | After |
|---|---|---|
| Cost calculation | `(TotalCost_$ / Shipped_kg) * 0.0077` → $/W | `(TotalCost_$ / Shipped_kg) * 1000` → $/ton |

> **Action required:** Re-run this script to regenerate the RTN recycling cost CSV with $/ton values before running the model.

`generate_landfill_costs.py` — no change needed (already outputs $/ton).

---

### `ABM_CE_PV_Model.py`

**Parameter defaults:**

| Parameter | Before ($/W) | After ($/ton) | Source |
|---|---|---|---|
| `original_recycling_cost` | `[0.0038-ε, 0.0038+ε, 0.0038]` | `[493-ε, 493+ε, 493]` | EPRI 2018 |
| `original_repairing_cost` | `[0.1, 0.45, 0.23]` | `[12987, 58442, 29870]` | IRENA-IEA 2016 |
| `hoarding_cost` | `[0, 0.001, 0.0005]` | `[0, 130, 65]` | cisco-eagle.com |
| `fsthand_mkt_pric` | `0.45` | `58442` | — |

**Transport cost formulas** (now correctly expressed as $/ton):

```python
# Before
transportation_cost_rcl = x * transportation_cost / 1E3 * dynamic_product_average_wght  # $/W
transportation_cost_rpr_ldf = dist * transportation_cost / 1E3 * dynamic_product_average_wght  # $/W

# After
transportation_cost_rcl = x * transportation_cost  # $/ton: dist [km] × cost [$/ton/km]
transportation_cost_rpr_ldf = dist * transportation_cost  # $/ton
```

---

### `ABM_CE_PV_ConsumerAgents.py`

**Landfill cost initialisation** — removed spurious W conversion:

```python
# Before
self.landfill_cost = raw_cost / 1E3 * dynamic_product_average_wght  # $/W

# After
self.landfill_cost = raw_cost  # $/ton — no conversion needed
```

**Transport cost formulas** — removed `/1E3` from all 5 assignments (init + update):

```python
# Before
self.recyc_transp_cost = dist * transportation_cost / 1E3  # $/kg

# After
self.recyc_transp_cost = dist * transportation_cost  # $/ton
```

**`get_landfill_cost()` — hazardous branch** (mixed-unit bug fix):

```python
# Before — BUG: $/ton + $/W cannot be summed
return hazardous_landfill_cost + hazardous_waste_management_cost['landfill'] / 1E3 * wght

# After — both $/ton; hazardous_waste_management_cost was already documented as $/ton
return hazardous_landfill_cost + hazardous_waste_management_cost['landfill']
```

**`get_landfill_cost()` — non-hazardous RTN path:**

```python
# Before
return _get_rtn_landfill_cost() / 1E3 * dynamic_product_average_wght  # $/ton → $/W

# After — RTN landfill CSV is in $/ton
return _get_rtn_landfill_cost()
```

**`get_hoarding_cost()` — hazardous branch:** same pattern, `/ 1E3 * wght` removed.

**`update_perceived_behavioral_control`** — removed hardcoded `0.0077` proxy:

```python
# Before — 0.0077 ≈ kg/W at year 2020, static approximation
self.perceived_behavioral_control[2] = recycling_cost + recyc_transp_cost * 0.0077
self.perceived_behavioral_control[3] = landfill_cost + landfill_transp_cost * 0.0077

# After — transport already $/ton, no factor needed
self.perceived_behavioral_control[2] = recycling_cost + recyc_transp_cost
self.perceived_behavioral_control[3] = landfill_cost + landfill_transp_cost
```

**`update_eol_volumes`** — no change at multiplication sites. With $/ton the arithmetic is already correct:
```python
self.consumer_costs += managed_waste [ton] * perceived_behavioral_control [$/ton]  # = $  ✓
```

---

### `ABM_CE_PV_RefurbisherAgents.py`

Transport cost formula in `economic_rationale_tpb` and `compute_refurbisher_costs`:

```python
# Before
revenue = -scd_hand_price + repairing_cost + dist * transportation_cost / 1E3 * wght  # $/W

# After
revenue = -scd_hand_price + repairing_cost + dist * transportation_cost  # $/ton
```

Volume multiplications in `compute_refurbisher_costs` need no `* 1000` since volumes are metric tons and costs are now $/ton:
```python
refurbisher_costs += (revenue * prod_sold + cost_recycling * prod_recycled + ...)  # ton × $/ton = $  ✓
```

---

### `ABM_CE_PV_RecyclerAgents.py`

No code changes at the multiplication site. `recycling_volume [ton] × recycling_cost [$/ton] = $` is automatically correct once the recycling cost CSV is regenerated in $/ton.

---

### `ABM_CE_PV_ProducerAgents.py`

**`recovered_volume_n_value`** — `recl_vol` is a mass quantity (kg), not a cost. Updated to convert tons → kg:

```python
# Before — wrong: dynamic_product_average_wght converts W→kg, not tons→kg
recl_vol = mass_fractions * tot_recycling_volume / num_neighbors * dynamic_product_average_wght * recovery_fractions

# After — correct: * 1000 converts metric tons → kg
recl_vol = mass_fractions * tot_recycling_volume * 1000 / num_neighbors * recovery_fractions
```

**`costs_producer`** — `industrial_waste_generated` remains in W (derived from `total_yearly_new_products`). Converted inline to metric tons at the cost site:

```python
# Before — dimensional mismatch
transport_cost = industrial_waste_W * (yearly_product_wght * transportation_cost / 1E3 * dist + average_landfill_cost)

# After — clean: W → tons inline, then ton × $/ton = $
industrial_waste_ton = industrial_waste_W * yearly_product_wght / 1000  # W × kg/W / 1000 = ton
transport_cost = industrial_waste_ton * (transportation_cost * dist + average_landfill_cost)
```

---

### `ABM_CE_PV_MultipleRun.py` and `ABM_CE_PV_BatchRun_Modified.py`

All scenario-level cost overrides converted to $/ton with inline provenance comments:

| Scenario value | Before ($/W) | After ($/ton) |
|---|---|---|
| `sa_landfill_costs` no-cost | `0.0000` | `0.0000` (unchanged) |
| `sa_landfill_costs` baseline | `0.0077` | `1000` |
| `sa_landfill_costs` high | `0.0134` | `1740` |
| `sa_landfill_costs` medium | `0.0115` | `1494` |
| `original_recycling_cost` baseline | `0.0077` | `1000` |
| `original_recycling_cost` high | `0.0134` | `1740` |
| `original_recycling_cost` calibration | `0.064` | `8312` |
| `original_recycling_cost` calibration | `0.085` | `11039` |
| `hoarding_cost` | `[0, 0.001, 0.0005]` | `[0, 130, 65]` |
| `original_repairing_cost` | `[0.1, 0.45, 0.23]` | `[12987, 58442, 29870]` |
| `fsthand_mkt_pric` | `0.45` | `58442` |

---

## Impact on Model Behaviour

### Financial outputs
All dollar-valued outputs (`consumer_costs`, `recycler_costs`, `refurbisher_costs`, `transport_cost_industrial_waste`) now correctly represent dollars. Previously, these were in `ton·$/W` — a nonsensical unit that silently produced order-of-magnitude errors.

### TPB pathway decisions
**Unaffected.** `tpb_perceived_behavioral_control` normalises all costs by `max(|cost|)` before weighting:

```
pbc_choice_normalised[i] = cost[i] / max(|cost[j]|)
```

Multiplying all costs by the same factor $k$ (the unit conversion) cancels in the ratio: $k \cdot c_i \,/\, k \cdot \max|c_j| = c_i \,/\, \max|c_j|$. Pathway rankings and behavioral intentions are identical to the pre-change model.

### kg reporting variables
**Unaffected.** The kg-valued reporters (`waste_kg_current_step`, `number_new_prod_repaired`, `pca_outputs`) are computed via a separate code path:
```python
new_eol_vol = number_product_EoL * 1000   # tons → kg (unchanged)
```
This `* 1000` is independent of cost units and was not modified.

### `recl_vol` (recovered material volume)
Changed from `* dynamic_product_average_wght` to `* 1000` — this is a **correctness fix** independent of the cost unit choice. `tot_recycling_volume` is in metric tons; `recl_vol` must be in kg for downstream material value calculations. The old formula used a kg/W factor on a tons input, which was incorrect.

### `_compute_synthetic_effective_capacity`
**Unaffected.** This method computes installed PV stock in Watts using internal tons→W conversion. It has no cost arithmetic and is orthogonal to this refactor.

---

## Verification Checklist

After re-running `generate_recycling_costs.py` to regenerate the RTN CSV:

- [ ] Trace one agent's `consumer_costs` for a single step: confirm `managed_waste [ton] × pbc [$/ton] = $` with sensible magnitude
- [ ] Check `recycler_costs` under RTN mode: confirm no extra conversion applied on top of $/ton CSV values
- [ ] Check `refurbisher_costs` sign: sell pathway should be negative (revenue), others positive (cost)
- [ ] Confirm `recl_vol` order of magnitude is in kg (not W and not tons)
- [ ] Confirm `transport_cost_industrial_waste` gives $ (units: `ton × [$/ton/km × km + $/ton] = $`)
- [ ] Run model baseline for 1 step; compare `consumer_costs`, `recycler_costs`, `refurbisher_costs` against pre-change run to verify same order of magnitude in $ terms

---

## Files Modified

| File | Phase | Change |
|---|---|---|
| `generate_recycling_costs.py` | 0 | `* 0.0077` → `* 1000`; re-run to regenerate CSV |
| `ABM_CE_PV_Model.py` | 1, 3 | Parameter defaults; transport cost formulas |
| `ABM_CE_PV_ConsumerAgents.py` | 2–6 | Landfill cost init; transport formulas; cost retrieval; PBC update |
| `ABM_CE_PV_RefurbisherAgents.py` | 7 | Transport cost formula in two methods |
| `ABM_CE_PV_RecyclerAgents.py` | 8 | No code change; depends on CSV regeneration |
| `ABM_CE_PV_ProducerAgents.py` | 9 | `recl_vol` tons→kg; `costs_producer` W→tons inline |
| `ABM_CE_PV_MultipleRun.py` | 10 | All scenario cost values |
| `ABM_CE_PV_BatchRun_Modified.py` | 10 | All scenario cost values |
| `generate_landfill_costs.py` | — | No change (already in $/ton) |
