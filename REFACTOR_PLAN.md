# ABSiCE Waste Pipeline Refactor Plan

## Overview

Replacing per-PCA PV ICE waste reads (in watts/m²) with a consolidated metric-ton CSV
loaded once at the model level, and updating the installed-capacity datain files to use
ReEDS StdScen24 data for years 2026+.

---

## Completed Work

### Script: `transform_pvice_waste_by_pca.py`
- Reshapes wide-format PV ICE waste CSV to long format
- Output columns: `Year` (int), `PCA` (str, e.g. `p31`), `Yearly_Waste_EOL_Ton` (float, metric tons)
- Output file: `PV_ICE/PVICE_PCA_WasteEOL_by_Year_and_PCA.csv`
- Status: **complete and verified**

---

### Script: `generate_combined_datain.py`
- Merges old Solar Futures datain CSVs (2010–2025) from `PV_ICE/TEMP/PCA/` with new
  ReEDS StdScen24 data (2026+) from
  `ReEDS/StdScen24_annual_balancingAreas_Mid_Case_CO2e_95by2035.xlsx`
- ReEDS reports at 3-year intervals → divide by 3 to get per-year values
- 0.85 silicon PV scaling factor applied to ReEDS values
- Output directory: `PV_ICE/TEMP/PCA_merged/`
- Units written: MW (consumer `__init__` divides by `agents_per_pca` then ×1E6 → W)
- Key formula: `reeds_raw["annual_MW"] = reeds_raw["total_MW"] / REEDS_INTERVAL * 0.85`
- Status: **complete and verified**
- **Pending**: `PCA_RENAME_MAP = {"z119": "p119"}` discussed but not yet added (see below)

---

### `ABM_CE_PV_Model.py` — All planned changes implemented

| Change | Description |
|--------|-------------|
| Load consolidated waste CSV | `pvice_waste_eol_df = pd.read_csv(_waste_eol_path)` + `transform_timeseries_timestep(..., scale=False)` at model `__init__` |
| Rename tracker | `pca_tot_waste_w` → `pca_tot_waste_ton`; `pca_tot_waste_m2` kept at 0 with deprecation comment |
| Reporter update | `"Tot waste (ton) by pca"` referencing `pca_tot_waste_ton` |
| Comments | "old PV ICE results" notes on datain/dataOut loop reads and `all_pca_dataOut` read |

---

### `ABM_CE_PV_ConsumerAgents.py` — All planned changes implemented except one block

| Phase | Location | Change |
|-------|----------|--------|
| 0 – Comments | `__init__` | "old PV ICE results" comment on `data_out_pca`/`data_in_pca` loads |
| 1 – Waste slice | `__init__` | Added `pv_ice_waste_df` slice from `model.pvice_waste_eol_df` by PCA; divided by `agents_per_pca` |
| 2 – Comment | `get_additional_capacity` | "old PV ICE results" comment on datain capacity read |
| 3 – Waste read | `update_product_stock` | Replaced `Yearly_Sum_Power_atEOL` reads with `pv_ice_waste_df['Yearly_Waste_EOL_Ton']`; m² vars set to 0 |
| 4 – Tons → kg | `update_eol_volumes` | `new_eol_vol = self.number_product_EoL * 1000 + storage * 1000`; `used_eol_vol = self.number_used_product_EoL * 1000`; hoarding branch `* 1000` |
| 5 – Hazardous | `update_generator_size` | `hazardous_waste_mass_kg = self.tot_prod_EoL * 1000` |
| 5 – Hazardous | `is_hazardous_waste_storage_limit_exceeded` | `total_mass_stored_kg = self.number_product_hoarded_hazardous * 1000` |
| 6 – Comments | `mass_per_function_model` | "old PV ICE results" comments on two `data_out_pca` reads (logic unchanged) |
| 7 – TPB decision | `tpb_decision` | `self.sold_waste * 1000 < used_volume_purchased * self.model.product_average_wght` |
| Model tracker | `update_product_stock` | `pca_tot_waste_ton[self.pca] += yearly_waste_ton` |
| Effective capacity block | `update_product_stock` | Block that overwrites `new_products`, `used_products`, `number_product` with `Effective_Capacity_[W]` from old dataOut is **commented out** (see note below) |
| datain/dataOut paths | `ABM_CE_PV_ConsumerAgents.__init__` | `data_out_pca` and `data_in_pca` now read from explicit absolute path `PV_ICE/TEMP/PCA_merged/` via `_pca_merged_dir = os.path.join(os.path.dirname(__file__), "PV_ICE", "TEMP", "PCA_merged")`; `testfolder` in model left unchanged at `PCA` |
| waste EOL CSV path | `ABM_CE_PV_Model.__init__` | `_waste_eol_path` updated to `PV_ICE/TEMP/PCA_merged/PVICE_PCA_WasteEOL_by_Year_and_PCA.csv` |

**Note on `Effective_Capacity_[W]` block**: The block was sourcing degradation-adjusted capacity stock from the old Solar Futures scenario `dataOut` files, creating an inconsistency with the new consolidated waste file (which reflects a different, newer installation history). It has been commented out with the following in-code note:

> `# TODO: Re-enable this block once PV ICE has been re-run with the merged`  
> `# datain files (PV_ICE/TEMP/PCA_merged/). The new dataOut files will provide`  
> `# Effective_Capacity_[W] consistent with the StdScen24 installation history.`  
> `# Until then, capacity stock is driven by raw datain values (no Weibull`  
> `# degradation applied), which is internally consistent with the waste file.`

---

## Pending Work

### 1. Add PCA rename map to `generate_combined_datain.py`  *(lower priority)*

**Why**: ReEDS StdScen24 uses `z119` where old Solar Futures datain files use `p119`.
Without the rename, `z119` rows are written to a separate unmatched file instead of
merged into the existing `p119` datain file.

**Where**: In `load_reeds_data`, after `reeds_annual = pd.DataFrame(rows)`:

```python
# PCA renames between old Solar Futures naming and new StdScen24 ReEDS file
PCA_RENAME_MAP: dict[str, str] = {
    "z119": "p119",
    # add other renames here if discovered
}
reeds_annual["pca"] = reeds_annual["pca"].replace(PCA_RENAME_MAP)
```

**Verification**: Run `generate_combined_datain.py` and check the unmatched PCAs list
printed to stdout; all entries should map to known Solar Futures PCA names.

---

## Future Work (not yet started)

### 2. Compute synthetic `Effective_Capacity_[W]` column in `data_out_pca`

**Motivation**: The `Effective_Capacity_[W]` column in old Solar Futures `dataOut` files reflects
Weibull-degraded capacity based on the *old* installation history and is inconsistent with the
StdScen24-merged `datain` files. Rather than waiting for new PV ICE dataOut files, we can
synthesize an approximate column from the merged datain installations and the consolidated
waste EOL series, and re-enable the product-stock overwrite block.

**Approach**:
1. After dividing `data_out_pca['Yearly_Sum_Power_atEOL']` and
   `data_out_pca['Yearly_Sum_Area_atEOL']` by `agents_per_pca` (existing step), but
   **before** calling `transform_timeseries_timestep` on `data_out_pca`:
   - Temporarily load the per-agent waste EOL series (ton/year, divided by `agents_per_pca`)
     from `self.model.pvice_waste_eol_df` for this PCA.
   - Load the raw `data_in_pca` (MW → W per agent) for cumulative installed capacity.
   - Merge with `self.model.pvice_mat_factor` on `year` to get `total_massperm2` (kg/m²).

2. **Convert waste tons → W (per year)**:
   ```
   mass_to_area_ratio(t) = Yearly_Sum_Area_atEOL(t) / Yearly_Sum_Power_atEOL(t)  # m²/W
   kg_per_W(t)           = total_massperm2(t) * mass_to_area_ratio(t)            # kg/W
   waste_W(t)            = waste_ton(t) * 1000 / kg_per_W(t)                     # W
   ```
   If `kg_per_W(t)` is zero or NaN, fall back to 0 for that year.

3. **Compute `Effective_Capacity_[W]`**:
   ```
   cumulative_installed_W(t) = cumsum(new_Installed_Capacity_[MW](t) * 1E6)   # W
   cumulative_waste_W(t)     = cumsum(waste_W(t))
   Effective_Capacity_[W](t) = cumulative_installed_W(t) - cumulative_waste_W(t)
   ```
   Clip to ≥ 0 to avoid negatives from timing mismatches.

4. Add the computed column to `data_out_pca` **on the `year` column** so it aligns correctly.

5. Then proceed with `transform_timeseries_timestep(data_out_pca, ...)` as normal — the
   new column will be disaggregated by timestep alongside the existing columns.

6. **Uncomment** the `Effective_Capacity_[W]` block in `update_product_stock`.

**Accuracy caveat** (add as comment in code):
> `# NOTE: Effective_Capacity_[W] is synthetically derived from cumulative merged-datain`
> `# installs minus cumulative EOL waste converted to W using the W→m² ratios from the`
> `# old Solar Futures dataOut files (Yearly_Sum_Area_atEOL / Yearly_Sum_Power_atEOL)`
> `# and material kg/m² factors from pvice_mat_factor. This mass conversion reflects`
> `# the old scenario's panel efficiency and geometry; the resulting W values may`
> `# underestimate or overestimate true effective capacity by 5–15% depending on year.`
> `# Replace with actual Effective_Capacity_[W] from a new PV ICE run once available.`

**Files to change**: `ABM_CE_PV_ConsumerAgents.py` — `__init__` (add column computation
between the `agents_per_pca` divisions and the `transform_timeseries_timestep` call) and
`update_product_stock` (uncomment the `Effective_Capacity_[W]` block).

---

### Re-run PV ICE with merged datain files — longer-term accuracy improvement
- Pending task 2 above gives an *approximation*. A proper re-run of PV ICE with the merged
  `datain` files would yield Weibull-degraded `Effective_Capacity_[W]` values fully
  consistent with the StdScen24 installation history.
- Once those files are available, replace the synthetic column computation with a direct
  read from the new `dataOut` files and remove the approximation comment.
- The `Effective_Capacity_[W]` block in `update_product_stock` can remain uncommented
  regardless; only the column source changes.

---

## Data Flow Summary

```
ReEDS StdScen24 (MW, 3-yr intervals)
    │ ÷3, ×0.85
    ▼
generate_combined_datain.py
    │
    ▼
PV_ICE/TEMP/PCA_merged/datain_95-by-35.Adv_pXX_.csv  (MW, annual)
    │ ÷agents_per_pca, ×1E6
    ▼
consumer.__init__ → self.data_in_pca (W per agent)
    │
    ├──► number_product (capacity stock list, W)
    └──► get_additional_capacity() → new installs per step

PV_ICE/TEMP/PCA_merged/PVICE_PCA_WasteEOL_by_Year_and_PCA.csv  (metric tons, annual)
    │ loaded once at model level → pvice_waste_eol_df
    │ sliced by PCA, ÷agents_per_pca
    ▼
consumer.pv_ice_waste_df → Yearly_Waste_EOL_Ton (metric tons per agent)
    │ ×1000
    ▼
waste in kg → update_eol_volumes, update_generator_size,
              is_hazardous_waste_storage_limit_exceeded,
              tpb_decision
```

---

## Unit Reference

| Variable | Unit | Where |
|----------|------|--------|
| `Yearly_Waste_EOL_Ton` | metric tons | `pv_ice_waste_df`, `pvice_waste_eol_df` |
| `pca_tot_waste_ton` | metric tons | model-level tracker |
| `tot_prod_EoL` | metric tons | per-agent, before ×1000 |
| `number_product_EoL` | metric tons | per-agent |
| `new_eol_vol`, `used_eol_vol` | kg | `update_eol_volumes` |
| `hazardous_waste_mass_kg` | kg | `update_generator_size` |
| `total_mass_stored_kg` | kg | `is_hazardous_waste_storage_limit_exceeded` |
| `new_Installed_Capacity_[MW]` in datain | W (after ×1E6 in `__init__`) | `data_in_pca` |
| `number_product` elements | W | capacity stock list |
