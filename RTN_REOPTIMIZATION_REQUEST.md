# RTN Re-Optimization Request: Apex Regional Landfill Fee Correction

## Summary
Round_3_2 RTN shipments were optimized with an **incorrect Apex Regional Landfill fee**. The routing decisions (which facility each site routes waste to and in what quantities) are now frozen with suboptimal cost assumptions. RTN optimization must be re-run with the corrected fee to ensure accurate cost basis for ABM sensitivity analysis.

---

## The Problem

### 1. Cost Correction Details
- **Old (incorrect) Apex fee:** $0.045359235 / kg
- **New (corrected) Apex fee:** $0.049603950 / kg  
- **Difference:** +9.3% increase

This 9.3% fee increase significantly changes the cost-benefit of routing waste to Apex Regional Landfill, especially for high-capacity NV facilities.

### 2. Concrete Example: Steamboat II (p12 PCA, Nevada)

**Current Round_3_2 Shipments (optimized with old fee):**

| Configuration | Landfill | Shipped_kg | Transport_$ | LandfillFee_$/kg | Total_$ | Cost/ton |
|---|---|---|---|---|---|---|
| All Landfills | Apex Regional | 105.17 | $9,642 | $0.0496 | $9,647 | **$91,734** |
| True Landfills | Apex Regional | 210.33 | $9,642 | $0.0496 | $9,653 | **$45,892** |

**Why the paradox?**
- True Landfills consolidates **2× more waste** (210.33 vs 105.17 kg) onto the **same truck** ($9,642 fixed transport cost)
- This halves the per-ton cost ($45,892 vs $91,734)
- **BUT:** RTN made this routing decision when Apex fee was only $0.0454/kg
- With the higher fee ($0.0496/kg), RTN should re-evaluate:
  - Is Apex still the optimal choice for true_landfills waste?
  - Should waste be routed to a different facility instead?
  - How does the consolidation trade-off change with the higher fee?

### 3. Impact of Manual Cost Update (Without Re-optimization)

The current approach manually updates the cost column in existing Round_3_2 shipments:
- ✗ Shipment routing decisions = **FROZEN** (made with old Apex fee)
- ✗ Quantities routed = **FROZEN** (made with old Apex fee)  
- ✗ Consolidation patterns = **FROZEN** (made with old Apex fee)
- ✓ Landfill cost files = Updated
- ⚠️ ABM receives cost data from **suboptimal** RTN routing

This means the ABM sensitivity analysis runs against cost data that doesn't represent true cost optimization.

---

## Required Action

**Re-run RTN optimization with corrected Apex Regional Landfill fee ($0.049603950 / kg)**

This will generate a new Round (e.g., Round_3_2_Corrected or Round_3_3) shipments file where:
1. Routing decisions reflect **true costs** (corrected Apex fee)
2. Quantities and consolidation patterns reflect **optimal transport utilization** under new fee structure
3. True_landfills vs all_landfills routing may differ substantially
4. Landfill cost files derived from these shipments will be cost-optimal

---

## Files Affected

### Input (to RTN optimization)
- Apex Regional Landfill fee: Update from $0.045359235 to $0.049603950 per kg

### Output (new RTN shipments)
- `shipments_landfill_alllandfills.csv` (re-generated)
- `shipments_landfill_truelandfills.csv` (re-generated)

### Downstream (ABM pipeline)
Once corrected shipments are available:
1. Run `ABSiCE/generate_landfill_costs.py` with corrected shipments
2. Run ABM sensitivity analysis with corrected cost files

---

## Timeline

- **Jun 23:** Round_3_2 shipments generated (with old Apex fee 0.0454)
- **Aug 13:** Apex fee correction identified and applied manually to Round_3_2 shipments
- **Aug 17:** Landfill cost files regenerated from manually-corrected shipments
- **Aug 18:** Discovered routing decisions were never re-optimized
- **Now:** Request re-run RTN optimization with corrected fee

---

## Questions for RTN Team

1. What is the standard workflow to update input costs and re-run optimization?
2. ETA for re-running RTN with corrected Apex fee?
3. Should we use a new Round naming (e.g., Round_3_2_Corrected) or overwrite Round_3_2?

