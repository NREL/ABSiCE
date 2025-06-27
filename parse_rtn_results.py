import pandas as pd
import os
import ast

EPS = 1e-6

name_conversion_map = {
    "FabTechSolar": "FabTech Solar Solutions",
    "SolarCycle": "SolarCycle",
    "WeRecycleSolar": "We Recycle Solar, Inc.",
    "OkonRecycling": "Okon Recycling",
}

state_conversiomn_map = {
   "Arizona": "AZ",
   "Texas": "TX",
}

total_cost_df = pd.DataFrame(
    {
        "Year": [
            2026, 2027, 2028, 2029, 2030,
            2026, 2027, 2028, 2029, 2030,
            2026, 2027, 2028, 2029, 2030,
            2026, 2027, 2028, 2029, 2030,
        ],

        "Recycler Name": [
            "FabTech Solar Solutions",
            "FabTech Solar Solutions",
            "FabTech Solar Solutions",
            "FabTech Solar Solutions",
            "FabTech Solar Solutions",
            "SolarCycle",
            "SolarCycle",
            "SolarCycle",
            "SolarCycle",
            "SolarCycle",
            "We Recycle Solar, Inc.",
            "We Recycle Solar, Inc.",
            "We Recycle Solar, Inc.",
            "We Recycle Solar, Inc.",
            "We Recycle Solar, Inc.",
            "Okon Recycling",
            "Okon Recycling",
            "Okon Recycling",
            "Okon Recycling",
            "Okon Recycling",
        ],
        "TotalCost": [ 
            77057.16,
            99800.68,
            135819.24,
            208813.94,
            295921.76,
            2716.85,
            3411.23,
            3026.45,
            5601.80,
            6286.21,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            10111.38,
            11508.60,
            12416.88,
            14802.11,
            16144.91,
        ]
    }
)

df = pd.read_excel(os.path.join("..", "solar.case.study.xlsx"))
df[["Recycler", "Flow"]] = df["Unnamed: 0"].str.split(":", expand=True)
df = df[["Recycler", "Flow"]]
df["Recycler"] = df["Recycler"].apply(ast.literal_eval)
df[["PCA", "Recycler Name", "Year"]] = pd.DataFrame(df["Recycler"].to_list())
df = df[["Year", "Recycler Name", "PCA", "Flow"]]
df["Recycler Name"] = df["Recycler Name"].replace(name_conversion_map)
df["Flow"] = df["Flow"].str.strip()
df["Flow"] = pd.to_numeric(df["Flow"])
df["Year"] = df["Year"].astype(int)
df = df.groupby(["Year", "Recycler Name"]).sum(numeric_only=True).reset_index()
df = pd.merge(df, total_cost_df, how='left', on=["Year", "Recycler Name"])
df["Cost"] = df["TotalCost"] / df["Flow"] * 0.0077 * 1000  # Convert to $/kg
df = df[["Year", "Recycler Name", "Cost", "Flow"]]
df.to_csv(os.path.join("RTN", "RecyclingCosts.csv"), index=False)

cost_df = pd.read_csv("/Users/pghosh/SOLAR/ABSiCE/RTN/RecyclingCosts.csv")
recycler_df = pd.read_csv("/Users/pghosh/SOLAR/ABSiCE/RTN/recycler_data.csv")
recycler_df["State"] = recycler_df["State"].replace(state_conversiomn_map)
pca_df = pd.read_excel("/Users/pghosh/SOLAR/ABSiCE/PV_ICE/baselines/SupportingMaterial/December Core Scenarios ReEDS Outputs Solar Futures v3a.xlsx")
state_pca_dict = {}
for group, df in pca_df.groupby("State"):
    state_pca_dict[group] = df["PCA"].unique()
state_pca_df = pd.DataFrame()
for state, pca in state_pca_dict.items():
    df = pd.DataFrame()
    df["PCA"] = pca
    df["State"] = state
    state_pca_df = pd.concat([state_pca_df, df])
merged_pca_df = pd.merge(recycler_df, state_pca_df, on=["State"])
merged_all_df = pd.merge(cost_df, merged_pca_df, how='left', on=["Recycler Name"])
merged_all_df[["Year", "PCA", "Recycler Name", "Cost"]].to_csv("RTN/RecyclingCostsbyYearPCA.csv")