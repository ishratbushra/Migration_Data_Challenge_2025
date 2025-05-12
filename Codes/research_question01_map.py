import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Load ADA-level data
df = pd.read_csv("merged_access_immigration2.csv")
df["pct_recent_immigrants"] = df["T1536"] / (df["T1528"] + df["T1536"]) * 100
df["ADA_code"] = df["ADA_code"].astype(str)

# Load shapefile
gdf = gpd.read_file("lada000b21a_e.shp")
gdf = gdf.rename(columns={"ADAUID": "ADA_code"})
gdf["ADA_code"] = gdf["ADA_code"].astype(str)

# Merge data with geometry
merged = gdf.merge(
    df[["ADA_code", "CMANAME", "PRNAME", "T1528", "T1536", "pct_recent_immigrants"]],
    on="ADA_code", how="left"
)

# Classify by quantile
quantiles = merged["pct_recent_immigrants"].quantile([0.9, 0.8, 0.7])
q90, q80, q70 = quantiles[0.9], quantiles[0.8], quantiles[0.7]

def classify(val):
    if pd.isna(val):
        return "No data"
    elif val >= q90:
        return "Top 10%"
    elif val >= q80:
        return "Next 10%"
    elif val >= q70:
        return "Next 20%"
    else:
        return "Other"

merged["map_category"] = merged["pct_recent_immigrants"].apply(classify)

# Color mapping
color_dict = {
    "Top 10%": "red",
    "Next 10%": "blue",
    "Next 20%": "green",
    "Other": "beige",
    "No data": "lightgrey"
}

# CMA-level % immigrant calculation
cma_data = (
    merged.groupby("CMANAME")[["T1528", "T1536"]]
    .sum()
    .reset_index()
)
# Add province name
province_map = merged.drop_duplicates("CMANAME")[["CMANAME", "PRNAME"]].set_index("CMANAME")["PRNAME"]
cma_data["PRNAME"] = cma_data["CMANAME"].map(province_map)
cma_data["CMA_label"] = cma_data["CMANAME"] + " (" + cma_data["PRNAME"] + ")"
cma_data["pct_recent_immigrants"] = cma_data["T1536"] / (cma_data["T1528"] + cma_data["T1536"]) * 100
cma_data = cma_data.dropna(subset=["pct_recent_immigrants"]).sort_values("pct_recent_immigrants", ascending=False)
cma_data["category"] = cma_data["pct_recent_immigrants"].apply(classify)
cma_data["color"] = cma_data["category"].map(color_dict)

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10), gridspec_kw={'width_ratios': [1.5, 1]})

# Choropleth
merged.plot(
    column="map_category",
    ax=ax1,
    categorical=True,
    color=merged["map_category"].map(color_dict),
    edgecolor="black",
    linewidth=0.2
)
ax1.set_title("% of Recent Immigrants by ADA (2016–2021)", fontsize=14, pad=20)
ax1.axis("off")

# Legend
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Top 10%', markerfacecolor='red', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Next 10%', markerfacecolor='blue', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Next 20%', markerfacecolor='green', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Other', markerfacecolor='beige', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='No data', markerfacecolor='lightgrey', markersize=10)
]
ax1.legend(
    handles=legend_elements,
    title="% Recent Immigrants",
    loc='lower center',
    bbox_to_anchor=(0.5, 1.05),
    ncol=5,
    frameon=False
)

# Bar chart: Top 10 CMAs
top_cmas = cma_data.head(10).iloc[::-1]  # Reverse for horizontal bar order
ax2.barh(top_cmas["CMA_label"], top_cmas["pct_recent_immigrants"], color=top_cmas["color"])
for i, val in enumerate(top_cmas["pct_recent_immigrants"]):
    ax2.text(val + 0.3, i, f"{val:.1f}%", va="center", fontsize=10)

ax2.set_xlabel("% Recent Immigrants")
ax2.set_title("Top CMAs by % Recent Immigrants (with Province)", fontsize=14)
ax2.set_xlim(0, top_cmas["pct_recent_immigrants"].max() + 5)

# Output
plt.tight_layout()
plt.savefig("choropleth_with_top_legend_and_CMANAME_bar_corrected.png", dpi=300, bbox_inches='tight')
plt.show()
