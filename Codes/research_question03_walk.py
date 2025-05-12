import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("merged_access_immigration_RQ3.csv")
df["region"] = df["CMANAME"] + " (" + df["PRNAME"] + ")"

# Compute walk access score
walk_avg = df.groupby("region")["walk_ef"].mean().reset_index()
walk_avg = walk_avg.sort_values("walk_ef", ascending=False).reset_index(drop=True)

# Filter top & bottom valid CMAs
def get_top_valid(df, label, n=3):
    valid_rows = df[~df["region"].str.contains("No metropolitan|Zone d'influence|Outside CAs|Territories", case=False, na=False)].copy()
    selected = valid_rows.head(n).copy()
    selected["WalkLevel"] = label
    return selected

top_valid = get_top_valid(walk_avg, "High Walk", 3)
bottom_valid = get_top_valid(walk_avg.sort_values("walk_ef"), "Low Walk", 3)
selected = pd.concat([top_valid, bottom_valid], ignore_index=True)
selected = selected.sort_values(["WalkLevel", "walk_ef"], ascending=[True, False])

# Merge with original dataset
df = df[df["region"].isin(selected["region"])].copy()
df = df.merge(selected[["region", "WalkLevel", "walk_ef"]], on="region", how="left")

# Compute generational composition
gen_cols = ["T1666", "T1667", "T1668"]
gen_labels = ["First Gen", "Second Gen", "Third Gen"]
grouped = df.groupby("region")[gen_cols].sum()
grouped_pct = grouped.div(grouped.sum(axis=1), axis=0) * 100
grouped_pct.columns = gen_labels
grouped_pct = grouped_pct.merge(selected.set_index("region")[["WalkLevel", "walk_ef"]], left_index=True, right_index=True)

# Append national census bar
census_row = pd.DataFrame([{
    "First Gen": 26.4,
    "Second Gen": 17.6,
    "Third Gen": 56.0,
    "WalkLevel": "National Census 2021",
    "walk_ef": 1.01
}], index=["National Census 2021"])
grouped_pct = pd.concat([grouped_pct, census_row], axis=0)

# Sort rows
order = pd.Categorical(grouped_pct["WalkLevel"], categories=["High Walk", "Low Walk", "National Census 2021"], ordered=True)
grouped_pct["WalkLevel"] = order
grouped_pct = grouped_pct.sort_values(["WalkLevel", "walk_ef"], ascending=[True, False])

# Plot
fig, ax = plt.subplots(figsize=(14, 9))
grouped_pct[gen_labels].plot(kind='barh', stacked=True, ax=ax, color=["#4c72b0", "#dd8452", "#55a868"])

# Annotate access scores
for i, (region, row) in enumerate(grouped_pct.iterrows()):
    label = row["WalkLevel"]
    if label == "National Census 2021":
        ax.text(101, i, str(label), va='center', fontsize=9)
    else:
        score = int(round(row["walk_ef"] * 100))
        ax.text(101, i, f"{label} ({score}%)", va='center', fontsize=9)

# Add % labels inside bar segments
for bar_group in ax.containers:
    for rect in bar_group:
        width = rect.get_width()
        if width > 10:
            x = rect.get_x() + width / 2
            y = rect.get_y() + rect.get_height() / 2
            ax.text(x, y, f"{width:.1f}%", ha='center', va='center', fontsize=8, color='white')

# Final styling
ax.set_title("Generational Composition in Regions by Walk Access to Primary and Secondary Educational Facilities", fontsize=14, pad=20)
ax.set_xlabel("% of Generational Composition")
ax.set_ylabel("Region")
ax.set_xlim(0, 100)
ax.legend(title="Generation", loc='upper left', bbox_to_anchor=(1.02, 0.5), fontsize=10, title_fontsize=11)

plt.tight_layout()
plt.savefig("Generational_Composition_Walk_with_Census.png", dpi=300, bbox_inches='tight')
plt.show()
