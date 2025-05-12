import pandas as pd
import matplotlib.pyplot as plt

# Load and filter data
df = pd.read_csv("merged_access_immigration_RQ2.csv")
exclude = ["No metropolitan influenced zone", "Strong metropolitan influenced zone", "Moderate metropolitan influenced zone"]
df = df[~df["CMANAME"].str.contains('|'.join(exclude), case=False, na=False)]
df["Region"] = df["CMANAME"] + " (" + df["PRNAME"] + ")"

# Metric and columns
metric = "public_emp"
immigrant_cols = ["T1670", "T1673", "T1674", "T1675"]

# Top and bottom 3 regions
cma_scores = df.groupby("Region")[metric].mean().reset_index()
top3 = cma_scores.nlargest(3, metric).copy()
bottom3 = cma_scores.nsmallest(3, metric).copy()

top3["AccessType"] = "High Access"
bottom3["AccessType"] = "Low Access"
combined = pd.concat([top3, bottom3])
df["AccessType"] = df["Region"].map(dict(zip(combined["Region"], combined["AccessType"])))
df = df[df["Region"].isin(combined["Region"])]

# Immigration class composition
comp = df.groupby("Region")[immigrant_cols].sum()
comp_pct = comp.div(comp.sum(axis=1), axis=0) * 100
comp_pct = comp_pct.rename(columns={
    "T1670": "Economic", "T1673": "Family", "T1674": "Refugee", "T1675": "Other"
})

# Add National Census
census = pd.Series({
    "Economic": 53.9,
    "Family": 29.6,
    "Refugee": 15.2,
    "Other": 1.3
}, name="National Census 2021")
comp_pct.loc["National Census 2021"] = census

# Bar order: top 3 → bottom 3 → census
label_order = (
    top3.sort_values(by=metric, ascending=False)["Region"].tolist() +
    bottom3.sort_values(by=metric, ascending=False)["Region"].tolist() +
    ["National Census 2021"]
)
comp_pct = comp_pct.loc[label_order]

# Plot
fig, ax = plt.subplots(figsize=(14, 7))
comp_pct.plot(
    kind='barh',
    stacked=True,
    ax=ax,
    color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
)

# Annotate bars with access level and score
for i, label in enumerate(comp_pct.index):
    if label == "National Census 2021":
        ax.text(101, i, label, va='center', fontsize=9)
    else:
        access_type = df[df["Region"] == label]["AccessType"].iloc[0]
        pct = df[df["Region"] == label][metric].mean()
        if pd.notna(pct):
            ax.text(101, i, f"{access_type} ({pct:.1%})", va='center', fontsize=9)


# Add % labels inside bar segments
for bar_group in ax.containers:
    for rect in bar_group:
        width = rect.get_width()
        if width > 3:
            x = rect.get_x() + width / 2
            y = rect.get_y() + rect.get_height() / 2
            ax.text(x, y, f"{width:.1f}%", ha='center', va='center', fontsize=8, color='white')

# Final plot styling
ax.set_title("Immigration Class Composition by Public Transit Access to Employment", fontsize=13, pad=20)
ax.set_xlabel("% Immigration Class Composition")
ax.set_xlim(0, 100)
ax.legend(title="Immigration Class", loc="upper right", bbox_to_anchor=(0, 1.05), ncol=4)

plt.tight_layout()
plt.savefig("Public_Transit_Immigration_Class_Composition_Horizontal_Cleaned.png", dpi=300)
plt.show()
