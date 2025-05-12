import pandas as pd
import plotly.express as px

# Load the immigration data
df = pd.read_csv("RQ1_What is the density of non-immigrants and recent immigrants (2016–2021) in each ADA - RQ1_What is the density of non-immigrants and recent immigrants (2016–2021) in each ADA.csv")

# Compute total population and recent immigrant percentage
df["total_pop"] = df["Sum of T1528"] + df["Sum of T1536"]
df["pct_recent_immigrants"] = (df["Sum of T1536"] / df["total_pop"]) * 100
df = df.dropna(subset=["pct_recent_immigrants"])

# Group by CMANAME to get top CMAs
grouped = df.groupby("CMANAME")[["Sum of T1528", "Sum of T1536"]].sum().reset_index()
grouped["pct_recent_immigrants"] = (grouped["Sum of T1536"] / (grouped["Sum of T1528"] + grouped["Sum of T1536"])) * 100

# Get top 20 CMAs by % recent immigrants
top20 = grouped.sort_values(by="pct_recent_immigrants", ascending=False).head(20)

# Create horizontal bar chart with color legend title
fig = px.bar(
    top20,
    x="pct_recent_immigrants",
    y="CMANAME",
    orientation="h",
    title="Top 20 CMAs by % of Recent Immigrants (2016–2021)",
    text="pct_recent_immigrants",
    color="pct_recent_immigrants",
    color_continuous_scale="Blues",
    labels={"pct_recent_immigrants": "% Recent Immigrants"}
)

fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")

fig.update_layout(
    xaxis_title="% Recent Immigrants",
    yaxis_title="Census Metropolitan Area",
    margin=dict(l=60, r=90, t=80, b=150),
    coloraxis_colorbar=dict(
        title="Color Scale:\n% of Recent Immigrants",
        ticksuffix="%",
        tickvals=[top20["pct_recent_immigrants"].min(), top20["pct_recent_immigrants"].max()],
        ticktext=["Light Blue = Low", "Dark Blue = High"],
        thickness=15,
        len=0.75
    )
)

# Save and show
fig.write_html("top20_recent_immigrants.html")
fig.show()
