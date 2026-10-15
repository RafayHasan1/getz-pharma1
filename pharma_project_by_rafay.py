
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(
    page_title="Getz Pharma | Customer Insights",
    page_icon="💊",
    layout="wide"
)

st.title("💊 Getz Pharma Customer Insights")
st.subheader("Customer Segmentation with Unsupervised Learning")
st.write(
    "This app uses K-Means to group customers based on their ordering, "
    "payment and purchasing patterns. The groups are created by the model, "
    "not from pre-existing customer labels."
)

DATA_FILE = "getz_pharma_customer_segmentation_1000.csv"

@st.cache_data
def load_data(file_name):
    return pd.read_csv(file_name)

try:
    df = load_data(DATA_FILE)
except FileNotFoundError:
    st.error(
        f"Could not find {DATA_FILE}. Put the CSV file in the same folder "
        "as pharma.py and run the app again."
    )
    st.stop()

required_columns = [
    "Customer_ID", "City", "Monthly_Orders", "Avg_Order_Value_PKR",
    "Products_Purchased", "Payment_Delay_Days", "Return_Rate_Percent",
    "Months_With_Company", "Days_Since_Last_Order", "Monthly_Spend_PKR"
]
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    st.error("These required columns are missing: " + ", ".join(missing_columns))
    st.stop()


a, b, c, d = st.columns(4)
a.metric("Total Customers", f"{len(df):,}")
b.metric("Cities Covered", df["City"].nunique())
c.metric("Orders per Month (avg.)", f"{df['Monthly_Orders'].mean():.1f}")
d.metric("Average Monthly Spend", f"PKR {df['Monthly_Spend_PKR'].mean():,.0f}")

data_tab, method_tab, results_tab, insights_tab = st.tabs([
    "📋 Data Overview",
    "🔎 Clustering Setup",
    "📊 Segmentation Results",
    "💡 Key Takeaways"
])

with data_tab:
    st.header("1. Get familiar with the dataset")
    st.write(
        "First, inspect a sample of the records and check whether any values "
        "are missing before training the model."
    )
    st.dataframe(df.head(15), use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.subheader("Missing values by column")
        missing_df = (
            df.isna().sum()
            .rename("Missing Values")
            .rename_axis("Column")
            .reset_index()
        )
        st.dataframe(missing_df, use_container_width=True, hide_index=True)
    with right:
        st.subheader("Customer type breakdown")
        type_counts = df["Customer_Type"].value_counts().rename_axis(
            "Customer Type"
        ).reset_index(name="Customers")
        st.bar_chart(type_counts, x="Customer Type", y="Customers")

    st.subheader("Basic numerical summary")
numeric_columns = df.select_dtypes(include="number")

if not numeric_columns.empty:
    summary = numeric_columns.describe().round(0)
    st.dataframe(summary, width="stretch")
else:
    st.warning("No numerical columns found in the dataset.")

features = [
    "Monthly_Orders",
    "Avg_Order_Value_PKR",
    "Products_Purchased",
    "Payment_Delay_Days",
    "Return_Rate_Percent",
    "Months_With_Company",
    "Days_Since_Last_Order"
]

X = df[features].copy()
X = X.fillna(X.median(numeric_only=True))
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

with method_tab:
    st.header("2. Decide how many segments to use")
    st.write(
        "The Elbow Method compares different cluster counts. We look for a "
        "point where adding more clusters gives smaller improvements."
    )

    max_k = min(8, len(df))
    k_values = list(range(1, max_k + 1))
    inertia_values = []

    for k in k_values:
        trial_model = KMeans(n_clusters=k, random_state=24, n_init=10)
        trial_model.fit(X_scaled)
        inertia_values.append(trial_model.inertia_)

    elbow_df = pd.DataFrame({"Number of Clusters": k_values, "Inertia": inertia_values})
    st.line_chart(elbow_df, x="Number of Clusters", y="Inertia")

    st.caption(
        "Use the bend in the curve as a guide. The best K depends on the data "
        "and should also make sense when you compare the segment profiles."
    )
    selected_k = st.slider(
        "Select cluster count (K)",
        min_value=2,
        max_value=min(6, len(df)),
        value=3,
        step=1
    )

    st.write("**Features used by the model:**")
    st.write(", ".join(features))
    st.info(
        "StandardScaler puts the selected features on a comparable scale so "
        "large-valued columns do not dominate only because of their units."
    )


model = KMeans(n_clusters=selected_k, random_state=24, n_init=10)
df_clustered = df.copy()
df_clustered["Cluster"] = model.fit_predict(X_scaled)


cluster_profile = df_clustered.groupby("Cluster").agg(
    Customers=("Customer_ID", "count"),
    Avg_Monthly_Orders=("Monthly_Orders", "mean"),
    Avg_Order_Value_PKR=("Avg_Order_Value_PKR", "mean"),
    Avg_Products=("Products_Purchased", "mean"),
    Avg_Payment_Delay_Days=("Payment_Delay_Days", "mean"),
    Avg_Return_Rate=("Return_Rate_Percent", "mean"),
    Avg_Days_Since_Last_Order=("Days_Since_Last_Order", "mean"),
    Avg_Monthly_Spend_PKR=("Monthly_Spend_PKR", "mean")
).round(1)

if selected_k == 3:
    spend_order = cluster_profile["Avg_Monthly_Spend_PKR"].sort_values().index.tolist()
    names = {
        spend_order[0]: "Lower Spending",
        spend_order[1]: "Mid-range Spending",
        spend_order[2]: "Higher Spending"
    }
    df_clustered["Segment"] = df_clustered["Cluster"].map(names)
else:
    df_clustered["Segment"] = "Segment " + (df_clustered["Cluster"] + 1).astype(str)

segment_profile = df_clustered.groupby("Segment").agg(
    Customers=("Customer_ID", "count"),
    Avg_Monthly_Orders=("Monthly_Orders", "mean"),
    Avg_Order_Value_PKR=("Avg_Order_Value_PKR", "mean"),
    Avg_Products=("Products_Purchased", "mean"),
    Avg_Payment_Delay_Days=("Payment_Delay_Days", "mean"),
    Avg_Return_Rate_Percent=("Return_Rate_Percent", "mean"),
    Avg_Months_With_Company=("Months_With_Company", "mean"),
    Avg_Days_Since_Last_Order=("Days_Since_Last_Order", "mean"),
    Avg_Monthly_Spend_PKR=("Monthly_Spend_PKR", "mean")
).round(1).reset_index()

with results_tab:
    st.header("3. Explore the segments")
    counts = df_clustered["Segment"].value_counts().rename_axis(
        "Segment"
    ).reset_index(name="Customers")

    st.subheader("How many customers are in each segment?")
    st.bar_chart(counts, x="Segment", y="Customers")

    st.subheader("Compare ordering behaviour")
    st.scatter_chart(
        df_clustered,
        x="Monthly_Orders",
        y="Avg_Order_Value_PKR",
        color="Segment",
        size="Products_Purchased",
        x_label="Orders per Month",
        y_label="Average Order Value (PKR)"
    )

    st.subheader("Average profile for each segment")
    st.dataframe(segment_profile, use_container_width=True, hide_index=True)

    st.subheader("View customers in a segment")
    chosen_segment = st.selectbox(
        "Select a segment to inspect",
        options=sorted(df_clustered["Segment"].unique())
    )
    st.dataframe(
        df_clustered[df_clustered["Segment"] == chosen_segment],
        use_container_width=True,
        hide_index=True
    )

with insights_tab:
    st.header("4. What can we learn?")
    st.write(
        "The segment names are descriptive labels added after clustering. "
        "They are not labels that were supplied to K-Means during training."
    )

    highest_spend_segment = segment_profile.loc[
        segment_profile["Avg_Monthly_Spend_PKR"].idxmax()
    ]
    lowest_spend_segment = segment_profile.loc[
        segment_profile["Avg_Monthly_Spend_PKR"].idxmin()
    ]

    st.markdown(
        f"- **Highest average monthly spend:** {highest_spend_segment['Segment']} "
        f"(about PKR {highest_spend_segment['Avg_Monthly_Spend_PKR']:,.0f})."
    )
    st.markdown(
        f"- **Lowest average monthly spend:** {lowest_spend_segment['Segment']} "
        f"(about PKR {lowest_spend_segment['Avg_Monthly_Spend_PKR']:,.0f})."
    )

    st.subheader("Ideas for the business")
    st.write(
        "- Review ordering frequency and recent activity to identify customers "
        "who may need follow-up.\n"
        "- Compare payment delays across segments before deciding where payment "
        "support or credit reviews may be useful.\n"
        "- Check return rates by segment to see whether any group needs a closer "
        "look at product or order issues."
    )
    st.caption(
        "These are exploratory findings, not proof of why customers behave this way. "
        "Discuss them with business teams before making decisions."
    )

st.divider()
st.caption("Student project • K-Means clustering • Getz Pharma sample dataset")
