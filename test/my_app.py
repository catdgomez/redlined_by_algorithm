"""
Redlined by Algorithm: Interactive Fair Lending Analysis
HMDA Mortgage Data - Atlanta, Georgia
"""


# ── Dependencies ───────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Redlined by Algorithm",
    page_icon="🏠",
    layout="wide"
)


# ── Title & intro ─────────────────────────────────────────────────────────────
st.title("🏠 Redlined by Algorithm")
st.subheader("Does race or ethnicity affect your chances of getting a mortgage in Atlanta?")
st.markdown("""
This tool looks at real mortgage application data from Atlanta, GA to explore a simple question:

**After accounting for financial factors like income and debt, does a person's race, ethnicity, 
gender, or age still affect whether their mortgage application gets approved?**

Use the sidebar to choose which groups to compare, which financial factors to account for, 
and which types of loans to include. The chart will update instantly.
""")

with st.expander("How does this work?"):
    st.markdown("""
    We're using a statistical model called logistic regression.
    
    We're looking at thousands of mortgage applications and figure out what predicts whether someone gets 
    approved or denied. We can tell it to hold constant certain financial factors, comparing people who 
    look financially similar on paper, and then see if race or ethnicity still makes a difference.
    
    **If it does, that's a sign that something other than finances might be influencing the decision.**
    """)


# ── Read in the data ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    # hmda_all_nf is the full OHE dataset before most/any loan type filters are applied
    df = pd.read_csv("hmda_all_nf.csv")
    return df

try:
    df_raw = load_data()
except FileNotFoundError:
    st.error("⚠️ Data file not found. Make sure `hmda_all_nf.csv` is in the same directory as this app.")
    st.stop()


# ── All available demographic variables ───────────────────────────────────────
all_ethnicity = {
    "ae_values_1":  "Hispanic or Latino",
    "ae_values_11": "Mexican",
    "ae_values_12": "Puerto Rican",
    "ae_values_13": "Cuban",
    "ae_values_14": "Other Hispanic or Latino",
    "ae_values_2":  "Not Hispanic or Latino",
}

all_race = {
    "ar_values_1":  "American Indian or Alaska Native",
    "ar_values_2":  "Asian",
    "ar_values_3":  "Black or African American",
    "ar_values_4":  "Native Hawaiian or Other Pacific Islander",
    "ar_values_5":  "White",
    "ar_values_21": "Asian Indian",
    "ar_values_22": "Chinese",
    "ar_values_23": "Filipino",
    "ar_values_24": "Japanese",
    "ar_values_25": "Korean",
    "ar_values_26": "Vietnamese",
    "ar_values_27": "Other Asian",
    "ar_values_41": "Native Hawaiian",
    "ar_values_42": "Guamanian or Chamorro",
    "ar_values_43": "Samoan",
    "ar_values_44": "Other Pacific Islander",
}


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Build Your Analysis Here")


# ── Dataset filters ───────────────────────────────────────────────────────────
st.sidebar.subheader("Which types of loans do you want to include?")
st.sidebar.markdown("""
By default we include all loan types in the dataset. 
Toggle these on to narrow down to specific kinds of loans.
Each one removes certain applications from the analysis entirely.
""")

filter_reverse = st.sidebar.checkbox(
    "Exclude reverse mortgages",
    value=False,
    help="Reverse mortgages work differently from regular home loans. Older homeowners borrow against equity they already have. Excluding them makes the comparison more apples-to-apples."
)
filter_open_end = st.sidebar.checkbox(
    "Exclude open-end lines of credit (HELOCs)",
    value=False,
    help="A HELOC is a revolving credit line secured by your home. More like a credit card than a traditional mortgage. Excluding these focuses the analysis on standard home loans."
)
filter_business = st.sidebar.checkbox(
    "Exclude business or commercial loans",
    value=False,
    help="Loans taken out for business purposes follow different rules and involve different applicants (companies, investors) than personal home loans."
)
filter_single_family = st.sidebar.checkbox(
    "Only include single-family homes",
    value=False,
    help="Single-family homes (one unit) are the most common type. Excluding multi-unit properties (duplexes, apartment buildings) focuses on individual homebuyers."
)
filter_home_purchase = st.sidebar.checkbox(
    "Only include home purchase loans",
    value=False,
    help="This excludes refinancing and home improvement loans. Home purchases are the clearest test of who can buy a home. Refinancing involves people who already own one."
)
filter_primary_residence = st.sidebar.checkbox(
    "Only include primary residences",
    value=False,
    help="Excludes second homes and investment properties. Focusing on primary residences zeroes in on people trying to buy the home they actually live in."
)

st.sidebar.markdown("---")


# ── Baseline selection ────────────────────────────────────────────────────────
st.sidebar.subheader("Who is the comparison group?")
st.sidebar.markdown("""
The **comparison group** (baseline) is who everyone else gets compared to.
All approval odds in the chart are shown relative to this group.
""")

eth_baseline_options  = {v: k for k, v in all_ethnicity.items()}
race_baseline_options = {v: k for k, v in all_race.items()}

eth_baseline_label = st.sidebar.selectbox(
    "Ethnic identity baseline",
    options=list(eth_baseline_options.keys()),
    index=list(eth_baseline_options.keys()).index("Not Hispanic or Latino"),
    help="All ethnic identity groups will be compared to this group"
)
eth_baseline_var = eth_baseline_options[eth_baseline_label]

race_baseline_label = st.sidebar.selectbox(
    "Racial identity baseline",
    options=list(race_baseline_options.keys()),
    index=list(race_baseline_options.keys()).index("White"),
    help="All racial identity groups will be compared to this group"
)
race_baseline_var = race_baseline_options[race_baseline_label]

st.sidebar.markdown("---")


# ── Ethnicity group selection ─────────────────────────────────────────────────
st.sidebar.subheader("Which ethnicity groups do you want to include?")
st.sidebar.markdown(f"Each group is compared to **{eth_baseline_label}**.")

selected_eth_vars = []
for var, label in all_ethnicity.items():
    if var == eth_baseline_var:
        continue
    if st.sidebar.checkbox(label, value=True, key=f"eth_{var}"):
        selected_eth_vars.append(var)

st.sidebar.markdown("---")


# ── Racial group selection ──────────────────────────────────────────────────────
st.sidebar.subheader("Which racial groups do you want to include?")
st.sidebar.markdown(f"Each group is compared to **{race_baseline_label}**.")

selected_race_vars = []
for var, label in all_race.items():
    if var == race_baseline_var:
        continue
    if st.sidebar.checkbox(label, value=True, key=f"race_{var}"):
        selected_race_vars.append(var)

st.sidebar.markdown("---")
