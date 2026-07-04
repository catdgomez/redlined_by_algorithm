"""
Redlined by Algorithm: Interactive Fair Lending Analysis
Home Mortgage Disclosure Act (HMDA) Mortgage Data from Atlanta, Georgia
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
st.subheader("Does race, ethnicity, or age affect your chances of getting a mortgage in Atlanta?")
with st.expander("Why are we here?"):
    st.markdown("""
    This tool looks at real mortgage application data from Atlanta, GA to explore a simple question:

    **After accounting for financial factors like income and debt, does a person's race, ethnicity, 
    or age still affect whether their mortgage application gets approved?**

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
    "is_hispanic":  "Any Hispanic or Latino (combined)",
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

# hmda age bins --> baseline will be selected by app user
all_age = {
    "applicant_age_lt25":  "Under 25",
    "applicant_age_25_34": "25-34",
    "applicant_age_35_44": "35-44",
    "applicant_age_45_54": "45-54",
    "applicant_age_55_64": "55-64",
    "applicant_age_65_74": "65-74",
    "applicant_age_gt74":  "Over 74",
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
    help="A reverse mortgage is when a homeowner, (usually older), borrows money using their home as collateral, and the bank pays them instead of the other way around. These work so differently from a regular home purchase loan that including them can muddy the results."
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
age_baseline_options  = {v: k for k, v in all_age.items()}

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

age_baseline_label = st.sidebar.selectbox(
    "Age identity baseline",
    options=list(age_baseline_options.keys()),
    index=list(age_baseline_options.keys()).index("35-44"),
    help="All age groups will be compared to this group"
)
age_baseline_var = age_baseline_options[age_baseline_label]

st.sidebar.markdown("---")


# ── Ethnicity group selection ─────────────────────────────────────────────────
st.sidebar.subheader("Which ethnicity groups do you want to include?")
st.sidebar.markdown(f"Each group is compared to **{eth_baseline_label}**.")

selected_eth_vars = []
for var, label in all_ethnicity.items():
    if var == eth_baseline_var:
        continue
    if st.sidebar.checkbox(label, value=False, key=f"eth_{var}"):
        selected_eth_vars.append(var)

st.sidebar.markdown("---")


# ── Racial group selection ──────────────────────────────────────────────────────
st.sidebar.subheader("Which racial groups do you want to include?")
st.sidebar.markdown(f"Each group is compared to **{race_baseline_label}**.")

selected_race_vars = []
for var, label in all_race.items():
    if var == race_baseline_var:
        continue
    if st.sidebar.checkbox(label, value=False, key=f"race_{var}"):
        selected_race_vars.append(var)

st.sidebar.markdown("---")


# ── Age group selection ─────────────────────────────────────────────────
st.sidebar.subheader("Which age groups do you want to include?")
st.sidebar.markdown(f"Each group is compared to **{age_baseline_label}**.")

# ── Age group selection ───────────────────────────────────────────────────────
st.sidebar.subheader("Age groups")
st.sidebar.markdown("""
Age is a protected class under the Equal Credit Opportunity Act.
""")

include_age_section = st.sidebar.checkbox(
    "Include age in the analysis", value=False,
    help="Add age as a demographic variable to examine"
)

selected_age_vars = []
age_baseline_label = "35-44"

if include_age_section:
    age_baseline_label = st.sidebar.selectbox(
        "Age baseline (comparison group)",
        options=list(age_baseline_options.keys()),
        index=list(age_baseline_options.keys()).index("35-44"),
        help="All other age groups will be compared to this age group"
    )
    age_baseline_var = age_baseline_options[age_baseline_label]

    st.sidebar.markdown(f"Each age group is compared to **{age_baseline_label}**.")

    also_include_62_flag = st.sidebar.checkbox(
        "Also include Age 62+ as a single indicator",
        value=False,
        help="Adds a simple yes or no flag for whether the applicant is 62 or older, in addition to the age bins"
    )

    for var, label in all_age.items():
        if var == age_baseline_var:
            continue
        if st.sidebar.checkbox(label, value=False, key=f"age_{var}"):
            selected_age_vars.append(var)

    if also_include_62_flag:
        selected_age_vars.append("applicant_age_above_62")

st.sidebar.markdown("---")

# ── Expanders ─────────────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("How to read this chart"):
    st.markdown("""
    **The dot** shows the approval odds for that group compared to the baseline group you selected.

    **The line** through the dot is the uncertainty range. Sort of like a margin of error. 
    A wider line means we have less certainty because there are fewer people in that group.

    **The dashed line** is the "no difference" mark. 
    - If the dot is to the **left**, that group has **lower** approval odds than the baseline
    - If the dot is to the **right**, that group has **higher** approval odds than the baseline
    - If the line **crosses** the dashed line, the difference might just be random chance

    **Red bars** mean the difference is large enough that we're confident it's real, not just noise.

    **Colors by group type:**
    - 🔵 Blue = Ethnicity groups
    - 🟢 Green = Race groups
    - 🟠 Orange = Age groups
    - 🔴 Red = Statistically significant (any type)

    **Try switching the baseline group** in the sidebar to see the data from a different perspective.

    **Try unchecking financial factors** and you might see the disparities get larger, 
    which suggests those factors were masking some of the gap.

    **Try toggling the loan filters** and narrowing to just home purchases or single-family homes 
    may sharpen or change the patterns you see.
    """)

with st.expander("A note on age in this data"):
    st.markdown("""
    Home Mortgage Disclosure Act (HMDA) reports age in bins rather than exact ages:
    -->  Under 25, 25-34, 35-44, 45-54, 55-64, 65-74, Over 74

    There is also an **Age 62+ flag** which is a simple yes or no used in fair lending analysis 
    because reverse mortgages (which work very differently from standard mortgages) are 
    only available to homeowners 62 and older.

    Under the **Equal Credit Opportunity Act**, lenders cannot discriminate based on age 
    (specifically protecting applicants 40 and older). If older applicants show lower 
    approval odds than younger applicants after controlling for financial factors, 
    that's worth investigating.
    """)

with st.expander("What do the loan filters mean?"):
    st.markdown("""
    **Reverse mortgages** is a special type of loan where older homeowners borrow against the equity 
    in their home. Very different from regular mortgages.

    **Open-end lines of credit or Home Equity Line of Credits (HELOCs)** are like a credit card 
    secured by your home. Very different from a standard mortgage loan.

    **Business or commercial purpose loans** are loans taken out by businesses or investors rather 
    than individual homebuyers.

    **Single-family homes only** will limit the analysis to houses with one unit.

    **Home purchase only** will exclude refinancing and home improvement loans.

    **Primary residences only** excludes second homes and investment properties.
    """)

with st.expander("Why does this matter?"):
    st.markdown("""
    Federal law, specifically the __**Equal Credit Opportunity Act**__, says lenders cannot discriminate 
    based on race, color, religion, national origin, sex, marital status, or age.

    But discrimination doesn't always look obvious. Sometimes it shows up in patterns across thousands 
    of decisions where one group consistently gets worse outcomes even when their finances look the same.

    That's what this tool helps you see. It's simliar to the kind of analysis that bank regulators and 
    civil rights lawyers use to investigate fair lending violations.

    **This data is from 2020-2024, covers the Atlanta metro area, and comes from the Home Mortgage 
    Disclosure Act (HMDA) which is a federal database that lenders are required to report to.**
    """)
    
st.markdown("---")

