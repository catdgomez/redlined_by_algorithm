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
st.title("Redlined by Algorithm")
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

with st.expander("What are we looking at?"):
    st.markdown("""
This tool looks at real mortgage application data from Atlanta, GA across four lenses:

1: Odds Chart - After controlling for financial factors, how much better or worse are approval odds by demographic group?\n
2: Probability Chart - What is the estimated approval probability for a typical applicant in each group?\n
3: Lenders Chart - Which specific lenders have the lowest predicted approval probabilities for the group you select?\n
4: Underwriting Systems Chart - Which automated underwriting systems produce the lowest predicted approval probabilities?\n

    """)

with st.expander("What's the break down?"):
    st.markdown("""
    The **odds and probability charts** use a logistic regression model, controlling for financial 
    factors like income and debt, to isolate the effect of race, ethnicity, and age on approval outcomes.
    
    The **lender and AUS charts** take the model's predicted probability for every individual 
    application and average those predictions by lender or underwriting system. This means the 
    lender and AUS charts are also model-based, they show predicted probabilities, not raw 
    approval rates, so financial differences between applicants are already baked in.
    
    **Approval rate** = raw percentage of applications approved (no adjustments).  

    **Predicted probability** = what the model estimates each applicant's approval chance to be, 
    given their financial profile and demographic group. Averaging these by lender or AUS tells 
    you where the model predicts the lowest chances of approval.
    """)


# ── Read in the data ──────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("hmda_all_nf.csv")
    return df

try:
    df_raw = load_data()
except FileNotFoundError:
    st.error("Data file not found. Make sure `hmda_all_nf.csv` is in the same folder as this app file.")
    st.stop()
except Exception as e:
    st.error(f"Something went wrong loading the data: {e}")
    st.stop()


# ── All available demographic variables ───────────────────────────────────────
all_ethnicity = {
    "is_hispanic":  "Any Hispanic or Latino (combined)",
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

age_bins_to_show = {
    "applicant_age_lt25":  "Under 25",
    "applicant_age_25_34": "25-34",
    "applicant_age_45_54": "45-54",
    "applicant_age_55_64": "55-64",
    "applicant_age_65_74": "65-74",
    "applicant_age_gt74":  "Over 74",
}

aus_labels = {
    1:    "Desktop Underwriter (DU / DO)",
    2:    "Loan Prospector / Loan Product Advisor (LP / LPA)",
    3:    "Technology Open to Approved Lenders (TOTAL)",
    4:    "Guaranteed Underwriting System (GUS)",
    5:    "Other",
    7:    "Internal Proprietary System",
}


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Build Your Analysis Here")


# ── Dataset filters ───────────────────────────────────────────────────────────
st.sidebar.subheader("Which types of loans to include?")
st.sidebar.markdown("""
By default we include all loan types in the dataset. 
Check these boxes to narrow down to specific kinds of loans.
Each one removes certain applications from the analysis entirely.
""")

filter_reverse = st.sidebar.checkbox("Exclude reverse mortgages", value=False,
    help="A reverse mortgage is when a homeowner borrows money using their home as collateral and the bank pays them. Very different from a regular home purchase loan.")
filter_open_end = st.sidebar.checkbox("Exclude open-end lines of credit (HELOCs)", value=False,
    help="A revolving credit line secured by your home - more like a credit card than a traditional mortgage.")
filter_business = st.sidebar.checkbox("Exclude business or commercial loans", value=False,
    help="Loans taken out for business purposes follow different rules than personal home loans.")
filter_single_family = st.sidebar.checkbox("Only include single-family homes", value=False,
    help="Excludes multi-unit properties to focus on individual homebuyers.")
filter_home_purchase = st.sidebar.checkbox("Only include home purchase loans", value=False,
    help="Excludes refinancing and home improvement loans.")
filter_primary_residence = st.sidebar.checkbox("Only include primary residences", value=False,
    help="Excludes second homes and investment properties.")

st.sidebar.markdown("---")


# ── Baseline selection ────────────────────────────────────────────────────────
st.sidebar.subheader("Comparison group or baseline")
st.sidebar.markdown("Everyone else is compared to this group.")

eth_baseline_options  = {v: k for k, v in all_ethnicity.items()}
race_baseline_options = {v: k for k, v in all_race.items()}

eth_baseline_label = st.sidebar.selectbox(
    "Ethnicity baseline",
    options=list(eth_baseline_options.keys()),
    index=list(eth_baseline_options.keys()).index("Not Hispanic or Latino"),
)
eth_baseline_var = eth_baseline_options[eth_baseline_label]

race_baseline_label = st.sidebar.selectbox(
    "Race baseline",
    options=list(race_baseline_options.keys()),
    index=list(race_baseline_options.keys()).index("White"),
)
race_baseline_var = race_baseline_options[race_baseline_label]

st.sidebar.markdown("---")


# ── Ethnicity group selection ─────────────────────────────────────────────────
st.sidebar.subheader("Ethnicity groups")
st.sidebar.markdown(f"Compared to **{eth_baseline_label}**.")

selected_eth_vars = []
for var, label in all_ethnicity.items():
    if var == eth_baseline_var:
        continue
    if st.sidebar.checkbox(label, value=False, key=f"eth_{var}"):
        selected_eth_vars.append(var)

st.sidebar.markdown("---")


# ── Race group selection ──────────────────────────────────────────────────────
st.sidebar.subheader("Racial groups")
st.sidebar.markdown(f"Compared to **{race_baseline_label}**.")

selected_race_vars = []
for var, label in all_race.items():
    if var == race_baseline_var:
        continue
    if st.sidebar.checkbox(label, value=False, key=f"race_{var}"):
        selected_race_vars.append(var)

st.sidebar.markdown("---")


# ── Age group selection ───────────────────────────────────────────────────────
st.sidebar.subheader("Age groups")

include_age_section = st.sidebar.checkbox("Include age in the analysis", value=False)
selected_age_vars = []

if include_age_section:
    st.sidebar.markdown("**Age groups** compared to **25-34** (baseline):")
    for var, label in age_bins_to_show.items():
        if st.sidebar.checkbox(label, value=False, key=f"age_{var}"):
            selected_age_vars.append(var)
    st.sidebar.markdown("**Age 62+ flag** is a separate column:")
    if st.sidebar.checkbox("Include Age 62 or older indicator", value=False, key="age_above_62",
                            help="A yes or no flag from a separate column can be used alongside or instead of the age groups."):
        selected_age_vars.append("applicant_age_above_62")

st.sidebar.markdown("---")


# ── Target variable ───────────────────────────────────────────────────────────
st.sidebar.subheader("Target Variable or Outcome to predict")
target_options = {"Approved and Originated vs. Denied": "approved_originated_or_denied"}
target_label = st.sidebar.selectbox("Target Variable", options=list(target_options.keys()), index=0)
target_var = target_options.get(target_label, target_label)

st.sidebar.markdown("---")

# ── Control variable checkboxes ──────────────────────────────────────────────────
st.sidebar.subheader("Financial factors to hold constant")
st.sidebar.markdown("The more you check, the more we isolate the demographic factor.")

include_income         = st.sidebar.checkbox("Income", value=True,
                                            help="The applicant's reported income")
include_dti            = st.sidebar.checkbox("Debt-to-Income Ratio", value=True,
                                              help="How much of their income goes toward debt payments each month")
include_ltv            = st.sidebar.checkbox("Loan-to-Value Ratio", value=True,
                                             help="How much they're borrowing compared to what the home is worth")
include_loan_type      = st.sidebar.checkbox("Loan Type - BUT ISNT THIS A FILTER?", value=False,
                                              help="Turn OFF to see full disparity including loan product steering effects")
include_loan_purpose   = st.sidebar.checkbox("Loan Purpose", value=False)
include_lien_status    = st.sidebar.checkbox("Lien Status", value=False)
include_occupancy      = st.sidebar.checkbox("Occupancy Type", value=False)
include_tract_minority = st.sidebar.checkbox("Neighborhood Racial Composition", value=True)
include_conforming     = st.sidebar.checkbox("Conforming Loan Limit", value=False)

st.sidebar.markdown("---")


# ── Institution and AUS chart settings ───────────────────────────────────────
st.sidebar.subheader("Lender and AUS chart settings")

min_applications = st.sidebar.slider(
    "Minimum applications per lender", min_value=10, max_value=500, value=50, step=10,
    help="Only show lenders with at least this many applications from the selected group"
)
top_n_lenders = st.sidebar.slider(
    "Number of lenders to show", min_value=5, max_value=30, value=15, step=5,
)
aus_min = st.sidebar.slider(
    "Minimum applications per AUS system", min_value=5, max_value=200, value=20, step=5,
)
show_comparison = st.sidebar.checkbox(
    "Show comparison group on lender and AUS charts", value=True,
    help="Adds a diamond/dot for the baseline group's predicted probability at each lender and AUS"
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"""
**Baselines:**  
Ethnicity → **{eth_baseline_label}**  
Race → **{race_baseline_label}**  
""")

st.sidebar.markdown(f"""
**Current comparison groups:**  \n
Ethnicity --> compared to **{eth_baseline_label}** \n
Race --> compared to **{race_baseline_label}** \n
Age groups → **25-34** \n
**Gender** --> I have chosen to exclude gender from this analysis because, in this data, gender had high mismatch rates between self-reported and lender-observed gender fields. This made reliable interpretation impossible without further investigation. 
""")


########   last two PULLED FROM MY_APP.PY NOT SAME AS REST                             ########
######## ── Apply dataset filters ─────────────────────────────────────────────────────########
df = df_raw.copy()

if filter_reverse and 'reverse_mortgage' in df.columns:
    df = df[df['reverse_mortgage'] == 2]
if filter_open_end and 'open_end_line_of_credit' in df.columns:
    df = df[df['open_end_line_of_credit'] == 2]
if filter_business and 'business_or_commercial_purpose' in df.columns:
    df = df[df['business_or_commercial_purpose'] == 2]
if filter_single_family and 'total_units' in df.columns:
    df = df[df['total_units'] == 1]
if filter_home_purchase and 'loan_purpose' in df.columns:
    df = df[df['loan_purpose'] == 1]
if filter_primary_residence and 'occupancy_type' in df.columns:
    df = df[df['occupancy_type'] == 1]
# if filter_lien_status and 'lien_status' in df.columns:
#     df = df[df['lien_status'] == 1]
# if filter_conventional_loan_type and 'loan_type' in df.columns:
#     df = df[df['loan_type'] == 1]    
st.sidebar.markdown("---")


# ── Create age group dummy columns if selected ────────────────────────────────────
if include_age_section and 'applicant_age' in df.columns:
    age_map = {
        '<25':   'applicant_age_lt25',
        '25-34': 'applicant_age_25_34',
        '35-44': 'applicant_age_35_44',
        '45-54': 'applicant_age_45_54',
        '55-64': 'applicant_age_55_64',
        '65-74': 'applicant_age_65_74',
        '>74':   'applicant_age_gt74',
    }
    for age_val, col in age_map.items():
        if col not in df.columns:
            df[col] = (df['applicant_age'] == age_val).astype(int)


# ── Show filter impact ────────────────────────────────────────────────────────
n_total    = len(df_raw)
n_filtered = len(df)
n_removed  = n_total - n_filtered

if n_removed > 0:
    st.info(f"After applying your loan filters, **{n_filtered:,}** applications remain out of {n_total:,} total ({n_removed:,} removed).")
else:
    st.info(f"No loan filters have been applied and we're analyzing all **{n_total:,}** applications.")


# ── Build combined demo vars and labels ───────────────────────────────────────
age_labels_map = {
    "applicant_age_lt25":     "Under 25",
    "applicant_age_25_34":    "25-34",
    "applicant_age_45_54":    "45-54",
    "applicant_age_55_64":    "55-64",
    "applicant_age_65_74":    "65-74",
    "applicant_age_gt74":     "Over 74",
    "applicant_age_above_62": "Age 62 or older",
}

demo_vars  = selected_eth_vars + selected_race_vars + selected_age_vars
var_labels = {
    **{v: all_ethnicity[v] for v in selected_eth_vars},
    **{v: all_race[v]      for v in selected_race_vars},
    **{v: age_labels_map[v] for v in selected_age_vars if v in age_labels_map},
}

def get_demo_type(v):
    if v in all_ethnicity: return "Ethnicity"
    if v in all_race:      return "Race"
    return "Age"

if not demo_vars:
    st.warning("Please select at least one group from the sidebar.")
    st.stop()


####### REVIEW FURTHER ONCE COMPLETED                                    #######
# ── Build control list ─────────────────────────────────────────────────#######
control_parts = []
if include_income:          control_parts.append("income")
if include_dti:             control_parts.append("C(debt_to_income_ratio)")
if include_ltv:             control_parts.append("loan_to_value_ratio")
if include_loan_type:       control_parts.append("C(loan_type)")
if include_loan_purpose:    control_parts.append("C(loan_purpose)")
if include_lien_status:     control_parts.append("C(lien_status)")
if include_occupancy:       control_parts.append("C(occupancy_type)")
if include_tract_minority:  control_parts.append("tract_minority_population_percent")
if include_conforming:      control_parts.append("C(conforming_loan_limit)")



