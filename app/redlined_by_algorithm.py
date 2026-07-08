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


####### REVIEW FURTHER ONCE COMPLETED POSSIBLY ADD MORE TO CHART 2 calc pred probs #######
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


# ── Build formula and run model ───────────────────────────────────────────────
def build_formula(target, demo_vars, controls):
    rhs = " + ".join(demo_vars + controls)
    return f"{target} ~ {rhs}"

formula = build_formula(target_var, demo_vars, control_parts)

@st.cache_data(show_spinner=False)
def run_model(formula, filtered_df):
    model  = smf.logit(formula=formula, data=filtered_df)
    result = model.fit(maxiter=200, disp=False)
    return result

with st.spinner("Calculating..."):
    try:
        result    = run_model(formula, df)
        converged = result.mle_retvals["converged"]
    except Exception as e:
        st.error(f"Model failed to fit: {e}")
        st.stop()

if not converged:
    st.warning("The model had trouble finding a stable answer. Try unchecking some groups or financial factors.")


# ── Add predicted probabilities to the dataframe ──────────────────────────────
df['predicted_prob'] = result.predict(df)


# ── Extract model results ─────────────────────────────────────────────────────
demo_vars   = [v for v in demo_vars if v in result.params.index]
params      = result.params[demo_vars]
conf        = result.conf_int().loc[demo_vars]
pvalues     = result.pvalues[demo_vars]
odds_ratios = np.exp(params)
or_conf     = np.exp(conf)

baseline_map = {
    "Ethnicity": eth_baseline_label,
    "Race":      race_baseline_label,
    "Age":       "25-34",
}

results_df = pd.DataFrame({
    "Group":         [var_labels[v] for v in demo_vars],
    "Type":          [get_demo_type(v) for v in demo_vars],
    "Approval Odds": odds_ratios.values,
    "OR Low":        or_conf[0].values,
    "OR High":       or_conf[1].values,
    "p-value":       pvalues.values,
    "Significant":   pvalues.values < 0.05,
}).set_index("Group")


# ── Calculate predicted probabilities for chart 2/probability chart ───────────────────
def get_typical_applicant(df, control_parts):
    profile = {}
    if "income" in control_parts:
        profile["income"] = df["income"].median()
    if "loan_to_value_ratio" in control_parts:
        profile["loan_to_value_ratio"] = df["loan_to_value_ratio"].median()
    if "tract_minority_population_percent" in control_parts:
        profile["tract_minority_population_percent"] = df["tract_minority_population_percent"].median()
    for part in control_parts:
        if part.startswith("C("):
            col = part[2:-1]
            if col in df.columns:
                profile[col] = df[col].mode()[0]
    return profile

typical = get_typical_applicant(df, control_parts)

def predict_prob(result, var, demo_vars, typical_profile, control_parts):
    intercept = result.params.get("Intercept", 0)
    log_odds  = intercept
    for dv in demo_vars:
        coef     = result.params.get(dv, 0)
        val      = 1 if dv == var else 0
        log_odds += coef * val
    for part in control_parts:
        if part.startswith("C("):
            col = part[2:-1]
            if col in typical_profile:
                ref_val  = typical_profile[col]
                coef_key = f"C({col})[T.{ref_val}]"
                coef     = result.params.get(coef_key, 0)
                log_odds += coef
        else:
            col  = part
            coef = result.params.get(col, 0)
            val  = typical_profile.get(col, 0)
            log_odds += coef * val
    return 1 / (1 + np.exp(-log_odds))

group_probs = {var_labels[v]: predict_prob(result, v, demo_vars, typical, control_parts)
               for v in demo_vars}

# Baseline probability
baseline_log_odds = result.params.get("Intercept", 0)
for part in control_parts:
    if part.startswith("C("):
        col = part[2:-1]
        if col in typical:
            coef_key = f"C({col})[T.{typical[col]}]"
            baseline_log_odds += result.params.get(coef_key, 0)
    else:
        col  = part
        coef = result.params.get(col, 0)
        baseline_log_odds += coef * typical.get(col, 0)
baseline_prob = 1 / (1 + np.exp(-baseline_log_odds))


# ── Chart size controls ───────────────────────────────────────────────────────
st.markdown("---")
type_colors = {
    "Ethnicity": "steelblue",
    "Race":      "#2ecc71",
    "Age":       "#e67e22",
}

pc1, pc2, pc3, pc4 = st.columns(4)
with pc1:
    fig_width  = st.number_input("Chart width",  min_value=4, max_value=24, value=10, step=1)
with pc2:
    fig_height = st.number_input("Chart height", min_value=3, max_value=20,
                                 value=max(4, int(len(demo_vars) * 0.65 + 1.5)), step=1)
with pc3:
    x_min = st.number_input("Odds chart left edge",  min_value=0.0, max_value=5.0,  value=0.0, step=0.5)
with pc4:
    x_max = st.number_input("Odds chart right edge", min_value=1.0, max_value=50.0, value=5.0, step=0.5)

labels_ordered = [var_labels[v] for v in demo_vars]


# ══════════════════════════════════════════════════════════════════════════════
# ── CHART 1: ODDS RATIOS ─────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("Chart 1 - Approval Odds")
st.markdown("How much better or worse are the approval odds for each group compared to the baseline, after controlling for financial factors?")

fig1, ax1 = plt.subplots(figsize=(fig_width, fig_height))

prev_type = None
for i, var in enumerate(demo_vars):
    is_sig    = pvalues[var] < 0.05
    demo_type = get_demo_type(var)
    dot_color = "#c0392b" if is_sig else type_colors.get(demo_type, "steelblue")
    ax1.plot([or_conf.loc[var, 0], or_conf.loc[var, 1]], [i, i],
             color=dot_color, linewidth=2, zorder=2)
    ax1.plot(odds_ratios[var], i, 'o', color=dot_color, markersize=8, zorder=3)
    if prev_type and demo_type != prev_type:
        ax1.axhline(i - 0.5, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    prev_type = demo_type

ax1.axvline(1, color="black", linestyle="--", linewidth=1)
ax1.set_yticks(range(len(demo_vars)))
ax1.set_yticklabels(labels_ordered)
ax1.set_xlabel("Approval odds relative to baseline\n(below 1.0 = worse odds  |  above 1.0 = better odds)")
ax1.set_xlim(x_min, x_max)
ax1.set_title("Odds Ratio - How much better or worse compared to the baseline?", fontweight='bold')

sig_patch  = mpatches.Patch(color="#c0392b",   label="Statistically significant difference (p < 0.05)")
eth_patch  = mpatches.Patch(color="steelblue", label="Ethnicity")
race_patch = mpatches.Patch(color="#2ecc71",   label="Race")
age_patch  = mpatches.Patch(color="#e67e22",   label="Age")
no_effect  = plt.Line2D([0], [0], color='black', linestyle='--', label='No difference from baseline')
ax1.legend(handles=[sig_patch, eth_patch, race_patch, age_patch, no_effect],
           fontsize=8, loc="lower right")

plt.tight_layout()
st.pyplot(fig1)
plt.close()


# ── Plain english summary ─────────────────────────────────────────────────────
st.markdown("---")
st.subheader("What the data shows")

sig_results = results_df[results_df["Significant"]]
if len(sig_results) == 0:
    st.info("With the current settings, no group shows a meaningful difference in approval odds.")
else:
    for group, row in sig_results.iterrows():
        or_val   = row["Approval Odds"]
        grp_type = row["Type"]
        baseline = baseline_map.get(grp_type, "the baseline group")
        prob     = group_probs.get(group, None)
        if or_val < 1:
            pct = round((1 - or_val) * 100)
            prob_str = f" Their estimated approval probability is **{prob:.1%}**." if prob else ""
            st.error(f"**{group}** applicants had about **{pct}% lower odds** of getting approved compared to **{baseline}** applicants.{prob_str}")
        else:
            pct = round((or_val - 1) * 100)
            prob_str = f" Their estimated approval probability is **{prob:.1%}**." if prob else ""
            st.success(f"**{group}** applicants had about **{pct}% higher odds** of getting approved compared to **{baseline}** applicants.{prob_str}")

st.markdown(f"""
Comparing all groups to their baselines:  
- **Ethnicity:** {eth_baseline_label}  
- **Race:** {race_baseline_label}  
- **Age bins:** 25-34  

**Estimated approval probability for a typical baseline applicant: {baseline_prob:.1%}**
""")


# ══════════════════════════════════════════════════════════════════════════════
# ── CHART 2: PREDICTED PROBABILITIES ─────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("Chart 2 - Estimated Approval Probability")
st.markdown(f"Estimated probability of approval for a typical applicant with median financial characteristics. Baseline ({eth_baseline_label if selected_eth_vars else race_baseline_label}): **{baseline_prob:.1%}**")

fig2, ax2 = plt.subplots(figsize=(fig_width, fig_height))

probs      = [group_probs.get(label, np.nan) for label in labels_ordered]
bar_colors = []
for var in demo_vars:
    is_sig    = pvalues[var] < 0.05
    demo_type = get_demo_type(var)
    bar_colors.append("#c0392b" if is_sig else type_colors.get(demo_type, "steelblue"))

bars = ax2.barh(range(len(demo_vars)), probs, color=bar_colors, height=0.6, zorder=2)
ax2.axvline(baseline_prob, color="black", linestyle="--", linewidth=1.5,
            label=f"Baseline: {baseline_prob:.1%}")

for i, prob in enumerate(probs):
    if not np.isnan(prob):
        ax2.text(prob + 0.005, i, f"{prob:.1%}", va='center', fontsize=9)

ax2.set_yticks(range(len(demo_vars)))
ax2.set_yticklabels(labels_ordered)
ax2.set_xlabel("Estimated probability of approval")
ax2.set_xlim(0, 1.15)
ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
ax2.axvline(0.5, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
ax2.set_title("Estimated Approval Probability - What are the chances?", fontweight='bold')
ax2.legend(fontsize=8, loc="lower right")

plt.tight_layout()
st.pyplot(fig2)
plt.close()


# ── Results table ─────────────────────────────────────────────────────────────
with st.expander("See full results table"):
    display_df = pd.DataFrame({
        "Type":              results_df["Type"],
        "Compared to":       results_df["Type"].map(baseline_map),
        "Odds Ratio":        results_df["Approval Odds"].round(3),
        "Est. Probability":  [f"{group_probs.get(g, np.nan):.1%}" for g in results_df.index],
        "p-value":           results_df["p-value"].round(4),
        "Statistically Significant?":       results_df["Significant"].map({True: "✅ Yes", False: "❌ Not sure"}),
    })
    st.dataframe(
        display_df.style.apply(
            lambda row: ["background-color: #e74c3c; color: white"
                         if "Yes" in str(row["Statistically Significant?"]) else "" for _ in row],
            axis=1
        ),
        use_container_width=True
    )
    st.markdown(f"""
    - Applications analyzed: {int(result.nobs):,}
    - Financial factors held constant: {len(control_parts)}
    - Baseline approval probability: {baseline_prob:.1%}
    - Model found a stable answer: {'✅ Yes' if converged else 'Not quite converged so interpret with caution'}
    """)


# ── Focus group selector which is shared by charts 3 and 4 ────────────────────
st.markdown("---")
focus_group_options = {var_labels[v]: v for v in demo_vars}

if not focus_group_options:
    st.info("Select at least one demographic group in the sidebar to see the lender and AUS charts.")
    st.stop()

focus_label = st.selectbox(
    "Which group do you want to examine in Charts 3 and 4?",
    options=list(focus_group_options.keys()),
    help="Both the lender chart and AUS chart will show predicted probabilities for this group"
)
focus_var = focus_group_options[focus_label]

# Determine comparison var for this group
focus_type = get_demo_type(focus_var)
if focus_type == "Ethnicity":
    comparison_var   = eth_baseline_var
    comparison_label = eth_baseline_label
elif focus_type == "Race":
    comparison_var   = race_baseline_var
    comparison_label = race_baseline_label
else:
    comparison_var   = "applicant_age_25_34"
    comparison_label = "25-34"

# Filter df to focus group
focus_df = df[df[focus_var] == 1].copy() if focus_var in df.columns else pd.DataFrame()

# Filter df to comparison group
comp_df = df[df[comparison_var] == 1].copy() if comparison_var in df.columns else pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# ── CHART 3: LENDER PREDICTED PROBABILITIES ───────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("Chart 3 - Lenders with the Lowest Predicted Approval Probabilities")
st.markdown(f"""
Average predicted probability of approval for **{focus_label}** applicants at each lender,  
ordered from lowest to highest. Based on the model's predictions for each application - not raw approval rates.
""")

if 'lei' not in df.columns:
    st.warning("LEI column not found in the dataset.")
elif focus_df.empty:
    st.warning(f"No applications found for {focus_label} in the current dataset.")
else:
    # Average predicted prob by lender for focus group
    lender_stats = (
        focus_df.groupby('lei')['predicted_prob']
        .agg(['mean', 'count'])
        .reset_index()
    )
    lender_stats.columns = ['lei', 'pred_prob', 'n_applications']
    lender_stats = lender_stats[lender_stats['n_applications'] >= min_applications]

    if lender_stats.empty:
        st.warning(f"No lenders have at least {min_applications} applications from {focus_label}. Try lowering the minimum applications slider.")
    else:
        bottom_lenders = lender_stats.nsmallest(top_n_lenders, 'pred_prob')

        # Add institution names
        if panel_df is not None:
            bottom_lenders = bottom_lenders.merge(panel_df, on='lei', how='left')
            bottom_lenders['institution'] = bottom_lenders.get('respondent_name', bottom_lenders['lei']).fillna(bottom_lenders['lei'])
        else:
            bottom_lenders['institution'] = bottom_lenders['lei']

        # Comparison group predicted prob by lender
        if show_comparison and not comp_df.empty:
            comp_lender_stats = (
                comp_df.groupby('lei')['predicted_prob']
                .agg(['mean', 'count'])
                .reset_index()
            )
            comp_lender_stats.columns = ['lei', 'comp_pred_prob', 'comp_n']
            bottom_lenders = bottom_lenders.merge(comp_lender_stats, on='lei', how='left')

        bottom_lenders = bottom_lenders.sort_values('pred_prob', ascending=True)

        lc1, lc2 = st.columns(2)
        with lc1:
            lender_width  = st.number_input("Lender chart width",  min_value=6, max_value=24, value=12, step=1, key="lw")
        with lc2:
            lender_height = st.number_input("Lender chart height", min_value=4, max_value=24,
                                            value=max(5, int(len(bottom_lenders) * 0.5 + 2)), step=1, key="lh")

        fig3, ax3 = plt.subplots(figsize=(lender_width, lender_height))
        y_pos = range(len(bottom_lenders))

        ax3.barh(y_pos, bottom_lenders['pred_prob'],
                 color="#c0392b", height=0.5,
                 label=f"{focus_label}", zorder=2)

        if show_comparison and 'comp_pred_prob' in bottom_lenders.columns:
            ax3.scatter(
                bottom_lenders['comp_pred_prob'], y_pos,
                color="steelblue", zorder=3, s=60,
                label=f"{comparison_label} (comparison)",
                marker='D'
            )

        for i, (_, row) in enumerate(bottom_lenders.iterrows()):
            prob = row['pred_prob']
            n    = int(row['n_applications'])
            ax3.text(prob + 0.005, i, f"{prob:.1%}  (n={n:,})", va='center', fontsize=8)

        ax3.set_yticks(list(y_pos))
        ax3.set_yticklabels(bottom_lenders['institution'], fontsize=9)
        ax3.set_xlabel("Average predicted probability of approval")
        ax3.set_xlim(0, 1.2)
        ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
        ax3.set_title(
            f"Lenders — Lowest Predicted Approval Probability for {focus_label}\n"
            f"(minimum {min_applications:,} applications | bottom {len(bottom_lenders)})",
            fontweight='bold'
        )
        ax3.axvline(0.5, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
        ax3.legend(fontsize=9, loc='lower right')
        ax3.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

        overall_pred = focus_df['predicted_prob'].mean()
        st.markdown(f"""
        **Average predicted approval probability for {focus_label} across all lenders:** {overall_pred:.1%}  
        **Lowest in this list:** {bottom_lenders['pred_prob'].min():.1%} — {bottom_lenders.iloc[0]['institution']}  
        **Highest in this list:** {bottom_lenders['pred_prob'].max():.1%} — {bottom_lenders.iloc[-1]['institution']}
        """)

        download_df = bottom_lenders[['lei', 'institution', 'pred_prob', 'n_applications']].copy()
        download_df.columns = ['LEI', 'Institution', 'Predicted Probability', 'Applications']
        download_df['Predicted Probability'] = download_df['Predicted Probability'].apply(lambda x: f"{x:.1%}")
        st.download_button(
            label="Download lender data as CSV",
            data=download_df.to_csv(index=False),
            file_name=f"lender_pred_prob_{focus_label.replace(' ', '_')}.csv",
            mime="text/csv"
        )


# ══════════════════════════════════════════════════════════════════════════════
# ── CHART 4: AUS PREDICTED PROBABILITIES ─────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("Chart 4 - Automated Underwriting Systems: Predicted Approval Probability")
st.markdown(f"""
Average predicted probability of approval for **{focus_label}** applicants processed through 
each automated underwriting system, ordered from lowest to highest.
""")

aus_cols = [c for c in ['aus-1', 'aus-2', 'aus-3', 'aus-4', 'aus-5'] if c in df.columns]

if not aus_cols:
    st.warning("No AUS columns found in the dataset.")
elif focus_df.empty:
    st.warning(f"No applications found for {focus_label}.")
else:
    # Melt AUS columns — include predicted_prob
    aus_melted = focus_df[['unique_id', target_var, 'predicted_prob'] + aus_cols].melt(
        id_vars=['unique_id', target_var, 'predicted_prob'],
        value_vars=aus_cols,
        var_name='aus_col',
        value_name='aus_code'
    )
    aus_melted = aus_melted.dropna(subset=['aus_code'])
    aus_melted['aus_code'] = pd.to_numeric(aus_melted['aus_code'], errors='coerce')
    aus_melted = aus_melted[~aus_melted['aus_code'].isin([6, 1111])]
    aus_melted = aus_melted.dropna(subset=['aus_code'])
    aus_melted['aus_code'] = aus_melted['aus_code'].astype(int)

    if aus_melted.empty:
        st.warning("No valid AUS data found for this group.")
    else:
        # Average predicted prob by AUS
        aus_stats = (
            aus_melted.groupby('aus_code')['predicted_prob']
            .agg(['mean', 'count'])
            .reset_index()
        )
        aus_stats.columns = ['aus_code', 'pred_prob', 'n_applications']
        aus_stats['aus_name'] = aus_stats['aus_code'].map(aus_labels).fillna(aus_stats['aus_code'].astype(str))
        aus_stats = aus_stats[aus_stats['n_applications'] >= aus_min]

        if aus_stats.empty:
            st.warning(f"No AUS systems have at least {aus_min} applications from {focus_label}. Try lowering the minimum.")
        else:
            # Comparison group AUS predicted probs
            if show_comparison and not comp_df.empty:
                aus_comp_melted = comp_df[['unique_id', target_var, 'predicted_prob'] + aus_cols].melt(
                    id_vars=['unique_id', target_var, 'predicted_prob'],
                    value_vars=aus_cols,
                    var_name='aus_col',
                    value_name='aus_code'
                )
                aus_comp_melted = aus_comp_melted.dropna(subset=['aus_code'])
                aus_comp_melted['aus_code'] = pd.to_numeric(aus_comp_melted['aus_code'], errors='coerce')
                aus_comp_melted = aus_comp_melted[~aus_comp_melted['aus_code'].isin([6, 1111])]
                aus_comp_melted = aus_comp_melted.dropna(subset=['aus_code'])
                aus_comp_melted['aus_code'] = aus_comp_melted['aus_code'].astype(int)

                aus_comp_stats = (
                    aus_comp_melted.groupby('aus_code')['predicted_prob']
                    .agg(['mean', 'count'])
                    .reset_index()
                )
                aus_comp_stats.columns = ['aus_code', 'comp_pred_prob', 'comp_n']
                aus_stats = aus_stats.merge(aus_comp_stats, on='aus_code', how='left')

            # Sort lowest to highest
            aus_stats = aus_stats.sort_values('pred_prob', ascending=True)

            ac1, ac2 = st.columns(2)
            with ac1:
                aus_width  = st.number_input("AUS chart width",  min_value=6, max_value=24, value=12, step=1, key="aw")
            with ac2:
                aus_height = st.number_input("AUS chart height", min_value=3, max_value=16,
                                             value=max(4, int(len(aus_stats) * 0.8 + 1.5)), step=1, key="ah")

            fig4, ax4 = plt.subplots(figsize=(aus_width, aus_height))
            y_pos = range(len(aus_stats))

            ax4.barh(y_pos, aus_stats['pred_prob'],
                     color="#9b59b6", height=0.5,
                     label=f"{focus_label}", zorder=2)

            if show_comparison and 'comp_pred_prob' in aus_stats.columns:
                ax4.scatter(
                    aus_stats['comp_pred_prob'], y_pos,
                    color="steelblue", zorder=3, s=60,
                    label=f"{comparison_label} (comparison)",
                    marker='D'
                )

            for i, (_, row) in enumerate(aus_stats.iterrows()):
                prob = row['pred_prob']
                n    = int(row['n_applications'])
                ax4.text(prob + 0.005, i, f"{prob:.1%}  (n={n:,})", va='center', fontsize=9)

            ax4.set_yticks(list(y_pos))
            ax4.set_yticklabels(aus_stats['aus_name'], fontsize=9)
            ax4.set_xlabel("Average predicted probability of approval (lowest to highest)")
            ax4.set_xlim(0, 1.2)
            ax4.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
            ax4.set_title(
                f"AUS — Predicted Approval Probability for {focus_label}\n"
                f"(minimum {aus_min} applications | ordered lowest to highest)",
                fontweight='bold'
            )
            ax4.axvline(0.5, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
            ax4.legend(fontsize=9, loc='lower right')
            ax4.grid(axis='x', alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig4)
            plt.close()

            best_aus  = aus_stats.iloc[-1]
            worst_aus = aus_stats.iloc[0]
            st.markdown(f"""
            **Highest predicted probability:** {best_aus['pred_prob']:.1%} — {best_aus['aus_name']} (n={int(best_aus['n_applications']):,})  
            **Lowest predicted probability:** {worst_aus['pred_prob']:.1%} — {worst_aus['aus_name']} (n={int(worst_aus['n_applications']):,})
            """)

            aus_download = aus_stats[['aus_name', 'pred_prob', 'n_applications']].copy()
            aus_download.columns = ['AUS System', 'Predicted Probability', 'Applications']
            aus_download['Predicted Probability'] = aus_download['Predicted Probability'].apply(lambda x: f"{x:.1%}")
            st.download_button(
                label="Download AUS data as CSV",
                data=aus_download.to_csv(index=False),
                file_name=f"aus_pred_prob_{focus_label.replace(' ', '_')}.csv",
                mime="text/csv"
            )



