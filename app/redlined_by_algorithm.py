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
    1: Odds Chart - After controlling for financial factors, how much better or worse are approval odds by demographic group?
    2: Probability Chart - What is the estimated approval probability for a typical applicant in each group?
    3: Lenders Chart - Which specific lenders have the lowest predicted approval probabilities for the group you select?
    4: Underwriting Systems Chart - Which automated underwriting systems produce the lowest predicted approval probabilities?

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
    st.error("⚠️ Data file not found. Make sure `hmda_all_nf.csv` is in the same folder as this app file.")
    st.stop()
except Exception as e:
    st.error(f"⚠️ Something went wrong loading the data: {e}")
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









