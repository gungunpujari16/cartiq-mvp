"""
Generates a SIMULATED B2B brand survey dataset for the Business Intelligence
page -- CartIQ has not actually run this survey. Fields follow the BRD's own
survey design (Section 7: "B2B Survey Questions -- Willingness to Pay").

Styled after IPBL/ecommerce_data_generator_v2.py (seeded, correlated, with
realistic noise) so the two synthetic datasets in this project follow the
same methodology.

Run once: python generate_brand_survey.py   (writes data/brand_survey.csv)
"""
from pathlib import Path

import numpy as np
import pandas as pd

np.random.seed(42)
N = 220

OUT_PATH = Path(__file__).resolve().parent / "data" / "brand_survey.csv"

# ── Identity ─────────────────────────────────────────────────────────────
brand_id = [f"BRAND{str(i).zfill(4)}" for i in range(1, N + 1)]

_prefixes = ["Urban", "Desert", "Nile", "Gulf", "Metro", "Coastal", "Falcon", "Oasis",
             "Prime", "Bloom", "Nova", "Summit", "Azure", "Golden", "Silk", "Palm"]
_suffixes = ["Trading Co.", "Retail Group", "E-Store", "Commerce Ltd", "Marketplace",
             "& Sons", "Collective", "Outlet", "Bazaar", "Direct"]
company_name = [f"{np.random.choice(_prefixes)} {np.random.choice(_suffixes)}" for _ in range(N)]

industry = np.random.choice(
    ["Fashion", "Electronics", "Grocery", "Beauty", "Home & Kitchen", "Sports", "Books", "Toys"],
    size=N, p=[0.22, 0.18, 0.10, 0.14, 0.14, 0.10, 0.06, 0.06],
)

# MENA-first per BRD constraint: "Initial target markets: UAE, Saudi Arabia, India"
region = np.random.choice(
    ["UAE", "Saudi Arabia", "India", "Egypt", "UK"],
    size=N, p=[0.30, 0.24, 0.22, 0.14, 0.10],
)

# ── Firmographics (ordinal tiers, WTP-03/07/09) ─────────────────────────
GMV_TIERS = ["<$10K/mo", "$10K-50K/mo", "$50K-200K/mo", "$200K-1M/mo", "$1M-5M/mo", ">$5M/mo"]
ORDER_TIERS = ["<100/mo", "100-500/mo", "500-2,000/mo", "2,000-10,000/mo", "10,000+/mo"]
TEAM_TIERS = ["1-5", "6-15", "16-50", "51-200", "200+"]

gmv_tier_enc = np.clip(np.random.beta(a=2.2, b=3.5, size=N) * 6, 0, 5).astype(int)
gmv_tier = np.array(GMV_TIERS)[gmv_tier_enc]

# orders and team size correlate with GMV tier, with noise
monthly_orders_enc = np.clip(gmv_tier_enc + np.random.randint(-1, 2, N), 0, 4)
monthly_orders = np.array(ORDER_TIERS)[monthly_orders_enc]

team_size_enc = np.clip(gmv_tier_enc + np.random.randint(-1, 2, N) - 1, 0, 4)
team_size = np.array(TEAM_TIERS)[team_size_enc]
company_size = np.where(team_size_enc <= 1, "Small", np.where(team_size_enc <= 2, "Medium", "Large"))

# ── Platform & operations (WTP-05/06/08) ────────────────────────────────
platform = np.random.choice(
    ["Shopify", "WooCommerce", "Magento", "BigCommerce", "Custom Built"],
    size=N, p=[0.40, 0.24, 0.14, 0.12, 0.10],
)

checkout_pain_point = np.random.choice(
    ["Payment Failures", "Complex Forms", "Trust Issues", "Shipping Costs", "Mobile UX"],
    size=N, p=[0.26, 0.18, 0.16, 0.18, 0.22],
)

current_tool_used = np.random.choice(["Yes", "No"], size=N, p=[0.38, 0.62])
current_tool_sophistication = np.where(
    current_tool_used == "Yes", np.random.randint(2, 11, N), 0
)

# ── Pain severity (WTP-04) ───────────────────────────────────────────────
# Less sophisticated brands (no tool / low team size) tend toward higher abandonment
cart_abandonment_rate = np.clip(
    45 + (10 - current_tool_sophistication) * 2.2 - team_size_enc * 1.5 + np.random.normal(0, 8, N),
    30, 92,
).round(1)

pricing_preference = np.random.choice(["Fixed Subscription", "Revenue Share"], size=N, p=[0.66, 0.34])

# ── Targets (WTP-10, WTP-01) ─────────────────────────────────────────────
# Adoption likelihood: pain (abandonment) and lack of an existing tool drive interest
adoption_score = (
    2.0
    + (cart_abandonment_rate - 45) / 47 * 4.5
    + (10 - current_tool_sophistication) / 10 * 2.5
    + gmv_tier_enc * 0.25
    + np.random.normal(0, 1.0, N)
)
adoption_likelihood = np.clip(np.round(adoption_score), 1, 10).astype(int)

# WTP: scales with firmographics + pain; log-right-skewed for enterprise outliers
# (mirrors TRD S3.2's own target transform: log(1+WTP), back-transformed for reporting)
wtp_base = (
    70
    + gmv_tier_enc * 190
    + monthly_orders_enc * 55
    + team_size_enc * 90
    + cart_abandonment_rate * 2.6
    + np.where(pricing_preference == "Revenue Share", -60, 0)
)
wtp_noise = np.random.normal(0, 0.18, N)  # multiplicative noise in log-space
willingness_to_pay_usd = np.clip(wtp_base * np.exp(wtp_noise), 49, 6000).round(0)

df = pd.DataFrame({
    "brand_id": brand_id,
    "company_name": company_name,
    "industry": industry,
    "region": region,
    "gmv_tier": gmv_tier,
    "monthly_orders": monthly_orders,
    "team_size": team_size,
    "company_size": company_size,
    "platform": platform,
    "cart_abandonment_rate": cart_abandonment_rate,
    "checkout_pain_point": checkout_pain_point,
    "current_tool_used": current_tool_used,
    "current_tool_sophistication": current_tool_sophistication,
    "pricing_preference": pricing_preference,
    "adoption_likelihood": adoption_likelihood,
    "willingness_to_pay_usd": willingness_to_pay_usd,
})

if __name__ == "__main__":
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df)} rows to {OUT_PATH}")
    print(df.describe(include="all").T[["count", "unique", "top", "mean"]].head(20))
