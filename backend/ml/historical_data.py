"""
Maps a row of IPBL/ecommerce_cleaned.csv onto ShopperSession-shaped fields.
Shared by train_model.py (builds the training matrix) and seed_demo_data.py
(replays historical sessions into the live DB so the dashboard has data
immediately) -- one mapping, so both stay in sync.
"""
import pandas as pd


def compute_price_proxy(df: pd.DataFrame) -> tuple[dict[str, float], float]:
    """Historical data has no live cart total, only final Order_Value for
    converted sessions. Estimate $/item per category from those so cart-adders
    who didn't convert still get a plausible cart_value (see features.py docstring)."""
    converted = df[df["Converted"] == "Yes"].copy()
    converted["price_per_item"] = converted["Order_Value"] / converted["Items_in_Cart"].clip(lower=1)
    category_price = (
        converted.groupby(df.loc[converted.index, "Product_Category"].str.lower())["price_per_item"]
        .mean()
        .to_dict()
    )
    global_price = float(converted["price_per_item"].mean())
    return category_price, global_price


def row_to_session_fields(row: pd.Series, category_price: dict[str, float], global_price: float) -> dict:
    added_to_cart = row["Added_to_Cart"] == "Yes"
    items_in_cart = float(row["Items_in_Cart"]) if pd.notna(row["Items_in_Cart"]) else 0.0
    category = str(row["Product_Category"]).lower()

    cart_value = 0.0
    if added_to_cart:
        price_per_item = category_price.get(category, global_price)
        cart_value = round(items_in_cart * price_per_item, 2)

    return {
        "device_type": row["Device_Type"].lower(),
        "traffic_source": row["Traffic_Source"].lower(),
        "time_of_day": row["Time_of_Day"].lower(),
        "product_category": category,
        "return_customer": row["Return_Customer"] == "Yes",
        "discount_used": row["Discount_Code_Used"] == "Yes",
        "added_to_cart": added_to_cart,
        "items_in_cart": items_in_cart,
        "cart_value": cart_value,
        "time_on_site": float(row["Time_Spent_on_Site"]),
        "pages_viewed": int(row["Pages_Viewed"]),
        "converted": row["Converted"] == "Yes",
        "order_value": float(row["Order_Value"]) if pd.notna(row["Order_Value"]) else None,
        "abandonment_point": row["Abandonment_Point"] if pd.notna(row["Abandonment_Point"]) else "None",
        # Not present in historical data -- captured live by the snippet instead.
        "scroll_depth_avg": 0.0,
        "exit_intent_count": 0.0,
        "payment_attempts": 0.0,
    }


def iter_historical_sessions(csv_path):
    """Yields (customer_id, fields_dict) for every row in the historical dataset."""
    df = pd.read_csv(csv_path)
    category_price, global_price = compute_price_proxy(df)
    for _, row in df.iterrows():
        yield row["Customer_ID"], row_to_session_fields(row, category_price, global_price)
