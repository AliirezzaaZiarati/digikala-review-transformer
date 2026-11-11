"""Utilities for creating deterministic product-grouped data splits."""

import numpy as np
import pandas as pd

from .config import SPLIT_SEED

TRAIN_CUMULATIVE_SHARE = 0.70
VALIDATION_CUMULATIVE_SHARE = 0.85


def build_product_split(
    prod_df: pd.DataFrame,
    seed: int = SPLIT_SEED,
) -> pd.DataFrame:
    """Build a deterministic train/validation/test mapping for products.

    Args:
        prod_df: DataFrame with one row per product and columns
            `product_id` and `n_rows`.
        seed: Random seed used to shuffle products.

    Returns:
        DataFrame with columns `product_id` and `split`.
    """
    required_columns = {"product_id", "n_rows"}
    missing_columns = required_columns.difference(prod_df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    if prod_df.empty:
        raise ValueError("prod_df is empty.")

    if prod_df["product_id"].isna().any():
        raise ValueError("product_id contains missing values.")

    if prod_df["product_id"].duplicated().any():
        raise ValueError("prod_df must contain exactly one row per product_id.")

    if prod_df["n_rows"].isna().any() or (prod_df["n_rows"] <= 0).any():
        raise ValueError("n_rows must contain positive, non-missing values.")

    # Use a stable initial order before applying the seeded shuffle.
    products = prod_df.sort_values("product_id").reset_index(drop=True)
    products = products.iloc[
        np.random.default_rng(seed).permutation(len(products))
    ].reset_index(drop=True)

    cumulative_row_share = (
        products["n_rows"].cumsum() / products["n_rows"].sum()
    )

    products["split"] = np.where(
        cumulative_row_share <= TRAIN_CUMULATIVE_SHARE,
        "train",
        np.where(
            cumulative_row_share <= VALIDATION_CUMULATIVE_SHARE,
            "validation",
            "test",
        ),
    )

    return products[["product_id", "split"]]