"""
Data Normalizer Module
Transforms raw, messy policy data into clean, standardized format
Demonstrates ETL pipeline skills from JD.com data infrastructure work
"""

import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np


class DataNormalizer:
    """
    Normalizes raw EV subsidy data into standardized format.
    Handles inconsistent naming, date formats, and value representations.
    """

    # Standardized column mappings
    COLUMN_MAPPINGS = {
        # State name variants
        "state": "state",
        "State": "state",
        "STATE": "state",
        "state_name": "state",
        # Rebate amount variants
        "rebate_amount": "rebate_usd",
        "incentive": "rebate_usd",
        "subsidy_usd": "rebate_usd",
        "amount": "rebate_usd",
        # Income cap variants
        "income_limit": "income_cap_usd",
        "income_cap": "income_cap_usd",
        "max_income": "income_cap_usd",
    }

    # Date format patterns to try
    DATE_PATTERNS = [
        ("%Y-%m-%d", r"^\d{4}-\d{2}-\d{2}$"),
        ("%m/%d/%Y", r"^\d{2}/\d{2}/\d{4}$"),
        ("%d-%m-%Y", r"^\d{2}-\d{2}-\d{4}$"),
        ("%B %d, %Y", r"^[A-Za-z]+ \d{1,2}, \d{4}$"),
        ("%m-%d-%Y", r"^\d{2}-\d{2}-\d{4}$"),
        ("%Y/%m/%d", r"^\d{4}/\d{2}/\d{2}$"),
    ]

    # Vehicle type standardization
    VEHICLE_TYPE_MAPPING = {
        "BEV": "battery_electric",
        "PHEV": "plug_in_hybrid",
        "FCEV": "fuel_cell",
        "Electric Vehicle": "battery_electric",
        "Plug-in Hybrid": "plug_in_hybrid",
        "Battery Electric": "battery_electric",
        "All EVs": "all_electric_vehicles",
    }

    def __init__(self, verbose: bool = True):
        """Initialize normalizer with processing statistics."""
        self.verbose = verbose
        self.stats = {
            "records_processed": 0,
            "records_normalized": 0,
            "parse_errors": 0,
            "missing_values_filled": 0,
            "duplicates_removed": 0,
            "processing_time_ms": 0,
        }

    def normalize_raw_data(self, raw_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Main normalization pipeline for raw policy data.

        Args:
            raw_data: List of raw policy records with inconsistent formats

        Returns:
            Normalized DataFrame with standardized schema
        """
        start_time = time.time()

        if self.verbose:
            print(f"Starting normalization of {len(raw_data)} records...")

        # Step 1: Standardize column names
        standardized = [self._standardize_columns(record) for record in raw_data]

        # Step 2: Parse and normalize values
        normalized = [self._normalize_values(record) for record in standardized]

        # Filter out failed normalizations
        normalized = [r for r in normalized if r is not None]

        # Step 3: Convert to DataFrame
        df = pd.DataFrame(normalized)

        # Step 4: Remove duplicates (keep most recent version per state)
        initial_count = len(df)
        df = self._deduplicate(df)
        self.stats["duplicates_removed"] = initial_count - len(df)

        # Step 5: Fill missing values with sensible defaults
        df = self._fill_missing_values(df)

        # Step 6: Add derived columns
        df = self._add_derived_columns(df)

        # Step 7: Final validation and sorting
        df = self._validate_and_sort(df)

        # Record statistics
        self.stats["records_processed"] = len(raw_data)
        self.stats["records_normalized"] = len(df)
        self.stats["processing_time_ms"] = int((time.time() - start_time) * 1000)

        if self.verbose:
            self._print_stats()

        return df

    def _standardize_columns(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Standardize column names using mapping."""
        standardized = {}

        for key, value in record.items():
            # Map to standard name or keep original
            standard_key = self.COLUMN_MAPPINGS.get(key, key)
            standardized[standard_key] = value

        return standardized

    def _normalize_values(self, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Normalize individual field values."""
        try:
            normalized = {}

            # State (required)
            normalized["state"] = record.get("state", "").strip()
            if not normalized["state"]:
                self.stats["parse_errors"] += 1
                return None

            # Rebate amount - clean currency formatting
            rebate_raw = record.get("rebate_usd", 0)
            normalized["rebate_usd"] = self._parse_currency(rebate_raw)

            # Income cap
            income_cap = record.get("income_cap_usd")
            normalized["income_cap_usd"] = (
                self._parse_currency(income_cap) if income_cap else None
            )

            # Effective date
            date_raw = record.get("effective_date", "")
            normalized["effective_date"] = self._parse_date(date_raw)

            # Policy type
            normalized["policy_type"] = record.get("policy_type", "unknown").lower()

            # Region
            normalized["region"] = record.get("region", "Unknown")

            # Vehicle type
            vehicle_raw = record.get("vehicle_type", "All EVs")
            normalized["vehicle_type"] = self.VEHICLE_TYPE_MAPPING.get(
                vehicle_raw, "all_electric_vehicles"
            )

            # Eligibility
            normalized["eligibility"] = record.get("eligibility", "Standard eligibility")

            # Source metadata
            normalized["data_source"] = record.get("source", "unknown")
            normalized["version"] = record.get("version", 1)

            # Additional metrics
            normalized["ev_adoption_rate"] = record.get("ev_adoption_rate", 0.0)
            normalized["population_millions"] = record.get("population_millions", 0.0)

            return normalized

        except Exception as e:
            self.stats["parse_errors"] += 1
            if self.verbose:
                print(f"  Warning: Failed to normalize record: {e}")
            return None

    def _parse_currency(self, value: Any) -> int:
        """Parse various currency formats to integer cents."""
        if value is None or value == "":
            return 0

        if isinstance(value, (int, float)):
            return int(value)

        # String parsing
        value_str = str(value)
        # Remove currency symbols and commas
        cleaned = re.sub(r"[$,]", "", value_str)
        # Remove decimal portion
        cleaned = re.sub(r"\.\d+$", "", cleaned)

        try:
            return int(cleaned)
        except ValueError:
            return 0

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date string trying multiple formats."""
        if not date_str:
            return None

        date_str = str(date_str).strip()

        for fmt, pattern in self.DATE_PATTERNS:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue

        # Fallback: try pandas parsing
        try:
            return pd.to_datetime(date_str)
        except:
            return None

    def _deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate records, keeping most recent version per state."""
        if df.empty:
            return df

        # Sort by state and version (descending) then effective_date (descending)
        df_sorted = df.sort_values(
            ["state", "version", "effective_date"],
            ascending=[True, False, False]
        )

        # Keep first (most recent) record per state
        df_deduped = df_sorted.drop_duplicates(subset=["state"], keep="first")

        return df_deduped.reset_index(drop=True)

    def _fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values with sensible defaults."""
        # Track fills
        fills = 0

        # Fill numeric columns
        if "rebate_usd" in df.columns:
            fills += df["rebate_usd"].isna().sum()
            df["rebate_usd"] = df["rebate_usd"].fillna(0).astype(int)

        # Fill categorical columns
        categorical_defaults = {
            "policy_type": "unknown",
            "region": "Unknown",
            "vehicle_type": "all_electric_vehicles",
            "eligibility": "Standard eligibility",
            "data_source": "unknown",
        }

        for col, default in categorical_defaults.items():
            if col in df.columns:
                fills += df[col].isna().sum()
                df[col] = df[col].fillna(default)

        self.stats["missing_values_filled"] = int(fills)
        return df

    def _add_derived_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add calculated/derived columns for analysis."""

        # Rebate tier categorization
        def categorize_rebate(amount):
            if amount == 0:
                return "none"
            elif amount < 2000:
                return "low"
            elif amount < 5000:
                return "medium"
            else:
                return "high"

        df["rebate_tier"] = df["rebate_usd"].apply(categorize_rebate)

        # Has income cap flag
        df["has_income_cap"] = df["income_cap_usd"].notna()

        # Estimated annual impact (population * adoption rate * rebate)
        df["est_annual_impact_millions"] = (
            df["population_millions"] *
            df["ev_adoption_rate"] *
            df["rebate_usd"] *
            0.1  # Assume 10% of adopters claim rebate
        ).round(2)

        # Processing timestamp
        df["processed_at"] = datetime.now()

        return df

    def _validate_and_sort(self, df: pd.DataFrame) -> pd.DataFrame:
        """Final validation and sorting of normalized data."""

        # Ensure required columns exist
        required_columns = ["state", "rebate_usd", "policy_type", "region"]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Sort by state name
        df = df.sort_values("state").reset_index(drop=True)

        # Reorder columns for readability
        column_order = [
            "state", "region", "rebate_usd", "rebate_tier", "income_cap_usd",
            "has_income_cap", "policy_type", "vehicle_type", "eligibility",
            "effective_date", "ev_adoption_rate", "population_millions",
            "est_annual_impact_millions", "data_source", "version", "processed_at"
        ]
        existing_cols = [c for c in column_order if c in df.columns]
        other_cols = [c for c in df.columns if c not in column_order]
        df = df[existing_cols + other_cols]

        return df

    def _print_stats(self):
        """Print processing statistics."""
        print("\n" + "=" * 50)
        print("NORMALIZATION STATISTICS")
        print("=" * 50)
        for key, value in self.stats.items():
            print(f"  {key}: {value}")
        print("=" * 50 + "\n")

    def save_normalized_data(
        self, df: pd.DataFrame, output_dir: str = "data/processed"
    ) -> Dict[str, str]:
        """Save normalized data to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save main normalized data
        csv_file = output_path / "ev_subsidies_normalized.csv"
        df.to_csv(csv_file, index=False)

        # Save as JSON for API
        json_file = output_path / "ev_subsidies_normalized.json"
        # Convert datetime columns to string for JSON
        df_json = df.copy()
        for col in df_json.select_dtypes(include=["datetime64"]).columns:
            df_json[col] = df_json[col].astype(str)
        df_json.to_json(json_file, orient="records", indent=2)

        # Save processing stats
        stats_file = output_path / "processing_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)

        return {
            "csv": str(csv_file),
            "json": str(json_file),
            "stats": str(stats_file),
        }

    def get_processing_metrics(self) -> Dict[str, Any]:
        """Return processing metrics for visualization."""
        return {
            **self.stats,
            "success_rate": (
                self.stats["records_normalized"] / max(1, self.stats["records_processed"])
            ) * 100,
            "error_rate": (
                self.stats["parse_errors"] / max(1, self.stats["records_processed"])
            ) * 100,
        }


def normalize_pipeline(
    input_file: str = "data/raw/ev_subsidies_raw.json",
    output_dir: str = "data/processed"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Run complete normalization pipeline.

    Args:
        input_file: Path to raw data JSON file
        output_dir: Directory for output files

    Returns:
        Tuple of (normalized DataFrame, processing metrics)
    """
    # Load raw data
    with open(input_file) as f:
        raw_data = json.load(f)

    # Initialize and run normalizer
    normalizer = DataNormalizer(verbose=True)
    df = normalizer.normalize_raw_data(raw_data)

    # Save results
    normalizer.save_normalized_data(df, output_dir)

    return df, normalizer.get_processing_metrics()


if __name__ == "__main__":
    df, metrics = normalize_pipeline()
    print(f"\nNormalized {len(df)} state policies")
    print(f"Processing time: {metrics['processing_time_ms']}ms")
    print(f"\nSample output:")
    print(df[["state", "rebate_usd", "rebate_tier", "region", "policy_type"]].head(10))
