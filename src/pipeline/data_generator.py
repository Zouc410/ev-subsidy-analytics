"""
EV Subsidy Data Generator
Generates realistic US state-level EV subsidy policy data
Mirrors provincial subsidy analysis work done at JD.com
"""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import numpy as np


class EVSubsidyDataGenerator:
    """
    Generates realistic EV subsidy data for US states.
    Simulates messy, real-world policy data requiring normalization.
    """

    # Realistic US state EV subsidy configurations
    STATE_POLICIES = {
        "California": {
            "base_rebate": 7500, "income_cap": 150000, "region": "West",
            "policy_type": "rebate", "ev_adoption_rate": 0.18, "population_m": 39.5
        },
        "New York": {
            "base_rebate": 2000, "income_cap": 250000, "region": "Northeast",
            "policy_type": "rebate", "ev_adoption_rate": 0.08, "population_m": 19.8
        },
        "Texas": {
            "base_rebate": 2500, "income_cap": None, "region": "South",
            "policy_type": "rebate", "ev_adoption_rate": 0.05, "population_m": 29.1
        },
        "Colorado": {
            "base_rebate": 5000, "income_cap": None, "region": "West",
            "policy_type": "tax_credit", "ev_adoption_rate": 0.12, "population_m": 5.8
        },
        "New Jersey": {
            "base_rebate": 4000, "income_cap": None, "region": "Northeast",
            "policy_type": "rebate", "ev_adoption_rate": 0.09, "population_m": 9.3
        },
        "Massachusetts": {
            "base_rebate": 3500, "income_cap": 175000, "region": "Northeast",
            "policy_type": "rebate", "ev_adoption_rate": 0.11, "population_m": 7.0
        },
        "Oregon": {
            "base_rebate": 7500, "income_cap": 125000, "region": "West",
            "policy_type": "rebate", "ev_adoption_rate": 0.14, "population_m": 4.2
        },
        "Washington": {
            "base_rebate": 7500, "income_cap": None, "region": "West",
            "policy_type": "sales_tax_exemption", "ev_adoption_rate": 0.13, "population_m": 7.7
        },
        "Connecticut": {
            "base_rebate": 3000, "income_cap": None, "region": "Northeast",
            "policy_type": "rebate", "ev_adoption_rate": 0.07, "population_m": 3.6
        },
        "Vermont": {
            "base_rebate": 4000, "income_cap": 100000, "region": "Northeast",
            "policy_type": "rebate", "ev_adoption_rate": 0.10, "population_m": 0.6
        },
        "Maryland": {
            "base_rebate": 3000, "income_cap": None, "region": "South",
            "policy_type": "tax_credit", "ev_adoption_rate": 0.06, "population_m": 6.2
        },
        "Pennsylvania": {
            "base_rebate": 3000, "income_cap": 200000, "region": "Northeast",
            "policy_type": "rebate", "ev_adoption_rate": 0.04, "population_m": 13.0
        },
        "Illinois": {
            "base_rebate": 4000, "income_cap": None, "region": "Midwest",
            "policy_type": "rebate", "ev_adoption_rate": 0.05, "population_m": 12.8
        },
        "Michigan": {
            "base_rebate": 2000, "income_cap": None, "region": "Midwest",
            "policy_type": "rebate", "ev_adoption_rate": 0.04, "population_m": 10.0
        },
        "Arizona": {
            "base_rebate": 1500, "income_cap": None, "region": "West",
            "policy_type": "tax_credit", "ev_adoption_rate": 0.06, "population_m": 7.3
        },
        "Georgia": {
            "base_rebate": 0, "income_cap": None, "region": "South",
            "policy_type": "none", "ev_adoption_rate": 0.03, "population_m": 10.7
        },
        "Florida": {
            "base_rebate": 0, "income_cap": None, "region": "South",
            "policy_type": "none", "ev_adoption_rate": 0.04, "population_m": 21.8
        },
        "Nevada": {
            "base_rebate": 2500, "income_cap": None, "region": "West",
            "policy_type": "rebate", "ev_adoption_rate": 0.08, "population_m": 3.1
        },
    }

    # Simulate messy data formats (like real-world policy documents)
    DATE_FORMATS = [
        "%Y-%m-%d", "%m/%d/%Y", "%d-%m-%Y", "%B %d, %Y",
        "%m-%d-%Y", "%Y/%m/%d"
    ]

    VEHICLE_TYPES = ["BEV", "PHEV", "FCEV", "Electric Vehicle",
                     "Plug-in Hybrid", "Battery Electric", "All EVs"]

    ELIGIBILITY_CRITERIA = [
        "New vehicle purchase only",
        "New or used vehicles",
        "Lease or purchase",
        "Must be registered in state for 2+ years",
        "Primary residence requirement",
        "First-time EV buyer priority",
    ]

    def __init__(self, seed: int = 42):
        """Initialize with random seed for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        self.generation_timestamp = datetime.now()

    def generate_raw_policy_data(self) -> List[Dict[str, Any]]:
        """
        Generate raw, messy policy data simulating real-world sources.
        Includes inconsistent formats, missing values, and varied structures.
        """
        raw_data = []

        for state, config in self.STATE_POLICIES.items():
            # Generate multiple policy records per state (updates over time)
            num_updates = random.randint(2, 5)
            base_date = datetime(2020, 1, 1)

            for i in range(num_updates):
                # Simulate policy evolution
                update_date = base_date + timedelta(days=random.randint(i*180, (i+1)*365))

                # Add randomness to rebate amounts (policy changes)
                rebate_variation = random.uniform(0.8, 1.2)
                current_rebate = int(config["base_rebate"] * rebate_variation)

                # Generate messy record with intentional inconsistencies
                record = self._generate_messy_record(
                    state, config, update_date, current_rebate, i
                )
                raw_data.append(record)

        # Shuffle to simulate unordered data ingestion
        random.shuffle(raw_data)
        return raw_data

    def _generate_messy_record(
        self, state: str, config: Dict, update_date: datetime,
        rebate: int, version: int
    ) -> Dict[str, Any]:
        """Generate a single messy policy record."""

        # Randomly select date format (simulating varied data sources)
        date_fmt = random.choice(self.DATE_FORMATS)

        # Create record with intentional messiness
        record = {
            # Inconsistent key naming
            random.choice(["state", "State", "STATE", "state_name"]): state,
            random.choice(["rebate_amount", "incentive", "subsidy_usd", "amount"]):
                random.choice([rebate, f"${rebate}", f"{rebate}.00", str(rebate)]),
            "effective_date": update_date.strftime(date_fmt),
            "policy_type": config["policy_type"],
            "region": config["region"],
        }

        # Randomly include/exclude optional fields
        if config["income_cap"] and random.random() > 0.3:
            record[random.choice(["income_limit", "income_cap", "max_income"])] = (
                random.choice([config["income_cap"], f"${config['income_cap']:,}",
                              str(config["income_cap"])])
            )

        if random.random() > 0.4:
            record["vehicle_type"] = random.choice(self.VEHICLE_TYPES)

        if random.random() > 0.5:
            record["eligibility"] = random.choice(self.ELIGIBILITY_CRITERIA)

        # Add metadata
        record["source"] = random.choice([
            "state_dmv_website", "energy_office", "legislative_record",
            "third_party_aggregator", "press_release"
        ])
        record["version"] = version + 1
        record["ev_adoption_rate"] = config["ev_adoption_rate"]
        record["population_millions"] = config["population_m"]

        # Occasionally add noise/errors
        if random.random() > 0.9:
            record["notes"] = random.choice([
                "PENDING LEGISLATIVE APPROVAL",
                "Subject to annual budget allocation",
                "Program may be discontinued",
                None
            ])

        return record

    def generate_time_series_data(self) -> pd.DataFrame:
        """Generate time series data for policy tracking."""
        records = []

        for state, config in self.STATE_POLICIES.items():
            # Generate monthly data points
            start_date = datetime(2021, 1, 1)

            for month in range(36):  # 3 years of data
                date = start_date + timedelta(days=month * 30)

                # Simulate gradual policy changes
                trend = 1 + (month / 100) * random.uniform(-0.5, 1.5)
                current_rebate = max(0, int(config["base_rebate"] * trend))

                # Simulate application volumes
                base_applications = int(config["population_m"] * 100 * config["ev_adoption_rate"])
                applications = int(base_applications * random.uniform(0.7, 1.3))

                records.append({
                    "state": state,
                    "date": date,
                    "rebate_amount": current_rebate,
                    "applications_processed": applications,
                    "avg_processing_days": random.randint(5, 45),
                    "approval_rate": random.uniform(0.75, 0.98),
                    "total_disbursed": applications * current_rebate * random.uniform(0.6, 0.9)
                })

        return pd.DataFrame(records)

    def save_raw_data(self, output_dir: str = "data/raw") -> Dict[str, str]:
        """Save generated raw data to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate and save raw policy data
        raw_policies = self.generate_raw_policy_data()
        policy_file = output_path / "ev_subsidies_raw.json"
        with open(policy_file, 'w') as f:
            json.dump(raw_policies, f, indent=2, default=str)

        # Generate and save time series data
        time_series = self.generate_time_series_data()
        ts_file = output_path / "policy_time_series.csv"
        time_series.to_csv(ts_file, index=False)

        # Save state metadata
        metadata = {
            "generation_timestamp": self.generation_timestamp.isoformat(),
            "num_states": len(self.STATE_POLICIES),
            "num_raw_records": len(raw_policies),
            "time_series_records": len(time_series),
            "date_range": {
                "start": time_series["date"].min().isoformat(),
                "end": time_series["date"].max().isoformat()
            }
        }
        meta_file = output_path / "generation_metadata.json"
        with open(meta_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        return {
            "raw_policies": str(policy_file),
            "time_series": str(ts_file),
            "metadata": str(meta_file)
        }


if __name__ == "__main__":
    # Generate sample data
    generator = EVSubsidyDataGenerator(seed=42)
    files = generator.save_raw_data()

    print("Generated data files:")
    for name, path in files.items():
        print(f"  {name}: {path}")

    # Preview raw data
    with open(files["raw_policies"]) as f:
        raw = json.load(f)
    print(f"\nGenerated {len(raw)} raw policy records")
    print(f"Sample record: {json.dumps(raw[0], indent=2, default=str)}")
