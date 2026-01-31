"""
AI-Powered Policy Summarizer
Uses NLP to generate human-readable policy summaries
Demonstrates AI/ML integration skills from JD.com analytics platform
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd

# Try importing transformers, fall back to rule-based if unavailable
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available, using rule-based summarization")


class PolicySummarizer:
    """
    Generates AI-powered summaries of EV subsidy policies.
    Supports both transformer-based and rule-based summarization.
    """

    # Template patterns for rule-based generation
    SUMMARY_TEMPLATES = {
        "high": (
            "{state} offers one of the nation's most generous EV incentives with up to "
            "${rebate:,} available. {income_clause} The state's {adoption_rate:.1%} EV "
            "adoption rate reflects strong consumer interest in electric vehicles."
        ),
        "medium": (
            "{state} provides a moderate EV incentive of ${rebate:,} through its "
            "{policy_type} program. {income_clause} With {pop:.1f}M residents, "
            "the program has significant potential reach."
        ),
        "low": (
            "{state} offers a limited EV incentive of ${rebate:,}. {income_clause} "
            "The state's {adoption_rate:.1%} adoption rate suggests room for growth "
            "in the EV market."
        ),
        "none": (
            "{state} currently does not offer a state-level EV purchase incentive. "
            "Residents may still qualify for federal tax credits. The state has "
            "{pop:.1f}M residents with a {adoption_rate:.1%} EV adoption rate."
        ),
    }

    INCOME_CLAUSES = {
        True: "Income eligibility is capped at ${cap:,} for individuals. ",
        False: "There are no income restrictions for this program. ",
    }

    POLICY_TYPE_DESCRIPTIONS = {
        "rebate": "direct rebate",
        "tax_credit": "tax credit",
        "sales_tax_exemption": "sales tax exemption",
        "none": "no state incentive",
        "unknown": "incentive",
    }

    def __init__(self, use_ai: bool = True, model_name: str = "facebook/bart-large-cnn"):
        """
        Initialize summarizer.

        Args:
            use_ai: Whether to use AI-based summarization (requires transformers)
            model_name: Hugging Face model for summarization
        """
        self.use_ai = use_ai and TRANSFORMERS_AVAILABLE
        self.model_name = model_name
        self._summarizer = None

        if self.use_ai:
            try:
                print(f"Loading summarization model: {model_name}...")
                self._summarizer = pipeline(
                    "summarization",
                    model=model_name,
                    max_length=100,
                    min_length=30,
                    truncation=True
                )
                print("Model loaded successfully!")
            except Exception as e:
                print(f"Failed to load AI model: {e}")
                print("Falling back to rule-based summarization")
                self.use_ai = False

    def summarize_policy(self, policy_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a summary for a single policy.

        Args:
            policy_data: Dictionary containing policy details

        Returns:
            Dictionary with original data plus generated summary
        """
        # Extract key fields
        state = policy_data.get("state", "Unknown")
        rebate = policy_data.get("rebate_usd", 0)
        rebate_tier = policy_data.get("rebate_tier", "medium")
        policy_type = policy_data.get("policy_type", "unknown")
        income_cap = policy_data.get("income_cap_usd")
        has_income_cap = policy_data.get("has_income_cap", False)
        adoption_rate = policy_data.get("ev_adoption_rate", 0.05)
        population = policy_data.get("population_millions", 0)
        region = policy_data.get("region", "Unknown")

        # Generate summary text
        if self.use_ai:
            summary_text = self._ai_summarize(policy_data)
        else:
            summary_text = self._rule_based_summarize(
                state=state,
                rebate=rebate,
                rebate_tier=rebate_tier,
                policy_type=policy_type,
                income_cap=income_cap,
                has_income_cap=has_income_cap,
                adoption_rate=adoption_rate,
                population=population
            )

        # Generate quick insights
        insights = self._generate_insights(policy_data)

        # Generate comparison context
        comparison = self._generate_comparison(policy_data)

        return {
            **policy_data,
            "summary": summary_text,
            "key_insights": insights,
            "regional_comparison": comparison,
            "summarization_method": "ai" if self.use_ai else "rule_based"
        }

    def _ai_summarize(self, policy_data: Dict[str, Any]) -> str:
        """Generate summary using AI model."""
        # Create input text for summarization
        input_text = self._create_policy_document(policy_data)

        try:
            result = self._summarizer(input_text)
            return result[0]["summary_text"]
        except Exception as e:
            print(f"AI summarization failed: {e}")
            return self._rule_based_summarize(**self._extract_params(policy_data))

    def _create_policy_document(self, policy_data: Dict[str, Any]) -> str:
        """Create a document-style text for AI summarization."""
        state = policy_data.get("state", "Unknown")
        rebate = policy_data.get("rebate_usd", 0)
        policy_type = policy_data.get("policy_type", "unknown")
        income_cap = policy_data.get("income_cap_usd")
        adoption_rate = policy_data.get("ev_adoption_rate", 0.05)
        population = policy_data.get("population_millions", 0)
        region = policy_data.get("region", "Unknown")
        eligibility = policy_data.get("eligibility", "Standard eligibility")

        document = f"""
        Electric Vehicle Subsidy Policy for {state}

        The state of {state}, located in the {region} region of the United States,
        has implemented an electric vehicle incentive program. The program offers
        a {self.POLICY_TYPE_DESCRIPTIONS.get(policy_type, 'incentive')} of up to
        ${rebate:,} for qualifying purchases.

        {"Income eligibility is capped at $" + f"{income_cap:,}" + " for individuals."
         if income_cap else "There are no income restrictions."}

        Eligibility requirements: {eligibility}

        The state has a population of {population:.1f} million residents and
        currently shows an EV adoption rate of {adoption_rate:.1%}. This positions
        {state} as a {"leader" if adoption_rate > 0.1 else "developing market"}
        in the electric vehicle transition.
        """
        return document.strip()

    def _extract_params(self, policy_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract parameters for rule-based summarization."""
        return {
            "state": policy_data.get("state", "Unknown"),
            "rebate": policy_data.get("rebate_usd", 0),
            "rebate_tier": policy_data.get("rebate_tier", "medium"),
            "policy_type": policy_data.get("policy_type", "unknown"),
            "income_cap": policy_data.get("income_cap_usd"),
            "has_income_cap": policy_data.get("has_income_cap", False),
            "adoption_rate": policy_data.get("ev_adoption_rate", 0.05),
            "population": policy_data.get("population_millions", 0),
        }

    def _rule_based_summarize(
        self,
        state: str,
        rebate: int,
        rebate_tier: str,
        policy_type: str,
        income_cap: Optional[int],
        has_income_cap: bool,
        adoption_rate: float,
        population: float
    ) -> str:
        """Generate summary using rule-based templates."""

        # Get appropriate template
        template = self.SUMMARY_TEMPLATES.get(rebate_tier, self.SUMMARY_TEMPLATES["medium"])

        # Generate income clause
        if has_income_cap and income_cap:
            income_clause = self.INCOME_CLAUSES[True].format(cap=income_cap)
        else:
            income_clause = self.INCOME_CLAUSES[False]

        # Get policy type description
        policy_desc = self.POLICY_TYPE_DESCRIPTIONS.get(policy_type, "incentive")

        # Format and return summary
        return template.format(
            state=state,
            rebate=rebate,
            income_clause=income_clause,
            policy_type=policy_desc,
            adoption_rate=adoption_rate,
            pop=population
        )

    def _generate_insights(self, policy_data: Dict[str, Any]) -> List[str]:
        """Generate key insights for the policy."""
        insights = []

        rebate = policy_data.get("rebate_usd", 0)
        rebate_tier = policy_data.get("rebate_tier", "none")
        adoption_rate = policy_data.get("ev_adoption_rate", 0)
        has_income_cap = policy_data.get("has_income_cap", False)
        impact = policy_data.get("est_annual_impact_millions", 0)

        if rebate_tier == "high":
            insights.append("Above-average incentive compared to national benchmarks")
        elif rebate_tier == "none":
            insights.append("No state-level incentive available")

        if adoption_rate > 0.10:
            insights.append(f"High EV adoption rate ({adoption_rate:.1%}) indicates strong market")
        elif adoption_rate < 0.05:
            insights.append(f"Below-average adoption rate ({adoption_rate:.1%}) shows growth potential")

        if has_income_cap:
            insights.append("Income-restricted program targets middle-income consumers")

        if impact > 10:
            insights.append(f"Significant estimated annual impact (${impact:.1f}M)")

        return insights if insights else ["Standard state incentive program"]

    def _generate_comparison(self, policy_data: Dict[str, Any]) -> str:
        """Generate regional comparison context."""
        region = policy_data.get("region", "Unknown")
        rebate = policy_data.get("rebate_usd", 0)
        rebate_tier = policy_data.get("rebate_tier", "medium")

        regional_context = {
            "West": "West Coast states typically offer the highest incentives",
            "Northeast": "Northeastern states show moderate to high incentive levels",
            "South": "Southern states generally have lower state-level incentives",
            "Midwest": "Midwestern states show varied incentive approaches",
        }

        base_context = regional_context.get(region, "Regional comparison unavailable")

        if rebate_tier == "high":
            return f"{base_context}. This state is among the regional leaders."
        elif rebate_tier == "none":
            return f"{base_context}. This state currently lacks a state incentive."
        else:
            return f"{base_context}. This state offers a moderate incentive."

    def summarize_all_policies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate summaries for all policies in DataFrame.

        Args:
            df: DataFrame with normalized policy data

        Returns:
            DataFrame with added summary columns
        """
        print(f"Generating summaries for {len(df)} policies...")

        summaries = []
        for _, row in df.iterrows():
            policy_dict = row.to_dict()
            summarized = self.summarize_policy(policy_dict)
            summaries.append(summarized)

        result_df = pd.DataFrame(summaries)
        print(f"Generated {len(result_df)} policy summaries")

        return result_df

    def save_summaries(
        self, df: pd.DataFrame, output_dir: str = "data/processed"
    ) -> Dict[str, str]:
        """Save summarized data to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save full data with summaries
        csv_file = output_path / "ev_subsidies_with_summaries.csv"
        df.to_csv(csv_file, index=False)

        # Save summary-focused JSON for API
        summary_data = []
        for _, row in df.iterrows():
            summary_data.append({
                "state": row.get("state"),
                "region": row.get("region"),
                "rebate_usd": row.get("rebate_usd"),
                "rebate_tier": row.get("rebate_tier"),
                "summary": row.get("summary"),
                "key_insights": row.get("key_insights"),
                "regional_comparison": row.get("regional_comparison"),
            })

        json_file = output_path / "policy_summaries.json"
        with open(json_file, 'w') as f:
            json.dump(summary_data, f, indent=2)

        return {
            "csv": str(csv_file),
            "json": str(json_file),
        }


if __name__ == "__main__":
    # Test with sample data
    sample_policy = {
        "state": "California",
        "region": "West",
        "rebate_usd": 7500,
        "rebate_tier": "high",
        "income_cap_usd": 150000,
        "has_income_cap": True,
        "policy_type": "rebate",
        "vehicle_type": "battery_electric",
        "eligibility": "New vehicle purchase only",
        "ev_adoption_rate": 0.18,
        "population_millions": 39.5,
        "est_annual_impact_millions": 53.3,
    }

    # Use rule-based for demo (faster, no model download)
    summarizer = PolicySummarizer(use_ai=False)
    result = summarizer.summarize_policy(sample_policy)

    print("\n" + "=" * 60)
    print("POLICY SUMMARY EXAMPLE")
    print("=" * 60)
    print(f"\nState: {result['state']}")
    print(f"\nSummary:\n{result['summary']}")
    print(f"\nKey Insights:")
    for insight in result['key_insights']:
        print(f"  - {insight}")
    print(f"\nRegional Context: {result['regional_comparison']}")
