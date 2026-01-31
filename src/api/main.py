"""
FastAPI REST API for EV Subsidy Analytics
Serves processed policy data with filtering, search, and analytics endpoints
Demonstrates API development skills from JD.com platform work
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
from enum import Enum

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import pandas as pd


# ============================================================================
# Pydantic Models
# ============================================================================

class Region(str, Enum):
    WEST = "West"
    NORTHEAST = "Northeast"
    SOUTH = "South"
    MIDWEST = "Midwest"


class RebateTier(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


class PolicyType(str, Enum):
    REBATE = "rebate"
    TAX_CREDIT = "tax_credit"
    SALES_TAX_EXEMPTION = "sales_tax_exemption"
    NONE = "none"


class PolicySummary(BaseModel):
    """Summarized policy response model"""
    state: str
    region: str
    rebate_usd: int
    rebate_tier: str
    policy_type: str
    summary: Optional[str] = None
    key_insights: Optional[List[str]] = None

    class Config:
        json_schema_extra = {
            "example": {
                "state": "California",
                "region": "West",
                "rebate_usd": 7500,
                "rebate_tier": "high",
                "policy_type": "rebate",
                "summary": "California offers one of the nation's most generous EV incentives...",
                "key_insights": ["Above-average incentive compared to national benchmarks"]
            }
        }


class PolicyDetail(BaseModel):
    """Full policy detail response model"""
    state: str
    region: str
    rebate_usd: int
    rebate_tier: str
    income_cap_usd: Optional[int] = None
    has_income_cap: bool
    policy_type: str
    vehicle_type: str
    eligibility: str
    ev_adoption_rate: float
    population_millions: float
    est_annual_impact_millions: float
    summary: Optional[str] = None
    key_insights: Optional[List[str]] = None
    regional_comparison: Optional[str] = None


class AnalyticsResponse(BaseModel):
    """Analytics summary response"""
    total_states: int
    states_with_incentives: int
    avg_rebate_usd: float
    max_rebate_usd: int
    total_estimated_impact_millions: float
    by_region: Dict[str, Any]
    by_tier: Dict[str, int]


class PipelineMetrics(BaseModel):
    """Pipeline processing metrics"""
    records_processed: int
    records_normalized: int
    processing_time_ms: int
    success_rate: float
    last_updated: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    data_loaded: bool
    record_count: int


# ============================================================================
# Application Setup
# ============================================================================

app = FastAPI(
    title="EV Subsidy Analytics API",
    description="""
    REST API for accessing US state-level electric vehicle subsidy data.

    **Features:**
    - Query policies by state, region, or rebate tier
    - Get AI-generated policy summaries
    - Access analytics and aggregated insights
    - Monitor pipeline processing metrics

    Built as part of MIT MBAn interview demonstration.
    """,
    version="1.0.0",
    contact={
        "name": "Charlotte",
        "email": "charlotte@example.com"
    },
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Data Loading
# ============================================================================

class DataStore:
    """In-memory data store for processed policy data"""

    def __init__(self):
        self.policies_df: Optional[pd.DataFrame] = None
        self.summaries: Optional[List[Dict]] = None
        self.metrics: Optional[Dict] = None
        self.last_loaded: Optional[datetime] = None

    def load_data(self, data_dir: str = "data/processed"):
        """Load processed data from files"""
        data_path = Path(data_dir)

        # Load normalized policies
        csv_file = data_path / "ev_subsidies_normalized.csv"
        if csv_file.exists():
            self.policies_df = pd.read_csv(csv_file)

        # Load summaries
        summary_file = data_path / "policy_summaries.json"
        if summary_file.exists():
            with open(summary_file) as f:
                self.summaries = json.load(f)

        # Load processing metrics
        metrics_file = data_path / "processing_stats.json"
        if metrics_file.exists():
            with open(metrics_file) as f:
                self.metrics = json.load(f)

        self.last_loaded = datetime.now()

        # If no processed data, try to run pipeline
        if self.policies_df is None:
            self._run_pipeline()

    def _run_pipeline(self):
        """Run data generation and normalization pipeline"""
        try:
            from src.pipeline.data_generator import EVSubsidyDataGenerator
            from src.pipeline.normalizer import normalize_pipeline
            from src.pipeline.summarizer import PolicySummarizer

            # Generate data
            generator = EVSubsidyDataGenerator(seed=42)
            generator.save_raw_data()

            # Normalize
            df, metrics = normalize_pipeline()
            self.policies_df = df
            self.metrics = metrics

            # Summarize
            summarizer = PolicySummarizer(use_ai=False)
            df_with_summaries = summarizer.summarize_all_policies(df)
            summarizer.save_summaries(df_with_summaries)

            # Reload summaries
            summary_file = Path("data/processed/policy_summaries.json")
            if summary_file.exists():
                with open(summary_file) as f:
                    self.summaries = json.load(f)

            self.last_loaded = datetime.now()

        except Exception as e:
            print(f"Failed to run pipeline: {e}")

    def is_loaded(self) -> bool:
        return self.policies_df is not None


# Global data store
data_store = DataStore()


@app.on_event("startup")
async def startup_event():
    """Load data on application startup"""
    data_store.load_data()


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", tags=["Root"])
async def root():
    """API root endpoint"""
    return {
        "message": "EV Subsidy Analytics API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "policies": "/api/v1/policies",
            "analytics": "/api/v1/analytics",
            "health": "/health"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if data_store.is_loaded() else "degraded",
        timestamp=datetime.now().isoformat(),
        data_loaded=data_store.is_loaded(),
        record_count=len(data_store.policies_df) if data_store.is_loaded() else 0
    )


@app.get("/api/v1/policies", response_model=List[PolicySummary], tags=["Policies"])
async def get_policies(
    region: Optional[Region] = Query(None, description="Filter by region"),
    tier: Optional[RebateTier] = Query(None, description="Filter by rebate tier"),
    min_rebate: Optional[int] = Query(None, ge=0, description="Minimum rebate amount"),
    max_rebate: Optional[int] = Query(None, ge=0, description="Maximum rebate amount"),
    has_income_cap: Optional[bool] = Query(None, description="Filter by income cap requirement"),
    limit: int = Query(50, ge=1, le=100, description="Maximum results to return"),
    offset: int = Query(0, ge=0, description="Results offset for pagination")
):
    """
    Get list of EV subsidy policies with optional filtering.

    Returns summarized policy information for quick overview.
    """
    if not data_store.is_loaded():
        raise HTTPException(status_code=503, detail="Data not loaded")

    df = data_store.policies_df.copy()

    # Apply filters
    if region:
        df = df[df["region"] == region.value]
    if tier:
        df = df[df["rebate_tier"] == tier.value]
    if min_rebate is not None:
        df = df[df["rebate_usd"] >= min_rebate]
    if max_rebate is not None:
        df = df[df["rebate_usd"] <= max_rebate]
    if has_income_cap is not None:
        df = df[df["has_income_cap"] == has_income_cap]

    # Pagination
    df = df.iloc[offset:offset + limit]

    # Merge with summaries
    results = []
    for _, row in df.iterrows():
        summary_data = next(
            (s for s in (data_store.summaries or []) if s["state"] == row["state"]),
            {}
        )
        results.append(PolicySummary(
            state=row["state"],
            region=row["region"],
            rebate_usd=int(row["rebate_usd"]),
            rebate_tier=row["rebate_tier"],
            policy_type=row["policy_type"],
            summary=summary_data.get("summary"),
            key_insights=summary_data.get("key_insights")
        ))

    return results


@app.get("/api/v1/policies/{state}", response_model=PolicyDetail, tags=["Policies"])
async def get_policy_by_state(state: str):
    """
    Get detailed policy information for a specific state.

    Returns full policy details including AI-generated summary and insights.
    """
    if not data_store.is_loaded():
        raise HTTPException(status_code=503, detail="Data not loaded")

    # Case-insensitive state lookup
    df = data_store.policies_df
    mask = df["state"].str.lower() == state.lower()

    if not mask.any():
        raise HTTPException(status_code=404, detail=f"State '{state}' not found")

    row = df[mask].iloc[0]

    # Get summary data
    summary_data = next(
        (s for s in (data_store.summaries or []) if s["state"].lower() == state.lower()),
        {}
    )

    return PolicyDetail(
        state=row["state"],
        region=row["region"],
        rebate_usd=int(row["rebate_usd"]),
        rebate_tier=row["rebate_tier"],
        income_cap_usd=int(row["income_cap_usd"]) if pd.notna(row.get("income_cap_usd")) else None,
        has_income_cap=bool(row.get("has_income_cap", False)),
        policy_type=row["policy_type"],
        vehicle_type=row.get("vehicle_type", "all_electric_vehicles"),
        eligibility=row.get("eligibility", "Standard eligibility"),
        ev_adoption_rate=float(row.get("ev_adoption_rate", 0)),
        population_millions=float(row.get("population_millions", 0)),
        est_annual_impact_millions=float(row.get("est_annual_impact_millions", 0)),
        summary=summary_data.get("summary"),
        key_insights=summary_data.get("key_insights"),
        regional_comparison=summary_data.get("regional_comparison")
    )


@app.get("/api/v1/analytics", response_model=AnalyticsResponse, tags=["Analytics"])
async def get_analytics():
    """
    Get aggregated analytics across all policies.

    Returns summary statistics, regional breakdowns, and tier distributions.
    """
    if not data_store.is_loaded():
        raise HTTPException(status_code=503, detail="Data not loaded")

    df = data_store.policies_df

    # Regional breakdown
    by_region = {}
    for region in df["region"].unique():
        region_df = df[df["region"] == region]
        by_region[region] = {
            "count": len(region_df),
            "avg_rebate": round(region_df["rebate_usd"].mean(), 2),
            "total_impact_millions": round(region_df["est_annual_impact_millions"].sum(), 2)
        }

    # Tier breakdown
    by_tier = df["rebate_tier"].value_counts().to_dict()

    return AnalyticsResponse(
        total_states=len(df),
        states_with_incentives=len(df[df["rebate_usd"] > 0]),
        avg_rebate_usd=round(df["rebate_usd"].mean(), 2),
        max_rebate_usd=int(df["rebate_usd"].max()),
        total_estimated_impact_millions=round(df["est_annual_impact_millions"].sum(), 2),
        by_region=by_region,
        by_tier=by_tier
    )


@app.get("/api/v1/analytics/regions", tags=["Analytics"])
async def get_regional_comparison():
    """Get detailed regional comparison data for visualizations"""
    if not data_store.is_loaded():
        raise HTTPException(status_code=503, detail="Data not loaded")

    df = data_store.policies_df
    regions = []

    for region in df["region"].unique():
        region_df = df[df["region"] == region]
        regions.append({
            "region": region,
            "states": region_df["state"].tolist(),
            "avg_rebate": round(region_df["rebate_usd"].mean(), 2),
            "max_rebate": int(region_df["rebate_usd"].max()),
            "min_rebate": int(region_df["rebate_usd"].min()),
            "avg_adoption_rate": round(region_df["ev_adoption_rate"].mean(), 4),
            "total_population_millions": round(region_df["population_millions"].sum(), 2),
            "total_estimated_impact": round(region_df["est_annual_impact_millions"].sum(), 2)
        })

    return {"regions": regions}


@app.get("/api/v1/metrics", response_model=PipelineMetrics, tags=["Metrics"])
async def get_pipeline_metrics():
    """
    Get data pipeline processing metrics.

    Shows performance statistics from the last data processing run.
    """
    if data_store.metrics is None:
        raise HTTPException(status_code=503, detail="Metrics not available")

    return PipelineMetrics(
        records_processed=data_store.metrics.get("records_processed", 0),
        records_normalized=data_store.metrics.get("records_normalized", 0),
        processing_time_ms=data_store.metrics.get("processing_time_ms", 0),
        success_rate=data_store.metrics.get("success_rate", 0),
        last_updated=data_store.last_loaded.isoformat() if data_store.last_loaded else "unknown"
    )


@app.post("/api/v1/reload", tags=["Admin"])
async def reload_data():
    """
    Reload data from processed files.

    Useful for refreshing data after pipeline re-run.
    """
    data_store.load_data()
    return {
        "status": "reloaded",
        "timestamp": datetime.now().isoformat(),
        "records": len(data_store.policies_df) if data_store.is_loaded() else 0
    }


# ============================================================================
# Run Application
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
