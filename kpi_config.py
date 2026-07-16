"""CBOE KPI tracker config (SEC-8/SEC-9, lightweight edition).

Company -> KPI list for the static snapshot builder (build_kpi_snapshot.py).
Members are given as bare local-names (suffix); kpi_pilot._matches_dims does
namespace-agnostic matching, so the same member string works across every
company's prefix. All mappings were verified live against each company's
latest 10-Q XBRL (2026-07-15).

Coverage:
- TIER A (every public company): eps_diluted, revenue (fallback chain),
  net_income; banks add provision_credit_losses.
- TIER B (segment/dimensional): the marquee CBOE-listed segment revenues,
  verified present on the revenue concept + segment/product/geo axis.
- TIER C (operational counts: Meta DAP, MARA BTC, Tesla units, Robinhood
  funded customers, ...) is NOT in XBRL and is out of scope here (SEC-13).
- SpaceX is private, no filings (SEC-15 manual entry).
"""

from typing import Any, Dict, List

REV = "RevenueFromContractWithCustomerExcludingAssessedTax"
# Revenue concept varies by company; try the contract-revenue tag, then the
# generic Revenues, then the bank net-of-interest variant.
REV_FALLBACKS = ["Revenues", "RevenuesNetOfInterestExpense"]

SEG = "dim_us-gaap_StatementBusinessSegmentsAxis"
PROD = "dim_srt_ProductOrServiceAxis"
GEO = "dim_srt_StatementGeographicalAxis"

NAMES: Dict[str, str] = {
    "AAPL": "Apple", "GOOGL": "Alphabet", "AMD": "AMD", "AMZN": "Amazon",
    "BAC": "Bank of America", "C": "Citigroup", "COIN": "Coinbase",
    "F": "Ford", "INTC": "Intel", "JPM": "JPMorgan Chase",
    "MARA": "MARA Holdings", "META": "Meta", "MSFT": "Microsoft",
    "NFLX": "Netflix", "NVDA": "NVIDIA", "PLTR": "Palantir",
    "HOOD": "Robinhood", "SOFI": "SoFi", "SMCI": "Super Micro",
    "TGT": "Target", "TSLA": "Tesla", "DIS": "Disney",
}


def _eps() -> Dict[str, Any]:
    return {"kpi_key": "eps_diluted", "label": "Diluted EPS", "unit": "usd_per_share",
            "concept": "EarningsPerShareDiluted", "dims": None}


def _revenue(label: str = "Revenue") -> Dict[str, Any]:
    return {"kpi_key": "revenue", "label": label, "unit": "usd",
            "concept": REV, "fallback_concepts": REV_FALLBACKS, "dims": None}


def _net_income() -> Dict[str, Any]:
    return {"kpi_key": "net_income", "label": "Net income", "unit": "usd",
            "concept": "NetIncomeLoss", "dims": None}


def _provision() -> Dict[str, Any]:
    return {"kpi_key": "provision_credit_losses", "label": "Provision for credit losses", "unit": "usd",
            "concept": "ProvisionForLoanLeaseAndOtherLosses",
            "fallback_concepts": ["ProvisionForCreditLosses", "ProvisionForLoanAndLeaseLosses"], "dims": None}


def _seg(kpi_key: str, label: str, axis: str, member: str) -> Dict[str, Any]:
    return {"kpi_key": kpi_key, "label": label, "unit": "usd",
            "concept": REV, "fallback_concepts": REV_FALLBACKS, "dims": {axis: member}}


def _nflx_region(kpi_key: str, label: str, region_member: str) -> Dict[str, Any]:
    return {"kpi_key": kpi_key, "label": label, "unit": "usd",
            "concept": REV, "fallback_concepts": REV_FALLBACKS,
            "dims": {PROD: "StreamingMember", GEO: region_member}}


def _tier_a(extra: List[Dict[str, Any]] | None = None, revenue_label: str = "Revenue") -> List[Dict[str, Any]]:
    return [_eps(), _revenue(revenue_label), _net_income(), *(extra or [])]


COMPANY_KPIS: Dict[str, List[Dict[str, Any]]] = {
    # Apple / Alphabet keep their originally-verified pilot mappings.
    "AAPL": [
        _eps(),
        {"kpi_key": "total_net_sales", "label": "Total net sales", "unit": "usd", "concept": REV, "dims": None},
        _seg("iphone_net_sales", "iPhone net sales", PROD, "IPhoneMember"),
        _seg("services_net_sales", "Services net sales", PROD, "ServiceMember"),
        _seg("americas_net_sales", "Americas net sales", SEG, "AmericasSegmentMember"),
        _seg("greater_china_net_sales", "Greater China net sales", SEG, "GreaterChinaSegmentMember"),
    ],
    "GOOGL": [
        _eps(),
        {"kpi_key": "revenues", "label": "Revenues", "unit": "usd", "concept": "Revenues", "fallback_concepts": [REV], "dims": None},
        _net_income(),
        # YouTube ads are disclosed inside the Google Services segment - two axes.
        {"kpi_key": "youtube_ads_revenues", "label": "YouTube ads revenues", "unit": "usd", "concept": REV,
         "fallback_concepts": REV_FALLBACKS, "dims": {PROD: "YouTubeAdvertisingRevenueMember", SEG: "GoogleServicesMember"}},
        _seg("google_cloud_revenues", "Google Cloud revenues", SEG, "GoogleCloudMember"),
    ],
    "AMD": _tier_a([
        _seg("data_center", "Data Center", SEG, "DataCenterMember"),
        _seg("client_and_gaming", "Client and Gaming", SEG, "ClientAndGamingMember"),
        _seg("embedded", "Embedded", SEG, "EmbeddedMember"),
    ]),
    "AMZN": _tier_a([
        _seg("aws", "AWS", SEG, "AmazonWebServicesSegmentMember"),
        _seg("north_america", "North America", SEG, "NorthAmericaSegmentMember"),
        _seg("international", "International", SEG, "InternationalSegmentMember"),
        _seg("advertising", "Advertising services", PROD, "AdvertisingServicesMember"),
    ]),
    # BAC/C tag provision under a concept this snapshot doesn't map yet
    # (JPM's ProvisionForLoanLeaseAndOtherLosses resolves); left to SEC-10.
    "BAC": _tier_a(),
    "C": _tier_a(),
    "COIN": _tier_a(),
    "F": _tier_a([
        _seg("ford_pro", "Ford Pro", SEG, "FordProMember"),
        _seg("ford_blue", "Ford Blue", SEG, "FordBlueMember"),
        _seg("ford_model_e", "Ford Model e", SEG, "FordModelEMember"),
        _seg("ford_credit", "Ford Credit", SEG, "FordCreditMember"),
    ]),
    "INTC": _tier_a([
        _seg("ccg", "Client Computing Group", SEG, "ClientComputingGroupMember"),
        _seg("dcai", "Data Center and AI", SEG, "DatacenterAndAIMember"),
        _seg("foundry", "Intel Foundry", SEG, "IntelFoundryMember"),
    ]),
    "JPM": _tier_a([_provision()]),
    "MARA": _tier_a(),
    "META": _tier_a([
        _seg("family_of_apps", "Family of Apps", SEG, "FamilyOfAppsMember"),
        _seg("reality_labs", "Reality Labs", SEG, "RealityLabsMember"),
        # Ad revenue is disclosed inside the Family of Apps segment - two axes.
        {"kpi_key": "advertising", "label": "Advertising", "unit": "usd", "concept": REV,
         "fallback_concepts": REV_FALLBACKS, "dims": {PROD: "AdvertisingMember", SEG: "FamilyOfAppsMember"}},
    ]),
    "MSFT": _tier_a([
        _seg("intelligent_cloud", "Intelligent Cloud", SEG, "IntelligentCloudMember"),
        _seg("productivity", "Productivity and Business Processes", SEG, "ProductivityAndBusinessProcessesMember"),
        _seg("more_personal_computing", "More Personal Computing", SEG, "MorePersonalComputingMember"),
    ]),
    # Netflix regional revenue carries a StreamingMember product co-dimension.
    "NFLX": _tier_a([
        _nflx_region("ucan", "United States and Canada", "UnitedStatesAndCanadaMember"),
        _nflx_region("emea", "EMEA", "EMEAMember"),
        _nflx_region("latam", "Latin America", "LatinAmericaMember"),
        _nflx_region("apac", "Asia-Pacific", "AsiaPacificMember"),
    ]),
    "NVDA": _tier_a([
        _seg("data_center", "Data Center", PROD, "DataCenterMember"),
        _seg("compute_networking", "Compute and Networking", SEG, "ComputeAndNetworkingSegmentMember"),
        _seg("graphics", "Graphics", SEG, "GraphicsSegmentMember"),
    ]),
    "PLTR": _tier_a(),
    "HOOD": _tier_a([
        _seg("transaction_based", "Transaction-based revenues", PROD, "TransactionBasedRevenuesMember"),
        _seg("gold_subscription", "Robinhood Gold subscriptions", PROD, "RobinhoodGoldSubscriptionRevenuesMember"),
    ]),
    "SOFI": _tier_a([
        _seg("lending", "Lending", SEG, "LendingSegmentMember"),
        _seg("financial_services", "Financial Services", SEG, "FinancialServicesSegmentMember"),
        _seg("technology_platform", "Technology Platform", SEG, "TechnologyPlatformSegmentMember"),
    ]),
    "SMCI": _tier_a(),
    "TGT": _tier_a([
        _seg("food_beverage", "Food and beverage", PROD, "FoodAndBeverageMember"),
        _seg("beauty", "Beauty", PROD, "BeautyMember"),
        _seg("household_essentials", "Household essentials", PROD, "HouseholdEssentialsMember"),
        _seg("hardlines", "Hardlines", PROD, "HardlinesMember"),
    ]),
    "TSLA": _tier_a([
        _seg("automotive_revenues", "Automotive revenues", PROD, "AutomotiveRevenuesMember"),
        _seg("energy_storage", "Energy generation and storage", PROD, "EnergyGenerationAndStorageMember"),
    ]),
    "DIS": _tier_a([
        _seg("entertainment", "Entertainment", SEG, "EntertainmentSegmentMember"),
        _seg("sports", "Sports", SEG, "SportsSegmentMember"),
        _seg("experiences", "Experiences", SEG, "ExperiencesSegmentMember"),
    ]),
}
