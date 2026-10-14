"""Feature engineering used by the employee-attrition model.

This lives in its own module (rather than inside the notebook) so that the saved
pipeline in models/final_model.joblib can be loaded from any Python session:
pickle stores a reference to this function, not its code.
"""
import numpy as np

ENGINEERED_FEATURES = ["TenureRatio", "AvgYearsPerCompany", "PromotionGapRatio",
                       "MeanSatisfaction", "IncomePerJobLevel"]


def add_engineered_features(frame):
    """Return a copy of `frame` with five ratio/aggregate features (stateless, no fitting)."""
    out = frame.copy()
    total_years = out["TotalWorkingYears"].replace(0, np.nan)
    out["TenureRatio"] = (out["YearsAtCompany"] / total_years).fillna(0.0)
    out["AvgYearsPerCompany"] = out["TotalWorkingYears"] / (out["NumCompaniesWorked"] + 1)
    out["PromotionGapRatio"] = out["YearsSinceLastPromotion"] / (out["YearsAtCompany"] + 1)
    out["MeanSatisfaction"] = out[["EnvironmentSatisfaction", "JobSatisfaction",
                                   "RelationshipSatisfaction", "WorkLifeBalance"]].mean(axis=1)
    out["IncomePerJobLevel"] = out["MonthlyIncome"] / out["JobLevel"]
    return out
