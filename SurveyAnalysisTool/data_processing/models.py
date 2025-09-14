"""
Core data models and structures for survey analysis.

This module defines the fundamental data structures used throughout the
survey analysis tool, including metadata, variable definitions, and
result containers for various statistical analyses.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Union, Tuple, Any
from enum import Enum
import pandas as pd
import numpy as np


class VariableType(Enum):
    """Enumeration of variable measurement levels."""
    NOMINAL = "nominal"
    ORDINAL = "ordinal"
    INTERVAL = "interval"
    RATIO = "ratio"
    BINARY = "binary"


class MissingDataPattern(Enum):
    """Enumeration of missing data mechanisms."""
    MCAR = "missing_completely_at_random"
    MAR = "missing_at_random"
    MNAR = "missing_not_at_random"


class CorrelationType(Enum):
    """Enumeration of correlation methods."""
    PEARSON = "pearson"
    SPEARMAN = "spearman"
    KENDALL = "kendall"
    POINT_BISERIAL = "point_biserial"
    PHI = "phi"
    POLYCHORIC = "polychoric"
    TETRACHORIC = "tetrachoric"
    PARTIAL = "partial"
    SEMI_PARTIAL = "semi_partial"


class TestType(Enum):
    """Enumeration of statistical test types."""
    CHI_SQUARE_INDEPENDENCE = "chi_square_independence"
    CHI_SQUARE_GOODNESS_OF_FIT = "chi_square_goodness_of_fit"
    CHI_SQUARE_HOMOGENEITY = "chi_square_homogeneity"
    FISHER_EXACT = "fisher_exact"
    MCNEMAR = "mcnemar"
    COCHRAN_ARMITAGE = "cochran_armitage"


@dataclass
class SurveyMetadata:
    """Container for survey metadata and administrative information."""
    survey_id: str
    title: str
    description: str
    collection_period: Tuple[datetime, datetime]
    target_population: str
    sampling_method: str
    expected_responses: int
    actual_responses: int
    completion_rate: float
    variables: Dict[str, 'VariableDefinition'] = field(default_factory=dict)
    weights_variable: Optional[str] = None
    strata_variables: List[str] = field(default_factory=list)
    cluster_variable: Optional[str] = None

    def add_variable(self, variable: 'VariableDefinition') -> None:
        """Add a variable definition to the survey metadata."""
        self.variables[variable.name] = variable

    def get_variable(self, name: str) -> Optional['VariableDefinition']:
        """Retrieve a variable definition by name."""
        return self.variables.get(name)

    def list_variables_by_type(self, var_type: VariableType) -> List[str]:
        """List all variables of a specific type."""
        return [name for name, var in self.variables.items()
                if var.type == var_type]


@dataclass
class VariableDefinition:
    """Definition of a survey variable with metadata and constraints."""
    name: str
    label: str
    type: VariableType
    categories: Optional[Dict[int, str]] = None
    valid_range: Optional[Tuple[float, float]] = None
    missing_codes: List[Union[int, str]] = field(default_factory=list)
    question_text: str = ""
    scale_type: Optional[str] = None
    recoded_from: Optional[str] = None
    transformation: Optional[str] = None

    def is_categorical(self) -> bool:
        """Check if variable is categorical (nominal, ordinal, or binary)."""
        return self.type in [VariableType.NOMINAL, VariableType.ORDINAL, VariableType.BINARY]

    def is_continuous(self) -> bool:
        """Check if variable is continuous (interval or ratio)."""
        return self.type in [VariableType.INTERVAL, VariableType.RATIO]

    def is_ordinal_or_higher(self) -> bool:
        """Check if variable has ordinal or higher measurement level."""
        return self.type in [VariableType.ORDINAL, VariableType.INTERVAL, VariableType.RATIO]


@dataclass
class CorrelationResult:
    """Container for correlation analysis results."""
    variable1: str
    variable2: str
    correlation_type: CorrelationType
    correlation_coefficient: float
    p_value: float
    confidence_interval: Tuple[float, float]
    sample_size: int
    degrees_of_freedom: Optional[int] = None
    effect_size_interpretation: str = ""
    assumptions_met: bool = True
    warnings: List[str] = field(default_factory=list)
    control_variables: List[str] = field(default_factory=list)

    def is_significant(self, alpha: float = 0.05) -> bool:
        """Check if correlation is statistically significant."""
        return self.p_value < alpha

    def get_effect_size_category(self) -> str:
        """Categorize effect size according to Cohen's conventions."""
        abs_r = abs(self.correlation_coefficient)
        if abs_r < 0.1:
            return "negligible"
        elif abs_r < 0.3:
            return "small"
        elif abs_r < 0.5:
            return "medium"
        else:
            return "large"


@dataclass
class ChiSquareResult:
    """Container for chi-square test results."""
    test_type: TestType
    chi_square_statistic: float
    p_value: float
    degrees_of_freedom: int
    effect_size: float
    effect_size_measure: str
    contingency_table: pd.DataFrame
    expected_frequencies: pd.DataFrame
    standardized_residuals: pd.DataFrame
    assumptions_met: bool = True
    minimum_expected_frequency: float = 0.0
    warnings: List[str] = field(default_factory=list)
    exact_test_used: bool = False

    def is_significant(self, alpha: float = 0.05) -> bool:
        """Check if test result is statistically significant."""
        return self.p_value < alpha

    def get_effect_size_category(self) -> str:
        """Categorize effect size according to conventional benchmarks."""
        if self.effect_size_measure.lower() == "cramer's v":
            if self.effect_size < 0.1:
                return "negligible"
            elif self.effect_size < 0.3:
                return "small"
            elif self.effect_size < 0.5:
                return "medium"
            else:
                return "large"
        return "unknown"


@dataclass
class CrosstabResult:
    """Container for cross-tabulation analysis results."""
    row_variable: str
    column_variable: str
    control_variables: List[str] = field(default_factory=list)
    observed_frequencies: pd.DataFrame = field(default_factory=pd.DataFrame)
    percentages: Dict[str, pd.DataFrame] = field(default_factory=dict)
    chi_square_test: Optional[ChiSquareResult] = None
    association_measures: Dict[str, float] = field(default_factory=dict)
    weighted: bool = False
    design_effect: Optional[float] = None
    effective_sample_size: Optional[int] = None

    def get_row_percentages(self) -> pd.DataFrame:
        """Get row percentages from the cross-tabulation."""
        return self.percentages.get('row', pd.DataFrame())

    def get_column_percentages(self) -> pd.DataFrame:
        """Get column percentages from the cross-tabulation."""
        return self.percentages.get('column', pd.DataFrame())

    def get_total_percentages(self) -> pd.DataFrame:
        """Get total percentages from the cross-tabulation."""
        return self.percentages.get('total', pd.DataFrame())


@dataclass
class MissingDataAnalysis:
    """Container for missing data analysis results."""
    total_missing: int
    missing_percentage: float
    missing_by_variable: Dict[str, Dict[str, Union[int, float]]]
    missing_patterns: pd.DataFrame
    pattern_frequencies: Dict[str, int]
    estimated_mechanism: MissingDataPattern
    little_mcar_test: Optional[Dict[str, float]] = None
    missing_correlations: Optional[pd.DataFrame] = None
    recommendations: List[str] = field(default_factory=list)


@dataclass
class DescriptiveStats:
    """Container for descriptive statistics results."""
    variable: str
    count: int
    mean: Optional[float] = None
    median: Optional[float] = None
    mode: Optional[Union[float, str, List]] = None
    std: Optional[float] = None
    variance: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    minimum: Optional[Union[float, str]] = None
    maximum: Optional[Union[float, str]] = None
    range_: Optional[float] = None
    iqr: Optional[float] = None
    percentiles: Dict[int, float] = field(default_factory=dict)
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    frequency_table: Optional[pd.DataFrame] = None
    normality_tests: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass
class QualityMetrics:
    """Container for survey response quality metrics."""
    total_responses: int
    complete_responses: int
    partial_responses: int
    completion_rate: float
    median_completion_time: Optional[float] = None
    straight_lining_count: int = 0
    straight_lining_percentage: float = 0.0
    speeding_count: int = 0
    speeding_percentage: float = 0.0
    duplicate_responses: int = 0
    response_consistency_score: Optional[float] = None
    item_nonresponse_rates: Dict[str, float] = field(default_factory=dict)
    quality_flags: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class WeightingResults:
    """Container for survey weighting results and diagnostics."""
    weights_variable: str
    weighting_method: str
    unweighted_n: int
    effective_sample_size: float
    design_effect: float
    weight_statistics: Dict[str, float]
    convergence_achieved: bool = True
    iterations: int = 0
    target_margins: Dict[str, Dict] = field(default_factory=dict)
    achieved_margins: Dict[str, Dict] = field(default_factory=dict)
    efficiency: float = 1.0
    warnings: List[str] = field(default_factory=list)


# Type aliases for convenience
AnalysisResults = Dict[str, Any]
VariableList = List[str]
CategoryMapping = Dict[int, str]
ConfidenceInterval = Tuple[float, float]