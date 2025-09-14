"""
Chi-square tests for categorical data analysis in surveys.

This module provides comprehensive chi-square testing including tests of
independence, goodness-of-fit, homogeneity, and related tests for
categorical survey data.
"""

import logging
import warnings
from typing import Dict, List, Optional, Union, Tuple, Any
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats.contingency import expected_freq

try:
    import pingouin as pg
    HAS_PINGOUIN = True
except ImportError:
    HAS_PINGOUIN = False

from ..data_processing.models import ChiSquareResult, TestType


class ChiSquareTests:
    """
    Comprehensive chi-square testing for categorical survey data.

    Features:
    - Chi-square test of independence
    - Chi-square goodness-of-fit test
    - Chi-square test of homogeneity
    - Fisher's exact test for small samples
    - McNemar's test for paired data
    - Cochran-Armitage test for trend
    - Effect size calculations (Cramer's V, phi, etc.)
    - Assumption checking and warnings
    """

    def __init__(self,
                 alpha: float = 0.05,
                 min_expected_freq: float = 5.0):
        """
        Initialize ChiSquareTests.

        Parameters
        ----------
        alpha : float, default 0.05
            Significance level for statistical tests
        min_expected_freq : float, default 5.0
            Minimum expected frequency for chi-square test validity
        """
        self.alpha = alpha
        self.min_expected_freq = min_expected_freq
        self.logger = logging.getLogger(__name__)

    def test_independence(self,
                         data: pd.DataFrame,
                         row_variable: str,
                         column_variable: str,
                         weights: Optional[pd.Series] = None,
                         correction: bool = True,
                         exact: bool = False) -> ChiSquareResult:
        """
        Test independence between two categorical variables.

        Parameters
        ----------
        data : pd.DataFrame
            Survey data
        row_variable : str
            Name of row variable
        column_variable : str
            Name of column variable
        weights : pd.Series, optional
            Survey weights
        correction : bool, default True
            Apply Yates' continuity correction for 2x2 tables
        exact : bool, default False
            Use Fisher's exact test instead of chi-square

        Returns
        -------
        ChiSquareResult
            Complete test results with effect sizes and diagnostics
        """
        # Create contingency table
        contingency_table = self._create_contingency_table(
            data, row_variable, column_variable, weights
        )

        if contingency_table.size == 0:
            raise ValueError("Empty contingency table")

        # Calculate expected frequencies
        expected_frequencies = self._calculate_expected_frequencies(contingency_table)

        # Check assumptions
        assumptions_met, warnings_list = self._check_assumptions(
            contingency_table, expected_frequencies
        )

        # Choose appropriate test
        if exact or (contingency_table.shape == (2, 2) and
                     expected_frequencies.min().min() < self.min_expected_freq):
            # Use Fisher's exact test
            return self._fishers_exact_test(
                contingency_table, row_variable, column_variable,
                expected_frequencies, assumptions_met, warnings_list
            )
        else:
            # Use chi-square test
            return self._chi_square_independence_test(
                contingency_table, row_variable, column_variable,
                expected_frequencies, assumptions_met, warnings_list, correction
            )

    def test_goodness_of_fit(self,
                           observed_frequencies: Union[pd.Series, np.ndarray],
                           expected_frequencies: Optional[Union[pd.Series, np.ndarray]] = None,
                           categories: Optional[List[str]] = None,
                           variable_name: str = "variable") -> ChiSquareResult:
        """
        Test goodness-of-fit for a single categorical variable.

        Parameters
        ----------
        observed_frequencies : pd.Series or np.ndarray
            Observed frequencies for each category
        expected_frequencies : pd.Series or np.ndarray, optional
            Expected frequencies. If None, assumes uniform distribution
        categories : list of str, optional
            Category names
        variable_name : str, default "variable"
            Name of the variable being tested

        Returns
        -------
        ChiSquareResult
            Complete test results
        """
        # Convert to numpy arrays for processing
        if isinstance(observed_frequencies, pd.Series):
            observed = observed_frequencies.values
            if categories is None:
                categories = observed_frequencies.index.tolist()
        else:
            observed = np.array(observed_frequencies)

        if expected_frequencies is None:
            # Uniform distribution
            total = observed.sum()
            expected = np.full(len(observed), total / len(observed))
        else:
            if isinstance(expected_frequencies, pd.Series):
                expected = expected_frequencies.values
            else:
                expected = np.array(expected_frequencies)

        # Check that arrays have same length
        if len(observed) != len(expected):
            raise ValueError("Observed and expected frequencies must have same length")

        # Perform chi-square goodness-of-fit test
        chi_square_stat, p_value = stats.chisquare(observed, expected)

        # Degrees of freedom
        df = len(observed) - 1

        # Calculate effect size (not standard for goodness-of-fit, but useful)
        effect_size = np.sqrt(chi_square_stat / observed.sum())

        # Create contingency table format for consistency
        if categories is None:
            categories = [f"Category_{i+1}" for i in range(len(observed))]

        contingency_table = pd.DataFrame({
            'Observed': observed,
            'Expected': expected
        }, index=categories)

        expected_df = pd.DataFrame({'Expected': expected}, index=categories)

        # Check assumptions
        min_expected = expected.min()
        assumptions_met = min_expected >= self.min_expected_freq
        warnings_list = []

        if not assumptions_met:
            warnings_list.append(
                f"Minimum expected frequency ({min_expected:.2f}) "
                f"is below recommended threshold ({self.min_expected_freq})"
            )

        # Calculate standardized residuals
        standardized_residuals = pd.DataFrame({
            'Residuals': (observed - expected) / np.sqrt(expected)
        }, index=categories)

        return ChiSquareResult(
            test_type=TestType.CHI_SQUARE_GOODNESS_OF_FIT,
            chi_square_statistic=float(chi_square_stat),
            p_value=float(p_value),
            degrees_of_freedom=df,
            effect_size=float(effect_size),
            effect_size_measure="effect_size_w",
            contingency_table=contingency_table,
            expected_frequencies=expected_df,
            standardized_residuals=standardized_residuals,
            assumptions_met=assumptions_met,
            minimum_expected_frequency=float(min_expected),
            warnings=warnings_list
        )

    def test_homogeneity(self,
                        data: pd.DataFrame,
                        response_variable: str,
                        group_variable: str,
                        weights: Optional[pd.Series] = None) -> ChiSquareResult:
        """
        Test homogeneity of distributions across groups.

        This is mathematically equivalent to the independence test but
        conceptually different (testing if response distributions are
        the same across groups).

        Parameters
        ----------
        data : pd.DataFrame
            Survey data
        response_variable : str
            Name of response variable
        group_variable : str
            Name of grouping variable
        weights : pd.Series, optional
            Survey weights

        Returns
        -------
        ChiSquareResult
            Complete test results
        """
        # Test of homogeneity is mathematically identical to test of independence
        result = self.test_independence(
            data, group_variable, response_variable, weights
        )

        # Update test type
        result.test_type = TestType.CHI_SQUARE_HOMOGENEITY

        return result

    def mcnemar_test(self,
                    data: pd.DataFrame,
                    before_variable: str,
                    after_variable: str,
                    correction: bool = True) -> ChiSquareResult:
        """
        McNemar's test for paired categorical data.

        Tests whether the marginal frequencies of two binary variables
        are equal in paired data (e.g., before/after measurements).

        Parameters
        ----------
        data : pd.DataFrame
            Survey data with paired observations
        before_variable : str
            Name of "before" variable
        after_variable : str
            Name of "after" variable
        correction : bool, default True
            Apply continuity correction

        Returns
        -------
        ChiSquareResult
            Complete test results
        """
        # Create 2x2 contingency table for paired data
        contingency_table = pd.crosstab(
            data[before_variable],
            data[after_variable],
            margins=False
        )

        if contingency_table.shape != (2, 2):
            raise ValueError("McNemar's test requires 2x2 table (binary variables)")

        # Extract cells (a, b, c, d)
        try:
            a = contingency_table.iloc[0, 0]  # (0,0)
            b = contingency_table.iloc[0, 1]  # (0,1)
            c = contingency_table.iloc[1, 0]  # (1,0)
            d = contingency_table.iloc[1, 1]  # (1,1)
        except IndexError:
            raise ValueError("Invalid contingency table structure for McNemar's test")

        # McNemar's test statistic
        if correction and (b + c) > 0:
            # With continuity correction
            mcnemar_stat = (abs(b - c) - 1)**2 / (b + c)
        else:
            # Without continuity correction
            if (b + c) == 0:
                mcnemar_stat = 0
                p_value = 1.0
            else:
                mcnemar_stat = (b - c)**2 / (b + c)

        if (b + c) > 0:
            # Chi-square distribution with 1 df
            p_value = 1 - stats.chi2.cdf(mcnemar_stat, 1)
        else:
            p_value = 1.0

        # Effect size (phi coefficient for 2x2 table)
        n = contingency_table.sum().sum()
        phi = np.sqrt(mcnemar_stat / n) if n > 0 else 0

        # Expected frequencies (not meaningful for McNemar's test)
        expected_frequencies = contingency_table.copy() * 0

        # Standardized residuals (simplified for McNemar)
        standardized_residuals = contingency_table.copy() * 0

        return ChiSquareResult(
            test_type=TestType.MCNEMAR,
            chi_square_statistic=float(mcnemar_stat),
            p_value=float(p_value),
            degrees_of_freedom=1,
            effect_size=float(phi),
            effect_size_measure="phi",
            contingency_table=contingency_table,
            expected_frequencies=expected_frequencies,
            standardized_residuals=standardized_residuals,
            assumptions_met=True,  # McNemar has fewer assumptions
            minimum_expected_frequency=float(min(b + c, 1)),
            warnings=[]
        )

    def cochran_armitage_trend_test(self,
                                  data: pd.DataFrame,
                                  response_variable: str,
                                  exposure_variable: str,
                                  weights: Optional[pd.Series] = None) -> ChiSquareResult:
        """
        Cochran-Armitage test for trend in proportions.

        Tests for linear trend in proportions across ordered exposure groups.

        Parameters
        ----------
        data : pd.DataFrame
            Survey data
        response_variable : str
            Binary response variable
        exposure_variable : str
            Ordered exposure variable
        weights : pd.Series, optional
            Survey weights

        Returns
        -------
        ChiSquareResult
            Complete test results
        """
        # Create contingency table
        contingency_table = self._create_contingency_table(
            data, exposure_variable, response_variable, weights
        )

        # Check that response variable is binary
        if contingency_table.shape[1] != 2:
            raise ValueError("Cochran-Armitage test requires binary response variable")

        # Extract data for trend test
        exposure_levels = contingency_table.index.values
        successes = contingency_table.iloc[:, 1].values  # Assuming second column is "success"
        totals = contingency_table.sum(axis=1).values

        # Assign scores to exposure levels (assuming they're ordered)
        scores = np.arange(len(exposure_levels))

        # Calculate test statistic
        # This is a simplified implementation
        n = totals.sum()
        r = successes.sum()

        if n == 0 or r == 0 or r == n:
            # Degenerate case
            trend_stat = 0
            p_value = 1.0
        else:
            # Calculate trend test statistic
            weighted_score_sum = np.sum(scores * totals)
            weighted_success_sum = np.sum(scores * successes)

            mean_score = weighted_score_sum / n
            variance_score = np.sum(totals * (scores - mean_score)**2) / n

            if variance_score == 0:
                trend_stat = 0
                p_value = 1.0
            else:
                # Trend test statistic
                numerator = weighted_success_sum - r * mean_score
                denominator = np.sqrt(r * (n - r) * variance_score / n)

                if denominator == 0:
                    trend_stat = 0
                    p_value = 1.0
                else:
                    trend_stat = numerator / denominator
                    trend_stat = trend_stat**2  # Square for chi-square distribution

                    # P-value from chi-square distribution with 1 df
                    p_value = 1 - stats.chi2.cdf(trend_stat, 1)

        # Effect size (simplified)
        effect_size = np.sqrt(trend_stat / n) if n > 0 else 0

        # Expected frequencies (not standard for trend test)
        expected_frequencies = contingency_table.copy() * 0

        # Standardized residuals (not standard for trend test)
        standardized_residuals = contingency_table.copy() * 0

        return ChiSquareResult(
            test_type=TestType.COCHRAN_ARMITAGE,
            chi_square_statistic=float(trend_stat),
            p_value=float(p_value),
            degrees_of_freedom=1,
            effect_size=float(effect_size),
            effect_size_measure="trend_effect_size",
            contingency_table=contingency_table,
            expected_frequencies=expected_frequencies,
            standardized_residuals=standardized_residuals,
            assumptions_met=True,  # Simplified assumption checking
            minimum_expected_frequency=0.0,
            warnings=[]
        )

    def _create_contingency_table(self,
                                data: pd.DataFrame,
                                row_variable: str,
                                column_variable: str,
                                weights: Optional[pd.Series] = None) -> pd.DataFrame:
        """Create contingency table with optional weights."""
        # Remove missing values
        clean_data = data[[row_variable, column_variable]].dropna()

        if len(clean_data) == 0:
            return pd.DataFrame()

        if weights is not None:
            # Weighted contingency table
            clean_weights = weights[clean_data.index]

            # Create weighted crosstab
            contingency_table = pd.crosstab(
                clean_data[row_variable],
                clean_data[column_variable],
                values=clean_weights,
                aggfunc='sum',
                margins=False
            ).fillna(0)
        else:
            # Unweighted contingency table
            contingency_table = pd.crosstab(
                clean_data[row_variable],
                clean_data[column_variable],
                margins=False
            )

        return contingency_table

    def _calculate_expected_frequencies(self, contingency_table: pd.DataFrame) -> pd.DataFrame:
        """Calculate expected frequencies under independence assumption."""
        row_totals = contingency_table.sum(axis=1)
        col_totals = contingency_table.sum(axis=0)
        grand_total = contingency_table.sum().sum()

        expected = pd.DataFrame(
            index=contingency_table.index,
            columns=contingency_table.columns
        )

        for i, row_total in row_totals.items():
            for j, col_total in col_totals.items():
                expected.loc[i, j] = (row_total * col_total) / grand_total

        return expected

    def _check_assumptions(self,
                          contingency_table: pd.DataFrame,
                          expected_frequencies: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Check chi-square test assumptions."""
        warnings_list = []
        assumptions_met = True

        # Check minimum expected frequency
        min_expected = expected_frequencies.min().min()
        if min_expected < self.min_expected_freq:
            assumptions_met = False
            warnings_list.append(
                f"Minimum expected frequency ({min_expected:.2f}) "
                f"is below recommended threshold ({self.min_expected_freq})"
            )

        # Check percentage of cells with low expected frequencies
        low_freq_cells = (expected_frequencies < self.min_expected_freq).sum().sum()
        total_cells = expected_frequencies.size
        low_freq_percentage = (low_freq_cells / total_cells) * 100

        if low_freq_percentage > 20:
            warnings_list.append(
                f"{low_freq_percentage:.1f}% of cells have expected frequency < {self.min_expected_freq}"
            )

        # Check for empty cells
        empty_cells = (contingency_table == 0).sum().sum()
        if empty_cells > 0:
            warnings_list.append(f"{empty_cells} cells are empty")

        return assumptions_met, warnings_list

    def _chi_square_independence_test(self,
                                    contingency_table: pd.DataFrame,
                                    row_variable: str,
                                    column_variable: str,
                                    expected_frequencies: pd.DataFrame,
                                    assumptions_met: bool,
                                    warnings_list: List[str],
                                    correction: bool = True) -> ChiSquareResult:
        """Perform chi-square test of independence."""
        # Perform chi-square test
        chi2_stat, p_value, dof, expected_array = stats.chi2_contingency(
            contingency_table.values,
            correction=correction and contingency_table.shape == (2, 2)
        )

        # Calculate effect sizes
        n = contingency_table.sum().sum()

        # Cramer's V
        min_dim = min(contingency_table.shape) - 1
        cramers_v = np.sqrt(chi2_stat / (n * min_dim)) if min_dim > 0 else 0

        # For 2x2 tables, also calculate phi coefficient
        if contingency_table.shape == (2, 2):
            phi = np.sqrt(chi2_stat / n)
            effect_size = phi
            effect_size_measure = "phi"
        else:
            effect_size = cramers_v
            effect_size_measure = "cramers_v"

        # Calculate standardized residuals
        standardized_residuals = self._calculate_standardized_residuals(
            contingency_table, expected_frequencies
        )

        return ChiSquareResult(
            test_type=TestType.CHI_SQUARE_INDEPENDENCE,
            chi_square_statistic=float(chi2_stat),
            p_value=float(p_value),
            degrees_of_freedom=int(dof),
            effect_size=float(effect_size),
            effect_size_measure=effect_size_measure,
            contingency_table=contingency_table,
            expected_frequencies=expected_frequencies,
            standardized_residuals=standardized_residuals,
            assumptions_met=assumptions_met,
            minimum_expected_frequency=float(expected_frequencies.min().min()),
            warnings=warnings_list
        )

    def _fishers_exact_test(self,
                          contingency_table: pd.DataFrame,
                          row_variable: str,
                          column_variable: str,
                          expected_frequencies: pd.DataFrame,
                          assumptions_met: bool,
                          warnings_list: List[str]) -> ChiSquareResult:
        """Perform Fisher's exact test for 2x2 tables."""
        if contingency_table.shape != (2, 2):
            raise ValueError("Fisher's exact test requires 2x2 table")

        # Extract values from 2x2 table
        table_values = contingency_table.values

        # Perform Fisher's exact test
        odds_ratio, p_value = stats.fisher_exact(table_values)

        # Calculate phi coefficient as effect size
        n = contingency_table.sum().sum()

        # For Fisher's exact, we don't have a chi-square statistic
        # but we can calculate phi from the odds ratio or contingency table
        a, b, c, d = table_values.flatten()
        phi = (a * d - b * c) / np.sqrt((a + b) * (c + d) * (a + c) * (b + d)) if n > 0 else 0

        # Create standardized residuals (set to zero for exact test)
        standardized_residuals = contingency_table.copy() * 0

        return ChiSquareResult(
            test_type=TestType.FISHER_EXACT,
            chi_square_statistic=0.0,  # Not applicable for exact test
            p_value=float(p_value),
            degrees_of_freedom=1,
            effect_size=float(abs(phi)),
            effect_size_measure="phi",
            contingency_table=contingency_table,
            expected_frequencies=expected_frequencies,
            standardized_residuals=standardized_residuals,
            assumptions_met=True,  # Exact test doesn't rely on asymptotic assumptions
            minimum_expected_frequency=float(expected_frequencies.min().min()),
            warnings=warnings_list,
            exact_test_used=True
        )

    def _calculate_standardized_residuals(self,
                                        observed: pd.DataFrame,
                                        expected: pd.DataFrame) -> pd.DataFrame:
        """Calculate standardized residuals for contingency table."""
        # Standardized residuals: (observed - expected) / sqrt(expected)
        residuals = (observed - expected) / np.sqrt(expected)

        # Handle division by zero
        residuals = residuals.fillna(0)

        return residuals