"""
Cross-tabulation analysis for survey data.

This module provides comprehensive cross-tabulation functionality including
frequency tables, percentage calculations, statistical testing, and
association measures for categorical survey data.
"""

import logging
import warnings
from typing import Dict, List, Optional, Union, Tuple, Any
import pandas as pd
import numpy as np
from scipy import stats

from .chi_square_tests import ChiSquareTests
from .association_measures import AssociationMeasures
from ..data_processing.models import CrosstabResult, ChiSquareResult


class CrossTabulation:
    """
    Comprehensive cross-tabulation analysis for survey data.

    Features:
    - Basic frequency and percentage tables
    - Weighted cross-tabulations for survey data
    - Multiple percentage types (row, column, total)
    - Statistical testing (chi-square, Fisher's exact)
    - Association measures (Cramer's V, phi, lambda, etc.)
    - Multi-way cross-tabulations with control variables
    - Design effect calculations for complex surveys
    """

    def __init__(self,
                 alpha: float = 0.05,
                 include_margins: bool = True):
        """
        Initialize CrossTabulation analyzer.

        Parameters
        ----------
        alpha : float, default 0.05
            Significance level for statistical tests
        include_margins : bool, default True
            Whether to include marginal totals in tables
        """
        self.alpha = alpha
        self.include_margins = include_margins
        self.logger = logging.getLogger(__name__)

        # Initialize dependent analyzers
        self.chi_square_tests = ChiSquareTests(alpha=alpha)
        self.association_measures = AssociationMeasures()

    def crosstab(self,
                data: pd.DataFrame,
                row_variable: str,
                column_variable: str,
                control_variables: Optional[List[str]] = None,
                weights: Optional[pd.Series] = None,
                normalize: Optional[str] = None,
                percentages: List[str] = ['row', 'column', 'total'],
                statistical_tests: bool = True) -> Union[CrosstabResult, Dict[str, CrosstabResult]]:
        """
        Create comprehensive cross-tabulation with statistics.

        Parameters
        ----------
        data : pd.DataFrame
            Survey data
        row_variable : str
            Variable for table rows
        column_variable : str
            Variable for table columns
        control_variables : list of str, optional
            Variables to control for (creates separate tables)
        weights : pd.Series, optional
            Survey weights
        normalize : str, optional
            Normalization method: None, 'index', 'columns', 'all'
        percentages : list of str, default ['row', 'column', 'total']
            Types of percentages to calculate
        statistical_tests : bool, default True
            Whether to perform statistical tests

        Returns
        -------
        CrosstabResult or dict
            Cross-tabulation results. If control variables specified,
            returns dict with results for each control group.
        """
        if control_variables:
            return self._stratified_crosstab(
                data, row_variable, column_variable, control_variables,
                weights, normalize, percentages, statistical_tests
            )
        else:
            return self._simple_crosstab(
                data, row_variable, column_variable,
                weights, normalize, percentages, statistical_tests
            )

    def _simple_crosstab(self,
                        data: pd.DataFrame,
                        row_variable: str,
                        column_variable: str,
                        weights: Optional[pd.Series] = None,
                        normalize: Optional[str] = None,
                        percentages: List[str] = ['row', 'column', 'total'],
                        statistical_tests: bool = True) -> CrosstabResult:
        """Create simple two-way cross-tabulation."""
        # Clean data
        clean_data = data[[row_variable, column_variable]].dropna()

        if len(clean_data) == 0:
            self.logger.warning("No valid data for cross-tabulation")
            return CrosstabResult(
                row_variable=row_variable,
                column_variable=column_variable
            )

        # Get weights for clean data
        clean_weights = None
        if weights is not None:
            clean_weights = weights[clean_data.index]

        # Create frequency table
        if clean_weights is not None:
            # Weighted cross-tabulation
            observed_frequencies = pd.crosstab(
                clean_data[row_variable],
                clean_data[column_variable],
                values=clean_weights,
                aggfunc='sum',
                margins=self.include_margins,
                normalize=normalize
            )
        else:
            # Unweighted cross-tabulation
            observed_frequencies = pd.crosstab(
                clean_data[row_variable],
                clean_data[column_variable],
                margins=self.include_margins,
                normalize=normalize
            )

        # Remove margins for percentage calculations if they exist
        freq_table = observed_frequencies.copy()
        if self.include_margins and 'All' in freq_table.index:
            freq_table = freq_table.drop('All', axis=0)
        if self.include_margins and 'All' in freq_table.columns:
            freq_table = freq_table.drop('All', axis=1)

        # Calculate different types of percentages
        percentage_tables = {}
        for pct_type in percentages:
            percentage_tables[pct_type] = self._calculate_percentages(
                freq_table, pct_type, clean_weights
            )

        # Perform statistical tests
        chi_square_result = None
        if statistical_tests:
            try:
                chi_square_result = self.chi_square_tests.test_independence(
                    clean_data, row_variable, column_variable, clean_weights
                )
            except Exception as e:
                self.logger.warning(f"Chi-square test failed: {e}")

        # Calculate association measures
        association_measures = {}
        if statistical_tests and chi_square_result:
            try:
                association_measures = self.association_measures.calculate_measures(
                    freq_table, chi_square_result.chi_square_statistic
                )
            except Exception as e:
                self.logger.warning(f"Association measures calculation failed: {e}")

        # Calculate design effect and effective sample size for weighted data
        design_effect = None
        effective_sample_size = None
        if clean_weights is not None:
            design_effect = self._calculate_design_effect(clean_weights)
            effective_sample_size = self._calculate_effective_sample_size(clean_weights)

        return CrosstabResult(
            row_variable=row_variable,
            column_variable=column_variable,
            observed_frequencies=observed_frequencies,
            percentages=percentage_tables,
            chi_square_test=chi_square_result,
            association_measures=association_measures,
            weighted=clean_weights is not None,
            design_effect=design_effect,
            effective_sample_size=effective_sample_size
        )

    def _stratified_crosstab(self,
                           data: pd.DataFrame,
                           row_variable: str,
                           column_variable: str,
                           control_variables: List[str],
                           weights: Optional[pd.Series] = None,
                           normalize: Optional[str] = None,
                           percentages: List[str] = ['row', 'column', 'total'],
                           statistical_tests: bool = True) -> Dict[str, CrosstabResult]:
        """Create stratified cross-tabulation by control variables."""
        results = {}

        # Create all combinations of control variable values
        control_data = data[control_variables].dropna()
        unique_combinations = control_data.drop_duplicates()

        for idx, combination in unique_combinations.iterrows():
            # Create filter for this combination
            filter_mask = pd.Series(True, index=data.index)
            stratum_label_parts = []

            for control_var in control_variables:
                filter_mask &= (data[control_var] == combination[control_var])
                stratum_label_parts.append(f"{control_var}={combination[control_var]}")

            stratum_label = "; ".join(stratum_label_parts)

            # Get subset of data for this stratum
            stratum_data = data[filter_mask]
            stratum_weights = weights[filter_mask] if weights is not None else None

            if len(stratum_data) < 4:  # Need minimum data for analysis
                self.logger.warning(f"Insufficient data for stratum: {stratum_label}")
                continue

            # Perform cross-tabulation for this stratum
            try:
                stratum_result = self._simple_crosstab(
                    stratum_data, row_variable, column_variable,
                    stratum_weights, normalize, percentages, statistical_tests
                )
                stratum_result.control_variables = [stratum_label]
                results[stratum_label] = stratum_result

            except Exception as e:
                self.logger.warning(f"Cross-tabulation failed for stratum {stratum_label}: {e}")

        return results

    def _calculate_percentages(self,
                             freq_table: pd.DataFrame,
                             percentage_type: str,
                             weights: Optional[pd.Series] = None) -> pd.DataFrame:
        """Calculate different types of percentages."""
        if percentage_type == 'row':
            # Row percentages (percentages within each row)
            return freq_table.div(freq_table.sum(axis=1), axis=0) * 100

        elif percentage_type == 'column':
            # Column percentages (percentages within each column)
            return freq_table.div(freq_table.sum(axis=0), axis=1) * 100

        elif percentage_type == 'total':
            # Total percentages (percentages of grand total)
            return (freq_table / freq_table.sum().sum()) * 100

        else:
            raise ValueError(f"Unknown percentage type: {percentage_type}")

    def _calculate_design_effect(self, weights: pd.Series) -> float:
        """
        Calculate design effect for weighted survey data.

        Design effect = (sum of weights)^2 / (n * sum of squared weights)
        where n is the number of observations.
        """
        try:
            n = len(weights)
            sum_weights = weights.sum()
            sum_squared_weights = (weights ** 2).sum()

            if sum_squared_weights == 0:
                return 1.0

            design_effect = (sum_weights ** 2) / (n * sum_squared_weights)
            return float(design_effect)

        except Exception:
            return 1.0

    def _calculate_effective_sample_size(self, weights: pd.Series) -> int:
        """
        Calculate effective sample size for weighted survey data.

        Effective n = (sum of weights)^2 / sum of squared weights
        """
        try:
            sum_weights = weights.sum()
            sum_squared_weights = (weights ** 2).sum()

            if sum_squared_weights == 0:
                return len(weights)

            effective_n = (sum_weights ** 2) / sum_squared_weights
            return int(round(effective_n))

        except Exception:
            return len(weights)

    def multi_way_crosstab(self,
                          data: pd.DataFrame,
                          variables: List[str],
                          weights: Optional[pd.Series] = None,
                          max_combinations: int = 1000) -> Dict[str, pd.DataFrame]:
        """
        Create multi-way cross-tabulation for multiple variables.

        Parameters
        ----------
        data : pd.DataFrame
            Survey data
        variables : list of str
            List of variables to cross-tabulate
        weights : pd.Series, optional
            Survey weights
        max_combinations : int, default 1000
            Maximum number of combinations to prevent memory issues

        Returns
        -------
        dict
            Dictionary of cross-tabulation tables for different variable combinations
        """
        if len(variables) < 2:
            raise ValueError("Need at least 2 variables for cross-tabulation")

        results = {}

        # Clean data
        clean_data = data[variables].dropna()
        clean_weights = weights[clean_data.index] if weights is not None else None

        # Check total number of unique combinations
        total_combinations = 1
        for var in variables:
            total_combinations *= clean_data[var].nunique()

        if total_combinations > max_combinations:
            self.logger.warning(
                f"Too many combinations ({total_combinations}). "
                f"Consider reducing variables or increasing max_combinations."
            )
            return {}

        # Create multi-way table
        try:
            if clean_weights is not None:
                # Weighted multi-way table
                multi_table = clean_data.assign(weights=clean_weights).groupby(
                    variables
                )['weights'].sum().unstack(fill_value=0)
            else:
                # Unweighted multi-way table
                multi_table = pd.crosstab(
                    [clean_data[var] for var in variables[:-1]],
                    clean_data[variables[-1]],
                    margins=self.include_margins
                )

            results['full_table'] = multi_table

            # Create pairwise tables
            from itertools import combinations
            for var1, var2 in combinations(variables, 2):
                table_name = f"{var1}_x_{var2}"

                if clean_weights is not None:
                    pairwise_table = pd.crosstab(
                        clean_data[var1],
                        clean_data[var2],
                        values=clean_weights,
                        aggfunc='sum',
                        margins=self.include_margins
                    )
                else:
                    pairwise_table = pd.crosstab(
                        clean_data[var1],
                        clean_data[var2],
                        margins=self.include_margins
                    )

                results[table_name] = pairwise_table

        except Exception as e:
            self.logger.error(f"Multi-way cross-tabulation failed: {e}")

        return results

    def mantel_haenszel_test(self,
                           data: pd.DataFrame,
                           row_variable: str,
                           column_variable: str,
                           stratum_variable: str,
                           weights: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Perform Mantel-Haenszel test for association across strata.

        Tests whether the association between row and column variables
        is consistent across levels of the stratum variable.

        Parameters
        ----------
        data : pd.DataFrame
            Survey data
        row_variable : str
            Row variable (should be binary)
        column_variable : str
            Column variable (should be binary)
        stratum_variable : str
            Stratification variable
        weights : pd.Series, optional
            Survey weights

        Returns
        -------
        dict
            Test results including statistic, p-value, and common odds ratio
        """
        try:
            # Clean data
            clean_data = data[[row_variable, column_variable, stratum_variable]].dropna()
            clean_weights = weights[clean_data.index] if weights is not None else None

            # Initialize accumulators for Mantel-Haenszel test
            mh_numerator = 0
            mh_denominator = 0
            variance_sum = 0

            # Common odds ratio calculation
            or_numerator = 0
            or_denominator = 0

            strata = clean_data[stratum_variable].unique()

            for stratum in strata:
                stratum_data = clean_data[clean_data[stratum_variable] == stratum]

                if len(stratum_data) < 4:  # Need at least 4 observations
                    continue

                # Create 2x2 table for this stratum
                if clean_weights is not None:
                    stratum_weights = clean_weights[stratum_data.index]
                    table = pd.crosstab(
                        stratum_data[row_variable],
                        stratum_data[column_variable],
                        values=stratum_weights,
                        aggfunc='sum'
                    )
                else:
                    table = pd.crosstab(
                        stratum_data[row_variable],
                        stratum_data[column_variable]
                    )

                if table.shape != (2, 2):
                    continue

                # Extract 2x2 table values
                a, b, c, d = table.values.flatten()
                n = a + b + c + d

                if n == 0:
                    continue

                # Mantel-Haenszel calculations
                expected_a = (a + b) * (a + c) / n
                mh_numerator += a - expected_a

                variance_a = ((a + b) * (c + d) * (a + c) * (b + d)) / (n**2 * (n - 1))
                variance_sum += variance_a

                # Common odds ratio calculation
                or_numerator += a * d / n
                or_denominator += b * c / n

            # Calculate test statistic
            if variance_sum == 0:
                mh_statistic = 0
                p_value = 1.0
            else:
                mh_statistic = (abs(mh_numerator) - 0.5)**2 / variance_sum
                p_value = 1 - stats.chi2.cdf(mh_statistic, 1)

            # Calculate common odds ratio
            common_or = or_numerator / or_denominator if or_denominator > 0 else np.inf

            return {
                'test_statistic': float(mh_statistic),
                'p_value': float(p_value),
                'degrees_of_freedom': 1,
                'common_odds_ratio': float(common_or),
                'n_strata': len(strata)
            }

        except Exception as e:
            self.logger.error(f"Mantel-Haenszel test failed: {e}")
            return {
                'test_statistic': 0.0,
                'p_value': 1.0,
                'degrees_of_freedom': 1,
                'common_odds_ratio': 1.0,
                'n_strata': 0
            }

    def create_summary_table(self,
                           crosstab_results: Union[CrosstabResult, Dict[str, CrosstabResult]],
                           include_statistics: bool = True) -> pd.DataFrame:
        """
        Create summary table of cross-tabulation results.

        Parameters
        ----------
        crosstab_results : CrosstabResult or dict
            Cross-tabulation results to summarize
        include_statistics : bool, default True
            Whether to include statistical test results

        Returns
        -------
        pd.DataFrame
            Summary table with key statistics
        """
        if isinstance(crosstab_results, CrosstabResult):
            # Single result
            return self._create_single_summary(crosstab_results, include_statistics)
        else:
            # Multiple results (stratified)
            return self._create_stratified_summary(crosstab_results, include_statistics)

    def _create_single_summary(self,
                             result: CrosstabResult,
                             include_statistics: bool) -> pd.DataFrame:
        """Create summary for single cross-tabulation result."""
        summary_data = {
            'Row Variable': [result.row_variable],
            'Column Variable': [result.column_variable],
            'Sample Size': [result.observed_frequencies.sum().sum() if not result.observed_frequencies.empty else 0]
        }

        if result.weighted:
            summary_data['Weighted'] = ['Yes']
            if result.design_effect:
                summary_data['Design Effect'] = [f"{result.design_effect:.3f}"]
            if result.effective_sample_size:
                summary_data['Effective N'] = [result.effective_sample_size]

        if include_statistics and result.chi_square_test:
            chi_sq = result.chi_square_test
            summary_data.update({
                'Chi-square': [f"{chi_sq.chi_square_statistic:.3f}"],
                'df': [chi_sq.degrees_of_freedom],
                'p-value': [f"{chi_sq.p_value:.4f}"],
                'Effect Size': [f"{chi_sq.effect_size:.3f}"],
                'Effect Measure': [chi_sq.effect_size_measure]
            })

        return pd.DataFrame(summary_data)

    def _create_stratified_summary(self,
                                 results: Dict[str, CrosstabResult],
                                 include_statistics: bool) -> pd.DataFrame:
        """Create summary for stratified cross-tabulation results."""
        summary_rows = []

        for stratum, result in results.items():
            row_data = {
                'Stratum': stratum,
                'Row Variable': result.row_variable,
                'Column Variable': result.column_variable,
                'Sample Size': result.observed_frequencies.sum().sum() if not result.observed_frequencies.empty else 0
            }

            if include_statistics and result.chi_square_test:
                chi_sq = result.chi_square_test
                row_data.update({
                    'Chi-square': f"{chi_sq.chi_square_statistic:.3f}",
                    'p-value': f"{chi_sq.p_value:.4f}",
                    'Effect Size': f"{chi_sq.effect_size:.3f}"
                })

            summary_rows.append(row_data)

        return pd.DataFrame(summary_rows)