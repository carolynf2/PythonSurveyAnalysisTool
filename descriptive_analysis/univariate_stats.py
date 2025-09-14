"""
Univariate descriptive statistics for survey data.

This module provides comprehensive univariate statistical analysis including
measures of central tendency, variability, distribution shape, and confidence
intervals, with special handling for survey data and different variable types.
"""

import logging
import warnings
from typing import Dict, List, Optional, Union, Tuple, Any
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import bootstrap
import math

from ..data_processing.models import DescriptiveStats, VariableType, VariableDefinition


class UnivariateStats:
    """
    Comprehensive univariate descriptive statistics for survey data.

    Features:
    - Central tendency measures (mean, median, mode) with confidence intervals
    - Variability measures (std, variance, IQR, range)
    - Distribution shape (skewness, kurtosis, normality tests)
    - Frequency tables and percentile calculations
    - Survey-weighted statistics
    - Variable type-specific analyses
    """

    def __init__(self,
                 confidence_level: float = 0.95,
                 bootstrap_samples: int = 1000,
                 random_state: int = 42):
        """
        Initialize UnivariateStats calculator.

        Parameters
        ----------
        confidence_level : float, default 0.95
            Confidence level for interval estimation
        bootstrap_samples : int, default 1000
            Number of bootstrap samples for confidence intervals
        random_state : int, default 42
            Random state for reproducible bootstrap sampling
        """
        self.confidence_level = confidence_level
        self.bootstrap_samples = bootstrap_samples
        self.random_state = random_state
        self.logger = logging.getLogger(__name__)

        # Calculate alpha for confidence intervals
        self.alpha = 1 - confidence_level

    def calculate_descriptive_stats(self,
                                  data: pd.DataFrame,
                                  variables: Optional[List[str]] = None,
                                  weights: Optional[pd.Series] = None,
                                  group_by: Optional[str] = None,
                                  variable_definitions: Optional[Dict[str, VariableDefinition]] = None) -> Dict[str, DescriptiveStats]:
        """
        Calculate comprehensive descriptive statistics.

        Parameters
        ----------
        data : pd.DataFrame
            Survey data
        variables : list of str, optional
            Variables to analyze. If None, analyzes all numeric variables
        weights : pd.Series, optional
            Survey weights for weighted statistics
        group_by : str, optional
            Variable to group analysis by
        variable_definitions : dict, optional
            Variable definitions for type-specific analysis

        Returns
        -------
        dict
            Dictionary mapping variable names to DescriptiveStats objects
        """
        if variables is None:
            # Select numeric and categorical variables
            numeric_vars = data.select_dtypes(include=[np.number]).columns.tolist()
            categorical_vars = data.select_dtypes(include=['object', 'category']).columns.tolist()
            variables = numeric_vars + categorical_vars

        results = {}

        for var in variables:
            if var not in data.columns:
                self.logger.warning(f"Variable {var} not found in data")
                continue

            self.logger.debug(f"Calculating descriptive statistics for {var}")

            # Get variable definition if available
            var_def = variable_definitions.get(var) if variable_definitions else None

            # Calculate statistics based on variable type
            if group_by and group_by in data.columns:
                # Group-wise analysis
                grouped_stats = {}
                for group_value in data[group_by].unique():
                    if pd.notna(group_value):
                        group_data = data[data[group_by] == group_value][var]
                        group_weights = weights[data[group_by] == group_value] if weights is not None else None

                        grouped_stats[group_value] = self._calculate_single_variable_stats(
                            group_data, var, var_def, group_weights
                        )

                results[var] = grouped_stats
            else:
                # Overall analysis
                results[var] = self._calculate_single_variable_stats(
                    data[var], var, var_def, weights
                )

        return results

    def _calculate_single_variable_stats(self,
                                       series: pd.Series,
                                       var_name: str,
                                       var_def: Optional[VariableDefinition] = None,
                                       weights: Optional[pd.Series] = None) -> DescriptiveStats:
        """Calculate descriptive statistics for a single variable."""
        # Remove missing values
        valid_data = series.dropna()
        if weights is not None:
            valid_weights = weights[valid_data.index]
        else:
            valid_weights = None

        if len(valid_data) == 0:
            return DescriptiveStats(
                variable=var_name,
                count=0
            )

        # Basic counts
        count = len(valid_data)

        # Determine if variable is numeric or categorical
        is_numeric = pd.api.types.is_numeric_dtype(valid_data)

        # Initialize result object
        stats_result = DescriptiveStats(
            variable=var_name,
            count=count
        )

        if is_numeric:
            stats_result = self._calculate_numeric_stats(
                valid_data, var_name, valid_weights, stats_result
            )
        else:
            stats_result = self._calculate_categorical_stats(
                valid_data, var_name, valid_weights, stats_result
            )

        # Generate frequency table
        stats_result.frequency_table = self._create_frequency_table(
            valid_data, valid_weights, is_numeric
        )

        # Perform normality tests for numeric variables
        if is_numeric and count >= 8:  # Need at least 8 observations
            stats_result.normality_tests = self._perform_normality_tests(valid_data)

        return stats_result

    def _calculate_numeric_stats(self,
                               data: pd.Series,
                               var_name: str,
                               weights: Optional[pd.Series],
                               stats_result: DescriptiveStats) -> DescriptiveStats:
        """Calculate statistics for numeric variables."""
        try:
            # Central tendency
            if weights is not None:
                stats_result.mean = self._weighted_mean(data, weights)
                stats_result.median = self._weighted_percentile(data, weights, 50)
            else:
                stats_result.mean = float(data.mean())
                stats_result.median = float(data.median())

            # Mode (most frequent value)
            mode_result = data.mode()
            if not mode_result.empty:
                stats_result.mode = float(mode_result.iloc[0])

            # Variability measures
            if weights is not None:
                stats_result.variance = self._weighted_variance(data, weights)
                stats_result.std = math.sqrt(stats_result.variance)
            else:
                stats_result.variance = float(data.var())
                stats_result.std = float(data.std())

            # Distribution shape
            if len(data) >= 3:  # Need at least 3 observations
                stats_result.skewness = float(stats.skew(data))
                if len(data) >= 4:  # Need at least 4 observations
                    stats_result.kurtosis = float(stats.kurtosis(data))

            # Range and extremes
            stats_result.minimum = float(data.min())
            stats_result.maximum = float(data.max())
            stats_result.range_ = stats_result.maximum - stats_result.minimum

            # Quartiles and IQR
            if weights is not None:
                q1 = self._weighted_percentile(data, weights, 25)
                q3 = self._weighted_percentile(data, weights, 75)
            else:
                q1 = float(data.quantile(0.25))
                q3 = float(data.quantile(0.75))

            stats_result.iqr = q3 - q1

            # Percentiles
            percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
            stats_result.percentiles = {}

            for p in percentiles:
                if weights is not None:
                    stats_result.percentiles[p] = self._weighted_percentile(data, weights, p)
                else:
                    stats_result.percentiles[p] = float(data.quantile(p / 100))

            # Confidence intervals
            stats_result.confidence_intervals = self._calculate_confidence_intervals(
                data, weights
            )

        except Exception as e:
            self.logger.warning(f"Error calculating numeric statistics for {var_name}: {e}")

        return stats_result

    def _calculate_categorical_stats(self,
                                   data: pd.Series,
                                   var_name: str,
                                   weights: Optional[pd.Series],
                                   stats_result: DescriptiveStats) -> DescriptiveStats:
        """Calculate statistics for categorical variables."""
        try:
            # Mode (most frequent category)
            if weights is not None:
                # Weighted mode
                value_counts = self._weighted_value_counts(data, weights)
                if not value_counts.empty:
                    stats_result.mode = value_counts.index[0]
            else:
                mode_result = data.mode()
                if not mode_result.empty:
                    stats_result.mode = mode_result.iloc[0]

            # For ordinal variables, we can calculate some additional statistics
            if data.dtype.name == 'category' and data.cat.ordered:
                # Convert to numeric for calculation, then convert back
                numeric_data = pd.Series(data.cat.codes, index=data.index)

                if weights is not None:
                    median_code = self._weighted_percentile(numeric_data, weights, 50)
                    # Convert back to category
                    if 0 <= median_code < len(data.cat.categories):
                        stats_result.median = data.cat.categories[int(round(median_code))]
                else:
                    median_code = numeric_data.median()
                    if 0 <= median_code < len(data.cat.categories):
                        stats_result.median = data.cat.categories[int(round(median_code))]

        except Exception as e:
            self.logger.warning(f"Error calculating categorical statistics for {var_name}: {e}")

        return stats_result

    def _create_frequency_table(self,
                              data: pd.Series,
                              weights: Optional[pd.Series],
                              is_numeric: bool) -> pd.DataFrame:
        """Create frequency table for variable."""
        try:
            if weights is not None:
                freq_counts = self._weighted_value_counts(data, weights)
                freq_table = pd.DataFrame({
                    'Value': freq_counts.index,
                    'Frequency': freq_counts.values,
                    'Percentage': (freq_counts.values / freq_counts.sum()) * 100
                })
            else:
                value_counts = data.value_counts()
                freq_table = pd.DataFrame({
                    'Value': value_counts.index,
                    'Frequency': value_counts.values,
                    'Percentage': (value_counts.values / len(data)) * 100
                })

            # Add cumulative percentages
            freq_table['Cumulative_Percentage'] = freq_table['Percentage'].cumsum()

            # For numeric variables with many unique values, bin the data
            if is_numeric and len(freq_table) > 20:
                freq_table = self._create_binned_frequency_table(data, weights)

            return freq_table.reset_index(drop=True)

        except Exception as e:
            self.logger.warning(f"Error creating frequency table: {e}")
            return pd.DataFrame()

    def _create_binned_frequency_table(self,
                                     data: pd.Series,
                                     weights: Optional[pd.Series],
                                     n_bins: int = 10) -> pd.DataFrame:
        """Create binned frequency table for numeric variables with many values."""
        try:
            # Create bins
            bins = pd.cut(data, bins=n_bins, include_lowest=True)

            if weights is not None:
                # Weighted binned frequencies
                binned_data = pd.DataFrame({'bins': bins, 'weights': weights})
                freq_counts = binned_data.groupby('bins')['weights'].sum()
            else:
                freq_counts = bins.value_counts().sort_index()

            # Create frequency table
            freq_table = pd.DataFrame({
                'Bin': freq_counts.index.astype(str),
                'Frequency': freq_counts.values,
                'Percentage': (freq_counts.values / freq_counts.sum()) * 100
            })

            freq_table['Cumulative_Percentage'] = freq_table['Percentage'].cumsum()

            return freq_table

        except Exception as e:
            self.logger.warning(f"Error creating binned frequency table: {e}")
            return pd.DataFrame()

    def _perform_normality_tests(self, data: pd.Series) -> Dict[str, Dict[str, float]]:
        """Perform normality tests on numeric data."""
        normality_tests = {}

        try:
            # Shapiro-Wilk test (good for small samples)
            if len(data) <= 5000:  # Shapiro-Wilk works best for smaller samples
                stat, p_value = stats.shapiro(data)
                normality_tests['shapiro_wilk'] = {
                    'statistic': float(stat),
                    'p_value': float(p_value)
                }

            # Anderson-Darling test
            result = stats.anderson(data, dist='norm')
            normality_tests['anderson_darling'] = {
                'statistic': float(result.statistic),
                'critical_values': result.critical_values.tolist(),
                'significance_levels': result.significance_levels.tolist()
            }

            # Kolmogorov-Smirnov test
            # First standardize the data
            standardized = (data - data.mean()) / data.std()
            stat, p_value = stats.kstest(standardized, 'norm')
            normality_tests['kolmogorov_smirnov'] = {
                'statistic': float(stat),
                'p_value': float(p_value)
            }

            # Jarque-Bera test (good for larger samples)
            if len(data) >= 8:
                stat, p_value = stats.jarque_bera(data)
                normality_tests['jarque_bera'] = {
                    'statistic': float(stat),
                    'p_value': float(p_value)
                }

        except Exception as e:
            self.logger.warning(f"Error performing normality tests: {e}")

        return normality_tests

    def _calculate_confidence_intervals(self,
                                      data: pd.Series,
                                      weights: Optional[pd.Series]) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals for various statistics."""
        confidence_intervals = {}

        try:
            # Mean confidence interval
            if weights is not None:
                # Weighted confidence interval (more complex)
                mean_ci = self._weighted_mean_confidence_interval(data, weights)
            else:
                # Standard confidence interval for mean
                mean_ci = self._bootstrap_confidence_interval(data, np.mean)

            confidence_intervals['mean'] = mean_ci

            # Median confidence interval (using bootstrap)
            if weights is not None:
                median_ci = self._bootstrap_confidence_interval_weighted(
                    data, weights, lambda d, w: self._weighted_percentile(d, w, 50)
                )
            else:
                median_ci = self._bootstrap_confidence_interval(data, np.median)

            confidence_intervals['median'] = median_ci

            # Standard deviation confidence interval
            if not weights:  # Only for unweighted data
                std_ci = self._bootstrap_confidence_interval(data, np.std)
                confidence_intervals['std'] = std_ci

        except Exception as e:
            self.logger.warning(f"Error calculating confidence intervals: {e}")

        return confidence_intervals

    def _bootstrap_confidence_interval(self, data: pd.Series, statistic_func) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval for a statistic."""
        try:
            # Use scipy's bootstrap function
            rng = np.random.default_rng(self.random_state)

            # Convert to numpy array
            data_array = data.values

            # Define the statistic function for bootstrap
            def stat_func(x):
                return statistic_func(x)

            # Perform bootstrap
            res = bootstrap(
                (data_array,),
                stat_func,
                n_resamples=self.bootstrap_samples,
                confidence_level=self.confidence_level,
                random_state=rng
            )

            return (float(res.confidence_interval.low), float(res.confidence_interval.high))

        except Exception as e:
            self.logger.warning(f"Bootstrap confidence interval failed: {e}")
            # Fallback to simple percentile method
            return self._simple_bootstrap_ci(data, statistic_func)

    def _simple_bootstrap_ci(self, data: pd.Series, statistic_func) -> Tuple[float, float]:
        """Simple bootstrap implementation as fallback."""
        np.random.seed(self.random_state)

        bootstrap_stats = []
        n = len(data)

        for _ in range(self.bootstrap_samples):
            # Sample with replacement
            sample_indices = np.random.choice(n, size=n, replace=True)
            sample = data.iloc[sample_indices]
            bootstrap_stats.append(statistic_func(sample))

        # Calculate percentiles
        lower_percentile = (self.alpha / 2) * 100
        upper_percentile = (1 - self.alpha / 2) * 100

        lower = np.percentile(bootstrap_stats, lower_percentile)
        upper = np.percentile(bootstrap_stats, upper_percentile)

        return (float(lower), float(upper))

    def _bootstrap_confidence_interval_weighted(self,
                                              data: pd.Series,
                                              weights: pd.Series,
                                              statistic_func) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval for weighted statistics."""
        np.random.seed(self.random_state)

        bootstrap_stats = []
        n = len(data)

        for _ in range(self.bootstrap_samples):
            # Sample with replacement (maintaining weights)
            sample_indices = np.random.choice(n, size=n, replace=True)
            sample_data = data.iloc[sample_indices].reset_index(drop=True)
            sample_weights = weights.iloc[sample_indices].reset_index(drop=True)

            stat_value = statistic_func(sample_data, sample_weights)
            bootstrap_stats.append(stat_value)

        # Calculate percentiles
        lower_percentile = (self.alpha / 2) * 100
        upper_percentile = (1 - self.alpha / 2) * 100

        lower = np.percentile(bootstrap_stats, lower_percentile)
        upper = np.percentile(bootstrap_stats, upper_percentile)

        return (float(lower), float(upper))

    def _weighted_mean_confidence_interval(self,
                                         data: pd.Series,
                                         weights: pd.Series) -> Tuple[float, float]:
        """Calculate confidence interval for weighted mean."""
        try:
            # Calculate weighted mean and variance
            weighted_mean = self._weighted_mean(data, weights)
            weighted_var = self._weighted_variance(data, weights)

            # Effective sample size
            effective_n = (weights.sum() ** 2) / (weights ** 2).sum()

            # Standard error
            se = math.sqrt(weighted_var / effective_n)

            # t-distribution critical value
            df = effective_n - 1
            t_critical = stats.t.ppf(1 - self.alpha / 2, df)

            # Confidence interval
            margin_error = t_critical * se
            lower = weighted_mean - margin_error
            upper = weighted_mean + margin_error

            return (float(lower), float(upper))

        except Exception as e:
            self.logger.warning(f"Weighted mean CI calculation failed: {e}")
            # Fallback to bootstrap
            return self._bootstrap_confidence_interval_weighted(
                data, weights, self._weighted_mean
            )

    def _weighted_mean(self, data: pd.Series, weights: pd.Series) -> float:
        """Calculate weighted mean."""
        return float((data * weights).sum() / weights.sum())

    def _weighted_variance(self, data: pd.Series, weights: pd.Series) -> float:
        """Calculate weighted variance."""
        weighted_mean = self._weighted_mean(data, weights)
        weighted_var = ((weights * (data - weighted_mean) ** 2).sum() /
                       weights.sum())
        return float(weighted_var)

    def _weighted_percentile(self, data: pd.Series, weights: pd.Series, percentile: float) -> float:
        """Calculate weighted percentile."""
        # Sort data and weights by data values
        sorted_indices = data.argsort()
        sorted_data = data.iloc[sorted_indices]
        sorted_weights = weights.iloc[sorted_indices]

        # Calculate cumulative weights
        cumulative_weights = sorted_weights.cumsum()
        total_weight = weights.sum()

        # Find the percentile position
        percentile_weight = (percentile / 100) * total_weight

        # Find the value at this position
        idx = np.searchsorted(cumulative_weights, percentile_weight)

        if idx >= len(sorted_data):
            return float(sorted_data.iloc[-1])
        elif idx == 0:
            return float(sorted_data.iloc[0])
        else:
            # Linear interpolation
            w1 = cumulative_weights.iloc[idx - 1]
            w2 = cumulative_weights.iloc[idx]
            v1 = sorted_data.iloc[idx - 1]
            v2 = sorted_data.iloc[idx]

            # Interpolation factor
            factor = (percentile_weight - w1) / (w2 - w1)
            value = v1 + factor * (v2 - v1)

            return float(value)

    def _weighted_value_counts(self, data: pd.Series, weights: pd.Series) -> pd.Series:
        """Calculate weighted value counts."""
        # Create DataFrame with data and weights
        df = pd.DataFrame({'value': data, 'weight': weights})

        # Group by value and sum weights
        weighted_counts = df.groupby('value')['weight'].sum()

        # Sort by weight (descending)
        weighted_counts = weighted_counts.sort_values(ascending=False)

        return weighted_counts