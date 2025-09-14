"""
Automated data cleaning and validation for survey data.

This module provides comprehensive data cleaning capabilities including
response validation, duplicate detection, straight-lining detection,
speeding detection, and open-end text cleaning.
"""

import re
import logging
import warnings
from typing import Dict, List, Optional, Union, Tuple, Any, Set
import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime, timedelta

from .models import SurveyMetadata, VariableDefinition, VariableType, QualityMetrics


class DataCleaner:
    """
    Comprehensive data cleaning and validation for survey responses.

    Features:
    - Response validation with range and logic checks
    - Duplicate response detection and handling
    - Straight-lining (satisficing) behavior detection
    - Response time analysis and speeding detection
    - Open-end text response cleaning and standardization
    - Data type validation and conversion
    - Outlier detection using multiple methods
    """

    def __init__(self,
                 metadata: Optional[SurveyMetadata] = None,
                 strict_validation: bool = False,
                 auto_clean: bool = True):
        """
        Initialize the DataCleaner.

        Parameters
        ----------
        metadata : SurveyMetadata, optional
            Survey metadata for validation rules
        strict_validation : bool, default False
            Whether to use strict validation (raise errors vs warnings)
        auto_clean : bool, default True
            Whether to automatically apply common cleaning operations
        """
        self.metadata = metadata
        self.strict_validation = strict_validation
        self.auto_clean = auto_clean
        self.logger = logging.getLogger(__name__)

        # Cleaning statistics
        self.cleaning_log = []
        self.quality_flags = {}

    def clean_data(self,
                   data: pd.DataFrame,
                   response_time_col: Optional[str] = None,
                   respondent_id_col: str = 'respondent_id',
                   cleaning_rules: Optional[Dict] = None) -> Tuple[pd.DataFrame, QualityMetrics]:
        """
        Perform comprehensive data cleaning.

        Parameters
        ----------
        data : pd.DataFrame
            Raw survey data
        response_time_col : str, optional
            Column name containing response times (in seconds)
        respondent_id_col : str, default 'respondent_id'
            Column name containing unique respondent identifiers
        cleaning_rules : dict, optional
            Custom cleaning rules to override defaults

        Returns
        -------
        tuple
            (Cleaned DataFrame, QualityMetrics object)
        """
        self.logger.info(f"Starting data cleaning for {len(data)} responses")

        # Initialize quality metrics
        quality_metrics = QualityMetrics(
            total_responses=len(data),
            complete_responses=0,
            partial_responses=0,
            completion_rate=0.0
        )

        # Make a copy to avoid modifying original data
        cleaned_data = data.copy()

        # 1. Basic data type validation and conversion
        if self.metadata:
            cleaned_data = self._validate_data_types(cleaned_data)

        # 2. Duplicate detection and removal
        cleaned_data, duplicate_count = self._remove_duplicates(
            cleaned_data, respondent_id_col
        )
        quality_metrics.duplicate_responses = duplicate_count

        # 3. Response validation (range checks, logic checks)
        if self.metadata:
            cleaned_data = self._validate_responses(cleaned_data)

        # 4. Straight-lining detection
        straightlining_flags = self._detect_straightlining(cleaned_data)
        quality_metrics.straight_lining_count = sum(straightlining_flags.values())
        quality_metrics.straight_lining_percentage = (
            quality_metrics.straight_lining_count / len(cleaned_data) * 100
        )

        # 5. Response time analysis and speeding detection
        if response_time_col and response_time_col in cleaned_data.columns:
            speeding_flags = self._detect_speeding(
                cleaned_data, response_time_col
            )
            quality_metrics.speeding_count = sum(speeding_flags)
            quality_metrics.speeding_percentage = (
                quality_metrics.speeding_count / len(cleaned_data) * 100
            )
            quality_metrics.median_completion_time = cleaned_data[response_time_col].median()

        # 6. Text cleaning for open-ended responses
        text_columns = self._identify_text_columns(cleaned_data)
        for col in text_columns:
            cleaned_data[col] = self._clean_text_responses(cleaned_data[col])

        # 7. Outlier detection for numeric variables
        outlier_flags = self._detect_outliers(cleaned_data)

        # 8. Calculate completion rates
        completion_info = self._calculate_completion_rates(cleaned_data)
        quality_metrics.complete_responses = completion_info['complete']
        quality_metrics.partial_responses = completion_info['partial']
        quality_metrics.completion_rate = completion_info['rate']

        # 9. Item non-response analysis
        quality_metrics.item_nonresponse_rates = self._calculate_item_nonresponse(cleaned_data)

        # 10. Compile quality flags
        quality_metrics.quality_flags = {
            'straightlining': [i for i, flag in straightlining_flags.items() if flag],
            'speeding': [i for i, flag in enumerate(speeding_flags) if flag] if response_time_col else [],
            'outliers': outlier_flags,
            'duplicates': [f"Removed {duplicate_count} duplicate responses"]
        }

        # Apply automatic cleaning if enabled
        if self.auto_clean:
            cleaned_data = self._apply_auto_cleaning(
                cleaned_data, quality_metrics, cleaning_rules
            )

        self.logger.info(f"Data cleaning completed. {len(cleaned_data)} responses retained.")

        return cleaned_data, quality_metrics

    def _validate_data_types(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate and convert data types based on metadata."""
        for col_name, var_def in self.metadata.variables.items():
            if col_name not in data.columns:
                continue

            try:
                if var_def.type in [VariableType.INTERVAL, VariableType.RATIO]:
                    # Convert to numeric, coercing errors to NaN
                    data[col_name] = pd.to_numeric(data[col_name], errors='coerce')

                elif var_def.type == VariableType.BINARY:
                    # Ensure binary variables are 0/1 or boolean
                    unique_vals = data[col_name].dropna().unique()
                    if len(unique_vals) > 2:
                        self._log_warning(f"Binary variable {col_name} has more than 2 unique values")

                elif var_def.type in [VariableType.NOMINAL, VariableType.ORDINAL]:
                    # Convert to categorical if categories are defined
                    if var_def.categories:
                        # Map numeric codes to labels
                        data[col_name] = data[col_name].map(var_def.categories)
                        if var_def.type == VariableType.ORDINAL:
                            ordered_categories = [var_def.categories[k]
                                                for k in sorted(var_def.categories.keys())]
                            data[col_name] = pd.Categorical(
                                data[col_name],
                                categories=ordered_categories,
                                ordered=True
                            )
                        else:
                            data[col_name] = data[col_name].astype('category')

            except Exception as e:
                self._log_warning(f"Could not convert {col_name} to expected type: {e}")

        return data

    def _remove_duplicates(self, data: pd.DataFrame, id_col: str) -> Tuple[pd.DataFrame, int]:
        """Remove duplicate responses based on respondent ID."""
        if id_col not in data.columns:
            self._log_warning(f"Respondent ID column '{id_col}' not found")
            return data, 0

        initial_count = len(data)

        # Check for exact duplicates first
        exact_duplicates = data.duplicated()
        if exact_duplicates.any():
            data = data[~exact_duplicates]
            self._log_info(f"Removed {exact_duplicates.sum()} exact duplicate rows")

        # Check for duplicate IDs (keeping first occurrence)
        id_duplicates = data.duplicated(subset=[id_col])
        if id_duplicates.any():
            data = data[~id_duplicates]
            self._log_info(f"Removed {id_duplicates.sum()} duplicate respondent IDs")

        duplicates_removed = initial_count - len(data)
        return data, duplicates_removed

    def _validate_responses(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate responses against metadata constraints."""
        for col_name, var_def in self.metadata.variables.items():
            if col_name not in data.columns:
                continue

            # Range validation for numeric variables
            if var_def.valid_range and var_def.type in [VariableType.INTERVAL, VariableType.RATIO]:
                min_val, max_val = var_def.valid_range
                out_of_range = (data[col_name] < min_val) | (data[col_name] > max_val)

                if out_of_range.any():
                    out_of_range_count = out_of_range.sum()
                    self._log_warning(
                        f"{col_name}: {out_of_range_count} values outside valid range "
                        f"[{min_val}, {max_val}]"
                    )

                    if self.auto_clean:
                        # Set out-of-range values to NaN
                        data.loc[out_of_range, col_name] = np.nan

            # Category validation for categorical variables
            if var_def.categories and var_def.type in [VariableType.NOMINAL, VariableType.ORDINAL]:
                valid_categories = set(var_def.categories.keys()) | set(var_def.categories.values())
                invalid_responses = ~data[col_name].isin(valid_categories) & data[col_name].notna()

                if invalid_responses.any():
                    invalid_count = invalid_responses.sum()
                    self._log_warning(
                        f"{col_name}: {invalid_count} responses not in valid categories"
                    )

                    if self.auto_clean:
                        data.loc[invalid_responses, col_name] = np.nan

            # Handle missing codes
            if var_def.missing_codes:
                missing_mask = data[col_name].isin(var_def.missing_codes)
                if missing_mask.any():
                    data.loc[missing_mask, col_name] = np.nan

        return data

    def _detect_straightlining(self, data: pd.DataFrame) -> Dict[int, bool]:
        """
        Detect straight-lining behavior (same response to consecutive questions).

        Returns dictionary mapping row indices to straightlining flags.
        """
        straightlining_flags = {}

        if not self.metadata:
            return {i: False for i in range(len(data))}

        # Group variables that might be susceptible to straightlining
        # (typically Likert scales and rating questions)
        likert_variables = []
        for var_name, var_def in self.metadata.variables.items():
            if (var_name in data.columns and
                var_def.type == VariableType.ORDINAL and
                var_def.categories and
                len(var_def.categories) >= 3):  # At least 3-point scale
                likert_variables.append(var_name)

        if len(likert_variables) < 3:  # Need at least 3 variables to detect straightlining
            return {i: False for i in range(len(data))}

        # Detect straightlining patterns
        for idx, row in data.iterrows():
            consecutive_same = 0
            max_consecutive = 0

            for i in range(len(likert_variables) - 1):
                var1, var2 = likert_variables[i], likert_variables[i + 1]

                if (pd.notna(row[var1]) and pd.notna(row[var2]) and
                    row[var1] == row[var2]):
                    consecutive_same += 1
                    max_consecutive = max(max_consecutive, consecutive_same)
                else:
                    consecutive_same = 0

            # Flag as straightlining if more than 70% of consecutive pairs are identical
            # or if there are more than 5 consecutive identical responses
            threshold = max(3, int(0.7 * (len(likert_variables) - 1)))
            straightlining_flags[idx] = max_consecutive >= threshold

        return straightlining_flags

    def _detect_speeding(self, data: pd.DataFrame, time_col: str) -> List[bool]:
        """
        Detect speeding behavior based on response times.

        Returns list of speeding flags for each response.
        """
        if time_col not in data.columns:
            return [False] * len(data)

        response_times = pd.to_numeric(data[time_col], errors='coerce')

        # Calculate percentiles for outlier detection
        q1 = response_times.quantile(0.25)
        q3 = response_times.quantile(0.75)
        iqr = q3 - q1

        # Define speeding threshold (responses below Q1 - 1.5*IQR or below 30 seconds)
        speeding_threshold = max(30, q1 - 1.5 * iqr)  # At least 30 seconds

        speeding_flags = response_times < speeding_threshold

        self._log_info(
            f"Speeding detection: {speeding_flags.sum()} responses below "
            f"{speeding_threshold:.1f} seconds threshold"
        )

        return speeding_flags.fillna(False).tolist()

    def _identify_text_columns(self, data: pd.DataFrame) -> List[str]:
        """Identify columns containing text responses."""
        text_columns = []

        for col in data.columns:
            # Check if column contains primarily string data
            if data[col].dtype == 'object':
                # Further check if it's actually text (not categorical codes)
                sample_values = data[col].dropna().head(100)
                if sample_values.empty:
                    continue

                # Check if values are text-like (contain spaces, letters, etc.)
                text_like = sample_values.astype(str).str.contains(r'[a-zA-Z\s]', regex=True)
                if text_like.mean() > 0.5:  # More than 50% contain text
                    text_columns.append(col)

        return text_columns

    def _clean_text_responses(self, series: pd.Series) -> pd.Series:
        """Clean and standardize text responses."""
        if series.dtype != 'object':
            return series

        cleaned = series.copy()

        # Convert to string and handle NaN
        cleaned = cleaned.astype(str).replace('nan', np.nan)

        # Apply text cleaning operations
        cleaned = cleaned.str.strip()  # Remove leading/trailing whitespace
        cleaned = cleaned.str.replace(r'\s+', ' ', regex=True)  # Normalize whitespace
        cleaned = cleaned.str.replace(r'[^\w\s\.\,\!\?\-\(\)]', '', regex=True)  # Remove special chars

        # Handle common non-responses
        non_responses = [
            'n/a', 'na', 'none', 'nothing', 'no comment', 'no response',
            'not applicable', 'not sure', 'don\'t know', 'dk', 'unsure',
            '', ' ', '  '
        ]

        for non_resp in non_responses:
            cleaned = cleaned.str.replace(non_resp, np.nan, case=False)

        # Convert empty strings to NaN
        cleaned = cleaned.replace('', np.nan)

        return cleaned

    def _detect_outliers(self, data: pd.DataFrame) -> Dict[str, List[int]]:
        """
        Detect outliers in numeric variables using multiple methods.

        Returns dictionary mapping variable names to lists of outlier indices.
        """
        outlier_flags = {}

        numeric_columns = data.select_dtypes(include=[np.number]).columns

        for col in numeric_columns:
            if data[col].isna().all():
                continue

            outliers = set()
            values = data[col].dropna()

            if len(values) < 10:  # Skip if too few values
                continue

            # Method 1: IQR method
            q1, q3 = values.quantile([0.25, 0.75])
            iqr = q3 - q1
            if iqr > 0:
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                iqr_outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)].index
                outliers.update(iqr_outliers)

            # Method 2: Z-score method (for normally distributed data)
            z_scores = np.abs(stats.zscore(values))
            z_outliers = values[z_scores > 3].index
            outliers.update(z_outliers)

            # Method 3: Modified Z-score using median absolute deviation
            median = values.median()
            mad = np.median(np.abs(values - median))
            if mad > 0:
                modified_z_scores = 0.6745 * (values - median) / mad
                mad_outliers = values[np.abs(modified_z_scores) > 3.5].index
                outliers.update(mad_outliers)

            outlier_flags[col] = sorted(list(outliers))

            if outliers:
                self._log_info(f"{col}: detected {len(outliers)} potential outliers")

        return outlier_flags

    def _calculate_completion_rates(self, data: pd.DataFrame) -> Dict[str, int]:
        """Calculate response completion rates."""
        total_variables = len(data.columns)

        # Calculate completion per respondent
        completion_per_resp = data.notna().sum(axis=1) / total_variables

        # Define completion thresholds
        complete_threshold = 0.9  # 90% of questions answered
        partial_threshold = 0.1   # At least 10% of questions answered

        complete_responses = (completion_per_resp >= complete_threshold).sum()
        partial_responses = (
            (completion_per_resp >= partial_threshold) &
            (completion_per_resp < complete_threshold)
        ).sum()

        completion_rate = complete_responses / len(data) * 100

        return {
            'complete': complete_responses,
            'partial': partial_responses,
            'rate': completion_rate
        }

    def _calculate_item_nonresponse(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate item non-response rates for each variable."""
        nonresponse_rates = {}

        for col in data.columns:
            missing_count = data[col].isna().sum()
            nonresponse_rate = (missing_count / len(data)) * 100
            nonresponse_rates[col] = nonresponse_rate

        return nonresponse_rates

    def _apply_auto_cleaning(self,
                            data: pd.DataFrame,
                            quality_metrics: QualityMetrics,
                            cleaning_rules: Optional[Dict] = None) -> pd.DataFrame:
        """Apply automatic cleaning based on quality flags and rules."""
        cleaned_data = data.copy()

        if cleaning_rules is None:
            cleaning_rules = {
                'remove_straightliners': True,
                'remove_speeders': True,
                'remove_high_missing': True,
                'missing_threshold': 0.7  # Remove if >70% missing
            }

        # Remove straightliners if enabled
        if cleaning_rules.get('remove_straightliners', False):
            straightlining_indices = quality_metrics.quality_flags.get('straightlining', [])
            if straightlining_indices:
                cleaned_data = cleaned_data.drop(index=straightlining_indices)
                self._log_info(f"Removed {len(straightlining_indices)} straightlining responses")

        # Remove speeders if enabled
        if cleaning_rules.get('remove_speeders', False):
            speeding_indices = quality_metrics.quality_flags.get('speeding', [])
            if speeding_indices:
                cleaned_data = cleaned_data.drop(index=speeding_indices)
                self._log_info(f"Removed {len(speeding_indices)} speeding responses")

        # Remove responses with high missing data
        if cleaning_rules.get('remove_high_missing', False):
            missing_threshold = cleaning_rules.get('missing_threshold', 0.7)
            total_vars = len(cleaned_data.columns)
            missing_per_resp = cleaned_data.isna().sum(axis=1) / total_vars
            high_missing_mask = missing_per_resp > missing_threshold

            if high_missing_mask.any():
                high_missing_count = high_missing_mask.sum()
                cleaned_data = cleaned_data[~high_missing_mask]
                self._log_info(
                    f"Removed {high_missing_count} responses with >"
                    f"{missing_threshold*100:.0f}% missing data"
                )

        return cleaned_data

    def _log_info(self, message: str):
        """Log info message and add to cleaning log."""
        self.logger.info(message)
        self.cleaning_log.append(f"INFO: {message}")

    def _log_warning(self, message: str):
        """Log warning message and add to cleaning log."""
        if self.strict_validation:
            raise ValueError(message)
        else:
            self.logger.warning(message)
            self.cleaning_log.append(f"WARNING: {message}")

    def get_cleaning_report(self) -> Dict[str, Any]:
        """Generate a comprehensive cleaning report."""
        return {
            'cleaning_log': self.cleaning_log,
            'auto_clean_enabled': self.auto_clean,
            'strict_validation': self.strict_validation,
            'total_operations': len(self.cleaning_log)
        }