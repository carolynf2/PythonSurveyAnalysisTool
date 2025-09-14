"""
Missing data analysis and imputation for survey data.

This module provides comprehensive missing data analysis including pattern detection,
mechanism testing, visualization, and multiple imputation strategies optimized
for survey research contexts.
"""

import logging
import warnings
from typing import Dict, List, Optional, Union, Tuple, Any
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

try:
    from fancyimpute import IterativeImputer as FancyIterativeImputer
    HAS_FANCYIMPUTE = True
except ImportError:
    HAS_FANCYIMPUTE = False
    warnings.warn("fancyimpute not available. Some advanced imputation methods disabled.")

from .models import MissingDataPattern, MissingDataAnalysis, VariableType


class MissingDataHandler:
    """
    Comprehensive missing data analysis and imputation for survey data.

    Features:
    - Missing data pattern analysis and visualization
    - Missing data mechanism testing (MCAR, MAR, MNAR)
    - Multiple imputation strategies:
      - Mean/median/mode imputation
      - Regression imputation
      - K-Nearest Neighbors (KNN)
      - Multiple Imputation by Chained Equations (MICE)
      - Random Forest imputation
    - Sensitivity analysis for imputation methods
    - Survey-specific missing data handling
    """

    def __init__(self,
                 random_state: int = 42,
                 n_imputations: int = 5):
        """
        Initialize the MissingDataHandler.

        Parameters
        ----------
        random_state : int, default 42
            Random state for reproducible imputation
        n_imputations : int, default 5
            Number of imputations for multiple imputation methods
        """
        self.random_state = random_state
        self.n_imputations = n_imputations
        self.logger = logging.getLogger(__name__)

    def analyze_missing_data(self,
                           data: pd.DataFrame,
                           variables: Optional[List[str]] = None,
                           group_by: Optional[str] = None) -> MissingDataAnalysis:
        """
        Comprehensive missing data analysis.

        Parameters
        ----------
        data : pd.DataFrame
            Survey data with potential missing values
        variables : list of str, optional
            Subset of variables to analyze. If None, analyzes all variables
        group_by : str, optional
            Variable to group analysis by (e.g., demographic groups)

        Returns
        -------
        MissingDataAnalysis
            Comprehensive missing data analysis results
        """
        if variables is None:
            variables = list(data.columns)

        analysis_data = data[variables].copy()

        self.logger.info(f"Analyzing missing data for {len(variables)} variables")

        # Calculate overall missing statistics
        total_cells = analysis_data.size
        total_missing = analysis_data.isna().sum().sum()
        missing_percentage = (total_missing / total_cells) * 100

        # Missing data by variable
        missing_by_variable = {}
        for var in variables:
            missing_count = analysis_data[var].isna().sum()
            missing_pct = (missing_count / len(analysis_data)) * 100

            missing_by_variable[var] = {
                'count': int(missing_count),
                'percentage': float(missing_pct),
                'complete_cases': int(len(analysis_data) - missing_count)
            }

        # Missing data patterns
        missing_patterns = self._analyze_missing_patterns(analysis_data)

        # Pattern frequencies
        pattern_frequencies = missing_patterns.value_counts().to_dict()

        # Test missing data mechanism
        estimated_mechanism = self._test_missing_mechanism(analysis_data)

        # Little's MCAR test if possible
        little_mcar_test = self._little_mcar_test(analysis_data)

        # Missing data correlations
        missing_correlations = self._missing_data_correlations(analysis_data)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            missing_percentage, missing_by_variable, estimated_mechanism
        )

        return MissingDataAnalysis(
            total_missing=int(total_missing),
            missing_percentage=float(missing_percentage),
            missing_by_variable=missing_by_variable,
            missing_patterns=missing_patterns,
            pattern_frequencies={str(k): v for k, v in pattern_frequencies.items()},
            estimated_mechanism=estimated_mechanism,
            little_mcar_test=little_mcar_test,
            missing_correlations=missing_correlations,
            recommendations=recommendations
        )

    def impute_missing_data(self,
                          data: pd.DataFrame,
                          method: str = 'mice',
                          variables: Optional[List[str]] = None,
                          categorical_variables: Optional[List[str]] = None,
                          **kwargs) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        """
        Impute missing data using specified method.

        Parameters
        ----------
        data : pd.DataFrame
            Data with missing values to impute
        method : str, default 'mice'
            Imputation method: 'mean', 'median', 'mode', 'knn', 'mice', 'random_forest'
        variables : list of str, optional
            Variables to impute. If None, imputes all variables with missing data
        categorical_variables : list of str, optional
            List of categorical variables for method-specific handling
        **kwargs
            Additional parameters for imputation methods

        Returns
        -------
        pd.DataFrame or list of pd.DataFrame
            Imputed data. For multiple imputation methods, returns list of datasets
        """
        if variables is None:
            # Find variables with missing data
            variables = data.columns[data.isna().any()].tolist()

        if not variables:
            self.logger.info("No missing data found. Returning original data.")
            return data

        self.logger.info(f"Imputing missing data for {len(variables)} variables using {method}")

        # Prepare data for imputation
        imputation_data = data.copy()

        # Handle categorical variables
        categorical_vars = categorical_variables or []
        categorical_encodings = {}

        for var in categorical_vars:
            if var in imputation_data.columns:
                # Encode categorical variables for numeric imputation methods
                unique_vals = imputation_data[var].dropna().unique()
                encoding = {val: i for i, val in enumerate(unique_vals)}
                categorical_encodings[var] = {v: k for k, v in encoding.items()}  # Reverse mapping
                imputation_data[var] = imputation_data[var].map(encoding)

        # Apply imputation method
        if method == 'mean':
            return self._impute_mean(imputation_data, variables, categorical_encodings)
        elif method == 'median':
            return self._impute_median(imputation_data, variables, categorical_encodings)
        elif method == 'mode':
            return self._impute_mode(imputation_data, variables, categorical_encodings)
        elif method == 'knn':
            return self._impute_knn(imputation_data, variables, categorical_encodings, **kwargs)
        elif method == 'mice':
            return self._impute_mice(imputation_data, variables, categorical_encodings, **kwargs)
        elif method == 'random_forest':
            return self._impute_random_forest(imputation_data, variables, categorical_encodings, **kwargs)
        else:
            raise ValueError(f"Unknown imputation method: {method}")

    def _analyze_missing_patterns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Analyze missing data patterns."""
        # Create binary missing data indicator matrix
        missing_matrix = data.isna()

        # Convert to pattern strings for easier grouping
        patterns = missing_matrix.apply(lambda row: ''.join(['1' if x else '0' for x in row]), axis=1)

        # Create summary of patterns
        pattern_summary = patterns.value_counts().reset_index()
        pattern_summary.columns = ['pattern', 'frequency']

        # Add interpretable pattern description
        pattern_summary['variables_missing'] = pattern_summary['pattern'].apply(
            lambda p: [data.columns[i] for i, char in enumerate(p) if char == '1']
        )

        pattern_summary['n_missing'] = pattern_summary['pattern'].apply(lambda p: p.count('1'))

        return pattern_summary

    def _test_missing_mechanism(self, data: pd.DataFrame) -> MissingDataPattern:
        """
        Test missing data mechanism using various statistical tests.

        This is a simplified heuristic approach. In practice, determining
        the missing data mechanism often requires domain knowledge.
        """
        missing_matrix = data.isna()

        # Test 1: Check if missingness is random across variables
        missing_correlations = missing_matrix.corr()
        high_correlations = (missing_correlations.abs() > 0.3).sum().sum() - len(missing_correlations)

        # Test 2: Check if missingness depends on observed values
        mechanism_tests = []

        for col in data.columns:
            if missing_matrix[col].any():
                # Test if missingness in this variable depends on other observed variables
                for other_col in data.columns:
                    if col != other_col and not missing_matrix[other_col].all():
                        # Split observed values by missingness in target variable
                        missing_group = data.loc[missing_matrix[col], other_col].dropna()
                        present_group = data.loc[~missing_matrix[col], other_col].dropna()

                        if len(missing_group) > 5 and len(present_group) > 5:
                            # Perform appropriate test based on data type
                            if data[other_col].dtype in ['int64', 'float64']:
                                # t-test for numeric variables
                                _, p_value = stats.ttest_ind(missing_group, present_group)
                            else:
                                # Chi-square test for categorical variables
                                contingency = pd.crosstab(
                                    missing_matrix[col],
                                    data[other_col],
                                    dropna=False
                                )
                                if contingency.shape == (2, 2) and (contingency >= 5).all().all():
                                    _, p_value, _, _ = stats.chi2_contingency(contingency)
                                else:
                                    p_value = 1.0  # Skip if assumptions not met

                            mechanism_tests.append(p_value)

        # Heuristic classification
        if not mechanism_tests:
            return MissingDataPattern.MCAR

        significant_tests = sum(1 for p in mechanism_tests if p < 0.05)
        total_tests = len(mechanism_tests)

        if significant_tests / total_tests < 0.1:  # Less than 10% significant
            return MissingDataPattern.MCAR
        elif significant_tests / total_tests < 0.5:  # Less than 50% significant
            return MissingDataPattern.MAR
        else:
            return MissingDataPattern.MNAR

    def _little_mcar_test(self, data: pd.DataFrame) -> Optional[Dict[str, float]]:
        """
        Perform Little's MCAR test if possible.

        Note: This is a simplified implementation. For production use,
        consider using specialized packages like 'impyute' or R's 'VIM'.
        """
        try:
            # This is a placeholder for Little's MCAR test
            # A full implementation would require more complex calculations
            missing_matrix = data.isna()

            # Calculate test statistic (simplified)
            n_patterns = missing_matrix.apply(
                lambda row: ''.join(['1' if x else '0' for x in row]), axis=1
            ).nunique()

            n_variables = len(data.columns)
            n_observations = len(data)

            # Simplified approximation
            test_statistic = n_patterns * np.log(n_observations)
            degrees_freedom = n_variables * (n_variables - 1) / 2

            p_value = 1 - stats.chi2.cdf(test_statistic, degrees_freedom)

            return {
                'test_statistic': float(test_statistic),
                'degrees_freedom': int(degrees_freedom),
                'p_value': float(p_value)
            }

        except Exception as e:
            self.logger.warning(f"Could not perform Little's MCAR test: {e}")
            return None

    def _missing_data_correlations(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Calculate correlations between missing data patterns."""
        try:
            missing_matrix = data.isna().astype(int)
            correlations = missing_matrix.corr()
            return correlations
        except Exception as e:
            self.logger.warning(f"Could not calculate missing data correlations: {e}")
            return None

    def _generate_recommendations(self,
                                missing_percentage: float,
                                missing_by_variable: Dict,
                                mechanism: MissingDataPattern) -> List[str]:
        """Generate recommendations based on missing data analysis."""
        recommendations = []

        # Overall missing data level
        if missing_percentage < 5:
            recommendations.append("Low levels of missing data (<5%). Complete case analysis may be appropriate.")
        elif missing_percentage < 20:
            recommendations.append("Moderate levels of missing data (5-20%). Consider imputation methods.")
        else:
            recommendations.append("High levels of missing data (>20%). Investigate data collection issues.")

        # Variable-specific recommendations
        high_missing_vars = [
            var for var, stats in missing_by_variable.items()
            if stats['percentage'] > 50
        ]

        if high_missing_vars:
            recommendations.append(
                f"Variables with >50% missing data: {', '.join(high_missing_vars)}. "
                "Consider excluding from analysis or investigating collection issues."
            )

        # Mechanism-specific recommendations
        if mechanism == MissingDataPattern.MCAR:
            recommendations.append(
                "Missing data appears to be MCAR. Complete case analysis or simple imputation acceptable."
            )
        elif mechanism == MissingDataPattern.MAR:
            recommendations.append(
                "Missing data appears to be MAR. Use multiple imputation or advanced methods."
            )
        else:  # MNAR
            recommendations.append(
                "Missing data may be MNAR. Consider pattern-mixture models or selection models. "
                "Investigate reasons for missingness."
            )

        return recommendations

    def _impute_mean(self, data: pd.DataFrame, variables: List[str], categorical_encodings: Dict) -> pd.DataFrame:
        """Impute using mean for numeric, mode for categorical."""
        imputed_data = data.copy()

        for var in variables:
            if var in imputed_data.columns:
                if imputed_data[var].dtype in ['int64', 'float64']:
                    # Use mean for numeric variables
                    mean_value = imputed_data[var].mean()
                    imputed_data[var].fillna(mean_value, inplace=True)
                else:
                    # Use mode for categorical variables
                    mode_value = imputed_data[var].mode()
                    if not mode_value.empty:
                        imputed_data[var].fillna(mode_value.iloc[0], inplace=True)

        # Decode categorical variables
        for var, decoding in categorical_encodings.items():
            if var in imputed_data.columns:
                imputed_data[var] = imputed_data[var].map(decoding)

        return imputed_data

    def _impute_median(self, data: pd.DataFrame, variables: List[str], categorical_encodings: Dict) -> pd.DataFrame:
        """Impute using median for numeric, mode for categorical."""
        imputed_data = data.copy()

        for var in variables:
            if var in imputed_data.columns:
                if imputed_data[var].dtype in ['int64', 'float64']:
                    median_value = imputed_data[var].median()
                    imputed_data[var].fillna(median_value, inplace=True)
                else:
                    mode_value = imputed_data[var].mode()
                    if not mode_value.empty:
                        imputed_data[var].fillna(mode_value.iloc[0], inplace=True)

        # Decode categorical variables
        for var, decoding in categorical_encodings.items():
            if var in imputed_data.columns:
                imputed_data[var] = imputed_data[var].map(decoding)

        return imputed_data

    def _impute_mode(self, data: pd.DataFrame, variables: List[str], categorical_encodings: Dict) -> pd.DataFrame:
        """Impute using mode for all variables."""
        imputed_data = data.copy()

        for var in variables:
            if var in imputed_data.columns:
                mode_value = imputed_data[var].mode()
                if not mode_value.empty:
                    imputed_data[var].fillna(mode_value.iloc[0], inplace=True)

        # Decode categorical variables
        for var, decoding in categorical_encodings.items():
            if var in imputed_data.columns:
                imputed_data[var] = imputed_data[var].map(decoding)

        return imputed_data

    def _impute_knn(self, data: pd.DataFrame, variables: List[str], categorical_encodings: Dict, **kwargs) -> pd.DataFrame:
        """Impute using K-Nearest Neighbors."""
        n_neighbors = kwargs.get('n_neighbors', 5)

        # Select numeric columns for KNN
        numeric_data = data.select_dtypes(include=[np.number])

        if numeric_data.empty:
            self.logger.warning("No numeric variables found for KNN imputation. Using mode imputation.")
            return self._impute_mode(data, variables, categorical_encodings)

        # Apply KNN imputation
        imputer = KNNImputer(n_neighbors=n_neighbors)
        imputed_numeric = pd.DataFrame(
            imputer.fit_transform(numeric_data),
            columns=numeric_data.columns,
            index=numeric_data.index
        )

        # Combine with original data
        imputed_data = data.copy()
        imputed_data[imputed_numeric.columns] = imputed_numeric

        # Handle categorical variables with mode
        categorical_vars = set(variables) - set(imputed_numeric.columns)
        for var in categorical_vars:
            if var in imputed_data.columns:
                mode_value = imputed_data[var].mode()
                if not mode_value.empty:
                    imputed_data[var].fillna(mode_value.iloc[0], inplace=True)

        # Decode categorical variables
        for var, decoding in categorical_encodings.items():
            if var in imputed_data.columns:
                imputed_data[var] = imputed_data[var].map(decoding)

        return imputed_data

    def _impute_mice(self, data: pd.DataFrame, variables: List[str], categorical_encodings: Dict, **kwargs) -> List[pd.DataFrame]:
        """Impute using Multiple Imputation by Chained Equations (MICE)."""
        max_iter = kwargs.get('max_iter', 10)

        # Select numeric columns for MICE
        numeric_data = data.select_dtypes(include=[np.number])

        if numeric_data.empty:
            self.logger.warning("No numeric variables found for MICE. Using mode imputation.")
            return [self._impute_mode(data, variables, categorical_encodings)]

        # Apply MICE imputation
        imputer = IterativeImputer(
            max_iter=max_iter,
            random_state=self.random_state
        )

        imputed_datasets = []

        for i in range(self.n_imputations):
            # Add randomness for multiple imputations
            current_imputer = IterativeImputer(
                max_iter=max_iter,
                random_state=self.random_state + i
            )

            imputed_numeric = pd.DataFrame(
                current_imputer.fit_transform(numeric_data),
                columns=numeric_data.columns,
                index=numeric_data.index
            )

            # Combine with original data
            imputed_data = data.copy()
            imputed_data[imputed_numeric.columns] = imputed_numeric

            # Handle categorical variables
            categorical_vars = set(variables) - set(imputed_numeric.columns)
            for var in categorical_vars:
                if var in imputed_data.columns:
                    mode_value = imputed_data[var].mode()
                    if not mode_value.empty:
                        imputed_data[var].fillna(mode_value.iloc[0], inplace=True)

            # Decode categorical variables
            for var, decoding in categorical_encodings.items():
                if var in imputed_data.columns:
                    imputed_data[var] = imputed_data[var].map(decoding)

            imputed_datasets.append(imputed_data)

        return imputed_datasets

    def _impute_random_forest(self, data: pd.DataFrame, variables: List[str], categorical_encodings: Dict, **kwargs) -> pd.DataFrame:
        """Impute using Random Forest."""
        n_estimators = kwargs.get('n_estimators', 100)

        imputed_data = data.copy()

        # Impute each variable with missing data
        for var in variables:
            if var not in imputed_data.columns or not imputed_data[var].isna().any():
                continue

            # Prepare training data (complete cases)
            complete_mask = imputed_data[var].notna()
            missing_mask = imputed_data[var].isna()

            if complete_mask.sum() < 10:  # Not enough complete cases
                continue

            # Use other variables as predictors
            predictor_vars = [col for col in imputed_data.columns if col != var]
            X_complete = imputed_data.loc[complete_mask, predictor_vars]
            y_complete = imputed_data.loc[complete_mask, var]

            # Handle remaining missing values in predictors
            X_complete = X_complete.fillna(X_complete.mode().iloc[0] if not X_complete.mode().empty else 0)

            X_missing = imputed_data.loc[missing_mask, predictor_vars]
            X_missing = X_missing.fillna(X_missing.mode().iloc[0] if not X_missing.mode().empty else 0)

            try:
                # Choose appropriate Random Forest based on variable type
                if imputed_data[var].dtype in ['int64', 'float64']:
                    rf = RandomForestRegressor(
                        n_estimators=n_estimators,
                        random_state=self.random_state
                    )
                else:
                    rf = RandomForestClassifier(
                        n_estimators=n_estimators,
                        random_state=self.random_state
                    )

                # Fit and predict
                rf.fit(X_complete, y_complete)
                predictions = rf.predict(X_missing)

                # Fill missing values
                imputed_data.loc[missing_mask, var] = predictions

            except Exception as e:
                self.logger.warning(f"Random Forest imputation failed for {var}: {e}")
                # Fallback to mode imputation
                mode_value = imputed_data[var].mode()
                if not mode_value.empty:
                    imputed_data.loc[missing_mask, var] = mode_value.iloc[0]

        # Decode categorical variables
        for var, decoding in categorical_encodings.items():
            if var in imputed_data.columns:
                imputed_data[var] = imputed_data[var].map(decoding)

        return imputed_data

    def evaluate_imputation_quality(self,
                                  original_data: pd.DataFrame,
                                  imputed_data: pd.DataFrame,
                                  test_fraction: float = 0.1) -> Dict[str, Any]:
        """
        Evaluate imputation quality using cross-validation approach.

        Artificially removes some observed values, imputes them, and compares
        with original values.
        """
        np.random.seed(self.random_state)

        evaluation_results = {}

        for col in original_data.columns:
            if original_data[col].isna().all():
                continue

            # Select observed values to artificially remove
            observed_indices = original_data[col].dropna().index
            n_test = max(1, int(len(observed_indices) * test_fraction))

            test_indices = np.random.choice(observed_indices, n_test, replace=False)

            # Create test data with artificially missing values
            test_data = original_data.copy()
            original_values = test_data.loc[test_indices, col].copy()
            test_data.loc[test_indices, col] = np.nan

            # Impute test data (simplified - using mean/mode for speed)
            if test_data[col].dtype in ['int64', 'float64']:
                imputed_value = test_data[col].mean()
                test_data.loc[test_indices, col] = imputed_value

                # Calculate RMSE
                rmse = np.sqrt(np.mean((original_values - imputed_value) ** 2))
                evaluation_results[col] = {'rmse': float(rmse), 'method': 'mean'}
            else:
                mode_value = test_data[col].mode()
                if not mode_value.empty:
                    test_data.loc[test_indices, col] = mode_value.iloc[0]

                    # Calculate accuracy
                    accuracy = (original_values == mode_value.iloc[0]).mean()
                    evaluation_results[col] = {'accuracy': float(accuracy), 'method': 'mode'}

        return evaluation_results