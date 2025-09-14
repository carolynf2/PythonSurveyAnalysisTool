"""
Main Survey Analysis Tool class.

This module provides the primary interface for comprehensive survey data analysis,
integrating all components for data loading, cleaning, analysis, and reporting.
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
import pandas as pd
import numpy as np

# Import all major components
from .data_processing import (
    DataLoader, DataCleaner, MissingDataHandler,
    SurveyMetadata, VariableDefinition, VariableType,
    CorrelationResult, ChiSquareResult, CrosstabResult,
    MissingDataAnalysis, DescriptiveStats, QualityMetrics
)
from .descriptive_analysis import UnivariateStats
from .correlation_analysis import CorrelationEngine
from .categorical_analysis import ChiSquareTests, CrossTabulation


class SurveyAnalysisTool:
    """
    Comprehensive survey data analysis tool.

    This is the main interface that integrates all analysis components,
    providing a unified API for survey data analysis workflows from
    data loading through reporting.

    Features:
    - Multi-format data loading with metadata extraction
    - Automated data cleaning and quality assessment
    - Comprehensive descriptive statistics
    - Advanced correlation analysis
    - Chi-square tests and cross-tabulation
    - Missing data analysis and imputation
    - Survey weighting support
    - Professional reporting
    """

    def __init__(self,
                 config_path: Optional[str] = None,
                 random_state: int = 42,
                 log_level: str = 'INFO'):
        """
        Initialize the Survey Analysis Tool.

        Parameters
        ----------
        config_path : str, optional
            Path to configuration file
        random_state : int, default 42
            Random state for reproducible results
        log_level : str, default 'INFO'
            Logging level
        """
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # Configuration
        self.random_state = random_state
        self.config = self._load_config(config_path) if config_path else {}

        # Initialize components
        self.data_loader = DataLoader()
        self.data_cleaner = DataCleaner()
        self.missing_data_handler = MissingDataHandler(random_state=random_state)
        self.univariate_stats = UnivariateStats(random_state=random_state)
        self.correlation_engine = CorrelationEngine(random_state=random_state)
        self.chi_square_tests = ChiSquareTests()
        self.cross_tabulation = CrossTabulation()

        # Data storage
        self.data = None
        self.metadata = None
        self.cleaned_data = None
        self.analysis_results = {}

        self.logger.info("Survey Analysis Tool initialized successfully")

    def load_survey_data(self,
                        file_path: str,
                        metadata_path: Optional[str] = None,
                        survey_weights: Optional[str] = None,
                        **kwargs) -> pd.DataFrame:
        """
        Load survey data from file with metadata.

        Parameters
        ----------
        file_path : str
            Path to survey data file
        metadata_path : str, optional
            Path to metadata file
        survey_weights : str, optional
            Column name containing survey weights
        **kwargs
            Additional arguments for data loading

        Returns
        -------
        pd.DataFrame
            Loaded survey data
        """
        self.logger.info(f"Loading survey data from {file_path}")

        try:
            # Load data and metadata
            self.data, self.metadata = self.data_loader.load_data(
                file_path, metadata_path, **kwargs
            )

            # Set weights variable if specified
            if survey_weights and self.metadata:
                self.metadata.weights_variable = survey_weights

            self.logger.info(
                f"Successfully loaded {len(self.data)} records with "
                f"{len(self.data.columns)} variables"
            )

            return self.data

        except Exception as e:
            self.logger.error(f"Failed to load survey data: {e}")
            raise

    def clean_survey_data(self,
                         data: Optional[pd.DataFrame] = None,
                         auto_clean: bool = True,
                         cleaning_rules: Optional[Dict] = None,
                         response_time_col: Optional[str] = None,
                         respondent_id_col: str = 'respondent_id') -> Tuple[pd.DataFrame, QualityMetrics]:
        """
        Clean and validate survey data.

        Parameters
        ----------
        data : pd.DataFrame, optional
            Data to clean. Uses self.data if not provided
        auto_clean : bool, default True
            Whether to apply automatic cleaning
        cleaning_rules : dict, optional
            Custom cleaning rules
        response_time_col : str, optional
            Column containing response times
        respondent_id_col : str, default 'respondent_id'
            Column containing respondent IDs

        Returns
        -------
        tuple
            (Cleaned data, Quality metrics)
        """
        if data is None:
            if self.data is None:
                raise ValueError("No data loaded. Call load_survey_data() first.")
            data = self.data

        self.logger.info("Starting data cleaning process")

        try:
            # Configure data cleaner
            self.data_cleaner.metadata = self.metadata
            self.data_cleaner.auto_clean = auto_clean

            # Clean data
            self.cleaned_data, quality_metrics = self.data_cleaner.clean_data(
                data,
                response_time_col=response_time_col,
                respondent_id_col=respondent_id_col,
                cleaning_rules=cleaning_rules
            )

            # Store quality metrics
            self.analysis_results['quality_metrics'] = quality_metrics

            self.logger.info(
                f"Data cleaning completed. {len(self.cleaned_data)} records retained "
                f"({quality_metrics.completion_rate:.1f}% completion rate)"
            )

            return self.cleaned_data, quality_metrics

        except Exception as e:
            self.logger.error(f"Data cleaning failed: {e}")
            raise

    def analyze_missing_data(self,
                           data: Optional[pd.DataFrame] = None,
                           variables: Optional[List[str]] = None) -> MissingDataAnalysis:
        """
        Analyze missing data patterns and mechanisms.

        Parameters
        ----------
        data : pd.DataFrame, optional
            Data to analyze. Uses cleaned_data if available, else self.data
        variables : list of str, optional
            Variables to analyze

        Returns
        -------
        MissingDataAnalysis
            Comprehensive missing data analysis
        """
        if data is None:
            data = self.cleaned_data if self.cleaned_data is not None else self.data

        if data is None:
            raise ValueError("No data available for analysis")

        self.logger.info("Analyzing missing data patterns")

        try:
            missing_analysis = self.missing_data_handler.analyze_missing_data(
                data, variables
            )

            self.analysis_results['missing_data_analysis'] = missing_analysis

            self.logger.info(
                f"Missing data analysis completed. {missing_analysis.missing_percentage:.1f}% "
                f"of data is missing ({missing_analysis.estimated_mechanism.value})"
            )

            return missing_analysis

        except Exception as e:
            self.logger.error(f"Missing data analysis failed: {e}")
            raise

    def descriptive_analysis(self,
                           data: Optional[pd.DataFrame] = None,
                           variables: Optional[List[str]] = None,
                           group_by: Optional[str] = None,
                           weights: Optional[str] = None) -> Dict[str, DescriptiveStats]:
        """
        Generate comprehensive descriptive statistics.

        Parameters
        ----------
        data : pd.DataFrame, optional
            Data to analyze
        variables : list of str, optional
            Variables to analyze
        group_by : str, optional
            Variable to group analysis by
        weights : str, optional
            Variable containing survey weights

        Returns
        -------
        dict
            Dictionary mapping variable names to DescriptiveStats
        """
        if data is None:
            data = self.cleaned_data if self.cleaned_data is not None else self.data

        if data is None:
            raise ValueError("No data available for analysis")

        self.logger.info("Generating descriptive statistics")

        try:
            # Get weights
            weight_series = None
            if weights:
                if weights in data.columns:
                    weight_series = data[weights]
                elif self.metadata and self.metadata.weights_variable:
                    if self.metadata.weights_variable in data.columns:
                        weight_series = data[self.metadata.weights_variable]

            # Get variable definitions
            variable_definitions = None
            if self.metadata:
                variable_definitions = self.metadata.variables

            # Calculate descriptive statistics
            descriptive_results = self.univariate_stats.calculate_descriptive_stats(
                data,
                variables=variables,
                weights=weight_series,
                group_by=group_by,
                variable_definitions=variable_definitions
            )

            self.analysis_results['descriptive_stats'] = descriptive_results

            self.logger.info(f"Descriptive analysis completed for {len(descriptive_results)} variables")

            return descriptive_results

        except Exception as e:
            self.logger.error(f"Descriptive analysis failed: {e}")
            raise

    def correlation_analysis(self,
                           data: Optional[pd.DataFrame] = None,
                           variables: Optional[List[str]] = None,
                           method: str = 'auto',
                           control_variables: Optional[List[str]] = None,
                           weights: Optional[str] = None) -> List[CorrelationResult]:
        """
        Perform correlation analysis with automatic method selection.

        Parameters
        ----------
        data : pd.DataFrame, optional
            Data to analyze
        variables : list of str, optional
            Variables to correlate
        method : str, default 'auto'
            Correlation method
        control_variables : list of str, optional
            Variables to control for
        weights : str, optional
            Variable containing survey weights

        Returns
        -------
        list of CorrelationResult
            Correlation results for all variable pairs
        """
        if data is None:
            data = self.cleaned_data if self.cleaned_data is not None else self.data

        if data is None:
            raise ValueError("No data available for analysis")

        self.logger.info(f"Starting correlation analysis using method: {method}")

        try:
            # Get weights
            weight_series = None
            if weights:
                if weights in data.columns:
                    weight_series = data[weights]
                elif self.metadata and self.metadata.weights_variable:
                    if self.metadata.weights_variable in data.columns:
                        weight_series = data[self.metadata.weights_variable]

            # Get variable definitions
            variable_definitions = None
            if self.metadata:
                variable_definitions = self.metadata.variables

            # Compute correlations
            correlation_results = self.correlation_engine.compute_correlations(
                data,
                variables=variables,
                method=method,
                variable_definitions=variable_definitions,
                weights=weight_series,
                control_variables=control_variables
            )

            self.analysis_results['correlations'] = correlation_results

            significant_correlations = sum(1 for r in correlation_results if r.is_significant())

            self.logger.info(
                f"Correlation analysis completed. {len(correlation_results)} correlations computed, "
                f"{significant_correlations} significant at α=0.05"
            )

            return correlation_results

        except Exception as e:
            self.logger.error(f"Correlation analysis failed: {e}")
            raise

    def chi_square_test(self,
                       data: Optional[pd.DataFrame] = None,
                       row_variable: str = None,
                       column_variable: Optional[str] = None,
                       expected_frequencies: Optional[np.ndarray] = None,
                       weights: Optional[str] = None,
                       test_type: str = 'independence') -> ChiSquareResult:
        """
        Perform chi-square tests for independence and goodness-of-fit.

        Parameters
        ----------
        data : pd.DataFrame, optional
            Data to analyze
        row_variable : str
            Row variable for contingency table
        column_variable : str, optional
            Column variable for contingency table
        expected_frequencies : np.ndarray, optional
            Expected frequencies for goodness-of-fit test
        weights : str, optional
            Variable containing survey weights
        test_type : str, default 'independence'
            Type of test: 'independence', 'goodness_of_fit', 'homogeneity'

        Returns
        -------
        ChiSquareResult
            Complete test results
        """
        if data is None:
            data = self.cleaned_data if self.cleaned_data is not None else self.data

        if data is None:
            raise ValueError("No data available for analysis")

        if row_variable is None:
            raise ValueError("row_variable is required")

        self.logger.info(f"Performing chi-square {test_type} test")

        try:
            # Get weights
            weight_series = None
            if weights:
                if weights in data.columns:
                    weight_series = data[weights]
                elif self.metadata and self.metadata.weights_variable:
                    if self.metadata.weights_variable in data.columns:
                        weight_series = data[self.metadata.weights_variable]

            # Perform appropriate test
            if test_type == 'independence':
                if column_variable is None:
                    raise ValueError("column_variable required for independence test")

                result = self.chi_square_tests.test_independence(
                    data, row_variable, column_variable, weight_series
                )

            elif test_type == 'goodness_of_fit':
                if row_variable not in data.columns:
                    raise ValueError(f"Variable {row_variable} not found in data")

                # Create observed frequencies
                if weight_series is not None:
                    observed_freq = data.groupby(row_variable)[weights].sum()
                else:
                    observed_freq = data[row_variable].value_counts()

                result = self.chi_square_tests.test_goodness_of_fit(
                    observed_freq, expected_frequencies, variable_name=row_variable
                )

            elif test_type == 'homogeneity':
                if column_variable is None:
                    raise ValueError("column_variable required for homogeneity test")

                result = self.chi_square_tests.test_homogeneity(
                    data, row_variable, column_variable, weight_series
                )

            else:
                raise ValueError(f"Unknown test type: {test_type}")

            # Store result
            if 'chi_square_tests' not in self.analysis_results:
                self.analysis_results['chi_square_tests'] = []
            self.analysis_results['chi_square_tests'].append(result)

            self.logger.info(
                f"Chi-square test completed. χ²={result.chi_square_statistic:.3f}, "
                f"p={result.p_value:.4f}"
            )

            return result

        except Exception as e:
            self.logger.error(f"Chi-square test failed: {e}")
            raise

    def cross_tabulation(self,
                        data: Optional[pd.DataFrame] = None,
                        row_variable: str = None,
                        column_variable: str = None,
                        control_variables: Optional[List[str]] = None,
                        weights: Optional[str] = None,
                        percentages: List[str] = ['row', 'column', 'total']) -> Union[CrosstabResult, Dict[str, CrosstabResult]]:
        """
        Create comprehensive cross-tabulation with statistical testing.

        Parameters
        ----------
        data : pd.DataFrame, optional
            Data to analyze
        row_variable : str
            Variable for table rows
        column_variable : str
            Variable for table columns
        control_variables : list of str, optional
            Variables to control for
        weights : str, optional
            Variable containing survey weights
        percentages : list of str, default ['row', 'column', 'total']
            Types of percentages to calculate

        Returns
        -------
        CrosstabResult or dict
            Cross-tabulation results
        """
        if data is None:
            data = self.cleaned_data if self.cleaned_data is not None else self.data

        if data is None:
            raise ValueError("No data available for analysis")

        if row_variable is None or column_variable is None:
            raise ValueError("Both row_variable and column_variable are required")

        self.logger.info(f"Creating cross-tabulation: {row_variable} × {column_variable}")

        try:
            # Get weights
            weight_series = None
            if weights:
                if weights in data.columns:
                    weight_series = data[weights]
                elif self.metadata and self.metadata.weights_variable:
                    if self.metadata.weights_variable in data.columns:
                        weight_series = data[self.metadata.weights_variable]

            # Create cross-tabulation
            crosstab_result = self.cross_tabulation.crosstab(
                data,
                row_variable=row_variable,
                column_variable=column_variable,
                control_variables=control_variables,
                weights=weight_series,
                percentages=percentages
            )

            # Store result
            if 'cross_tabulations' not in self.analysis_results:
                self.analysis_results['cross_tabulations'] = []
            self.analysis_results['cross_tabulations'].append(crosstab_result)

            if isinstance(crosstab_result, dict):
                self.logger.info(f"Cross-tabulation completed with {len(crosstab_result)} strata")
            else:
                self.logger.info("Cross-tabulation completed")

            return crosstab_result

        except Exception as e:
            self.logger.error(f"Cross-tabulation failed: {e}")
            raise

    def demographic_profiling(self,
                             data: Optional[pd.DataFrame] = None,
                             demographic_variables: Optional[List[str]] = None,
                             target_variable: Optional[str] = None,
                             weights: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate demographic profiles and breakdowns.

        Parameters
        ----------
        data : pd.DataFrame, optional
            Data to analyze
        demographic_variables : list of str, optional
            Demographic variables to profile
        target_variable : str, optional
            Target variable for demographic breakdowns
        weights : str, optional
            Variable containing survey weights

        Returns
        -------
        dict
            Demographic profiling results
        """
        if data is None:
            data = self.cleaned_data if self.cleaned_data is not None else self.data

        if data is None:
            raise ValueError("No data available for analysis")

        self.logger.info("Generating demographic profiles")

        try:
            # Auto-detect demographic variables if not specified
            if demographic_variables is None:
                demographic_variables = self._identify_demographic_variables(data)

            # Get weights
            weight_series = None
            if weights:
                if weights in data.columns:
                    weight_series = data[weights]
                elif self.metadata and self.metadata.weights_variable:
                    if self.metadata.weights_variable in data.columns:
                        weight_series = data[self.metadata.weights_variable]

            demographic_results = {}

            # Overall demographic profile
            for var in demographic_variables:
                if var not in data.columns:
                    continue

                # Basic frequency distribution
                if weight_series is not None:
                    freq_table = data.groupby(var)[weights].sum()
                else:
                    freq_table = data[var].value_counts()

                demographic_results[var] = {
                    'frequencies': freq_table,
                    'percentages': (freq_table / freq_table.sum()) * 100
                }

            # Cross-demographic analysis if target variable specified
            if target_variable and target_variable in data.columns:
                demographic_results['target_breakdowns'] = {}

                for var in demographic_variables:
                    if var not in data.columns:
                        continue

                    # Cross-tabulation with target variable
                    crosstab = self.cross_tabulation.crosstab(
                        data, var, target_variable, weights=weight_series
                    )

                    demographic_results['target_breakdowns'][var] = crosstab

            self.analysis_results['demographic_profile'] = demographic_results

            self.logger.info(f"Demographic profiling completed for {len(demographic_variables)} variables")

            return demographic_results

        except Exception as e:
            self.logger.error(f"Demographic profiling failed: {e}")
            raise

    def generate_report(self,
                       analysis_results: Optional[Dict] = None,
                       report_type: str = 'comprehensive',
                       format: str = 'html',
                       output_path: Optional[str] = None) -> str:
        """
        Generate professional analysis reports.

        Parameters
        ----------
        analysis_results : dict, optional
            Analysis results to include. Uses self.analysis_results if not provided
        report_type : str, default 'comprehensive'
            Type of report: 'comprehensive', 'summary', 'executive'
        format : str, default 'html'
            Output format: 'html', 'pdf', 'markdown'
        output_path : str, optional
            Path to save report

        Returns
        -------
        str
            Generated report content or path to saved file
        """
        if analysis_results is None:
            analysis_results = self.analysis_results

        if not analysis_results:
            self.logger.warning("No analysis results available for reporting")
            return ""

        self.logger.info(f"Generating {report_type} report in {format} format")

        try:
            # This is a simplified implementation
            # In a full implementation, you would use a proper reporting engine
            report_content = self._generate_basic_report(analysis_results, report_type)

            if output_path:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(report_content)
                self.logger.info(f"Report saved to {output_path}")
                return output_path
            else:
                return report_content

        except Exception as e:
            self.logger.error(f"Report generation failed: {e}")
            raise

    def get_analysis_summary(self) -> Dict[str, Any]:
        """
        Get summary of all completed analyses.

        Returns
        -------
        dict
            Summary of analysis results
        """
        summary = {
            'data_loaded': self.data is not None,
            'data_cleaned': self.cleaned_data is not None,
            'n_records': len(self.data) if self.data is not None else 0,
            'n_variables': len(self.data.columns) if self.data is not None else 0,
            'analyses_completed': list(self.analysis_results.keys()),
            'has_metadata': self.metadata is not None,
            'has_weights': False
        }

        if self.metadata and self.metadata.weights_variable:
            summary['has_weights'] = True
            summary['weights_variable'] = self.metadata.weights_variable

        if 'quality_metrics' in self.analysis_results:
            qm = self.analysis_results['quality_metrics']
            summary['completion_rate'] = qm.completion_rate
            summary['duplicate_responses'] = qm.duplicate_responses

        if 'correlations' in self.analysis_results:
            correlations = self.analysis_results['correlations']
            summary['n_correlations'] = len(correlations)
            summary['significant_correlations'] = sum(1 for r in correlations if r.is_significant())

        return summary

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
            self.logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            self.logger.warning(f"Failed to load configuration: {e}")
            return {}

    def _identify_demographic_variables(self, data: pd.DataFrame) -> List[str]:
        """Auto-identify likely demographic variables."""
        demographic_keywords = [
            'age', 'gender', 'sex', 'race', 'ethnicity', 'education', 'income',
            'employment', 'marital', 'region', 'state', 'city', 'urban', 'rural'
        ]

        demographic_vars = []
        for col in data.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in demographic_keywords):
                demographic_vars.append(col)

        return demographic_vars

    def _generate_basic_report(self, analysis_results: Dict, report_type: str) -> str:
        """Generate basic HTML report (simplified implementation)."""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Survey Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                h1, h2, h3 { color: #2c3e50; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                .summary { background-color: #f8f9fa; padding: 15px; border-radius: 5px; }
            </style>
        </head>
        <body>
            <h1>Survey Analysis Report</h1>
        """

        # Add summary section
        if 'quality_metrics' in analysis_results:
            qm = analysis_results['quality_metrics']
            html_content += f"""
            <div class="summary">
                <h2>Data Quality Summary</h2>
                <p><strong>Total Responses:</strong> {qm.total_responses}</p>
                <p><strong>Complete Responses:</strong> {qm.complete_responses}</p>
                <p><strong>Completion Rate:</strong> {qm.completion_rate:.1f}%</p>
                <p><strong>Duplicate Responses:</strong> {qm.duplicate_responses}</p>
            </div>
            """

        # Add other sections based on available results
        if 'descriptive_stats' in analysis_results:
            html_content += "<h2>Descriptive Statistics</h2>\n"
            html_content += "<p>Descriptive analysis completed for multiple variables.</p>\n"

        if 'correlations' in analysis_results:
            correlations = analysis_results['correlations']
            significant = sum(1 for r in correlations if r.is_significant())
            html_content += f"""
            <h2>Correlation Analysis</h2>
            <p><strong>Total Correlations:</strong> {len(correlations)}</p>
            <p><strong>Significant Correlations:</strong> {significant}</p>
            """

        html_content += """
        </body>
        </html>
        """

        return html_content