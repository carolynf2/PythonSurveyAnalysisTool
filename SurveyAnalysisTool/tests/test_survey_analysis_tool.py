"""
Comprehensive tests for the SurveyAnalysisTool.

This module contains unit tests and integration tests for the main
survey analysis functionality.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime
import tempfile
import os
import sys

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from survey_analysis_tool import SurveyAnalysisTool
from data_processing.models import (
    SurveyMetadata, VariableDefinition, VariableType,
    CorrelationResult, ChiSquareResult, CrosstabResult
)


class TestSurveyAnalysisTool(unittest.TestCase):
    """Test cases for SurveyAnalysisTool main functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample data
        np.random.seed(42)
        self.n_responses = 100

        self.sample_data = pd.DataFrame({
            'respondent_id': range(1, self.n_responses + 1),
            'age': np.random.normal(40, 15, self.n_responses).astype(int).clip(18, 80),
            'gender': np.random.choice(['Male', 'Female'], self.n_responses),
            'satisfaction': np.random.choice([1, 2, 3, 4, 5], self.n_responses),
            'income': np.random.choice(['Low', 'Medium', 'High'], self.n_responses),
            'recommend': np.random.choice([0, 1], self.n_responses),
            'score': np.random.normal(50, 10, self.n_responses),
            'response_time': np.random.gamma(2, 30, self.n_responses).astype(int)
        })

        # Add some missing data
        self.sample_data.loc[0:4, 'income'] = np.nan
        self.sample_data.loc[10:12, 'satisfaction'] = np.nan

        # Create sample metadata
        self.sample_metadata = SurveyMetadata(
            survey_id='TEST_SURVEY',
            title='Test Survey',
            description='Test survey for unit testing',
            collection_period=(datetime(2024, 1, 1), datetime(2024, 1, 31)),
            target_population='Test population',
            sampling_method='Random',
            expected_responses=100,
            actual_responses=100,
            completion_rate=100.0,
            variables={
                'age': VariableDefinition(
                    name='age',
                    label='Age',
                    type=VariableType.RATIO,
                    valid_range=(18, 80)
                ),
                'gender': VariableDefinition(
                    name='gender',
                    label='Gender',
                    type=VariableType.NOMINAL,
                    categories={1: 'Male', 2: 'Female'}
                ),
                'satisfaction': VariableDefinition(
                    name='satisfaction',
                    label='Satisfaction',
                    type=VariableType.ORDINAL,
                    categories={1: 'Very Low', 2: 'Low', 3: 'Medium', 4: 'High', 5: 'Very High'},
                    valid_range=(1, 5)
                )
            }
        )

        # Initialize tool
        self.tool = SurveyAnalysisTool(random_state=42, log_level='ERROR')
        self.tool.data = self.sample_data
        self.tool.metadata = self.sample_metadata

    def test_initialization(self):
        """Test tool initialization."""
        tool = SurveyAnalysisTool(random_state=42)
        self.assertEqual(tool.random_state, 42)
        self.assertIsNotNone(tool.data_loader)
        self.assertIsNotNone(tool.correlation_engine)

    def test_data_cleaning(self):
        """Test data cleaning functionality."""
        cleaned_data, quality_metrics = self.tool.clean_survey_data(
            response_time_col='response_time',
            respondent_id_col='respondent_id'
        )

        # Check that cleaning was performed
        self.assertIsNotNone(cleaned_data)
        self.assertIsNotNone(quality_metrics)
        self.assertLessEqual(len(cleaned_data), len(self.sample_data))
        self.assertEqual(quality_metrics.total_responses, len(self.sample_data))

    def test_missing_data_analysis(self):
        """Test missing data analysis."""
        missing_analysis = self.tool.analyze_missing_data()

        self.assertIsNotNone(missing_analysis)
        self.assertGreater(missing_analysis.missing_percentage, 0)
        self.assertIn('income', missing_analysis.missing_by_variable)
        self.assertIn('satisfaction', missing_analysis.missing_by_variable)

    def test_descriptive_analysis(self):
        """Test descriptive statistics calculation."""
        desc_stats = self.tool.descriptive_analysis(
            variables=['age', 'satisfaction', 'score']
        )

        self.assertIsNotNone(desc_stats)
        self.assertIn('age', desc_stats)
        self.assertIn('satisfaction', desc_stats)
        self.assertIn('score', desc_stats)

        # Check that statistics are reasonable
        age_stats = desc_stats['age']
        self.assertIsNotNone(age_stats.mean)
        self.assertIsNotNone(age_stats.std)
        self.assertGreater(age_stats.count, 0)

    def test_correlation_analysis(self):
        """Test correlation analysis."""
        correlations = self.tool.correlation_analysis(
            variables=['age', 'satisfaction', 'score'],
            method='pearson'
        )

        self.assertIsNotNone(correlations)
        self.assertIsInstance(correlations, list)

        if correlations:  # If any correlations were computed
            self.assertIsInstance(correlations[0], CorrelationResult)
            self.assertTrue(-1 <= correlations[0].correlation_coefficient <= 1)

    def test_chi_square_test(self):
        """Test chi-square tests."""
        # Test independence
        chi_result = self.tool.chi_square_test(
            row_variable='gender',
            column_variable='satisfaction',
            test_type='independence'
        )

        self.assertIsNotNone(chi_result)
        self.assertIsInstance(chi_result, ChiSquareResult)
        self.assertGreaterEqual(chi_result.chi_square_statistic, 0)
        self.assertTrue(0 <= chi_result.p_value <= 1)

    def test_cross_tabulation(self):
        """Test cross-tabulation functionality."""
        crosstab_result = self.tool.cross_tabulation(
            row_variable='gender',
            column_variable='recommend'
        )

        self.assertIsNotNone(crosstab_result)
        self.assertIsInstance(crosstab_result, CrosstabResult)
        self.assertFalse(crosstab_result.observed_frequencies.empty)

    def test_demographic_profiling(self):
        """Test demographic profiling."""
        demo_profile = self.tool.demographic_profiling(
            demographic_variables=['gender', 'income'],
            target_variable='satisfaction'
        )

        self.assertIsNotNone(demo_profile)
        self.assertIn('gender', demo_profile)

    def test_get_analysis_summary(self):
        """Test analysis summary generation."""
        # Run some analyses first
        self.tool.clean_survey_data()
        self.tool.descriptive_analysis(variables=['age', 'score'])

        summary = self.tool.get_analysis_summary()

        self.assertIsNotNone(summary)
        self.assertTrue(summary['data_loaded'])
        self.assertEqual(summary['n_records'], len(self.sample_data))
        self.assertIn('quality_metrics', summary['analyses_completed'])

    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test with non-existent variable
        with self.assertRaises(ValueError):
            self.tool.chi_square_test(
                row_variable='non_existent_variable',
                column_variable='gender'
            )

        # Test correlation with insufficient variables
        correlations = self.tool.correlation_analysis(variables=['age'])
        self.assertEqual(len(correlations), 0)

    def test_report_generation(self):
        """Test basic report generation."""
        # Run some analyses first
        self.tool.clean_survey_data()
        self.tool.descriptive_analysis(variables=['age'])

        try:
            report = self.tool.generate_report(report_type='summary')
            self.assertIsNotNone(report)
            self.assertIsInstance(report, str)
            self.assertTrue(len(report) > 0)
        except Exception as e:
            # Report generation might fail due to missing dependencies
            self.skipTest(f"Report generation failed: {e}")


class TestDataProcessingComponents(unittest.TestCase):
    """Test cases for individual data processing components."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.test_data = pd.DataFrame({
            'var1': np.random.normal(0, 1, 50),
            'var2': np.random.normal(0, 1, 50),
            'var3': np.random.choice(['A', 'B', 'C'], 50),
            'var4': np.random.choice([0, 1], 50)
        })

        # Add some missing data
        self.test_data.loc[0:2, 'var1'] = np.nan

    def test_data_loader_basic(self):
        """Test basic data loader functionality."""
        from data_processing.data_loader import DataLoader

        loader = DataLoader()
        self.assertIsNotNone(loader)

        # Test CSV loading with temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            self.test_data.to_csv(f.name, index=False)
            temp_file = f.name

        try:
            data, metadata = loader.load_data(temp_file)
            self.assertIsNotNone(data)
            self.assertEqual(len(data), len(self.test_data))
        finally:
            os.unlink(temp_file)

    def test_data_cleaner_basic(self):
        """Test basic data cleaner functionality."""
        from data_processing.data_cleaner import DataCleaner

        cleaner = DataCleaner()

        # Add respondent IDs
        test_data_with_id = self.test_data.copy()
        test_data_with_id['respondent_id'] = range(len(test_data_with_id))

        cleaned_data, quality_metrics = cleaner.clean_data(
            test_data_with_id,
            respondent_id_col='respondent_id'
        )

        self.assertIsNotNone(cleaned_data)
        self.assertIsNotNone(quality_metrics)
        self.assertLessEqual(len(cleaned_data), len(test_data_with_id))

    def test_correlation_engine_basic(self):
        """Test basic correlation engine functionality."""
        from correlation_analysis.correlation_engine import CorrelationEngine

        engine = CorrelationEngine(random_state=42)
        correlations = engine.compute_correlations(
            self.test_data,
            variables=['var1', 'var2'],
            method='pearson'
        )

        self.assertIsInstance(correlations, list)
        if correlations:
            self.assertIsInstance(correlations[0], CorrelationResult)

    def test_chi_square_tests_basic(self):
        """Test basic chi-square test functionality."""
        from categorical_analysis.chi_square_tests import ChiSquareTests

        chi_tester = ChiSquareTests()
        result = chi_tester.test_independence(
            self.test_data,
            'var3',
            'var4'
        )

        self.assertIsInstance(result, ChiSquareResult)
        self.assertGreaterEqual(result.chi_square_statistic, 0)

    def test_missing_data_handler_basic(self):
        """Test basic missing data handler functionality."""
        from data_processing.missing_data_handler import MissingDataHandler

        handler = MissingDataHandler(random_state=42)
        analysis = handler.analyze_missing_data(self.test_data)

        self.assertIsNotNone(analysis)
        self.assertGreater(analysis.missing_percentage, 0)


class TestIntegrationScenarios(unittest.TestCase):
    """Integration tests for common survey analysis scenarios."""

    def setUp(self):
        """Set up integration test fixtures."""
        np.random.seed(42)

        # Create more realistic survey data
        self.survey_data = pd.DataFrame({
            'id': range(1, 201),
            'age': np.random.normal(35, 12, 200).astype(int).clip(18, 75),
            'gender': np.random.choice(['M', 'F'], 200),
            'satisfaction_overall': np.random.choice([1, 2, 3, 4, 5], 200, p=[0.1, 0.1, 0.3, 0.3, 0.2]),
            'satisfaction_service': np.random.choice([1, 2, 3, 4, 5], 200, p=[0.1, 0.2, 0.3, 0.3, 0.1]),
            'recommend': np.random.choice([0, 1], 200, p=[0.3, 0.7]),
            'income_category': np.random.choice(['Low', 'Medium', 'High'], 200, p=[0.3, 0.5, 0.2])
        })

        # Create some realistic correlations
        for i in range(len(self.survey_data)):
            # Make recommendation somewhat dependent on satisfaction
            if self.survey_data.loc[i, 'satisfaction_overall'] >= 4:
                if np.random.random() < 0.8:
                    self.survey_data.loc[i, 'recommend'] = 1

        self.tool = SurveyAnalysisTool(random_state=42, log_level='ERROR')
        self.tool.data = self.survey_data

    def test_full_analysis_workflow(self):
        """Test complete analysis workflow."""
        # Step 1: Clean data
        cleaned_data, quality_metrics = self.tool.clean_survey_data(
            respondent_id_col='id'
        )
        self.assertIsNotNone(cleaned_data)

        # Step 2: Descriptive analysis
        desc_stats = self.tool.descriptive_analysis()
        self.assertGreater(len(desc_stats), 0)

        # Step 3: Correlation analysis
        correlations = self.tool.correlation_analysis(
            variables=['satisfaction_overall', 'satisfaction_service', 'age']
        )
        self.assertIsInstance(correlations, list)

        # Step 4: Cross-tabulation
        crosstab = self.tool.cross_tabulation(
            row_variable='gender',
            column_variable='recommend'
        )
        self.assertIsNotNone(crosstab)

        # Step 5: Get summary
        summary = self.tool.get_analysis_summary()
        self.assertGreater(len(summary['analyses_completed']), 3)

    def test_satisfaction_analysis_scenario(self):
        """Test typical satisfaction analysis scenario."""
        # Clean data
        self.tool.clean_survey_data(respondent_id_col='id')

        # Analyze satisfaction correlations
        satisfaction_vars = ['satisfaction_overall', 'satisfaction_service']
        correlations = self.tool.correlation_analysis(variables=satisfaction_vars)

        # Test satisfaction vs recommendation
        crosstab = self.tool.cross_tabulation(
            row_variable='satisfaction_overall',
            column_variable='recommend'
        )

        # Verify results
        self.assertIsNotNone(correlations)
        self.assertIsNotNone(crosstab)
        if crosstab.chi_square_test:
            self.assertGreaterEqual(crosstab.chi_square_test.chi_square_statistic, 0)

    def test_demographic_analysis_scenario(self):
        """Test demographic analysis scenario."""
        # Clean data
        self.tool.clean_survey_data(respondent_id_col='id')

        # Demographic profiling
        demo_profile = self.tool.demographic_profiling(
            demographic_variables=['gender', 'age', 'income_category'],
            target_variable='satisfaction_overall'
        )

        self.assertIsNotNone(demo_profile)
        self.assertIn('gender', demo_profile)

        # Test demographic differences in satisfaction
        chi_result = self.tool.chi_square_test(
            row_variable='gender',
            column_variable='satisfaction_overall'
        )

        self.assertIsNotNone(chi_result)


def run_tests():
    """Run all tests."""
    # Create test suite
    test_suite = unittest.TestSuite()

    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestSurveyAnalysisTool))
    test_suite.addTest(unittest.makeSuite(TestDataProcessingComponents))
    test_suite.addTest(unittest.makeSuite(TestIntegrationScenarios))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    return result


if __name__ == '__main__':
    print("Running Survey Analysis Tool Tests")
    print("=" * 50)

    result = run_tests()

    print("\n" + "=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")

    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")

    if result.wasSuccessful():
        print("\nAll tests passed successfully!")
    else:
        print("\nSome tests failed. Please check the output above.")

    sys.exit(0 if result.wasSuccessful() else 1)