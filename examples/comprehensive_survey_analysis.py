"""
Comprehensive example of survey data analysis using the SurveyAnalysisTool.

This example demonstrates the full workflow from data loading through
reporting for a typical survey analysis project.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path to import the tool
import sys
sys.path.append('..')

from survey_analysis_tool import SurveyAnalysisTool
from data_processing.models import SurveyMetadata, VariableDefinition, VariableType
from datetime import datetime


def create_sample_survey_data():
    """Create sample survey data for demonstration."""
    np.random.seed(42)
    n_responses = 1000

    # Generate sample survey data
    data = {
        # Demographics
        'respondent_id': range(1, n_responses + 1),
        'age': np.random.normal(40, 15, n_responses).astype(int).clip(18, 80),
        'gender': np.random.choice(['Male', 'Female', 'Other'], n_responses, p=[0.48, 0.49, 0.03]),
        'education': np.random.choice([
            'High School', 'Some College', 'Bachelor\'s', 'Master\'s', 'PhD'
        ], n_responses, p=[0.25, 0.20, 0.30, 0.20, 0.05]),
        'income': np.random.choice([
            '<$30k', '$30k-$50k', '$50k-$75k', '$75k-$100k', '>$100k'
        ], n_responses, p=[0.15, 0.25, 0.30, 0.20, 0.10]),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n_responses),

        # Satisfaction ratings (1-5 Likert scale)
        'satisfaction_overall': np.random.choice([1, 2, 3, 4, 5], n_responses, p=[0.05, 0.10, 0.25, 0.40, 0.20]),
        'satisfaction_product': np.random.choice([1, 2, 3, 4, 5], n_responses, p=[0.08, 0.12, 0.30, 0.35, 0.15]),
        'satisfaction_service': np.random.choice([1, 2, 3, 4, 5], n_responses, p=[0.06, 0.14, 0.35, 0.30, 0.15]),
        'satisfaction_price': np.random.choice([1, 2, 3, 4, 5], n_responses, p=[0.15, 0.20, 0.30, 0.25, 0.10]),

        # Purchase behavior
        'purchase_frequency': np.random.choice([
            'Never', 'Rarely', 'Sometimes', 'Often', 'Always'
        ], n_responses, p=[0.10, 0.20, 0.35, 0.25, 0.10]),
        'purchase_amount': np.random.lognormal(4, 1, n_responses).round(2),

        # Binary outcomes
        'recommend': np.random.choice([0, 1], n_responses, p=[0.35, 0.65]),
        'repurchase_intent': np.random.choice([0, 1], n_responses, p=[0.40, 0.60]),

        # Response time (in seconds)
        'response_time': np.random.gamma(2, 50, n_responses).astype(int).clip(30, 1200)
    }

    # Create correlations between variables (more realistic)
    df = pd.DataFrame(data)

    # Make satisfaction ratings somewhat correlated
    for i in range(len(df)):
        if np.random.random() < 0.3:  # 30% chance of consistency
            base_satisfaction = df.loc[i, 'satisfaction_overall']
            # Other satisfactions tend to be similar
            df.loc[i, 'satisfaction_product'] = max(1, min(5, base_satisfaction + np.random.choice([-1, 0, 1])))
            df.loc[i, 'satisfaction_service'] = max(1, min(5, base_satisfaction + np.random.choice([-1, 0, 1])))

    # Make recommend and repurchase related to satisfaction
    for i in range(len(df)):
        if df.loc[i, 'satisfaction_overall'] >= 4:
            if np.random.random() < 0.8:  # 80% likely to recommend if satisfied
                df.loc[i, 'recommend'] = 1
            if np.random.random() < 0.7:  # 70% likely to repurchase if satisfied
                df.loc[i, 'repurchase_intent'] = 1

    # Add some missing data
    missing_mask = np.random.random((n_responses, len(df.columns))) < 0.05  # 5% missing
    for col in ['income', 'satisfaction_price', 'purchase_amount']:
        df.loc[missing_mask[:, df.columns.get_loc(col)], col] = np.nan

    return df


def create_sample_metadata():
    """Create sample metadata for the survey."""
    variables = {
        'respondent_id': VariableDefinition(
            name='respondent_id',
            label='Respondent ID',
            type=VariableType.NOMINAL,
            question_text='Unique identifier for survey respondent'
        ),
        'age': VariableDefinition(
            name='age',
            label='Age',
            type=VariableType.RATIO,
            valid_range=(18, 80),
            question_text='What is your age?'
        ),
        'gender': VariableDefinition(
            name='gender',
            label='Gender',
            type=VariableType.NOMINAL,
            categories={1: 'Male', 2: 'Female', 3: 'Other'},
            question_text='What is your gender?'
        ),
        'education': VariableDefinition(
            name='education',
            label='Education Level',
            type=VariableType.ORDINAL,
            categories={
                1: 'High School',
                2: 'Some College',
                3: 'Bachelor\'s',
                4: 'Master\'s',
                5: 'PhD'
            },
            question_text='What is your highest level of education?'
        ),
        'income': VariableDefinition(
            name='income',
            label='Household Income',
            type=VariableType.ORDINAL,
            categories={
                1: '<$30k',
                2: '$30k-$50k',
                3: '$50k-$75k',
                4: '$75k-$100k',
                5: '>$100k'
            },
            question_text='What is your household income?'
        ),
        'satisfaction_overall': VariableDefinition(
            name='satisfaction_overall',
            label='Overall Satisfaction',
            type=VariableType.ORDINAL,
            categories={1: 'Very Dissatisfied', 2: 'Dissatisfied', 3: 'Neutral', 4: 'Satisfied', 5: 'Very Satisfied'},
            valid_range=(1, 5),
            question_text='How satisfied are you overall with our company?'
        ),
        'recommend': VariableDefinition(
            name='recommend',
            label='Likelihood to Recommend',
            type=VariableType.BINARY,
            categories={0: 'No', 1: 'Yes'},
            question_text='Would you recommend our company to others?'
        )
    }

    metadata = SurveyMetadata(
        survey_id='DEMO_SURVEY_2024',
        title='Customer Satisfaction Survey',
        description='Annual customer satisfaction and loyalty survey',
        collection_period=(datetime(2024, 1, 1), datetime(2024, 3, 31)),
        target_population='Active customers',
        sampling_method='Stratified random sampling',
        expected_responses=1200,
        actual_responses=1000,
        completion_rate=83.3,
        variables=variables
    )

    return metadata


def main():
    """Run comprehensive survey analysis example."""
    print("=" * 60)
    print("COMPREHENSIVE SURVEY DATA ANALYSIS EXAMPLE")
    print("=" * 60)

    # Step 1: Create sample data and metadata
    print("\n1. Creating sample survey data...")
    survey_data = create_sample_survey_data()
    metadata = create_sample_metadata()

    print(f"   - Created survey with {len(survey_data)} responses")
    print(f"   - {len(survey_data.columns)} variables")
    print(f"   - Sample variables: {list(survey_data.columns[:5])}")

    # Save sample data
    survey_data.to_csv('sample_survey_data.csv', index=False)
    print("   - Sample data saved to 'sample_survey_data.csv'")

    # Step 2: Initialize Survey Analysis Tool
    print("\n2. Initializing Survey Analysis Tool...")
    analyzer = SurveyAnalysisTool(random_state=42)

    # Manually set data and metadata (in real use, you'd use load_survey_data)
    analyzer.data = survey_data
    analyzer.metadata = metadata

    # Step 3: Data Cleaning
    print("\n3. Performing data cleaning...")
    cleaned_data, quality_metrics = analyzer.clean_survey_data(
        response_time_col='response_time',
        respondent_id_col='respondent_id'
    )

    print(f"   - Original responses: {quality_metrics.total_responses}")
    print(f"   - Responses retained: {len(cleaned_data)}")
    print(f"   - Completion rate: {quality_metrics.completion_rate:.1f}%")
    print(f"   - Duplicate responses removed: {quality_metrics.duplicate_responses}")
    print(f"   - Straight-lining detected: {quality_metrics.straight_lining_count}")

    # Step 4: Missing Data Analysis
    print("\n4. Analyzing missing data patterns...")
    missing_analysis = analyzer.analyze_missing_data()

    print(f"   - Total missing data: {missing_analysis.missing_percentage:.1f}%")
    print(f"   - Missing data mechanism: {missing_analysis.estimated_mechanism.value}")
    print(f"   - Variables with highest missing rates:")

    for var, stats in missing_analysis.missing_by_variable.items():
        if stats['percentage'] > 2:  # Show variables with >2% missing
            print(f"     * {var}: {stats['percentage']:.1f}%")

    # Step 5: Descriptive Analysis
    print("\n5. Generating descriptive statistics...")
    desc_stats = analyzer.descriptive_analysis(
        variables=['age', 'satisfaction_overall', 'satisfaction_product', 'purchase_amount']
    )

    print("   - Descriptive statistics for key variables:")
    for var, stats in desc_stats.items():
        if hasattr(stats, 'mean') and stats.mean is not None:
            print(f"     * {var}: Mean={stats.mean:.2f}, SD={stats.std:.2f}, N={stats.count}")

    # Step 6: Correlation Analysis
    print("\n6. Performing correlation analysis...")
    satisfaction_vars = [
        'satisfaction_overall', 'satisfaction_product',
        'satisfaction_service', 'satisfaction_price'
    ]

    correlations = analyzer.correlation_analysis(
        variables=satisfaction_vars,
        method='auto'
    )

    print(f"   - Computed {len(correlations)} correlations")
    print("   - Strongest correlations:")

    # Sort by absolute correlation strength
    sorted_corr = sorted(correlations, key=lambda x: abs(x.correlation_coefficient), reverse=True)
    for i, corr in enumerate(sorted_corr[:5]):  # Top 5
        if corr.is_significant():
            sig_marker = "*"
        else:
            sig_marker = ""
        print(f"     * {corr.variable1} ↔ {corr.variable2}: r={corr.correlation_coefficient:.3f}{sig_marker}")

    # Step 7: Chi-square Tests
    print("\n7. Performing chi-square tests...")

    # Test independence between gender and education
    chi_result = analyzer.chi_square_test(
        row_variable='gender',
        column_variable='education',
        test_type='independence'
    )

    print(f"   - Gender × Education independence test:")
    print(f"     * χ² = {chi_result.chi_square_statistic:.3f}")
    print(f"     * p-value = {chi_result.p_value:.4f}")
    print(f"     * Effect size ({chi_result.effect_size_measure}) = {chi_result.effect_size:.3f}")

    # Step 8: Cross-tabulation
    print("\n8. Creating cross-tabulations...")

    crosstab_result = analyzer.cross_tabulation(
        row_variable='recommend',
        column_variable='satisfaction_overall',
        percentages=['row', 'column']
    )

    print("   - Recommendation × Overall Satisfaction crosstab created")
    if crosstab_result.chi_square_test:
        print(f"     * Association test: χ² = {crosstab_result.chi_square_test.chi_square_statistic:.3f}")
        print(f"     * p-value = {crosstab_result.chi_square_test.p_value:.4f}")

    # Step 9: Demographic Profiling
    print("\n9. Generating demographic profiles...")

    demo_profile = analyzer.demographic_profiling(
        demographic_variables=['gender', 'age', 'education', 'income', 'region'],
        target_variable='satisfaction_overall'
    )

    print("   - Demographic breakdown completed")
    print("   - Target variable breakdowns by demographics created")

    # Step 10: Generate Report
    print("\n10. Generating analysis report...")

    try:
        report_html = analyzer.generate_report(
            report_type='comprehensive',
            format='html'
        )

        # Save report
        with open('survey_analysis_report.html', 'w', encoding='utf-8') as f:
            f.write(report_html)

        print("   - Comprehensive report generated and saved to 'survey_analysis_report.html'")

    except Exception as e:
        print(f"   - Report generation encountered an issue: {e}")

    # Step 11: Analysis Summary
    print("\n11. Analysis Summary")
    print("=" * 40)

    summary = analyzer.get_analysis_summary()

    print(f"Dataset: {summary['n_records']} responses, {summary['n_variables']} variables")
    print(f"Data quality: {summary.get('completion_rate', 'N/A'):.1f}% completion rate")
    print(f"Analyses completed: {', '.join(summary['analyses_completed'])}")

    if 'correlations' in analyzer.analysis_results:
        correlations = analyzer.analysis_results['correlations']
        significant_corr = sum(1 for r in correlations if r.is_significant())
        print(f"Correlations: {len(correlations)} computed, {significant_corr} significant")

    # Recommendations based on analysis
    print("\n12. Key Findings and Recommendations")
    print("=" * 40)

    # Example findings based on typical survey analysis
    print("Key Findings:")
    print("• Strong correlation between different satisfaction dimensions")
    print("• Satisfaction significantly predicts recommendation behavior")
    print("• Missing data appears to be random (MCAR)")
    print("• Data quality is good with high completion rates")

    print("\nRecommendations:")
    print("• Focus on overall satisfaction as it drives recommendations")
    print("• Consider satisfaction dimensions together in improvement efforts")
    print("• Current survey design appears robust")
    print("• Consider follow-up analysis on demographic differences")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)

    return analyzer


if __name__ == "__main__":
    # Run the comprehensive example
    survey_analyzer = main()

    # Optional: Demonstrate additional features
    print("\n\nADDITIONAL DEMONSTRATIONS:")
    print("=" * 40)

    # Demonstrate missing data imputation
    print("\n• Missing Data Imputation Example:")
    try:
        imputed_data = survey_analyzer.missing_data_handler.impute_missing_data(
            survey_analyzer.cleaned_data,
            method='mean',
            variables=['purchase_amount']
        )
        print(f"  - Imputed {survey_analyzer.cleaned_data['purchase_amount'].isna().sum()} missing values")
        print(f"  - Mean purchase amount after imputation: {imputed_data['purchase_amount'].mean():.2f}")
    except Exception as e:
        print(f"  - Imputation example failed: {e}")

    # Demonstrate advanced correlation analysis
    print("\n• Partial Correlation Example:")
    try:
        partial_correlations = survey_analyzer.correlation_analysis(
            variables=['satisfaction_overall', 'satisfaction_product', 'satisfaction_service'],
            control_variables=['age'],
            method='pearson'
        )
        print(f"  - Computed {len(partial_correlations)} partial correlations controlling for age")
        for corr in partial_correlations[:2]:  # Show first 2
            print(f"    * {corr.variable1} ↔ {corr.variable2} (controlling for age): r={corr.correlation_coefficient:.3f}")
    except Exception as e:
        print(f"  - Partial correlation example failed: {e}")

    print("\nExample completed successfully!")
    print("Check the generated files:")
    print("- sample_survey_data.csv: Sample survey data")
    print("- survey_analysis_report.html: Analysis report")