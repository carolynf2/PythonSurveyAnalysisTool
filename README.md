# Survey Data Analysis Tool

A comprehensive Python toolkit for automated survey data analysis, from data cleaning and quality assessment to advanced statistical analysis, correlation studies, and cross-tabulation.

## Overview

The Survey Data Analysis Tool provides a complete workflow for survey research, integrating data processing, statistical analysis, and reporting in a single, easy-to-use package. It's designed for researchers, analysts, and professionals who need robust, automated survey analysis capabilities.

## Features

### 🔧 **Data Processing**
- **Multi-format support**: CSV, Excel, SPSS, SAS, Stata, JSON
- **Automated data cleaning**: Response validation, duplicate detection, outlier identification
- **Quality assessment**: Straight-lining detection, speeding analysis, completion rate calculation
- **Missing data analysis**: Pattern detection, mechanism testing (MCAR/MAR/MNAR)
- **Multiple imputation**: Mean/median, KNN, MICE, Random Forest methods

### 📊 **Statistical Analysis**
- **Descriptive statistics**: Central tendency, variability, distribution shape with confidence intervals
- **Correlation analysis**: Pearson, Spearman, Kendall, polychoric, tetrachoric, partial correlations
- **Chi-square tests**: Independence, goodness-of-fit, homogeneity testing
- **Cross-tabulation**: Weighted tables, association measures, multi-way analysis
- **Survey weighting**: Design effects, effective sample sizes, complex sampling support

### 📈 **Advanced Features**
- **Factor analysis**: Exploratory and confirmatory analysis
- **Reliability analysis**: Cronbach's alpha, item analysis
- **Demographic profiling**: Population breakdowns, quota analysis
- **Effect size calculations**: Cohen's conventions, practical significance
- **Bootstrap confidence intervals**: Robust uncertainty estimation

### 📋 **Reporting & Visualization**
- **Interactive visualizations**: Correlation matrices, network diagrams, heatmaps
- **Professional reports**: HTML, PDF, Markdown formats
- **Executive summaries**: Key findings and recommendations
- **Comprehensive documentation**: Statistical methodology and assumptions

## Quick Start

### Installation

```bash
# Clone the repository
git clone [repository-url]
cd SurveyAnalysisTool

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from survey_analysis_tool import SurveyAnalysisTool

# Initialize the tool
analyzer = SurveyAnalysisTool()

# Load survey data
data = analyzer.load_survey_data('survey_data.csv')

# Clean the data
cleaned_data, quality_metrics = analyzer.clean_survey_data()

# Perform descriptive analysis
desc_stats = analyzer.descriptive_analysis()

# Run correlation analysis
correlations = analyzer.correlation_analysis(method='auto')

# Create cross-tabulations
crosstab = analyzer.cross_tabulation('gender', 'satisfaction')

# Generate comprehensive report
report = analyzer.generate_report(format='html')\n```

### Example Analysis

See `examples/comprehensive_survey_analysis.py` for a complete workflow demonstration including:

- Loading and cleaning survey data
- Missing data analysis and imputation
- Descriptive statistics calculation
- Correlation and association analysis
- Chi-square testing and cross-tabulation
- Demographic profiling
- Professional report generation

## Project Structure

```
SurveyAnalysisTool/
├── data_processing/           # Data loading, cleaning, and validation
│   ├── data_loader.py        # Multi-format data ingestion
│   ├── data_cleaner.py       # Automated cleaning and validation
│   ├── missing_data_handler.py # Missing data analysis and imputation
│   └── models.py             # Core data structures
├── descriptive_analysis/     # Univariate and descriptive statistics
│   └── univariate_stats.py   # Comprehensive descriptive analysis
├── correlation_analysis/     # Correlation and association analysis
│   └── correlation_engine.py # Multiple correlation methods
├── categorical_analysis/     # Categorical data analysis
│   ├── chi_square_tests.py   # Chi-square and exact tests
│   ├── cross_tabulation.py   # Contingency table analysis
│   └── association_measures.py # Association strength measures
├── visualization/            # Plotting and visualization tools
│   └── correlation_plots.py  # Correlation visualizations
├── examples/                 # Usage examples and tutorials
│   └── comprehensive_survey_analysis.py
├── tests/                    # Unit and integration tests
│   └── test_survey_analysis_tool.py
└── survey_analysis_tool.py   # Main API interface
```

## Key Components

### SurveyAnalysisTool (Main Interface)
The primary class that orchestrates all analysis components:

```python
# Initialize with custom configuration
analyzer = SurveyAnalysisTool(
    random_state=42,
    log_level='INFO'
)

# Complete analysis workflow
analyzer.load_survey_data('data.csv', metadata_path='metadata.json')
analyzer.clean_survey_data(auto_clean=True)
analyzer.analyze_missing_data()
analyzer.descriptive_analysis(group_by='demographic')
analyzer.correlation_analysis(method='auto', control_variables=['age'])
analyzer.generate_report(report_type='comprehensive')
```

### Data Processing Pipeline
1. **Data Loading**: Automatic format detection and metadata extraction
2. **Data Cleaning**: Quality assessment and automated cleaning
3. **Missing Data**: Pattern analysis and imputation strategies
4. **Validation**: Range checks, logic validation, and assumption testing

### Statistical Analysis Engine
- **Automatic method selection**: Chooses appropriate tests based on variable types
- **Effect size calculation**: Practical significance beyond statistical significance
- **Assumption checking**: Validates statistical prerequisites with warnings
- **Multiple testing correction**: Controls family-wise error rates

## Statistical Methods

### Correlation Analysis
- **Pearson**: Linear relationships between continuous variables
- **Spearman**: Rank-based correlation for ordinal/non-normal data
- **Kendall's τ**: Alternative rank correlation with different properties
- **Point-biserial**: Continuous-binary variable associations
- **Polychoric/Tetrachoric**: Latent continuous variable correlations
- **Partial**: Correlation controlling for confounding variables

### Categorical Analysis
- **Chi-square independence**: Association between categorical variables
- **Chi-square goodness-of-fit**: Compare observed vs expected distributions
- **Fisher's exact**: Exact tests for small samples
- **McNemar's test**: Paired categorical data analysis
- **Association measures**: Cramer's V, phi, lambda, gamma coefficients

### Missing Data Methods
- **Mechanism testing**: Little's MCAR test, pattern analysis
- **Simple imputation**: Mean, median, mode replacement
- **Advanced imputation**: MICE, KNN, Random Forest methods
- **Sensitivity analysis**: Impact assessment of missing data handling

## Survey-Specific Features

### Complex Survey Design Support
- **Survey weights**: Post-stratification, raking, propensity weights
- **Design effects**: Efficiency loss due to complex sampling
- **Effective sample sizes**: Weighted sample size calculations
- **Stratified analysis**: Multi-group comparisons with proper statistics

### Quality Assessment
- **Response time analysis**: Speeding detection and quality flags
- **Straight-lining detection**: Satisficing behavior identification
- **Completion rate analysis**: Partial vs complete response patterns
- **Item non-response**: Question-level missing data assessment

## Requirements

### Core Dependencies
- Python 3.8+
- pandas >= 1.5.0
- numpy >= 1.21.0
- scipy >= 1.9.0
- statsmodels >= 0.13.0
- scikit-learn >= 1.1.0

### Statistical Packages
- pingouin >= 0.5.0 (advanced statistical tests)
- factor-analyzer >= 0.4.0 (factor analysis)
- weightedstats >= 0.4.0 (weighted statistics)

### Visualization
- plotly >= 5.0.0 (interactive plots)
- matplotlib >= 3.5.0 (static plots)
- seaborn >= 0.11.0 (statistical visualizations)

### Optional Dependencies
- pyreadstat >= 1.1.0 (SPSS/SAS/Stata support)
- rpy2 >= 3.5.0 (R integration)
- fancyimpute >= 0.7.0 (advanced imputation)

## Testing

Run the comprehensive test suite:

```bash
cd tests
python test_survey_analysis_tool.py
```

The test suite includes:
- Unit tests for individual components
- Integration tests for complete workflows
- Statistical validation tests
- Error handling and edge case tests

## Examples and Tutorials

### Basic Analysis
```python
# Quick start example
analyzer = SurveyAnalysisTool()
data = analyzer.load_survey_data('survey.csv')
analyzer.clean_survey_data()
stats = analyzer.descriptive_analysis()
corrs = analyzer.correlation_analysis()
report = analyzer.generate_report()
```

### Advanced Workflow
See `examples/comprehensive_survey_analysis.py` for:
- Complete survey analysis workflow
- Advanced statistical testing
- Custom visualization creation
- Professional report generation

### Custom Analysis
```python
# Custom correlation analysis with controls
correlations = analyzer.correlation_analysis(
    variables=['satisfaction_*'],  # Pattern matching
    method='spearman',
    control_variables=['age', 'gender'],
    weights='survey_weight'
)

# Stratified cross-tabulation
crosstab = analyzer.cross_tabulation(
    row_variable='product_rating',
    column_variable='recommend',
    control_variables=['customer_segment'],
    percentages=['row', 'column', 'total']
)
```

## Contributing

We welcome contributions! Please see our contributing guidelines for:
- Code style and standards
- Testing requirements
- Documentation expectations
- Feature request process

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this tool in academic research, please cite:

```
Survey Data Analysis Tool (2024). Comprehensive Python toolkit for automated survey analysis.
Version 1.0.0. https://github.com/[repository-url]
```

## Support

For questions, bug reports, or feature requests:
- Create an issue on GitHub
- Check the documentation in the `docs/` directory
- Review examples in the `examples/` directory

## Changelog

### Version 1.0.0 (2024)
- Initial release
- Complete survey analysis workflow
- Multi-format data support
- Comprehensive statistical testing
- Professional reporting capabilities
- Extensive documentation and examples

---

**Survey Data Analysis Tool** - Making survey research more efficient, accurate, and accessible.