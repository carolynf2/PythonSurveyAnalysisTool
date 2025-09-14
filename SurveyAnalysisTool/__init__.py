"""
Survey Data Analysis Tool

A comprehensive survey data analysis tool that automates the complete survey analysis
workflow: from data cleaning and quality assessment to advanced statistical analysis,
correlation studies, and cross-tabulation.

Author: Claude Code Assistant
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Claude Code Assistant"

from .survey_analysis_tool import SurveyAnalysisTool
from .advanced_analysis import AdvancedSurveyAnalysis
from .data_processing.models import (
    VariableType,
    MissingDataPattern,
    SurveyMetadata,
    VariableDefinition,
    CorrelationResult,
    ChiSquareResult,
    CrosstabResult
)

__all__ = [
    'SurveyAnalysisTool',
    'AdvancedSurveyAnalysis',
    'VariableType',
    'MissingDataPattern',
    'SurveyMetadata',
    'VariableDefinition',
    'CorrelationResult',
    'ChiSquareResult',
    'CrosstabResult'
]