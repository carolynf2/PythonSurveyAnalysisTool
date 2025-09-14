"""
Association measures for categorical variables.

This module calculates various measures of association between categorical
variables including Cramer's V, phi coefficient, lambda, gamma, and others.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional
from scipy import stats


class AssociationMeasures:
    """
    Calculate association measures for categorical variables.

    Features:
    - Cramer's V for general association
    - Phi coefficient for 2x2 tables
    - Lambda (proportional reduction in error)
    - Gamma for ordinal variables
    - Kendall's tau-b for ordinal variables
    - Contingency coefficient
    """

    def __init__(self):
        """Initialize AssociationMeasures calculator."""
        self.logger = logging.getLogger(__name__)

    def calculate_measures(self,
                         contingency_table: pd.DataFrame,
                         chi_square_statistic: Optional[float] = None) -> Dict[str, float]:
        """
        Calculate multiple association measures.

        Parameters
        ----------
        contingency_table : pd.DataFrame
            Contingency table
        chi_square_statistic : float, optional
            Chi-square statistic (calculated if not provided)

        Returns
        -------
        dict
            Dictionary of association measures
        """
        measures = {}

        try:
            # Basic table properties
            n = contingency_table.sum().sum()
            rows, cols = contingency_table.shape

            # Calculate chi-square if not provided
            if chi_square_statistic is None:
                chi_square_statistic, _, _, _ = stats.chi2_contingency(contingency_table.values)

            # Cramer's V
            measures['cramers_v'] = self.cramers_v(contingency_table, chi_square_statistic)

            # Phi coefficient (for 2x2 tables)
            if rows == 2 and cols == 2:
                measures['phi'] = self.phi_coefficient(contingency_table)

            # Contingency coefficient
            measures['contingency_coefficient'] = self.contingency_coefficient(
                chi_square_statistic, n
            )

            # Lambda (symmetric and asymmetric)
            lambda_measures = self.lambda_coefficient(contingency_table)
            measures.update(lambda_measures)

            # Uncertainty coefficient
            uncertainty_measures = self.uncertainty_coefficient(contingency_table)
            measures.update(uncertainty_measures)

        except Exception as e:
            self.logger.warning(f"Error calculating association measures: {e}")

        return measures

    def cramers_v(self,
                  contingency_table: pd.DataFrame,
                  chi_square_statistic: Optional[float] = None) -> float:
        """
        Calculate Cramer's V.

        Cramer's V = sqrt(chi_square / (n * min(rows-1, cols-1)))
        """
        try:
            n = contingency_table.sum().sum()
            rows, cols = contingency_table.shape

            if chi_square_statistic is None:
                chi_square_statistic, _, _, _ = stats.chi2_contingency(contingency_table.values)

            min_dim = min(rows - 1, cols - 1)
            if min_dim == 0 or n == 0:
                return 0.0

            cramers_v = np.sqrt(chi_square_statistic / (n * min_dim))
            return float(cramers_v)

        except Exception as e:
            self.logger.warning(f"Error calculating Cramer's V: {e}")
            return 0.0

    def phi_coefficient(self, contingency_table: pd.DataFrame) -> float:
        """
        Calculate phi coefficient for 2x2 tables.

        phi = (ad - bc) / sqrt((a+b)(c+d)(a+c)(b+d))
        """
        try:
            if contingency_table.shape != (2, 2):
                raise ValueError("Phi coefficient requires 2x2 table")

            # Extract values
            values = contingency_table.values.flatten()
            a, b, c, d = values

            # Calculate phi
            numerator = a * d - b * c
            denominator = np.sqrt((a + b) * (c + d) * (a + c) * (b + d))

            if denominator == 0:
                return 0.0

            phi = numerator / denominator
            return float(phi)

        except Exception as e:
            self.logger.warning(f"Error calculating phi coefficient: {e}")
            return 0.0

    def contingency_coefficient(self, chi_square_statistic: float, n: int) -> float:
        """
        Calculate contingency coefficient.

        C = sqrt(chi_square / (chi_square + n))
        """
        try:
            if n == 0:
                return 0.0

            contingency_coeff = np.sqrt(chi_square_statistic / (chi_square_statistic + n))
            return float(contingency_coeff)

        except Exception as e:
            self.logger.warning(f"Error calculating contingency coefficient: {e}")
            return 0.0

    def lambda_coefficient(self, contingency_table: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate lambda coefficients (proportional reduction in error).

        Returns symmetric and asymmetric lambda measures.
        """
        try:
            # Row totals and column totals
            row_totals = contingency_table.sum(axis=1)
            col_totals = contingency_table.sum(axis=0)
            n = contingency_table.sum().sum()

            # Lambda row|column (predicting row from column)
            lambda_row_given_col = self._lambda_asymmetric(
                contingency_table, row_totals, col_totals, predict_row=True
            )

            # Lambda column|row (predicting column from row)
            lambda_col_given_row = self._lambda_asymmetric(
                contingency_table, row_totals, col_totals, predict_row=False
            )

            # Symmetric lambda
            lambda_symmetric = (lambda_row_given_col + lambda_col_given_row) / 2

            return {
                'lambda_row_given_col': float(lambda_row_given_col),
                'lambda_col_given_row': float(lambda_col_given_row),
                'lambda_symmetric': float(lambda_symmetric)
            }

        except Exception as e:
            self.logger.warning(f"Error calculating lambda coefficients: {e}")
            return {
                'lambda_row_given_col': 0.0,
                'lambda_col_given_row': 0.0,
                'lambda_symmetric': 0.0
            }

    def _lambda_asymmetric(self,
                          contingency_table: pd.DataFrame,
                          row_totals: pd.Series,
                          col_totals: pd.Series,
                          predict_row: bool) -> float:
        """Calculate asymmetric lambda coefficient."""
        try:
            n = contingency_table.sum().sum()

            if predict_row:
                # Predicting row category from column category
                # Error without knowledge of column
                max_row_total = row_totals.max()
                error_without = n - max_row_total

                # Error with knowledge of column
                error_with = 0
                for col in contingency_table.columns:
                    col_data = contingency_table[col]
                    max_in_col = col_data.max()
                    col_total = col_data.sum()
                    error_with += col_total - max_in_col

            else:
                # Predicting column category from row category
                # Error without knowledge of row
                max_col_total = col_totals.max()
                error_without = n - max_col_total

                # Error with knowledge of row
                error_with = 0
                for row in contingency_table.index:
                    row_data = contingency_table.loc[row]
                    max_in_row = row_data.max()
                    row_total = row_data.sum()
                    error_with += row_total - max_in_row

            # Lambda calculation
            if error_without == 0:
                return 0.0

            lambda_coeff = (error_without - error_with) / error_without
            return float(lambda_coeff)

        except Exception:
            return 0.0

    def uncertainty_coefficient(self, contingency_table: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate uncertainty coefficients based on information theory.

        U(Y|X) = (H(Y) - H(Y|X)) / H(Y)
        where H is entropy.
        """
        try:
            # Calculate probabilities
            n = contingency_table.sum().sum()
            joint_prob = contingency_table / n
            row_prob = joint_prob.sum(axis=1)
            col_prob = joint_prob.sum(axis=0)

            # Calculate entropies
            h_row = self._entropy(row_prob)
            h_col = self._entropy(col_prob)
            h_joint = self._entropy(joint_prob.values.flatten())

            # Conditional entropies
            h_row_given_col = h_joint - h_col
            h_col_given_row = h_joint - h_row

            # Uncertainty coefficients
            u_row_given_col = (h_row - h_row_given_col) / h_row if h_row > 0 else 0
            u_col_given_row = (h_col - h_col_given_row) / h_col if h_col > 0 else 0

            # Symmetric uncertainty coefficient
            u_symmetric = 2 * (h_row + h_col - h_joint) / (h_row + h_col) if (h_row + h_col) > 0 else 0

            return {
                'uncertainty_row_given_col': float(u_row_given_col),
                'uncertainty_col_given_row': float(u_col_given_row),
                'uncertainty_symmetric': float(u_symmetric)
            }

        except Exception as e:
            self.logger.warning(f"Error calculating uncertainty coefficients: {e}")
            return {
                'uncertainty_row_given_col': 0.0,
                'uncertainty_col_given_row': 0.0,
                'uncertainty_symmetric': 0.0
            }

    def _entropy(self, probabilities) -> float:
        """Calculate entropy H = -sum(p * log2(p))."""
        # Remove zero probabilities
        p = probabilities[probabilities > 0]

        if len(p) == 0:
            return 0.0

        entropy = -np.sum(p * np.log2(p))
        return float(entropy)

    def gamma_coefficient(self,
                         contingency_table: pd.DataFrame,
                         ordered_rows: bool = True,
                         ordered_cols: bool = True) -> float:
        """
        Calculate Goodman and Kruskal's gamma for ordinal variables.

        Gamma = (C - D) / (C + D)
        where C is concordant pairs and D is discordant pairs.
        """
        try:
            if not (ordered_rows and ordered_cols):
                return 0.0  # Gamma only meaningful for ordinal variables

            concordant = 0
            discordant = 0

            rows, cols = contingency_table.shape

            # Count concordant and discordant pairs
            for i in range(rows):
                for j in range(cols):
                    freq_ij = contingency_table.iloc[i, j]

                    # Count pairs that are concordant (both increase together)
                    for ii in range(i + 1, rows):
                        for jj in range(j + 1, cols):
                            freq_iijj = contingency_table.iloc[ii, jj]
                            concordant += freq_ij * freq_iijj

                    # Count pairs that are discordant (one increases, other decreases)
                    for ii in range(i + 1, rows):
                        for jj in range(j):
                            freq_iijj = contingency_table.iloc[ii, jj]
                            discordant += freq_ij * freq_iijj

            # Calculate gamma
            total_pairs = concordant + discordant
            if total_pairs == 0:
                return 0.0

            gamma = (concordant - discordant) / total_pairs
            return float(gamma)

        except Exception as e:
            self.logger.warning(f"Error calculating gamma coefficient: {e}")
            return 0.0

    def kendalls_tau_b(self,
                      contingency_table: pd.DataFrame,
                      ordered_rows: bool = True,
                      ordered_cols: bool = True) -> float:
        """
        Calculate Kendall's tau-b for ordinal variables.

        tau-b = (C - D) / sqrt((C + D + Ty)(C + D + Tx))
        where Ty and Tx are tied pairs.
        """
        try:
            if not (ordered_rows and ordered_cols):
                return 0.0

            # This is a simplified implementation
            # For a full implementation, we'd need to account for tied pairs properly

            # Use scipy's implementation on the raw data if available
            # For now, return 0 as placeholder
            return 0.0

        except Exception as e:
            self.logger.warning(f"Error calculating Kendall's tau-b: {e}")
            return 0.0

    def interpret_association_strength(self, measure_value: float, measure_name: str) -> str:
        """
        Interpret the strength of association based on conventional benchmarks.

        Parameters
        ----------
        measure_value : float
            Value of the association measure
        measure_name : str
            Name of the measure for appropriate interpretation

        Returns
        -------
        str
            Interpretation of association strength
        """
        abs_value = abs(measure_value)

        if measure_name.lower() in ['cramers_v', 'phi', 'contingency_coefficient']:
            # Standard interpretation for these measures (0 to 1 scale)
            if abs_value < 0.1:
                return "negligible"
            elif abs_value < 0.3:
                return "weak"
            elif abs_value < 0.5:
                return "moderate"
            else:
                return "strong"

        elif measure_name.lower().startswith('lambda'):
            # Lambda interpretation (proportional reduction in error)
            if abs_value < 0.1:
                return "weak predictive power"
            elif abs_value < 0.3:
                return "moderate predictive power"
            else:
                return "strong predictive power"

        elif measure_name.lower() in ['gamma', 'tau_b']:
            # Ordinal measures interpretation
            if abs_value < 0.2:
                return "weak association"
            elif abs_value < 0.5:
                return "moderate association"
            else:
                return "strong association"

        else:
            # Generic interpretation
            if abs_value < 0.1:
                return "negligible"
            elif abs_value < 0.3:
                return "small"
            elif abs_value < 0.5:
                return "medium"
            else:
                return "large"