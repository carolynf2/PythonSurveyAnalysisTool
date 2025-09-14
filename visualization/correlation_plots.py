"""
Correlation visualization module for survey data.

This module provides various visualization methods for correlation analysis
including correlation matrices, heatmaps, network diagrams, and scatterplot matrices.
"""

import logging
import warnings
from typing import Dict, List, Optional, Union, Tuple, Any
import pandas as pd
import numpy as np

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.figure_factory as ff
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    warnings.warn("plotly not available. Interactive plots disabled.")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    warnings.warn("matplotlib/seaborn not available. Static plots disabled.")

from ..data_processing.models import CorrelationResult


class CorrelationPlots:
    """
    Visualization tools for correlation analysis.

    Features:
    - Interactive correlation matrix heatmaps
    - Correlation network diagrams
    - Scatterplot matrices for pairwise relationships
    - Hierarchical clustering of correlation matrices
    - Significance overlays and filtering
    """

    def __init__(self, style: str = 'plotly'):
        """
        Initialize correlation visualization tools.

        Parameters
        ----------
        style : str, default 'plotly'
            Plotting style: 'plotly' for interactive, 'matplotlib' for static
        """
        self.style = style
        self.logger = logging.getLogger(__name__)

        # Check availability
        if style == 'plotly' and not HAS_PLOTLY:
            if HAS_MATPLOTLIB:
                self.style = 'matplotlib'
                self.logger.warning("plotly not available, falling back to matplotlib")
            else:
                raise ImportError("Neither plotly nor matplotlib available")

        elif style == 'matplotlib' and not HAS_MATPLOTLIB:
            if HAS_PLOTLY:
                self.style = 'plotly'
                self.logger.warning("matplotlib not available, falling back to plotly")
            else:
                raise ImportError("Neither plotly nor matplotlib available")

    def correlation_matrix_heatmap(self,
                                 correlation_results: List[CorrelationResult],
                                 variables: Optional[List[str]] = None,
                                 show_significance: bool = True,
                                 cluster_variables: bool = False,
                                 title: str = "Correlation Matrix",
                                 **kwargs) -> Any:
        """
        Create correlation matrix heatmap.

        Parameters
        ----------
        correlation_results : list of CorrelationResult
            Correlation analysis results
        variables : list of str, optional
            Variables to include in matrix
        show_significance : bool, default True
            Show significance indicators
        cluster_variables : bool, default False
            Apply hierarchical clustering to reorder variables
        title : str, default "Correlation Matrix"
            Plot title
        **kwargs
            Additional plotting parameters

        Returns
        -------
        plotly.graph_objects.Figure or matplotlib.figure.Figure
            Correlation matrix heatmap
        """
        # Convert correlation results to matrix
        corr_matrix, significance_matrix = self._results_to_matrix(
            correlation_results, variables
        )

        if corr_matrix.empty:
            self.logger.warning("No correlation data to plot")
            return None

        # Apply clustering if requested
        if cluster_variables:
            corr_matrix, significance_matrix = self._cluster_correlation_matrix(
                corr_matrix, significance_matrix
            )

        if self.style == 'plotly':
            return self._plotly_correlation_heatmap(
                corr_matrix, significance_matrix, show_significance, title, **kwargs
            )
        else:
            return self._matplotlib_correlation_heatmap(
                corr_matrix, significance_matrix, show_significance, title, **kwargs
            )

    def correlation_network(self,
                          correlation_results: List[CorrelationResult],
                          threshold: float = 0.3,
                          significance_only: bool = True,
                          layout: str = 'spring',
                          title: str = "Correlation Network") -> Any:
        """
        Create correlation network diagram.

        Parameters
        ----------
        correlation_results : list of CorrelationResult
            Correlation analysis results
        threshold : float, default 0.3
            Minimum correlation strength to show
        significance_only : bool, default True
            Only show significant correlations
        layout : str, default 'spring'
            Network layout algorithm
        title : str, default "Correlation Network"
            Plot title

        Returns
        -------
        plotly.graph_objects.Figure
            Interactive network diagram
        """
        if not HAS_PLOTLY:
            self.logger.warning("Network plots require plotly")
            return None

        # Filter correlations
        filtered_results = []
        for result in correlation_results:
            if abs(result.correlation_coefficient) >= threshold:
                if not significance_only or result.is_significant():
                    filtered_results.append(result)

        if not filtered_results:
            self.logger.warning("No correlations meet the criteria for network plot")
            return None

        # Create network data
        nodes, edges = self._create_network_data(filtered_results)

        # Create network plot
        return self._plotly_network_plot(nodes, edges, layout, title)

    def scatterplot_matrix(self,
                         data: pd.DataFrame,
                         variables: Optional[List[str]] = None,
                         color_by: Optional[str] = None,
                         title: str = "Scatterplot Matrix") -> Any:
        """
        Create scatterplot matrix for pairwise relationships.

        Parameters
        ----------
        data : pd.DataFrame
            Survey data
        variables : list of str, optional
            Variables to include
        color_by : str, optional
            Variable to color points by
        title : str, default "Scatterplot Matrix"
            Plot title

        Returns
        -------
        plotly.graph_objects.Figure or matplotlib.figure.Figure
            Scatterplot matrix
        """
        if variables is None:
            # Use numeric variables
            variables = data.select_dtypes(include=[np.number]).columns.tolist()

        if len(variables) < 2:
            self.logger.warning("Need at least 2 variables for scatterplot matrix")
            return None

        # Limit number of variables to prevent overcrowding
        if len(variables) > 8:
            variables = variables[:8]
            self.logger.warning("Limited to first 8 variables to prevent overcrowding")

        if self.style == 'plotly':
            return self._plotly_scatterplot_matrix(data, variables, color_by, title)
        else:
            return self._matplotlib_scatterplot_matrix(data, variables, color_by, title)

    def correlation_significance_plot(self,
                                    correlation_results: List[CorrelationResult],
                                    title: str = "Correlation Significance") -> Any:
        """
        Create plot showing correlation coefficients vs p-values.

        Parameters
        ----------
        correlation_results : list of CorrelationResult
            Correlation analysis results
        title : str, default "Correlation Significance"
            Plot title

        Returns
        -------
        plotly.graph_objects.Figure or matplotlib.figure.Figure
            Scatter plot of correlations vs significance
        """
        if not correlation_results:
            self.logger.warning("No correlation results to plot")
            return None

        # Extract data
        correlations = [abs(r.correlation_coefficient) for r in correlation_results]
        p_values = [r.p_value for r in correlation_results]
        variable_pairs = [f"{r.variable1} × {r.variable2}" for r in correlation_results]

        if self.style == 'plotly':
            return self._plotly_significance_plot(correlations, p_values, variable_pairs, title)
        else:
            return self._matplotlib_significance_plot(correlations, p_values, variable_pairs, title)

    def _results_to_matrix(self,
                          correlation_results: List[CorrelationResult],
                          variables: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Convert correlation results to correlation and significance matrices."""
        if not correlation_results:
            return pd.DataFrame(), pd.DataFrame()

        # Get all variables
        all_variables = set()
        for result in correlation_results:
            all_variables.add(result.variable1)
            all_variables.add(result.variable2)

        if variables:
            all_variables = set(variables) & all_variables

        all_variables = sorted(list(all_variables))

        # Initialize matrices
        corr_matrix = pd.DataFrame(
            index=all_variables,
            columns=all_variables,
            dtype=float
        )
        sig_matrix = pd.DataFrame(
            index=all_variables,
            columns=all_variables,
            dtype=float
        )

        # Fill diagonal with 1.0 (perfect correlation with self)
        for var in all_variables:
            corr_matrix.loc[var, var] = 1.0
            sig_matrix.loc[var, var] = 0.0

        # Fill matrix with correlation results
        for result in correlation_results:
            if result.variable1 in all_variables and result.variable2 in all_variables:
                corr_matrix.loc[result.variable1, result.variable2] = result.correlation_coefficient
                corr_matrix.loc[result.variable2, result.variable1] = result.correlation_coefficient

                sig_matrix.loc[result.variable1, result.variable2] = result.p_value
                sig_matrix.loc[result.variable2, result.variable1] = result.p_value

        return corr_matrix.fillna(0), sig_matrix.fillna(1)

    def _cluster_correlation_matrix(self,
                                  corr_matrix: pd.DataFrame,
                                  sig_matrix: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Apply hierarchical clustering to reorder correlation matrix."""
        try:
            from scipy.cluster.hierarchy import linkage, leaves_list
            from scipy.spatial.distance import squareform

            # Convert correlation to distance
            distance_matrix = 1 - abs(corr_matrix)
            condensed_distances = squareform(distance_matrix, checks=False)

            # Perform hierarchical clustering
            linkage_matrix = linkage(condensed_distances, method='average')
            clustered_indices = leaves_list(linkage_matrix)

            # Reorder matrices
            reordered_vars = [corr_matrix.index[i] for i in clustered_indices]
            corr_matrix_clustered = corr_matrix.loc[reordered_vars, reordered_vars]
            sig_matrix_clustered = sig_matrix.loc[reordered_vars, reordered_vars]

            return corr_matrix_clustered, sig_matrix_clustered

        except ImportError:
            self.logger.warning("scipy required for clustering, returning original matrix")
            return corr_matrix, sig_matrix

    def _plotly_correlation_heatmap(self,
                                  corr_matrix: pd.DataFrame,
                                  sig_matrix: pd.DataFrame,
                                  show_significance: bool,
                                  title: str,
                                  **kwargs) -> go.Figure:
        """Create Plotly correlation heatmap."""
        # Create annotations for significance
        annotations = []
        if show_significance:
            for i, row in enumerate(corr_matrix.index):
                for j, col in enumerate(corr_matrix.columns):
                    corr_val = corr_matrix.loc[row, col]
                    p_val = sig_matrix.loc[row, col]

                    # Add significance stars
                    stars = ""
                    if p_val < 0.001:
                        stars = "***"
                    elif p_val < 0.01:
                        stars = "**"
                    elif p_val < 0.05:
                        stars = "*"

                    text = f"{corr_val:.3f}{stars}"
                    annotations.append(
                        dict(
                            x=j, y=i,
                            text=text,
                            showarrow=False,
                            font=dict(color="white" if abs(corr_val) > 0.5 else "black")
                        )
                    )

        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu_r',
            zmid=0,
            zmin=-1,
            zmax=1,
            colorbar=dict(title="Correlation Coefficient"),
            hovertemplate='%{y} × %{x}<br>Correlation: %{z:.3f}<extra></extra>'
        ))

        fig.update_layout(
            title=title,
            xaxis_title="Variables",
            yaxis_title="Variables",
            annotations=annotations,
            width=max(600, len(corr_matrix.columns) * 50),
            height=max(600, len(corr_matrix.index) * 50)
        )

        return fig

    def _matplotlib_correlation_heatmap(self,
                                      corr_matrix: pd.DataFrame,
                                      sig_matrix: pd.DataFrame,
                                      show_significance: bool,
                                      title: str,
                                      **kwargs) -> plt.Figure:
        """Create matplotlib correlation heatmap."""
        fig, ax = plt.subplots(figsize=(max(8, len(corr_matrix.columns) * 0.8),
                                       max(8, len(corr_matrix.index) * 0.8)))

        # Create heatmap
        sns.heatmap(
            corr_matrix,
            annot=show_significance,
            cmap='RdBu_r',
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            cbar_kws={'label': 'Correlation Coefficient'},
            ax=ax
        )

        # Add significance annotations if requested
        if show_significance:
            for i in range(len(corr_matrix.index)):
                for j in range(len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    p_val = sig_matrix.iloc[i, j]

                    stars = ""
                    if p_val < 0.001:
                        stars = "***"
                    elif p_val < 0.01:
                        stars = "**"
                    elif p_val < 0.05:
                        stars = "*"

                    ax.text(j + 0.5, i + 0.7, stars,
                           ha='center', va='center',
                           color='red', fontweight='bold')

        ax.set_title(title)
        plt.tight_layout()
        return fig

    def _create_network_data(self, correlation_results: List[CorrelationResult]) -> Tuple[List[Dict], List[Dict]]:
        """Create node and edge data for network plot."""
        # Get unique variables
        variables = set()
        for result in correlation_results:
            variables.add(result.variable1)
            variables.add(result.variable2)

        # Create nodes
        nodes = [{'id': var, 'label': var} for var in sorted(variables)]

        # Create edges
        edges = []
        for result in correlation_results:
            edge = {
                'source': result.variable1,
                'target': result.variable2,
                'weight': abs(result.correlation_coefficient),
                'correlation': result.correlation_coefficient,
                'p_value': result.p_value,
                'significant': result.is_significant()
            }
            edges.append(edge)

        return nodes, edges

    def _plotly_network_plot(self,
                           nodes: List[Dict],
                           edges: List[Dict],
                           layout: str,
                           title: str) -> go.Figure:
        """Create Plotly network plot."""
        try:
            import networkx as nx

            # Create NetworkX graph
            G = nx.Graph()

            # Add nodes
            for node in nodes:
                G.add_node(node['id'])

            # Add edges
            for edge in edges:
                G.add_edge(edge['source'], edge['target'], weight=edge['weight'])

            # Get layout positions
            if layout == 'spring':
                pos = nx.spring_layout(G, k=1, iterations=50)
            elif layout == 'circular':
                pos = nx.circular_layout(G)
            else:
                pos = nx.spring_layout(G)

            # Create edge traces
            edge_x = []
            edge_y = []
            edge_info = []

            for edge in edges:
                x0, y0 = pos[edge['source']]
                x1, y1 = pos[edge['target']]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                edge_info.append(f"{edge['source']} - {edge['target']}: r={edge['correlation']:.3f}")

            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=0.5, color='#888'),
                hoverinfo='none',
                mode='lines'
            )

            # Create node trace
            node_x = []
            node_y = []
            node_text = []

            for node in nodes:
                x, y = pos[node['id']]
                node_x.append(x)
                node_y.append(y)
                node_text.append(node['label'])

            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                hoverinfo='text',
                text=node_text,
                textposition="middle center",
                marker=dict(
                    showscale=True,
                    colorscale='YlGnBu',
                    reversescale=True,
                    color=[],
                    size=15,
                    colorbar=dict(
                        thickness=15,
                        len=0.5,
                        xanchor="left",
                        title="Node Connections"
                    ),
                    line=dict(width=2)
                )
            )

            # Color nodes by number of connections
            node_adjacencies = []
            for node in nodes:
                adjacencies = list(G.neighbors(node['id']))
                node_adjacencies.append(len(adjacencies))

            node_trace.marker.color = node_adjacencies

            # Create figure
            fig = go.Figure(data=[edge_trace, node_trace],
                          layout=go.Layout(
                              title=title,
                              titlefont_size=16,
                              showlegend=False,
                              hovermode='closest',
                              margin=dict(b=20, l=5, r=5, t=40),
                              annotations=[dict(
                                  text="Correlation network: node size = number of connections",
                                  showarrow=False,
                                  xref="paper", yref="paper",
                                  x=0.005, y=-0.002,
                                  xanchor='left', yanchor='bottom',
                                  font=dict(size=12)
                              )],
                              xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                              yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                          ))

            return fig

        except ImportError:
            self.logger.warning("networkx required for network plots")
            return None

    def _plotly_scatterplot_matrix(self,
                                 data: pd.DataFrame,
                                 variables: List[str],
                                 color_by: Optional[str],
                                 title: str) -> go.Figure:
        """Create Plotly scatterplot matrix."""
        # Clean data
        plot_data = data[variables].dropna()

        if color_by and color_by in data.columns:
            color_data = data.loc[plot_data.index, color_by]
        else:
            color_data = None

        # Create scatterplot matrix
        fig = ff.create_scatterplotmatrix(
            plot_data,
            diag='histogram',
            height=800,
            width=800,
            title=title
        )

        return fig

    def _matplotlib_scatterplot_matrix(self,
                                     data: pd.DataFrame,
                                     variables: List[str],
                                     color_by: Optional[str],
                                     title: str) -> plt.Figure:
        """Create matplotlib scatterplot matrix."""
        plot_data = data[variables].dropna()

        fig, axes = plt.subplots(
            len(variables), len(variables),
            figsize=(len(variables) * 3, len(variables) * 3)
        )

        for i, var1 in enumerate(variables):
            for j, var2 in enumerate(variables):
                ax = axes[i, j]

                if i == j:
                    # Diagonal: histogram
                    ax.hist(plot_data[var1], alpha=0.7, bins=20)
                    ax.set_ylabel('Frequency')
                else:
                    # Off-diagonal: scatter plot
                    if color_by and color_by in data.columns:
                        color_data = data.loc[plot_data.index, color_by]
                        scatter = ax.scatter(plot_data[var2], plot_data[var1],
                                           c=color_data, alpha=0.6)
                        if i == 0 and j == len(variables) - 1:
                            plt.colorbar(scatter, ax=ax, label=color_by)
                    else:
                        ax.scatter(plot_data[var2], plot_data[var1], alpha=0.6)

                # Set labels
                if i == len(variables) - 1:
                    ax.set_xlabel(var2)
                if j == 0:
                    ax.set_ylabel(var1)

        plt.suptitle(title)
        plt.tight_layout()
        return fig

    def _plotly_significance_plot(self,
                                correlations: List[float],
                                p_values: List[float],
                                variable_pairs: List[str],
                                title: str) -> go.Figure:
        """Create Plotly correlation vs significance plot."""
        # Create significance categories
        significance_levels = []
        for p in p_values:
            if p < 0.001:
                significance_levels.append("p < 0.001")
            elif p < 0.01:
                significance_levels.append("p < 0.01")
            elif p < 0.05:
                significance_levels.append("p < 0.05")
            else:
                significance_levels.append("p ≥ 0.05")

        fig = go.Figure()

        # Add scatter plot
        fig.add_trace(go.Scatter(
            x=correlations,
            y=[-np.log10(p) for p in p_values],
            mode='markers',
            marker=dict(
                color=significance_levels,
                size=8,
                colorscale='Viridis'
            ),
            text=variable_pairs,
            hovertemplate='%{text}<br>|r| = %{x:.3f}<br>p = %{customdata:.4f}<extra></extra>',
            customdata=p_values
        ))

        # Add significance threshold lines
        fig.add_hline(y=-np.log10(0.05), line_dash="dash", line_color="red",
                     annotation_text="p = 0.05")
        fig.add_hline(y=-np.log10(0.01), line_dash="dash", line_color="orange",
                     annotation_text="p = 0.01")
        fig.add_hline(y=-np.log10(0.001), line_dash="dash", line_color="darkred",
                     annotation_text="p = 0.001")

        fig.update_layout(
            title=title,
            xaxis_title="Absolute Correlation Coefficient",
            yaxis_title="-log10(p-value)",
            showlegend=False,
            width=800,
            height=600
        )

        return fig

    def _matplotlib_significance_plot(self,
                                    correlations: List[float],
                                    p_values: List[float],
                                    variable_pairs: List[str],
                                    title: str) -> plt.Figure:
        """Create matplotlib correlation vs significance plot."""
        fig, ax = plt.subplots(figsize=(10, 6))

        # Create significance categories for coloring
        colors = []
        for p in p_values:
            if p < 0.001:
                colors.append('darkred')
            elif p < 0.01:
                colors.append('red')
            elif p < 0.05:
                colors.append('orange')
            else:
                colors.append('gray')

        # Create scatter plot
        scatter = ax.scatter(correlations, [-np.log10(p) for p in p_values],
                           c=colors, alpha=0.7)

        # Add significance threshold lines
        ax.axhline(y=-np.log10(0.05), color='red', linestyle='--', alpha=0.7, label='p = 0.05')
        ax.axhline(y=-np.log10(0.01), color='orange', linestyle='--', alpha=0.7, label='p = 0.01')
        ax.axhline(y=-np.log10(0.001), color='darkred', linestyle='--', alpha=0.7, label='p = 0.001')

        ax.set_xlabel('Absolute Correlation Coefficient')
        ax.set_ylabel('-log10(p-value)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig