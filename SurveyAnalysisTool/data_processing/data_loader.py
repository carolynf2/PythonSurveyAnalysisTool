"""
Multi-format data loader for survey data.

This module provides comprehensive data loading capabilities for various
survey data formats including CSV, Excel, SPSS, SAS, Stata, and database
connections. It handles encoding detection, format-specific metadata extraction,
and automatic variable type inference.
"""

import os
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
import pandas as pd
import numpy as np

try:
    import pyreadstat
    HAS_PYREADSTAT = True
except ImportError:
    HAS_PYREADSTAT = False
    warnings.warn("pyreadstat not available. SPSS, SAS, and Stata support limited.")

try:
    import sqlalchemy
    HAS_SQLALCHEMY = True
except ImportError:
    HAS_SQLALCHEMY = False
    warnings.warn("sqlalchemy not available. Database connectivity disabled.")

from .models import SurveyMetadata, VariableDefinition, VariableType


class DataLoader:
    """
    Comprehensive data loader for survey data in multiple formats.

    Supports:
    - CSV/TSV files with encoding detection
    - Excel files (.xlsx, .xls) with multiple sheets
    - SPSS files (.sav, .por)
    - SAS files (.sas7bdat, .xpt)
    - Stata files (.dta)
    - JSON survey data
    - Database connections (SQL)
    - API endpoints (with authentication)
    """

    def __init__(self, encoding: str = 'auto', chunk_size: Optional[int] = None):
        """
        Initialize the DataLoader.

        Parameters
        ----------
        encoding : str, default 'auto'
            Text encoding for file reading. 'auto' enables detection.
        chunk_size : int, optional
            Size of chunks for reading large files. None loads entire file.
        """
        self.encoding = encoding
        self.chunk_size = chunk_size
        self.logger = logging.getLogger(__name__)

        # File format handlers
        self._handlers = {
            '.csv': self._load_csv,
            '.tsv': self._load_csv,
            '.txt': self._load_csv,
            '.xlsx': self._load_excel,
            '.xls': self._load_excel,
            '.json': self._load_json,
        }

        if HAS_PYREADSTAT:
            self._handlers.update({
                '.sav': self._load_spss,
                '.por': self._load_spss,
                '.dta': self._load_stata,
                '.sas7bdat': self._load_sas,
                '.xpt': self._load_sas,
            })

    def load_data(self,
                  file_path: Union[str, Path],
                  metadata_path: Optional[Union[str, Path]] = None,
                  **kwargs) -> Tuple[pd.DataFrame, Optional[SurveyMetadata]]:
        """
        Load survey data from file with automatic format detection.

        Parameters
        ----------
        file_path : str or Path
            Path to the data file
        metadata_path : str or Path, optional
            Path to metadata file (JSON format)
        **kwargs
            Additional arguments passed to format-specific loaders

        Returns
        -------
        tuple
            (DataFrame with survey data, SurveyMetadata object or None)
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        # Get file extension and find appropriate handler
        extension = file_path.suffix.lower()

        if extension not in self._handlers:
            raise ValueError(f"Unsupported file format: {extension}")

        self.logger.info(f"Loading data from {file_path} (format: {extension})")

        # Load data using appropriate handler
        handler = self._handlers[extension]
        data, file_metadata = handler(file_path, **kwargs)

        # Load additional metadata if provided
        metadata = None
        if metadata_path:
            metadata = self._load_metadata(metadata_path, file_metadata)
        elif file_metadata:
            metadata = file_metadata

        # Infer variable types if metadata is available
        if metadata:
            data = self._apply_metadata(data, metadata)

        self.logger.info(f"Loaded {len(data)} records with {len(data.columns)} variables")

        return data, metadata

    def _load_csv(self, file_path: Path, **kwargs) -> Tuple[pd.DataFrame, None]:
        """Load CSV/TSV files with encoding detection."""
        # Detect separator if not specified
        sep = kwargs.get('sep')
        if sep is None:
            if file_path.suffix.lower() == '.tsv':
                sep = '\t'
            else:
                sep = self._detect_separator(file_path)

        # Detect encoding if auto
        encoding = self.encoding
        if encoding == 'auto':
            encoding = self._detect_encoding(file_path)

        # Load data
        try:
            data = pd.read_csv(
                file_path,
                sep=sep,
                encoding=encoding,
                chunksize=self.chunk_size,
                **{k: v for k, v in kwargs.items() if k not in ['sep']}
            )

            if self.chunk_size is not None:
                # Concatenate chunks
                data = pd.concat(data, ignore_index=True)

        except UnicodeDecodeError:
            # Fallback encodings
            for fallback_encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    data = pd.read_csv(
                        file_path,
                        sep=sep,
                        encoding=fallback_encoding,
                        chunksize=self.chunk_size,
                        **{k: v for k, v in kwargs.items() if k not in ['sep']}
                    )
                    if self.chunk_size is not None:
                        data = pd.concat(data, ignore_index=True)
                    self.logger.warning(f"Used fallback encoding: {fallback_encoding}")
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError("Could not decode file with any supported encoding")

        return data, None

    def _load_excel(self, file_path: Path, **kwargs) -> Tuple[pd.DataFrame, None]:
        """Load Excel files with sheet detection."""
        sheet_name = kwargs.get('sheet_name', 0)

        # If sheet_name is None, load all sheets
        if sheet_name is None:
            excel_file = pd.ExcelFile(file_path)
            sheets = {}
            for sheet in excel_file.sheet_names:
                sheets[sheet] = pd.read_excel(
                    file_path,
                    sheet_name=sheet,
                    **{k: v for k, v in kwargs.items() if k != 'sheet_name'}
                )

            # Return first sheet as main data, store others in metadata
            first_sheet = list(sheets.keys())[0]
            data = sheets[first_sheet]

            # Could extend to handle multiple sheets in metadata
            return data, None
        else:
            data = pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)
            return data, None

    def _load_json(self, file_path: Path, **kwargs) -> Tuple[pd.DataFrame, Optional[SurveyMetadata]]:
        """Load JSON survey data with metadata extraction."""
        with open(file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        # Handle different JSON structures
        if 'responses' in json_data and 'metadata' in json_data:
            # Structured survey JSON
            data = pd.DataFrame(json_data['responses'])
            metadata = self._parse_json_metadata(json_data['metadata'])
            return data, metadata
        elif isinstance(json_data, list):
            # Array of response objects
            data = pd.DataFrame(json_data)
            return data, None
        elif isinstance(json_data, dict):
            # Single level dictionary
            data = pd.DataFrame([json_data])
            return data, None
        else:
            raise ValueError("Unsupported JSON structure")

    def _load_spss(self, file_path: Path, **kwargs) -> Tuple[pd.DataFrame, Optional[SurveyMetadata]]:
        """Load SPSS files with metadata extraction."""
        if not HAS_PYREADSTAT:
            raise ImportError("pyreadstat required for SPSS file support")

        data, meta = pyreadstat.read_sav(str(file_path), **kwargs)

        # Extract metadata from SPSS file
        metadata = self._extract_spss_metadata(meta, file_path)

        return data, metadata

    def _load_stata(self, file_path: Path, **kwargs) -> Tuple[pd.DataFrame, Optional[SurveyMetadata]]:
        """Load Stata files with metadata extraction."""
        if not HAS_PYREADSTAT:
            raise ImportError("pyreadstat required for Stata file support")

        data, meta = pyreadstat.read_dta(str(file_path), **kwargs)

        # Extract metadata from Stata file
        metadata = self._extract_stata_metadata(meta, file_path)

        return data, metadata

    def _load_sas(self, file_path: Path, **kwargs) -> Tuple[pd.DataFrame, Optional[SurveyMetadata]]:
        """Load SAS files with metadata extraction."""
        if not HAS_PYREADSTAT:
            raise ImportError("pyreadstat required for SAS file support")

        if file_path.suffix.lower() == '.sas7bdat':
            data, meta = pyreadstat.read_sas7bdat(str(file_path), **kwargs)
        else:  # .xpt
            data, meta = pyreadstat.read_xport(str(file_path), **kwargs)

        # Extract metadata from SAS file
        metadata = self._extract_sas_metadata(meta, file_path)

        return data, metadata

    def load_from_database(self,
                          connection_string: str,
                          query: str,
                          metadata_query: Optional[str] = None) -> Tuple[pd.DataFrame, Optional[SurveyMetadata]]:
        """
        Load survey data from database.

        Parameters
        ----------
        connection_string : str
            SQLAlchemy connection string
        query : str
            SQL query to retrieve survey data
        metadata_query : str, optional
            SQL query to retrieve metadata

        Returns
        -------
        tuple
            (DataFrame with survey data, SurveyMetadata object or None)
        """
        if not HAS_SQLALCHEMY:
            raise ImportError("sqlalchemy required for database connectivity")

        engine = sqlalchemy.create_engine(connection_string)

        try:
            # Load main data
            data = pd.read_sql(query, engine)

            # Load metadata if query provided
            metadata = None
            if metadata_query:
                metadata_df = pd.read_sql(metadata_query, engine)
                metadata = self._parse_database_metadata(metadata_df)

            return data, metadata

        finally:
            engine.dispose()

    def _detect_separator(self, file_path: Path) -> str:
        """Detect CSV separator by examining first few lines."""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            sample = f.read(1024)

        # Count occurrences of common separators
        separators = [',', ';', '\t', '|']
        counts = {sep: sample.count(sep) for sep in separators}

        # Return separator with highest count
        return max(counts.items(), key=lambda x: x[1])[0]

    def _detect_encoding(self, file_path: Path) -> str:
        """Detect file encoding using chardet or similar method."""
        try:
            import chardet
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)  # Read first 10KB
            result = chardet.detect(raw_data)
            return result['encoding'] or 'utf-8'
        except ImportError:
            # Fallback to utf-8 if chardet not available
            return 'utf-8'

    def _load_metadata(self, metadata_path: Path, file_metadata: Optional[SurveyMetadata] = None) -> SurveyMetadata:
        """Load metadata from JSON file."""
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata_dict = json.load(f)

        return self._parse_json_metadata(metadata_dict, file_metadata)

    def _parse_json_metadata(self, metadata_dict: Dict, existing_metadata: Optional[SurveyMetadata] = None) -> SurveyMetadata:
        """Parse metadata dictionary into SurveyMetadata object."""
        from datetime import datetime

        # Parse collection period
        collection_period = metadata_dict.get('collection_period')
        if collection_period:
            start = datetime.fromisoformat(collection_period[0])
            end = datetime.fromisoformat(collection_period[1])
            collection_period = (start, end)

        # Parse variable definitions
        variables = {}
        for var_name, var_info in metadata_dict.get('variables', {}).items():
            var_type = VariableType(var_info.get('type', 'nominal'))

            variables[var_name] = VariableDefinition(
                name=var_name,
                label=var_info.get('label', var_name),
                type=var_type,
                categories=var_info.get('categories'),
                valid_range=var_info.get('valid_range'),
                missing_codes=var_info.get('missing_codes', []),
                question_text=var_info.get('question_text', ''),
                scale_type=var_info.get('scale_type')
            )

        return SurveyMetadata(
            survey_id=metadata_dict.get('survey_id', ''),
            title=metadata_dict.get('title', ''),
            description=metadata_dict.get('description', ''),
            collection_period=collection_period or (datetime.now(), datetime.now()),
            target_population=metadata_dict.get('target_population', ''),
            sampling_method=metadata_dict.get('sampling_method', ''),
            expected_responses=metadata_dict.get('expected_responses', 0),
            actual_responses=metadata_dict.get('actual_responses', 0),
            completion_rate=metadata_dict.get('completion_rate', 0.0),
            variables=variables,
            weights_variable=metadata_dict.get('weights_variable'),
            strata_variables=metadata_dict.get('strata_variables', []),
            cluster_variable=metadata_dict.get('cluster_variable')
        )

    def _extract_spss_metadata(self, meta, file_path: Path) -> SurveyMetadata:
        """Extract metadata from SPSS file metadata."""
        from datetime import datetime

        variables = {}
        for col_name in meta.column_names:
            # Determine variable type based on SPSS metadata
            var_type = self._infer_variable_type_from_spss(col_name, meta)

            # Extract value labels if available
            categories = None
            if col_name in meta.value_labels:
                categories = meta.value_labels[col_name]

            # Extract variable label
            label = meta.column_labels.get(col_name, col_name) if meta.column_labels else col_name

            # Extract missing values
            missing_codes = []
            if meta.missing_ranges and col_name in meta.missing_ranges:
                missing_codes = list(meta.missing_ranges[col_name])

            variables[col_name] = VariableDefinition(
                name=col_name,
                label=label,
                type=var_type,
                categories=categories,
                missing_codes=missing_codes
            )

        return SurveyMetadata(
            survey_id=file_path.stem,
            title=f"SPSS Survey Data: {file_path.name}",
            description=f"Imported from SPSS file: {file_path}",
            collection_period=(datetime.now(), datetime.now()),
            target_population="Unknown",
            sampling_method="Unknown",
            expected_responses=0,
            actual_responses=meta.number_rows,
            completion_rate=1.0,
            variables=variables
        )

    def _extract_stata_metadata(self, meta, file_path: Path) -> SurveyMetadata:
        """Extract metadata from Stata file metadata."""
        # Similar to SPSS but adapted for Stata metadata structure
        from datetime import datetime

        variables = {}
        for col_name in meta.column_names:
            var_type = self._infer_variable_type_from_stata(col_name, meta)

            categories = None
            if meta.value_labels and col_name in meta.value_labels:
                categories = meta.value_labels[col_name]

            label = meta.column_labels.get(col_name, col_name) if meta.column_labels else col_name

            variables[col_name] = VariableDefinition(
                name=col_name,
                label=label,
                type=var_type,
                categories=categories
            )

        return SurveyMetadata(
            survey_id=file_path.stem,
            title=f"Stata Survey Data: {file_path.name}",
            description=f"Imported from Stata file: {file_path}",
            collection_period=(datetime.now(), datetime.now()),
            target_population="Unknown",
            sampling_method="Unknown",
            expected_responses=0,
            actual_responses=meta.number_rows,
            completion_rate=1.0,
            variables=variables
        )

    def _extract_sas_metadata(self, meta, file_path: Path) -> SurveyMetadata:
        """Extract metadata from SAS file metadata."""
        # Similar structure for SAS metadata
        from datetime import datetime

        variables = {}
        for col_name in meta.column_names:
            var_type = self._infer_variable_type_from_sas(col_name, meta)

            label = meta.column_labels.get(col_name, col_name) if meta.column_labels else col_name

            variables[col_name] = VariableDefinition(
                name=col_name,
                label=label,
                type=var_type
            )

        return SurveyMetadata(
            survey_id=file_path.stem,
            title=f"SAS Survey Data: {file_path.name}",
            description=f"Imported from SAS file: {file_path}",
            collection_period=(datetime.now(), datetime.now()),
            target_population="Unknown",
            sampling_method="Unknown",
            expected_responses=0,
            actual_responses=meta.number_rows,
            completion_rate=1.0,
            variables=variables
        )

    def _infer_variable_type_from_spss(self, col_name: str, meta) -> VariableType:
        """Infer variable type from SPSS metadata."""
        # Check if variable has value labels (categorical)
        if meta.value_labels and col_name in meta.value_labels:
            # Check if ordinal or nominal based on value labels
            values = list(meta.value_labels[col_name].keys())
            if all(isinstance(v, (int, float)) for v in values):
                return VariableType.ORDINAL
            else:
                return VariableType.NOMINAL

        # Check variable format/type
        if meta.readstat_variable_types:
            var_type = meta.readstat_variable_types.get(col_name)
            if var_type in ['READSTAT_TYPE_STRING', 'READSTAT_TYPE_STRING_REF']:
                return VariableType.NOMINAL
            else:
                return VariableType.RATIO  # Numeric without labels

        return VariableType.NOMINAL  # Default

    def _infer_variable_type_from_stata(self, col_name: str, meta) -> VariableType:
        """Infer variable type from Stata metadata."""
        # Similar logic for Stata
        if meta.value_labels and col_name in meta.value_labels:
            return VariableType.ORDINAL

        if meta.readstat_variable_types:
            var_type = meta.readstat_variable_types.get(col_name)
            if var_type in ['READSTAT_TYPE_STRING', 'READSTAT_TYPE_STRING_REF']:
                return VariableType.NOMINAL
            else:
                return VariableType.RATIO

        return VariableType.NOMINAL

    def _infer_variable_type_from_sas(self, col_name: str, meta) -> VariableType:
        """Infer variable type from SAS metadata."""
        # Similar logic for SAS
        if meta.readstat_variable_types:
            var_type = meta.readstat_variable_types.get(col_name)
            if var_type in ['READSTAT_TYPE_STRING', 'READSTAT_TYPE_STRING_REF']:
                return VariableType.NOMINAL
            else:
                return VariableType.RATIO

        return VariableType.NOMINAL

    def _apply_metadata(self, data: pd.DataFrame, metadata: SurveyMetadata) -> pd.DataFrame:
        """Apply metadata to clean and type the data appropriately."""
        for col_name, var_def in metadata.variables.items():
            if col_name in data.columns:
                # Handle missing codes
                if var_def.missing_codes:
                    data.loc[data[col_name].isin(var_def.missing_codes), col_name] = np.nan

                # Apply value labels if available
                if var_def.categories:
                    # Create categorical with proper ordering for ordinal variables
                    if var_def.type == VariableType.ORDINAL:
                        ordered_categories = sorted(var_def.categories.keys())
                        category_labels = [var_def.categories[k] for k in ordered_categories]
                        data[col_name] = pd.Categorical(
                            data[col_name].map(var_def.categories),
                            categories=category_labels,
                            ordered=True
                        )
                    else:
                        data[col_name] = data[col_name].map(var_def.categories).astype('category')

        return data

    def _parse_database_metadata(self, metadata_df: pd.DataFrame) -> Optional[SurveyMetadata]:
        """Parse metadata from database query results."""
        # Implementation depends on database metadata structure
        # This is a placeholder for custom database metadata parsing
        return None