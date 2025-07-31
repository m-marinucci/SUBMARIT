"""General data I/O utilities for SUBMARIT."""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
import yaml
from numpy.typing import NDArray

from submarit.core.substitution_matrix import SubstitutionMatrix


def load_substitution_data(
    filepath: Union[str, Path],
    format: Optional[str] = None,
    **kwargs
) -> SubstitutionMatrix:
    """Load substitution data from various file formats.
    
    Args:
        filepath: Input file path
        format: File format (auto-detected if None)
        **kwargs: Additional arguments passed to format-specific loaders
        
    Returns:
        SubstitutionMatrix object
    """
    filepath = Path(filepath)
    
    if format is None:
        format = filepath.suffix.lower().lstrip('.')
    
    loaders = {
        'csv': _load_csv,
        'xlsx': _load_excel,
        'xls': _load_excel,
        'npy': _load_numpy,
        'npz': _load_numpy_compressed,
        'txt': _load_text,
        'mat': _load_matlab,
        'h5': _load_hdf5,
        'hdf5': _load_hdf5,
        'parquet': _load_parquet,
    }
    
    if format not in loaders:
        raise ValueError(f"Unsupported format: {format}")
    
    data = loaders[format](filepath, **kwargs)
    
    return SubstitutionMatrix(data)


def _load_csv(filepath: Path, **kwargs) -> NDArray[np.float64]:
    """Load data from CSV file."""
    df = pd.read_csv(filepath, **kwargs)
    
    # If first column is index, use it
    if df.iloc[:, 0].dtype == object:
        df = df.set_index(df.columns[0])
    
    return df.values.astype(np.float64)


def _load_excel(filepath: Path, sheet_name: Union[str, int] = 0, **kwargs) -> NDArray[np.float64]:
    """Load data from Excel file."""
    df = pd.read_excel(filepath, sheet_name=sheet_name, **kwargs)
    
    # If first column is index, use it
    if df.iloc[:, 0].dtype == object:
        df = df.set_index(df.columns[0])
    
    return df.values.astype(np.float64)


def _load_numpy(filepath: Path, **kwargs) -> NDArray[np.float64]:
    """Load data from NumPy binary file."""
    return np.load(filepath, **kwargs).astype(np.float64)


def _load_numpy_compressed(filepath: Path, key: Optional[str] = None, **kwargs) -> NDArray[np.float64]:
    """Load data from compressed NumPy file."""
    data = np.load(filepath, **kwargs)
    
    if key is not None:
        return data[key].astype(np.float64)
    
    # If only one array, return it
    if len(data.files) == 1:
        return data[data.files[0]].astype(np.float64)
    
    # Otherwise, look for common names
    for name in ['substitution_matrix', 'matrix', 'data', 'X']:
        if name in data:
            return data[name].astype(np.float64)
    
    raise ValueError(f"Multiple arrays found, specify key: {data.files}")


def _load_text(filepath: Path, delimiter: Optional[str] = None, **kwargs) -> NDArray[np.float64]:
    """Load data from text file."""
    return np.loadtxt(filepath, delimiter=delimiter, **kwargs).astype(np.float64)


def _load_matlab(filepath: Path, variable: Optional[str] = None, **kwargs) -> NDArray[np.float64]:
    """Load data from MATLAB file."""
    from submarit.io.matlab_io import load_mat
    
    data = load_mat(filepath)
    
    if variable is not None:
        return data[variable].astype(np.float64)
    
    # Look for common variable names
    for name in ['substitution_matrix', 'matrix', 'data', 'X', 'S']:
        if name in data:
            return data[name].astype(np.float64)
    
    # If only one variable (excluding metadata), return it
    vars = [k for k in data.keys() if not k.startswith('__')]
    if len(vars) == 1:
        return data[vars[0]].astype(np.float64)
    
    raise ValueError(f"Multiple variables found, specify variable: {vars}")


def _load_hdf5(filepath: Path, key: str = '/data', **kwargs) -> NDArray[np.float64]:
    """Load data from HDF5 file."""
    import h5py
    
    with h5py.File(filepath, 'r') as f:
        return f[key][()].astype(np.float64)


def _load_parquet(filepath: Path, **kwargs) -> NDArray[np.float64]:
    """Load data from Parquet file."""
    df = pd.read_parquet(filepath, **kwargs)
    return df.values.astype(np.float64)


def save_results(
    results: Dict[str, Any],
    filepath: Union[str, Path],
    format: Optional[str] = None,
    **kwargs
) -> None:
    """Save clustering results to file.
    
    Args:
        results: Dictionary of results to save
        filepath: Output file path
        format: Output format (auto-detected if None)
        **kwargs: Additional arguments for format-specific savers
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if format is None:
        format = filepath.suffix.lower().lstrip('.')
    
    savers = {
        'json': _save_json,
        'yaml': _save_yaml,
        'yml': _save_yaml,
        'npz': _save_numpy_compressed,
        'csv': _save_csv,
        'xlsx': _save_excel,
        'h5': _save_hdf5,
        'hdf5': _save_hdf5,
    }
    
    if format not in savers:
        raise ValueError(f"Unsupported format: {format}")
    
    savers[format](results, filepath, **kwargs)


def _save_json(results: Dict[str, Any], filepath: Path, **kwargs) -> None:
    """Save results to JSON file."""
    # Convert numpy arrays to lists
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj
    
    with open(filepath, 'w') as f:
        json.dump(convert(results), f, indent=2, **kwargs)


def _save_yaml(results: Dict[str, Any], filepath: Path, **kwargs) -> None:
    """Save results to YAML file."""
    # Convert numpy arrays
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj
    
    with open(filepath, 'w') as f:
        yaml.dump(convert(results), f, **kwargs)


def _save_numpy_compressed(results: Dict[str, Any], filepath: Path, **kwargs) -> None:
    """Save results to compressed NumPy file."""
    # Only save numpy arrays and numeric values
    save_dict = {}
    for key, value in results.items():
        if isinstance(value, (np.ndarray, int, float, np.integer, np.floating)):
            save_dict[key] = value
    
    np.savez_compressed(filepath, **save_dict, **kwargs)


def _save_csv(results: Dict[str, Any], filepath: Path, **kwargs) -> None:
    """Save results to CSV file."""
    # Convert to DataFrame
    df_data = {}
    
    # Find arrays of same length
    array_len = None
    for key, value in results.items():
        if isinstance(value, np.ndarray) and value.ndim == 1:
            if array_len is None:
                array_len = len(value)
            if len(value) == array_len:
                df_data[key] = value
    
    # Add scalars as columns
    for key, value in results.items():
        if np.isscalar(value):
            df_data[key] = [value] * (array_len or 1)
    
    df = pd.DataFrame(df_data)
    df.to_csv(filepath, index=False, **kwargs)


def _save_excel(results: Dict[str, Any], filepath: Path, **kwargs) -> None:
    """Save results to Excel file."""
    with pd.ExcelWriter(filepath, **kwargs) as writer:
        # Summary sheet
        summary = {k: v for k, v in results.items() 
                  if np.isscalar(v) or (isinstance(v, np.ndarray) and v.size == 1)}
        if summary:
            pd.DataFrame([summary]).to_excel(writer, sheet_name='Summary', index=False)
        
        # Array sheets
        for key, value in results.items():
            if isinstance(value, np.ndarray) and value.size > 1:
                if value.ndim == 1:
                    pd.DataFrame({key: value}).to_excel(writer, sheet_name=key[:31], index=False)
                else:
                    pd.DataFrame(value).to_excel(writer, sheet_name=key[:31], index=False)


def _save_hdf5(results: Dict[str, Any], filepath: Path, **kwargs) -> None:
    """Save results to HDF5 file."""
    import h5py
    
    with h5py.File(filepath, 'w') as f:
        for key, value in results.items():
            if isinstance(value, (np.ndarray, int, float, np.integer, np.floating)):
                f.create_dataset(key, data=value, **kwargs)
            elif isinstance(value, dict):
                group = f.create_group(key)
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, (np.ndarray, int, float, np.integer, np.floating)):
                        group.create_dataset(subkey, data=subvalue)