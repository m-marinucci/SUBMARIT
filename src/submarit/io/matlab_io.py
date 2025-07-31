"""MATLAB file I/O utilities."""

import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Union

import h5py
import numpy as np
import scipy.io as sio
from numpy.typing import NDArray


def load_mat(
    filepath: Union[str, Path],
    variable_names: Optional[list] = None,
    matlab_compatible: bool = True
) -> Dict[str, Any]:
    """Load data from a MATLAB .mat file.
    
    Handles both old-style (< v7.3) and new-style (>= v7.3) .mat files.
    
    Args:
        filepath: Path to the .mat file
        variable_names: List of variable names to load (None = load all)
        matlab_compatible: Whether to maintain MATLAB compatibility
        
    Returns:
        Dictionary of loaded variables
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    try:
        # Try loading with scipy (works for v4, v6, v7 up to v7.2)
        mat_data = sio.loadmat(
            str(filepath),
            squeeze_me=not matlab_compatible,
            mat_dtype=True
        )
        
        # Remove metadata keys
        mat_data = {k: v for k, v in mat_data.items() 
                   if not k.startswith('__')}
        
    except NotImplementedError:
        # Fall back to h5py for v7.3 files
        mat_data = _load_mat_hdf5(filepath, variable_names)
    
    # Filter variables if requested
    if variable_names is not None:
        mat_data = {k: v for k, v in mat_data.items() 
                   if k in variable_names}
    
    return mat_data


def _load_mat_hdf5(
    filepath: Path,
    variable_names: Optional[list] = None
) -> Dict[str, Any]:
    """Load MATLAB v7.3 files using HDF5."""
    data = {}
    
    with h5py.File(filepath, 'r') as f:
        for key in f.keys():
            if variable_names is None or key in variable_names:
                data[key] = _read_hdf5_dataset(f[key])
    
    return data


def _read_hdf5_dataset(dataset: h5py.Dataset) -> Any:
    """Read HDF5 dataset handling MATLAB specifics."""
    if isinstance(dataset, h5py.Dataset):
        data = dataset[()]
        
        # Handle MATLAB's column-major storage
        if data.ndim > 1:
            data = data.T
            
        # Handle MATLAB strings
        if data.dtype.type is np.bytes_:
            data = data.tobytes().decode('utf-8')
            
        return data
    elif isinstance(dataset, h5py.Group):
        # Handle MATLAB structures
        return {key: _read_hdf5_dataset(dataset[key]) 
                for key in dataset.keys()}
    else:
        return dataset


def save_mat(
    filepath: Union[str, Path],
    data: Dict[str, Any],
    format: str = '5',
    do_compression: bool = False
) -> None:
    """Save data to a MATLAB .mat file.
    
    Args:
        filepath: Output file path
        data: Dictionary of variables to save
        format: MATLAB file format ('5' or '7.3')
        do_compression: Whether to compress the data
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if format == '7.3':
        _save_mat_hdf5(filepath, data, do_compression)
    else:
        sio.savemat(
            str(filepath),
            data,
            format='5',
            do_compression=do_compression
        )


def _save_mat_hdf5(
    filepath: Path,
    data: Dict[str, Any],
    do_compression: bool = False
) -> None:
    """Save MATLAB v7.3 files using HDF5."""
    compression = 'gzip' if do_compression else None
    
    with h5py.File(filepath, 'w') as f:
        for key, value in data.items():
            _write_hdf5_dataset(f, key, value, compression)


def _write_hdf5_dataset(
    group: h5py.Group,
    name: str,
    data: Any,
    compression: Optional[str] = None
) -> None:
    """Write data to HDF5 handling MATLAB specifics."""
    if isinstance(data, (np.ndarray, list)):
        data = np.asarray(data)
        
        # Convert to column-major for MATLAB
        if data.ndim > 1:
            data = data.T
            
        group.create_dataset(name, data=data, compression=compression)
        
    elif isinstance(data, dict):
        # Handle structures
        subgroup = group.create_group(name)
        for key, value in data.items():
            _write_hdf5_dataset(subgroup, key, value, compression)
            
    elif isinstance(data, str):
        # Handle strings
        group.create_dataset(
            name,
            data=np.bytes_(data),
            compression=compression
        )
    else:
        # Handle scalars
        group.create_dataset(name, data=data, compression=compression)


def convert_mat_to_npz(
    mat_filepath: Union[str, Path],
    npz_filepath: Union[str, Path],
    compressed: bool = True
) -> None:
    """Convert a .mat file to NumPy .npz format.
    
    Args:
        mat_filepath: Input .mat file path
        npz_filepath: Output .npz file path
        compressed: Whether to use compression
    """
    data = load_mat(mat_filepath)
    
    if compressed:
        np.savez_compressed(npz_filepath, **data)
    else:
        np.savez(npz_filepath, **data)


def validate_mat_file(filepath: Union[str, Path]) -> Dict[str, Any]:
    """Validate and get information about a .mat file.
    
    Args:
        filepath: Path to the .mat file
        
    Returns:
        Dictionary with file information
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    info = {
        'filepath': str(filepath),
        'size_bytes': filepath.stat().st_size,
        'variables': {},
        'format': None,
        'loadable': True,
        'errors': []
    }
    
    try:
        # Try scipy.io first
        mat_info = sio.whosmat(str(filepath))
        info['format'] = 'v5/v7'
        info['variables'] = {
            name: {'shape': shape, 'dtype': dtype}
            for name, shape, dtype in mat_info
        }
    except:
        try:
            # Try HDF5
            with h5py.File(filepath, 'r') as f:
                info['format'] = 'v7.3'
                for key in f.keys():
                    dataset = f[key]
                    if isinstance(dataset, h5py.Dataset):
                        info['variables'][key] = {
                            'shape': dataset.shape,
                            'dtype': str(dataset.dtype)
                        }
        except Exception as e:
            info['loadable'] = False
            info['errors'].append(str(e))
    
    return info