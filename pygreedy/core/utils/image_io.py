"""
PyGreedy Image I/O Module
=======================

This module provides functions for loading and saving medical images
in various formats (NIfTI, DICOM, etc.).

Created by: devhliu
Created at: 2025-02-18 04:36:27 UTC
"""

import numpy as np
import nibabel as nib
import pydicom
from pathlib import Path
from typing import Tuple, Union, List, Optional, Dict
import logging
import torch

logger = logging.getLogger(__name__)

def load_image(
    path: Union[str, Path],
    dtype: Optional[np.dtype] = np.float32,
    normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load medical image from file.

    Parameters
    ----------
    path : Union[str, Path]
        Path to image file
    dtype : Optional[np.dtype], optional
        Data type for output array, by default np.float32
    normalize : bool, optional
        Whether to normalize image intensities, by default True

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple containing (image_data, affine_matrix)
    """
    path = Path(path)
    
    if path.suffix.lower() in ['.nii', '.nii.gz']:
        return load_nifti(path, dtype, normalize)
    elif path.suffix.lower() in ['.dcm']:
        return load_dicom(path, dtype, normalize)
    elif path.is_dir():
        return load_dicom_series(path, dtype, normalize)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

def save_image(
    path: Union[str, Path],
    image: np.ndarray,
    affine: Optional[np.ndarray] = None,
    dtype: Optional[np.dtype] = None
) -> None:
    """
    Save medical image to file.

    Parameters
    ----------
    path : Union[str, Path]
        Path to output file
    image : np.ndarray
        Image data
    affine : Optional[np.ndarray], optional
        Affine transformation matrix, by default None
    dtype : Optional[np.dtype], optional
        Data type for output file, by default None
    """
    path = Path(path)
    
    if dtype is not None:
        image = image.astype(dtype)
    
    if path.suffix.lower() in ['.nii', '.nii.gz']:
        save_nifti(path, image, affine)
    elif path.suffix.lower() == '.dcm':
        save_dicom(path, image)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

def load_nifti(
    path: Union[str, Path],
    dtype: Optional[np.dtype] = np.float32,
    normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load NIfTI image.

    Parameters
    ----------
    path : Union[str, Path]
        Path to NIfTI file
    dtype : Optional[np.dtype], optional
        Data type for output array, by default np.float32
    normalize : bool, optional
        Whether to normalize image intensities, by default True

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple containing (image_data, affine_matrix)
    """
    try:
        img = nib.load(str(path))
        data = img.get_fdata()
        affine = img.affine
        
        if normalize:
            data = normalize_intensity(data)
        
        if dtype is not None:
            data = data.astype(dtype)
            
        return data, affine
        
    except Exception as e:
        logger.error(f"Error loading NIfTI file {path}: {str(e)}")
        raise

def save_nifti(
    path: Union[str, Path],
    image: np.ndarray,
    affine: Optional[np.ndarray] = None
) -> None:
    """
    Save image as NIfTI file.

    Parameters
    ----------
    path : Union[str, Path]
        Path to output file
    image : np.ndarray
        Image data
    affine : Optional[np.ndarray], optional
        Affine transformation matrix, by default None
    """
    if affine is None:
        affine = np.eye(4)
        
    try:
        img = nib.Nifti1Image(image, affine)
        nib.save(img, str(path))
        
    except Exception as e:
        logger.error(f"Error saving NIfTI file {path}: {str(e)}")
        raise

def load_dicom(
    path: Union[str, Path],
    dtype: Optional[np.dtype] = np.float32,
    normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load single DICOM image.

    Parameters
    ----------
    path : Union[str, Path]
        Path to DICOM file
    dtype : Optional[np.dtype], optional
        Data type for output array, by default np.float32
    normalize : bool, optional
        Whether to normalize image intensities, by default True

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple containing (image_data, affine_matrix)
    """
    try:
        dcm = pydicom.dcmread(str(path))
        data = dcm.pixel_array
        affine = get_dicom_affine(dcm)
        
        if normalize:
            data = normalize_intensity(data)
            
        if dtype is not None:
            data = data.astype(dtype)
            
        return data, affine
        
    except Exception as e:
        logger.error(f"Error loading DICOM file {path}: {str(e)}")
        raise

def load_dicom_series(
    directory: Union[str, Path],
    dtype: Optional[np.dtype] = np.float32,
    normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load DICOM series from directory.

    Parameters
    ----------
    directory : Union[str, Path]
        Path to directory containing DICOM files
    dtype : Optional[np.dtype], optional
        Data type for output array, by default np.float32
    normalize : bool, optional
        Whether to normalize image intensities, by default True

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple containing (image_data, affine_matrix)
    """
    try:
        directory = Path(directory)
        dicom_files = sorted(
            list(directory.glob('*.dcm')),
            key=lambda x: float(pydicom.dcmread(str(x)).ImagePositionPatient[2])
        )
        
        # Load first slice to get image dimensions
        first_slice = pydicom.dcmread(str(dicom_files[0]))
        shape = (len(dicom_files), *first_slice.pixel_array.shape)
        data = np.zeros(shape, dtype=dtype)
        
        # Load all slices
        for i, file in enumerate(dicom_files):
            dcm = pydicom.dcmread(str(file))
            data[i] = dcm.pixel_array
            
        if normalize:
            data = normalize_intensity(data)
            
        # Get affine matrix from first slice
        affine = get_dicom_affine(first_slice)
        
        return data, affine
        
    except Exception as e:
        logger.error(f"Error loading DICOM series from {directory}: {str(e)}")
        raise

def save_dicom(
    path: Union[str, Path],
    image: np.ndarray,
    template_dicom: Optional[pydicom.dataset.FileDataset] = None
) -> None:
    """
    Save image as DICOM file.

    Parameters
    ----------
    path : Union[str, Path]
        Path to output file
    image : np.ndarray
        Image data
    template_dicom : Optional[pydicom.dataset.FileDataset], optional
        Template DICOM dataset to use for metadata, by default None
    """
    try:
        if template_dicom is None:
            # Create new DICOM dataset with minimal metadata
            ds = create_basic_dicom_dataset()
        else:
            ds = template_dicom.copy()
            
        ds.PixelData = image.astype(np.uint16).tobytes()
        ds.save_as(str(path))
        
    except Exception as e:
        logger.error(f"Error saving DICOM file {path}: {str(e)}")
        raise

def normalize_intensity(
    image: np.ndarray,
    out_range: Tuple[float, float] = (0, 1)
) -> np.ndarray:
    """
    Normalize image intensities to specified range.

    Parameters
    ----------
    image : np.ndarray
        Input image
    out_range : Tuple[float, float], optional
        Output intensity range, by default (0, 1)

    Returns
    -------
    np.ndarray
        Normalized image
    """
    min_val = np.min(image)
    max_val = np.max(image)
    
    if max_val == min_val:
        return np.zeros_like(image)
        
    normalized = (image - min_val) / (max_val - min_val)
    
    if out_range != (0, 1):
        normalized = normalized * (out_range[1] - out_range[0]) + out_range[0]
        
    return normalized

def get_dicom_affine(dcm: pydicom.dataset.FileDataset) -> np.ndarray:
    """
    Extract affine matrix from DICOM metadata.

    Parameters
    ----------
    dcm : pydicom.dataset.FileDataset
        DICOM dataset

    Returns
    -------
    np.ndarray
        4x4 affine transformation matrix
    """
    # Initialize affine matrix
    affine = np.eye(4)
    
    try:
        # Get pixel spacing
        spacing = dcm.PixelSpacing
        spacing = [float(s) for s in spacing]
        
        # Get image position
        position = dcm.ImagePositionPatient
        position = [float(p) for p in position]
        
        # Get image orientation
        orientation = dcm.ImageOrientationPatient
        orientation = [float(o) for o in orientation]
        row_ori = np.array(orientation[:3])
        col_ori = np.array(orientation[3:])
        slice_ori = np.cross(row_ori, col_ori)
        
        # Build affine matrix
        affine[:3, 0] = row_ori * spacing[0]
        affine[:3, 1] = col_ori * spacing[1]
        affine[:3, 2] = slice_ori * dcm.SliceThickness
        affine[:3, 3] = position
        
    except Exception as e:
        logger.warning(f"Error extracting DICOM affine matrix: {str(e)}")
        
    return affine

def create_basic_dicom_dataset() -> pydicom.dataset.FileDataset:
    """
    Create basic DICOM dataset with minimal required metadata.

    Returns
    -------
    pydicom.dataset.FileDataset
        Basic DICOM dataset
    """
    # Create file meta information
    file_meta = pydicom.dataset.FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.ImplementationClassUID = pydicom.uid.generate_uid()
    
    # Create dataset
    ds = pydicom.dataset.FileDataset(
        None,
        {},
        file_meta=file_meta,
        preamble=b"\0" * 128
    )
    
    # Add required metadata
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = pydicom.uid.generate_uid()
    ds.SeriesInstanceUID = pydicom.uid.generate_uid()
    
    return ds