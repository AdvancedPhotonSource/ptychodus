# Manual: Implementing and Using HDF5 File Loaders

## Introduction

This manual is designed to guide developers through the process of implementing and using HDF5 file loaders within the repository's existing framework for handling diffraction data. The repository employs a plugin system for diffraction file readers, with `DiffractionFileReader` serving as the base class for all plugins.

## Prerequisites

- Familiarity with Python and object-oriented programming.
- Basic understanding of the HDF5 file format and the `h5py` library.
- Knowledge of the repository's structure and the `DiffractionFileReader` interface.

## Overview of the File Reader Interface

The `DiffractionFileReader` abstract base class defines the interface for all file reader plugins. Implementations must provide methods for reading diffraction data and metadata from files and constructing appropriate data structures for further processing.

### Key Methods

- `read(filePath: Path) -> DiffractionDataset`: Reads diffraction data from the specified file and returns a `DiffractionDataset` object containing the data and metadata.

## Implementing a Custom HDF5 File Loader

To implement a custom HDF5 file loader, follow these steps:

### Step 1: Define Your Loader Class

Create a new Python class that inherits from `DiffractionFileReader`. This class will implement the logic for reading diffraction data from your specific HDF5 file format.

```python
from pathlib import Path
import h5py
from .data import DiffractionFileReader, SimpleDiffractionDataset

class MyHDF5FileReader(DiffractionFileReader):
    def __init__(self, dataPath: str):
        self._dataPath = dataPath
```

### Step 2: Implement the Read Method

Implement the `read` method to open the HDF5 file, navigate its structure, and extract diffraction patterns and metadata. Use the `h5py` library to interact with the HDF5 file.

```python
def read(self, filePath: Path) -> SimpleDiffractionDataset:
    with h5py.File(filePath, 'r') as h5File:
        # Navigate the file and extract data
        data = h5File[self._dataPath][()]

        # Construct and return a SimpleDiffractionDataset object
        return SimpleDiffractionDataset(...)
```

### Step 3: Handle Metadata and Errors

Extract relevant metadata from the HDF5 file and handle potential errors, such as missing data paths or incompatible file formats.

### Step 4: Register Your Loader

Once your loader is implemented, register it with the plugin system to make it available for use.

```python
from .plugins import PluginRegistry

def registerPlugins(registry: PluginRegistry) -> None:
    registry.diffractionFileReaders.registerPlugin(
        MyHDF5FileReader(dataPath='/my/data/path'),
        simpleName='MyFormat',
        displayName='My Custom HDF5 Format (*.h5)',
    )
```

## Using the HDF5 File Loader

With the custom HDF5 file loader implemented and registered, it can be used to read diffraction data from files in the specified format. The loader will be automatically available to any part of the application that utilizes the plugin system for reading diffraction files.

## Conclusion

This manual provides a starting point for extending the repository's capabilities to handle additional HDF5 file formats for diffraction data. By following the outlined steps, developers can implement custom file loaders tailored to their specific data structures and experimental setups.
