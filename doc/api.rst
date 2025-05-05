#!/usr/bin/env python

from pathlib import Path
from ptychodus.model import ModelCore

def main() -> int:
    settings_file = Path("path/to/settings.ini")

    with ModelCore(settings_file) as model:
        input_product_api = model.workflow_api.create_product("new_product_name")
        output_product_api = input_product_api.reconstruct_local()
        output_product_api.save_product("/path/to/file.h5", file_type="HDF5")
