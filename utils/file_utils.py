# utils/file_utils.py

import os
import shutil
from fastapi import UploadFile
from typing import List

def save_uploaded_files(files: List[UploadFile], data_folder: str) -> list:
    """
    保存上传的文件到指定的文件夹，并返回文件路径和扩展名的列表。

    :param files: 上传的文件列表 (FastAPI UploadFile 实例)。
    :param data_folder: 保存文件的目标文件夹。
    :return: 保存的文件路径和文件扩展名的列表。
    """
    os.makedirs(data_folder, exist_ok=True)

    saved_files = []
    for file in files:
        try:
            file_extension = os.path.splitext(file.filename)[1]
            if not file_extension:
                raise ValueError("Uploaded file must have an extension.")

            file_path = os.path.join(data_folder, file.filename)

            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            saved_files.append({"file_path": file_path, "file_extension": file_extension.lower()})
        except Exception as e:
            raise ValueError(f"Error saving file {file.filename}: {e}")

    return saved_files
