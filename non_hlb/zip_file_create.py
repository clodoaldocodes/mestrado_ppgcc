import zipfile
import os

def zip_folder(folder_path, zip_path):
    with zipfile.ZipFile(zip_path, 'w') as zip_file:
        for folder_root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(folder_root, file)
                arcname = os.path.relpath(file_path, folder_path)
                zip_file.write(file_path, arcname=arcname)

# Example usage:
folder_to_zip = "/home/clodoaldo/Documentos/non_HLB/yolo_test/"
zip_file_path = "/home/clodoaldo/Documentos/non_HLB/yolo_test/archive.zip"

zip_folder(folder_to_zip, zip_file_path)
