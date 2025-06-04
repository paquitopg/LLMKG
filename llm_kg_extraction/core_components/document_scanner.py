from pathlib import Path
from typing import List

def discover_pdf_files(folder_path: Path, recursive: bool = True) -> List[Path]:
    """
    Discovers PDF files in the given folder.

    Args:
        folder_path (Path): The path to the folder to scan.
        recursive (bool): Whether to scan subfolders recursively. Defaults to True.

    Returns:
        List[Path]: A list of Path objects for the found PDF files.
    """
    print(f"Scanning for PDF files in: {folder_path} (recursive: {recursive})")
    pdf_files: List[Path] = []

    if not folder_path.is_dir():
        print(f"Error: Provided path '{folder_path}' is not a directory or does not exist.")
        return pdf_files

    if recursive:
        for p in folder_path.rglob('*.pdf'):
            if p.is_file(): # Make sure it's a file
                pdf_files.append(p)
    else:
        for p in folder_path.glob('*.pdf'):
            if p.is_file(): # Make sure it's a file
                pdf_files.append(p)
    
    print(f"Found {len(pdf_files)} PDF files in '{folder_path}'.")
    return pdf_files