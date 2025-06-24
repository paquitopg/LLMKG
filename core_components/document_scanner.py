# File: llm_kg_extraction/core_components/document_scanner.py

from pathlib import Path
from typing import List

def discover_pdf_files(input_path_str: str, recursive: bool = True) -> List[Path]:
    """
    Scans a given path for PDF files. The path can be a directory or a single PDF file.

    Args:
        input_path_str (str): The path to the folder to scan, or a direct path to a PDF file.
        recursive (bool): If True and input_path is a directory, scans subdirectories recursively.

    Returns:
        List[Path]: A list of Path objects for the discovered PDF files.
    """
    # Convert the input string path to a Path object immediately
    input_path = Path(input_path_str) 

    print(f"Scanning for PDF files in: {input_path} (recursive: {recursive})")
    
    pdf_files: List[Path] = []

    if input_path.is_file():
        # If the input is a single file, check if it's a PDF
        if input_path.suffix.lower() == '.pdf':
            pdf_files.append(input_path)
            print(f"Found single PDF file: {input_path.name}")
        else:
            print(f"Provided path '{input_path.name}' is a file but not a PDF. Skipping.")
    elif input_path.is_dir():
        # If the input is a directory, scan for PDF files
        if recursive:
            # Use rglob for recursive search
            for p in input_path.rglob('*.pdf'):
                pdf_files.append(p)
        else:
            # Use glob for non-recursive search (only top-level)
            for p in input_path.glob('*.pdf'):
                pdf_files.append(p)
        print(f"Found {len(pdf_files)} PDF files in directory: {input_path}")
    else:
        # Handle cases where the path doesn't exist or is not a file/directory
        print(f"Error: Provided path '{input_path_str}' is not a valid file or directory. Please check the path.")
    
    return pdf_files