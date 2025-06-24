import argparse
import json
import shutil
import re
from pathlib import Path

def sanitize_filename_component(name_part: str) -> str:
    """
    Sanitizes a string component to be safe for use in filenames.
    Replaces common problematic characters with an underscore.
    """
    if not isinstance(name_part, str):
        name_part = str(name_part)
    # Replace common OS-problematic characters for filenames
    return re.sub(r'[<>:"/\\|?*]', '_', name_part)

def create_detailed_structure(transformed_output_dir_path: str):
    """
    Creates a detailed directory structure from the transformed KG output files.
    """
    input_dir = Path(transformed_output_dir_path)

    if not input_dir.is_dir():
        print(f"Error: Input path '{input_dir}' is not a valid directory.")
        return

    # Create base 'detailed_outputs' directory and subdirectories
    detailed_outputs_dir = input_dir / "detailed_outputs"
    index1_dir = detailed_outputs_dir / "index1"
    index2_dir = detailed_outputs_dir / "index2"
    index3_dir = detailed_outputs_dir / "index3"

    try:
        detailed_outputs_dir.mkdir(exist_ok=True)
        index1_dir.mkdir(exist_ok=True)
        index2_dir.mkdir(exist_ok=True)
        index3_dir.mkdir(exist_ok=True)
        print(f"Successfully created directory structure under '{detailed_outputs_dir}'.")
    except OSError as e:
        print(f"Error creating directories: {e}")
        return

    # --- 1. Process meta file for index1 ---
    # transform_json.py creates filenames like: {extraction_mode}_{model_name}_{llm_provider}_{construction_mode}_meta.json
    meta_files = list(input_dir.glob("*_meta.json"))
    if not meta_files:
        print(f"Warning: No '*_meta.json' file found in '{input_dir}'. Skipping index1 population.")
    else:
        source_meta_file = meta_files[0]
        if len(meta_files) > 1:
            print(f"Warning: Multiple '*_meta.json' files found. Using '{source_meta_file.name}'.")
        try:
            destination_meta_file = index1_dir / source_meta_file.name
            shutil.copy2(source_meta_file, destination_meta_file)
            print(f"Copied '{source_meta_file.name}' to '{destination_meta_file}'.")
        except Exception as e:
            print(f"Error copying meta file '{source_meta_file.name}': {e}")

    # --- 2. Process nodes file for index2 ---
    # transform_json.py creates filenames like: {extraction_mode}_{model_name}_{llm_provider}_{construction_mode}_nodes.json
    # Each node object in this file has an "id" field (the composite ID).
    nodes_files = list(input_dir.glob("*_nodes.json"))
    if not nodes_files:
        print(f"Warning: No '*_nodes.json' file found in '{input_dir}'. Skipping index2 population.")
    else:
        source_nodes_file = nodes_files[0]
        if len(nodes_files) > 1:
            print(f"Warning: Multiple '*_nodes.json' files found. Using '{source_nodes_file.name}'.")
        try:
            with open(source_nodes_file, 'r', encoding='utf-8') as f:
                nodes_data = json.load(f)

            if not isinstance(nodes_data, list):
                print(f"Error: Content of '{source_nodes_file.name}' is not a list of nodes.")
            else:
                node_count = 0
                for i, node_obj in enumerate(nodes_data):
                    if not isinstance(node_obj, dict):
                        print(f"Warning: Item at index {i} in '{source_nodes_file.name}' is not a dictionary. Skipping.")
                        continue
                    
                    node_id = node_obj.get("id") # This "id" is the composite ID from transform_json.py
                    if node_id is None:
                        print(f"Warning: Node object at index {i} in '{source_nodes_file.name}' is missing an 'id'. Skipping. Node: {str(node_obj)[:100]}")
                        continue
                    
                    safe_node_id_for_filename = sanitize_filename_component(node_id)
                    node_filename = f"actor-{safe_node_id_for_filename}.node.json"
                    node_filepath = index2_dir / node_filename
                    
                    try:
                        with open(node_filepath, 'w', encoding='utf-8') as nf:
                            json.dump(node_obj, nf, indent=2, ensure_ascii=False)
                        node_count += 1
                    except Exception as e:
                        print(f"Error writing node file for ID '{node_id}': {e}")
                print(f"Processed {node_count} nodes from '{source_nodes_file.name}' into '{index2_dir}'.")

        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from '{source_nodes_file.name}'.")
        except Exception as e:
            print(f"Error processing nodes file '{source_nodes_file.name}': {e}")

    # --- 3. Process links file for index3 ---
    # transform_json.py creates filenames like: {extraction_mode}_{model_name}_{llm_provider}_{construction_mode}_links.json
    # Each link object in this file has a "key" field (a UUID).
    links_files = list(input_dir.glob("*_links.json"))
    if not links_files:
        print(f"Warning: No '*_links.json' file found in '{input_dir}'. Skipping index3 population.")
    else:
        source_links_file = links_files[0]
        if len(links_files) > 1:
            print(f"Warning: Multiple '*_links.json' files found. Using '{source_links_file.name}'.")
        try:
            with open(source_links_file, 'r', encoding='utf-8') as f:
                links_data = json.load(f)

            if not isinstance(links_data, list):
                print(f"Error: Content of '{source_links_file.name}' is not a list of links.")
            else:
                link_count = 0
                # This loop creates one file per LINK object.
                for i, link_obj in enumerate(links_data):
                    if not isinstance(link_obj, dict):
                        print(f"Warning: Item at index {i} in '{source_links_file.name}' is not a dictionary. Skipping.")
                        continue

                    link_obj_key = link_obj.get("key") # Correctly gets the field named "key"
                    if link_obj_key is None:
                        print(f"Warning: Link object at index {i} in '{source_links_file.name}' is missing a 'key'. Skipping. Link: {str(link_obj)[:100]}")
                        continue
                    
                    safe_link_key_for_filename = sanitize_filename_component(link_obj_key)
                    # Filename is "link-" followed by the value of the "key" field.
                    link_filename = f"{safe_link_key_for_filename}.link.json"
                    link_filepath = index3_dir / link_filename

                    try:
                        with open(link_filepath, 'w', encoding='utf-8') as lf:
                            json.dump(link_obj, lf, indent=2, ensure_ascii=False)
                        link_count += 1
                    except Exception as e:
                        print(f"Error writing link file for key '{link_obj_key}': {e}")
                print(f"Processed {link_count} links from '{source_links_file.name}' into '{index3_dir}'.")
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from '{source_links_file.name}'.")
        except Exception as e:
            print(f"Error processing links file '{source_links_file.name}': {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Creates a detailed output structure from transformed knowledge graph JSON files."
    )
    parser.add_argument(
        "transformed_output_dir",
        help="Path to the directory containing the _meta.json, _nodes.json, and _links.json files."
    )
    args = parser.parse_args()

    create_detailed_structure(args.transformed_output_dir)

if __name__ == "__main__":
    main()