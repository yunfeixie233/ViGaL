import argparse
import os
import subprocess
import glob
import shutil
from huggingface_hub import snapshot_download

def main(repo_id, local_dir, filename=None):
    """
    Downloads a Hugging Face dataset snapshot and extracts .tar.gz files
    directly to the target directory without linking.
    
    If filename is specified, only that file will be downloaded from the repository.
    """
    # --- Configuration ---
    # Enable hf_transfer for potentially faster downloads
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    repo_type = "dataset"  # Assuming it's always a dataset for this script

    # Expand the user's home directory if '~' is used
    expanded_local_dir = os.path.expanduser(local_dir)
    print(f"Target local directory: {expanded_local_dir}")

    # Ensure the target directory exists before downloading/extracting
    os.makedirs(expanded_local_dir, exist_ok=True)

    # --- Step 1: Download the snapshot ---
    if filename:
        file_path = os.path.join(expanded_local_dir, filename)    
        # If file exists, remove it for re-download
        if os.path.exists(file_path):
            print(f"File '{filename}' already exists in '{expanded_local_dir}'. Removing for re-download.")
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Error removing existing file: {e}")
                return
        
        print(f"\nDownloading file '{filename}' from repo '{repo_id}' ({repo_type}) to '{expanded_local_dir}'...")
        try:
            # Direct download to target directory, no symlinks
            snapshot_download(
                repo_id=repo_id,
                repo_type=repo_type,
                local_dir=expanded_local_dir,
                local_dir_use_symlinks=False,  # Ensure no symlinks are used
                allow_patterns=[filename]
            )
            print(f"Download of file '{filename}' completed successfully.")
        except Exception as e:
            print(f"Error during download: {e}")
            return  # Stop execution if download fails
        
        # Check if the downloaded file is a .tar.gz file that needs extraction
        if filename.endswith('.tar.gz') and os.path.exists(file_path):
            file_name = os.path.basename(file_path)
            expected_output_dir_name = file_name[:-len(".tar.gz")]
            expected_output_path = os.path.join(expanded_local_dir, expected_output_dir_name)
            
            # Check if target directory exists and remove it
            if os.path.isdir(expected_output_path):
                print(f"  Target directory '{expected_output_path}' exists. Removing for fresh extraction...")
                try:
                    shutil.rmtree(expected_output_path)
                    print(f"  Successfully removed existing directory '{expected_output_path}'.")
                except Exception as e:
                    print(f"  Error removing directory '{expected_output_path}': {e}")
                    return
            
            # Extract directly to target directory
            print(f"  Extracting '{file_name}' directly to '{expanded_local_dir}'...")
            try:
                # Direct extraction to target directory
                cmd = ["tar", "-xzf", file_path, "-C", expanded_local_dir]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"  Successfully extracted '{file_name}' to '{expanded_local_dir}'.")
                else:
                    print(f"  An error occurred during extraction of '{file_name}': {result.stderr}")
            except Exception as e:
                print(f"  An error occurred during extraction of '{file_name}': {e}.")
        return  # Done processing the specific file
    else:
        print(f"\nDownloading snapshot for repo '{repo_id}' ({repo_type}) to '{expanded_local_dir}'...")
        try:
            # Direct download to target directory, no symlinks
            snapshot_download(
                repo_id=repo_id,
                repo_type=repo_type,
                local_dir=expanded_local_dir,
                local_dir_use_symlinks=False  # Ensure no symlinks are used
            )
            print("Download completed successfully.")
        except Exception as e:
            print(f"Error during download: {e}")
            return  # Stop execution if download fails

    # --- Step 2: Unzip .tar.gz files ---
    print(f"\nSearching for .tar.gz files in '{expanded_local_dir}'...")

    # Use glob to find all files ending with .tar.gz in the target directory
    search_pattern = os.path.join(expanded_local_dir, "*.tar.gz")
    tar_files = glob.glob(search_pattern)

    if not tar_files:
        print("No .tar.gz files found to extract.")
        return

    print(f"Found {len(tar_files)} '.tar.gz' file(s). Starting extraction...")

    extracted_count = 0
    skipped_count = 0

    for tar_path in tar_files:
        file_name = os.path.basename(tar_path)

        # --- Skip specific files if needed ---
        if "meshy" in file_name:
             print(f"  Skipping specific file '{file_name}' based on name.")
             skipped_count += 1
             continue

        # --- Check if already extracted ---
        if file_name.endswith(".tar.gz"):
             expected_output_dir_name = file_name[:-len(".tar.gz")]
        else:
             # Should not happen with glob pattern, but handle defensively
             print(f"  Skipping unexpected file '{file_name}' (doesn't end with .tar.gz).")
             continue

        expected_output_path = os.path.join(expanded_local_dir, expected_output_dir_name)

        # Check if target directory exists and remove it
        if os.path.isdir(expected_output_path):
            print(f"  Target directory '{expected_output_path}' exists. Removing for fresh extraction...")
            try:
                shutil.rmtree(expected_output_path)
                print(f"  Successfully removed existing directory '{expected_output_path}'.")
            except Exception as e:
                print(f"  Error removing directory '{expected_output_path}': {e}")
                skipped_count += 1
                continue

        # --- Proceed with Extraction ---
        print(f"  Extracting '{file_name}' directly to '{expanded_local_dir}'...")
        try:
            # Direct extraction to target directory
            cmd = ["tar", "-xzf", tar_path, "-C", expanded_local_dir]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"  Successfully extracted '{file_name}' to '{expanded_local_dir}'.")
                extracted_count += 1
            else:
                print(f"  Error extracting tar file '{file_name}': {result.stderr}. Skipping.")
                skipped_count += 1

        except Exception as e:
            print(f"  An unexpected error occurred during extraction of '{file_name}': {e}. Skipping.")
            skipped_count += 1

    print(f"\nExtraction process finished. Extracted: {extracted_count}, Skipped: {skipped_count}.")

if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Download a Hugging Face dataset snapshot and extract all .tar.gz files directly to the target directory."
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="SCMayS/hydata",
        help="The repository ID on Hugging Face Hub (e.g., 'SCMayS/hydata')."
    )
    parser.add_argument(
        "--local_dir",
        type=str,
        default="./data",
        help="The local directory to download the dataset to and extract archives into (e.g., '~/my_datasets/hydata' or './data'). Supports '~' for home directory."
    )
    parser.add_argument(
        "filename",
        type=str,
        nargs="?",  # Make this an optional positional argument
        default="",
        help="The filename to download from the dataset."
    )
    args = parser.parse_args()

    # --- Run the main function ---
    main(args.repo_id, args.local_dir, args.filename)