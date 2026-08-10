import os
import json
import gzip
import subprocess
from huggingface_hub import HfApi, hf_hub_download

def main():
    hf_token = os.environ.get("HF_TOKEN")
    hf_repo_id = os.environ.get("HF_REPO_ID")
    
    if not hf_token or not hf_repo_id:
        print("Error: HF_TOKEN and HF_REPO_ID environment variables must be set.")
        return
        
    api = HfApi(token=hf_token)
    
    # 1. Fetch State
    state_path = "SCAN_RESULTS/state.json"
    state = {"current_part": 0, "current_offset": 0}
    try:
        local_state = hf_hub_download(
            repo_id=hf_repo_id,
            repo_type="dataset",
            filename=state_path
        )
        with open(local_state, "r") as f:
            state = json.load(f)
        print(f"Loaded existing state: {state}")
    except Exception as e:
        print(f"State file not found or error loading ({e}). Starting from default state: {state}")
        
    current_part = state.get("current_part", 0)
    current_offset = state.get("current_offset", 0)
    
    part_filename = f"GLOBAL_MASTER_part_{current_part:02d}.gz"
    repo_part_path = f"GLOBAL_MASTERS/{part_filename}"
    
    # 2. Download Data
    print(f"Downloading {repo_part_path}...")
    try:
        local_part_path = hf_hub_download(
            repo_id=hf_repo_id,
            repo_type="dataset",
            filename=repo_part_path
        )
    except Exception as e:
        print(f"Failed to download {repo_part_path}. We may have reached the end of all parts! Error: {e}")
        return
        
    # 3. Extract Batch
    batch_size = 10_000
    batch_file = "batch_domains.txt"
    lines_extracted = 0
    
    print(f"Extracting {batch_size} lines starting from offset {current_offset}...")
    with gzip.open(local_part_path, 'rt', encoding='utf-8') as gz, open(batch_file, 'w', encoding='utf-8') as out:
        # Skip to offset
        for _ in range(current_offset):
            try:
                next(gz)
            except StopIteration:
                break
                
        # Read batch
        for _ in range(batch_size):
            try:
                line = next(gz)
                out.write(line)
                lines_extracted += 1
            except StopIteration:
                break
                
    print(f"Extracted {lines_extracted} domains.")
    
    if lines_extracted == 0:
        print("No lines extracted. The part may be empty or offset is past EOF. Trying next part...")
        # Update state to next part
        new_state = {
            "current_part": current_part + 1,
            "current_offset": 0
        }
        _upload_state(api, hf_repo_id, state_path, new_state)
        print("State updated to next part. Please run the workflow again.")
        return
        
    # 4. Scan
    output_csv = "results.csv"
    exe_path = os.path.join("bin", "dip-cli.exe")
    sig_path = os.path.join("bin", "signatures.yml")
    
    if not os.path.exists(exe_path):
        print(f"ERROR: Executable not found at {exe_path}")
        return
        
    cmd = [
        exe_path,
        "--input", batch_file,
        "--output", output_csv,
        "--signatures", sig_path,
        "--headless"
    ]
    
    print(f"Running scanner: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    
    # 5. Process and Upload Results
    import pandas as pd
    
    cms_counts = state.get("cms_counts", {})
    
    if os.path.exists(output_csv):
        df = pd.read_csv(output_csv)
        
        # Filter for success (Error column is empty or NaN)
        df_success = df[df['Error'].fillna('') == '']
        
        # Filter for known CMS
        df_known = df_success[df_success['Primary CMS'] != 'Unknown']
        
        # Update CMS counts
        for cms, count in df_known['Primary CMS'].value_counts().items():
            cms_counts[cms] = cms_counts.get(cms, 0) + int(count)
            
        # 5a. Save and Upload Part Results (Success only)
        result_path_in_repo = f"SCAN_RESULTS/results_part_{current_part:02d}_offset_{current_offset}.parquet"
        local_result_pq = f"results_part_{current_part:02d}_offset_{current_offset}.parquet"
        df_success.to_parquet(local_result_pq, index=False)
        
        print(f"Uploading part results to {result_path_in_repo}...")
        api.upload_file(
            path_or_fileobj=local_result_pq,
            path_in_repo=result_path_in_repo,
            repo_id=hf_repo_id,
            repo_type="dataset",
            commit_message=f"Add scan results for part {current_part} offset {current_offset}"
        )
        os.remove(local_result_pq)
        
        # 5b. Save and Upload Combined Results (Known CMS only)
        combined_path_in_repo = "SCAN_RESULTS/combined_results.parquet"
        local_combined = "combined_results.parquet"
        
        try:
            print(f"Downloading existing {combined_path_in_repo} to append...")
            downloaded_combined = hf_hub_download(
                repo_id=hf_repo_id,
                repo_type="dataset",
                filename=combined_path_in_repo
            )
            df_combined = pd.read_parquet(downloaded_combined)
            df_combined = pd.concat([df_combined, df_known], ignore_index=True)
            print("Appended to existing combined results.")
        except Exception:
            print("No existing combined results found. Creating new one.")
            df_combined = df_known
            
        df_combined.to_parquet(local_combined, index=False)
        print(f"Uploading {combined_path_in_repo}...")
        api.upload_file(
            path_or_fileobj=local_combined,
            path_in_repo=combined_path_in_repo,
            repo_id=hf_repo_id,
            repo_type="dataset",
            commit_message=f"Update combined results up to part {current_part} offset {current_offset}"
        )
        os.remove(local_combined)
    else:
        print(f"Warning: {output_csv} not found locally! Skipping results upload.")
    
    # 6. Update State
    next_offset = current_offset + lines_extracted
    if lines_extracted < batch_size:
        print(f"Reached end of part {current_part}. Advancing to next part.")
        new_state = {
            "current_part": current_part + 1,
            "current_offset": 0,
            "cms_counts": cms_counts
        }
    else:
        new_state = {
            "current_part": current_part,
            "current_offset": next_offset,
            "cms_counts": cms_counts
        }
        
    print(f"Updating state to: {new_state}")
    _upload_state(api, hf_repo_id, state_path, new_state)
    
    print("Batch scan complete!")

def _upload_state(api, repo_id, state_path, state_dict):
    local_state_file = "state.json"
    with open(local_state_file, "w") as f:
        json.dump(state_dict, f, indent=4)
        
    if os.path.exists(local_state_file):
        api.upload_file(
            path_or_fileobj=local_state_file,
            path_in_repo=state_path,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Update scan state"
        )

if __name__ == "__main__":
    main()
