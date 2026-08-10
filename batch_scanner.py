import os
import json
import gzip
import subprocess
import argparse
import time
import random
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker-id', type=int, required=True, help="ID of this worker in the matrix")
    parser.add_argument('--total-workers', type=int, required=True, help="Total number of workers")
    parser.add_argument('--batch-size', type=int, default=100000, help="Domains to process per run")
    parser.add_argument('--threads', type=int, default=100, help="Number of concurrent threads for dip-cli")
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    hf_repo_id = os.environ.get("HF_REPO_ID")
    
    if not hf_token or not hf_repo_id:
        print("Error: HF_TOKEN and HF_REPO_ID environment variables must be set.")
        return
        
    api = HfApi(token=hf_token)
    
    # 1. Fetch State for this specific worker
    state_path = f"SCAN_RESULTS/state/worker_{args.worker_id:02d}.json"
    
    # Default initialization
    state = {
        "current_part": 0, 
        "current_offset": args.worker_id * args.batch_size,
        "cms_counts": {}
    }
    
    try:
        local_state = hf_hub_download(
            repo_id=hf_repo_id,
            repo_type="dataset",
            filename=state_path
        )
        with open(local_state, "r") as f:
            state = json.load(f)
        print(f"Worker {args.worker_id} loaded existing state: {state}")
    except Exception as e:
        print(f"Worker {args.worker_id} state file not found ({e}). Starting from default state: {state}")
        
    current_part = state.get("current_part", 0)
    current_offset = state.get("current_offset", 0)
    cms_counts = state.get("cms_counts", {})
    
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
    batch_file = "batch_domains.txt"
    lines_extracted = 0
    
    print(f"Worker {args.worker_id} extracting {args.batch_size} lines starting from offset {current_offset}...")
    with gzip.open(local_part_path, 'rt', encoding='utf-8') as gz, open(batch_file, 'w', encoding='utf-8') as out:
        # Skip to offset
        for _ in range(current_offset):
            try:
                next(gz)
            except StopIteration:
                break
                
        # Read batch
        for _ in range(args.batch_size):
            try:
                line = next(gz)
                out.write(line)
                lines_extracted += 1
            except StopIteration:
                break
                
    print(f"Worker {args.worker_id} extracted {lines_extracted} domains.")
    
    if lines_extracted == 0:
        print("No lines extracted. The part is empty or offset is past EOF. Advancing to next part...")
        # Update state to next part
        new_state = {
            "current_part": current_part + 1,
            "current_offset": args.worker_id * args.batch_size,
            "cms_counts": cms_counts
        }
        _upload_state(api, hf_repo_id, state_path, new_state, args.worker_id)
        print("State updated. Run workflow again to process next part.")
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
        "--threads", str(args.threads),
        "--headless"
    ]
    
    print(f"Running scanner: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    
    # 5. Process and Upload Results
    if os.path.exists(output_csv):
        df = pd.read_csv(output_csv)
        
        # Filter for success (Error column is empty or NaN)
        df_success = df[df['Error'].fillna('') == '']
        
        # Filter for known CMS
        df_known = df_success[df_success['Primary CMS'] != 'Unknown']
        
        # Update CMS counts
        for cms, count in df_known['Primary CMS'].value_counts().items():
            cms_counts[cms] = cms_counts.get(cms, 0) + int(count)
            
        # Add Jitter to prevent Hugging Face API rate limits / conflicts during matrix upload
        jitter = random.uniform(1, 30)
        print(f"Applying {jitter:.1f}s jitter before uploading results...")
        time.sleep(jitter)
        
        # 5a. Save and Upload Full Part Results (Success only)
        result_path_in_repo = f"SCAN_RESULTS/full/results_part_{current_part:02d}_worker_{args.worker_id:02d}_offset_{current_offset}.parquet"
        local_result_pq = "temp_results.parquet"
        df_success.to_parquet(local_result_pq, index=False)
        
        print(f"Uploading part results to {result_path_in_repo}...")
        api.upload_file(
            path_or_fileobj=local_result_pq,
            path_in_repo=result_path_in_repo,
            repo_id=hf_repo_id,
            repo_type="dataset",
            commit_message=f"Add scan results part {current_part} W{args.worker_id} offset {current_offset}"
        )
        os.remove(local_result_pq)
        
        # 5b. Save and Upload Combined Results (Known CMS only)
        # HF automatically merges all parquet files in a directory!
        combined_path_in_repo = f"SCAN_RESULTS/combined/known_part_{current_part:02d}_worker_{args.worker_id:02d}_offset_{current_offset}.parquet"
        local_combined = "temp_combined.parquet"
        
        df_known.to_parquet(local_combined, index=False)
        print(f"Uploading combined block to {combined_path_in_repo}...")
        api.upload_file(
            path_or_fileobj=local_combined,
            path_in_repo=combined_path_in_repo,
            repo_id=hf_repo_id,
            repo_type="dataset",
            commit_message=f"Add known CMS block part {current_part} W{args.worker_id} offset {current_offset}"
        )
        os.remove(local_combined)
    else:
        print(f"Warning: {output_csv} not found locally! Skipping results upload.")
    
    # 6. Update State
    next_offset = current_offset + (args.total_workers * args.batch_size)
    
    if lines_extracted < args.batch_size:
        print(f"Worker {args.worker_id} reached end of part {current_part}. Advancing to next part.")
        new_state = {
            "current_part": current_part + 1,
            "current_offset": args.worker_id * args.batch_size,
            "cms_counts": cms_counts
        }
    else:
        new_state = {
            "current_part": current_part,
            "current_offset": next_offset,
            "cms_counts": cms_counts
        }
        
    print(f"Updating state to: {new_state}")
    _upload_state(api, hf_repo_id, state_path, new_state, args.worker_id)
    
    print(f"Worker {args.worker_id} batch scan complete!")

def _upload_state(api, repo_id, state_path, state_dict, worker_id):
    local_state_file = f"state_worker_{worker_id}.json"
    with open(local_state_file, "w") as f:
        json.dump(state_dict, f, indent=4)
        
    if os.path.exists(local_state_file):
        api.upload_file(
            path_or_fileobj=local_state_file,
            path_in_repo=state_path,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Update scan state for worker {worker_id}"
        )

if __name__ == "__main__":
    main()
