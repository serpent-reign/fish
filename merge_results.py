import os
import glob
import pandas as pd
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub import CommitOperationAdd

def merge_and_save(repo_id, folder_path, output_filename):
    print(f"Downloading shards from {folder_path}...")
    try:
        local_dir = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns=f"{folder_path}/*.parquet"
        )
    except Exception as e:
        print(f"Failed to download from {folder_path}: {e}")
        return None, None

    # snapshot_download preserves the folder structure locally
    shard_pattern = os.path.join(local_dir, folder_path, "*.parquet")
    shard_files = glob.glob(shard_pattern)
    
    if not shard_files:
        print(f"No parquet files found in {folder_path}.")
        return None, None

    print(f"Found {len(shard_files)} shards. Merging...")
    
    # Read and concatenate all parquets
    dfs = []
    for file in shard_files:
        try:
            dfs.append(pd.read_parquet(file))
        except Exception as e:
            print(f"Error reading {file}: {e}")
            
    if not dfs:
        print("No valid dataframes to merge.")
        return None, None
        
    merged_df = pd.concat(dfs, ignore_index=True)
    
    if 'Domain' in merged_df.columns:
        merged_df = merged_df.drop_duplicates(subset=['Domain'], keep='last')
        
    print(f"Merged into {len(merged_df)} total rows. Saving to {output_filename}...")
    merged_df.to_parquet(output_filename, index=False)
    
    path_in_repo = f"SCAN_RESULTS/{output_filename}"
    return output_filename, path_in_repo, merged_df

def main():
    hf_token = os.environ.get("HF_TOKEN")
    hf_repo_id = os.environ.get("HF_REPO_ID")
    
    if not hf_token or not hf_repo_id:
        print("Error: HF_TOKEN and HF_REPO_ID environment variables must be set.")
        return
        
    api = HfApi(token=hf_token)
    operations = []
    
    # Merge combined (Known CMS)
    res_combined = merge_and_save(
        repo_id=hf_repo_id,
        folder_path="SCAN_RESULTS/combined",
        output_filename="combined_results_merged.parquet"
    )
    
    df_combined = None
    if res_combined != (None, None):
        local_file, repo_path, df_combined = res_combined
        operations.append(CommitOperationAdd(path_in_repo=repo_path, path_or_fileobj=local_file))
    
    # Calculate global CMS counts and metadata
    if df_combined is not None and 'Primary CMS' in df_combined.columns:
        print("Calculating total CMS counts across all data...")
        counts_series = df_combined['Primary CMS'].value_counts()
        global_cms_counts = {str(k): int(v) for k, v in counts_series.items()}
        
        # Download all worker states to aggregate totals
        print("Downloading worker states to aggregate global scan metrics...")
        import json
        
        total_scanned = 0
        total_live = 0
        total_fail = 0
        
        try:
            state_dir = snapshot_download(
                repo_id=hf_repo_id,
                repo_type="dataset",
                allow_patterns="SCAN_RESULTS/state/*.json"
            )
            state_files = glob.glob(os.path.join(state_dir, "SCAN_RESULTS/state", "*.json"))
            for state_file in state_files:
                try:
                    with open(state_file, "r") as f:
                        worker_state = json.load(f)
                        total_scanned += worker_state.get("total_scanned", 0)
                        total_live += worker_state.get("total_live", 0)
                        total_fail += worker_state.get("total_fail", 0)
                except Exception as e:
                    print(f"Error reading state file {state_file}: {e}")
        except Exception as e:
            print(f"Failed to download state files: {e}")
            
        global_metadata = {
            "total_scanned": total_scanned,
            "total_live": total_live,
            "total_fail": total_fail,
            "cms_counts": global_cms_counts
        }
        
        with open("global_metadata.json", "w") as f:
            json.dump(global_metadata, f, indent=4)
            
        operations.append(CommitOperationAdd(path_in_repo="SCAN_RESULTS/global_metadata.json", path_or_fileobj="global_metadata.json"))
    
    # Merge full (All Successes)
    res_full = merge_and_save(
        repo_id=hf_repo_id,
        folder_path="SCAN_RESULTS/full",
        output_filename="full_results_merged.parquet"
    )
    
    if res_full != (None, None):
        local_file, repo_path, _ = res_full
        operations.append(CommitOperationAdd(path_in_repo=repo_path, path_or_fileobj=local_file))
        
    if operations:
        print(f"Uploading {len(operations)} files in a single commit...")
        api.create_commit(
            repo_id=hf_repo_id,
            repo_type="dataset",
            operations=operations,
            commit_message="Update merged results and global metadata"
        )
        print("Merge process and upload complete!")
    else:
        print("Nothing to upload.")

if __name__ == "__main__":
    main()
