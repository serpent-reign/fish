import os
import glob
import pandas as pd
from huggingface_hub import HfApi, snapshot_download

def merge_and_upload(repo_id, folder_path, output_filename, commit_message):
    print(f"Downloading shards from {folder_path}...")
    try:
        local_dir = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns=f"{folder_path}/*.parquet"
        )
    except Exception as e:
        print(f"Failed to download from {folder_path}: {e}")
        return

    # snapshot_download preserves the folder structure locally
    shard_pattern = os.path.join(local_dir, folder_path, "*.parquet")
    shard_files = glob.glob(shard_pattern)
    
    if not shard_files:
        print(f"No parquet files found in {folder_path}.")
        return

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
        return
        
    merged_df = pd.concat(dfs, ignore_index=True)
    
    # If it's the known CMS file, we might want to drop duplicate domains keeping the latest
    # Since these are distinct slices, duplicates shouldn't happen unless a domain was scanned twice
    # but dropping duplicates by domain is a safe practice for the final file.
    if 'Domain' in merged_df.columns:
        merged_df = merged_df.drop_duplicates(subset=['Domain'], keep='last')
        
    print(f"Merged into {len(merged_df)} total rows. Saving to {output_filename}...")
    merged_df.to_parquet(output_filename, index=False)
    
    # Upload the single file to Hugging Face
    api = HfApi()
    upload_path = f"SCAN_RESULTS/{output_filename}"
    print(f"Uploading merged file to {upload_path}...")
    api.upload_file(
        path_or_fileobj=output_filename,
        path_in_repo=upload_path,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=commit_message
    )
    print(f"Successfully uploaded {upload_path}!")
    return merged_df

def main():
    hf_token = os.environ.get("HF_TOKEN")
    hf_repo_id = os.environ.get("HF_REPO_ID")
    
    if not hf_token or not hf_repo_id:
        print("Error: HF_TOKEN and HF_REPO_ID environment variables must be set.")
        return
        
    api = HfApi(token=hf_token)
    
    # Merge combined (Known CMS)
    df_combined = merge_and_upload(
        repo_id=hf_repo_id,
        folder_path="SCAN_RESULTS/combined",
        output_filename="combined_results_merged.parquet",
        commit_message="Update merged combined results"
    )
    
    # Calculate global CMS counts
    if df_combined is not None and 'Primary CMS' in df_combined.columns:
        print("Calculating total CMS counts across all data...")
        counts_series = df_combined['Primary CMS'].value_counts()
        global_cms_counts = {str(k): int(v) for k, v in counts_series.items()}
        
        import json
        with open("global_cms_counts.json", "w") as f:
            json.dump(global_cms_counts, f, indent=4)
            
        print("Uploading global_cms_counts.json...")
        api.upload_file(
            path_or_fileobj="global_cms_counts.json",
            path_in_repo="SCAN_RESULTS/global_cms_counts.json",
            repo_id=hf_repo_id,
            repo_type="dataset",
            commit_message="Update total global CMS counts"
        )
    
    # Merge full (All Successes)
    merge_and_upload(
        repo_id=hf_repo_id,
        folder_path="SCAN_RESULTS/full",
        output_filename="full_results_merged.parquet",
        commit_message="Update merged full results"
    )
    
    print("Merge process complete! The original shards have been kept intact on Hugging Face.")

if __name__ == "__main__":
    main()
