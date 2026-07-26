import os
import gzip
import random
import json
import datetime
from huggingface_hub import HfApi

# Sample sizes definition requested by user
SAMPLE_SIZES = [
    (100, "100"),
    (1000, "1k"),
    (50000, "50k"),
    (100000, "100k"),
    (500000, "500k"),
    (1000000, "1m"),
    (5000000, "5m"),
    (10000000, "10m"),
    (20000000, "20m")
]

def main():
    hf_token = os.environ.get("HF_TOKEN")
    hf_repo_id = os.environ.get("HF_REPO_ID")
    
    if not hf_token or not hf_repo_id:
        print("Error: HF_TOKEN and HF_REPO_ID environment variables must be set.")
        return

    api = HfApi(token=hf_token)

    print(f"Fetching repository file list from HuggingFace ({hf_repo_id})...")
    try:
        all_files = api.list_repo_files(repo_id=hf_repo_id, repo_type="dataset")
    except Exception as e:
        print(f"Error fetching file list from HF repo: {e}")
        return

    # Look for Global Master files
    master_files = [f for f in all_files if "GLOBAL_MASTER_part_" in f and f.endswith(".gz")]
    
    if not master_files:
        print("Error: No GLOBAL_MASTER_part_*.gz files found in the dataset repository.")
        return

    print(f"Found {len(master_files)} Global Master chunk file(s): {master_files}")

    max_sample_needed = max(size for size, _ in SAMPLE_SIZES)
    print(f"Targeting reservoir sampling of up to {max_sample_needed:,} random domains...")

    reservoir = []
    total_count = 0

    for i, file_path in enumerate(master_files, 1):
        print(f"[{i}/{len(master_files)}] Downloading and streaming {file_path}...")
        local_path = api.hf_hub_download(
            repo_id=hf_repo_id,
            filename=file_path,
            repo_type="dataset",
            token=hf_token
        )

        with gzip.open(local_path, 'rt', encoding='utf-8', errors='ignore') as gz:
            for line in gz:
                d = line.strip()
                if not d:
                    continue
                total_count += 1
                if len(reservoir) < max_sample_needed:
                    reservoir.append(d)
                else:
                    # Reservoir sampling (Algorithm R)
                    idx = random.randint(0, total_count - 1)
                    if idx < max_sample_needed:
                        reservoir[idx] = d

        # Remove local chunk immediately to save disk space
        if os.path.exists(local_path):
            os.remove(local_path)

        print(f"   Processed {total_count:,} total domains so far (Reservoir filled: {len(reservoir):,}).")

    if not reservoir:
        print("Error: Global Master chunks contained no valid domain lines.")
        return

    print(f"\nCompleted reading Global Master! Total unique domains scanned: {total_count:,}.")
    print("Shuffling reservoir to produce randomized mixed domain samples...")
    random.shuffle(reservoir)

    sample_meta = {}

    print("\nGenerating and uploading sample files to 'Samples' folder...")
    for size_num, label in SAMPLE_SIZES:
        actual_count = min(size_num, len(reservoir))
        filename = f"sample_{label}.txt.gz"
        path_in_repo = f"Samples/{filename}"

        print(f"Creating {filename} with {actual_count:,} random domains...")
        sample_subset = reservoir[:actual_count]

        with gzip.open(filename, 'wt', encoding='utf-8') as gz:
            for d in sample_subset:
                gz.write(f"{d}\n")

        print(f"Uploading {path_in_repo} to HuggingFace...")
        api.upload_file(
            path_or_fileobj=filename,
            path_in_repo=path_in_repo,
            repo_id=hf_repo_id,
            repo_type="dataset",
            commit_message=f"Upload mixed sample dataset: {label} ({actual_count:,} domains)"
        )

        sample_meta[label] = {
            "requested_count": size_num,
            "actual_count": actual_count,
            "file": filename,
            "path_in_repo": path_in_repo
        }

        # Cleanup local file
        if os.path.exists(filename):
            os.remove(filename)

    # Upload metadata json
    meta_filename = "metadata_samples.json"
    metadata = {
        "dataset": "SAMPLES",
        "generated_at": str(datetime.datetime.now()),
        "total_global_domains_scanned": total_count,
        "sample_sizes": sample_meta
    }

    with open(meta_filename, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"Uploading metadata to Samples/{meta_filename}...")
    api.upload_file(
        path_or_fileobj=meta_filename,
        path_in_repo=f"Samples/{meta_filename}",
        repo_id=hf_repo_id,
        repo_type="dataset",
        commit_message="Update sample dataset metadata"
    )
    if os.path.exists(meta_filename):
        os.remove(meta_filename)

    print("\nALL SAMPLE GENERATION & UPLOADS COMPLETED SUCCESSFULLY!")

if __name__ == '__main__':
    main()
