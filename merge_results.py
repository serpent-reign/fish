import os
import json
import glob
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub import CommitOperationAdd
from huggingface_hub.utils import EntryNotFoundError

TRACKER_REPO_PATH = "SCAN_RESULTS/merged_shards_tracker.json"
GLOBAL_META_REPO_PATH = "SCAN_RESULTS/global_metadata.json"


def load_tracker(api, hf_repo_id):
    """Download the tracker file from HF. Returns (tracker_dict, is_first_run)."""
    try:
        local_path = hf_hub_download(
            repo_id=hf_repo_id,
            repo_type="dataset",
            filename=TRACKER_REPO_PATH,
        )
        with open(local_path, "r") as f:
            tracker = json.load(f)
        # Ensure both keys exist
        tracker.setdefault("combined", [])
        tracker.setdefault("full", [])
        return tracker, False
    except EntryNotFoundError:
        print("No tracker found. This is the FIRST RUN of incremental merge.")
        return {"combined": [], "full": []}, True
    except Exception as e:
        print(f"Warning: Could not load tracker: {e}. Starting fresh.")
        return {"combined": [], "full": []}, True


def load_global_metadata(api, hf_repo_id):
    """Download global_metadata.json from HF. Returns dict."""
    try:
        local_path = hf_hub_download(
            repo_id=hf_repo_id,
            repo_type="dataset",
            filename=GLOBAL_META_REPO_PATH,
        )
        with open(local_path, "r") as f:
            return json.load(f)
    except EntryNotFoundError:
        print("No global metadata found. Starting fresh.")
        return {"total_scanned": 0, "total_live": 0, "total_fail": 0, "cms_counts": {}}
    except Exception as e:
        print(f"Warning: Could not load global metadata: {e}. Starting fresh.")
        return {"total_scanned": 0, "total_live": 0, "total_fail": 0, "cms_counts": {}}


def get_new_shards(api, hf_repo_id, folder_path, already_merged):
    """Lists all shards in the HF folder and returns only the ones not yet merged."""
    try:
        all_files = api.list_repo_files(repo_id=hf_repo_id, repo_type="dataset")
        prefix = f"{folder_path}/"
        all_shards = [f for f in all_files if f.startswith(prefix) and f.endswith(".parquet")]
        shard_names = {os.path.basename(s) for s in all_shards}
        already_set = set(already_merged)
        new_shard_names = shard_names - already_set
        new_shard_paths = [f for f in all_shards if os.path.basename(f) in new_shard_names]
        print(f"[{folder_path}] Total shards: {len(all_shards)} | Already merged: {len(already_set)} | New: {len(new_shard_paths)}")
        return new_shard_paths, list(shard_names)
    except Exception as e:
        print(f"Error listing shards in {folder_path}: {e}")
        return [], []


def download_shards(api, hf_repo_id, shard_paths):
    """Downloads a list of shard repo paths one by one. Returns list of local file paths."""
    local_files = []
    for path in shard_paths:
        try:
            local = hf_hub_download(
                repo_id=hf_repo_id,
                repo_type="dataset",
                filename=path,
            )
            local_files.append(local)
        except Exception as e:
            print(f"Warning: Failed to download shard {path}: {e}")
    return local_files


def incremental_merge(api, hf_repo_id, folder_path, output_filename, already_merged):
    """
    Incremental merge strategy:
    1. Find new shards only.
    2. Download the existing merged parquet (if any) + only the new shards.
    3. Stream-concatenate using pyarrow (never loads everything into pandas RAM).
    4. Overwrite output file.
    Returns (output_filename, repo_path, list_of_all_shard_names) or (None, None, [])
    """
    new_shard_paths, all_shard_names = get_new_shards(api, hf_repo_id, folder_path, already_merged)

    if not new_shard_paths:
        print(f"[{folder_path}] No new shards to merge. Skipping.")
        return None, None, all_shard_names

    print(f"[{folder_path}] Downloading {len(new_shard_paths)} new shards...")
    new_local_files = download_shards(api, hf_repo_id, new_shard_paths)

    if not new_local_files:
        print(f"[{folder_path}] Failed to download any new shards.")
        return None, None, all_shard_names

    # Build list of all parquet sources: existing merged file + new shards
    sources = []
    repo_merged_path = f"SCAN_RESULTS/{output_filename}"

    try:
        existing_merged_local = hf_hub_download(
            repo_id=hf_repo_id,
            repo_type="dataset",
            filename=repo_merged_path,
        )
        sources.append(existing_merged_local)
        print(f"[{folder_path}] Found existing merged file. Appending to it.")
    except EntryNotFoundError:
        print(f"[{folder_path}] No existing merged file. Creating fresh.")
    except Exception as e:
        print(f"[{folder_path}] Warning: Could not download existing merged file: {e}. Creating fresh.")

    sources.extend(new_local_files)

    # Stream-concatenate using pyarrow dataset — no pandas RAM blow-up
    print(f"[{folder_path}] Stream-merging {len(sources)} file(s) with pyarrow...")
    try:
        dataset = ds.dataset(sources, format="parquet")
        # Deduplicate on Domain if present — sort by Domain and keep last
        schema = dataset.schema
        if "Domain" in schema.names:
            table = dataset.to_table()
            df = table.to_pandas()
            del table  # Free Arrow table memory immediately
            import gc; gc.collect()
            
            df.drop_duplicates(subset=["Domain"], keep="last", inplace=True)
            table = pa.Table.from_pandas(df, preserve_index=False)
            del df  # Free Pandas dataframe memory immediately
            gc.collect()
        else:
            table = dataset.to_table()

        pq.write_table(table, output_filename, compression="snappy")
        print(f"[{folder_path}] Merged {table.num_rows:,} total rows → {output_filename}")
        del table
        del dataset
        import gc; gc.collect()
    except Exception as e:
        print(f"[{folder_path}] Error during pyarrow merge: {e}")
        return None, None, all_shard_names

    return output_filename, repo_merged_path, all_shard_names


def compute_metadata_from_workers(api, hf_repo_id):
    """
    Discovers and downloads all worker_XX.json state files dynamically from HF.
    Sums their cumulative totals. Immune to scaling workers up or down.
    """
    all_files = api.list_repo_files(repo_id=hf_repo_id, repo_type="dataset")
    state_files = sorted([
        f for f in all_files
        if f.startswith("SCAN_RESULTS/state/worker_") and f.endswith(".json")
    ])

    if not state_files:
        print("No worker state files found.")
        return {"total_scanned": 0, "total_live": 0, "total_fail": 0, "cms_counts": {}}

    print(f"Found {len(state_files)} worker state file(s). Downloading and summing...")
    total_scanned = 0
    total_live = 0
    total_fail = 0
    cms_counts = {}

    for state_path in state_files:
        try:
            local_path = hf_hub_download(
                repo_id=hf_repo_id,
                repo_type="dataset",
                filename=state_path,
            )
            with open(local_path, "r") as f:
                w = json.load(f)
            total_scanned += w.get("total_scanned", 0)
            total_live += w.get("total_live", 0)
            total_fail += w.get("total_fail", 0)
            for cms, count in w.get("cms_counts", {}).items():
                cms_counts[cms] = cms_counts.get(cms, 0) + int(count)
        except Exception as e:
            print(f"  Warning: Could not read {state_path}: {e}")

    return {
        "total_scanned": total_scanned,
        "total_live": total_live,
        "total_fail": total_fail,
        "cms_counts": cms_counts
    }


def main():
    hf_token = os.environ.get("HF_TOKEN")
    hf_repo_id = os.environ.get("HF_REPO_ID")

    if not hf_token or not hf_repo_id:
        print("Error: HF_TOKEN and HF_REPO_ID environment variables must be set.")
        return

    api = HfApi(token=hf_token)
    operations = []

    # --- 1. Load tracker and existing global metadata ---
    tracker, is_first_run = load_tracker(api, hf_repo_id)
    existing_metadata = load_global_metadata(api, hf_repo_id)

    # Bootstrap detection: if this is the first run of the NEW incremental script
    # but metadata already exists on HF (from the old script), we must NOT add any
    # delta on top. We treat all currently-existing shards as already accounted for.
    # The metadata is kept exactly as-is and the tracker is bootstrapped with all
    # existing shard names so future runs only pick up truly new shards.
    if is_first_run and existing_metadata.get("total_scanned", 0) > 0:
        print(f"\n[BOOTSTRAP] Existing progress detected ({existing_metadata['total_scanned']:,} domains scanned).")
        print("[BOOTSTRAP] Treating all existing shards as already merged. No metadata delta will be applied.")
        print("[BOOTSTRAP] Future runs will only process NEW shards going forward.\n")

    # --- 2. Incremental merge: combined (Known CMS only) ---
    comb_file, comb_repo_path, comb_all_shards = incremental_merge(
        api=api,
        hf_repo_id=hf_repo_id,
        folder_path="SCAN_RESULTS/combined",
        output_filename="combined_results_merged.parquet",
        already_merged=tracker["combined"]
    )
    if comb_file:
        operations.append(CommitOperationAdd(path_in_repo=comb_repo_path, path_or_fileobj=comb_file))

    # --- 3. Incremental merge: full (All successes) ---
    full_file, full_repo_path, full_all_shards = incremental_merge(
        api=api,
        hf_repo_id=hf_repo_id,
        folder_path="SCAN_RESULTS/full",
        output_filename="full_results_merged.parquet",
        already_merged=tracker["full"]
    )
    if full_file:
        operations.append(CommitOperationAdd(path_in_repo=full_repo_path, path_or_fileobj=full_file))

    # --- 4. Metadata: discover and re-sum ALL worker states on HF ---
    updated_metadata = compute_metadata_from_workers(api, hf_repo_id)
    with open("global_metadata.json", "w") as f:
        json.dump(updated_metadata, f, indent=4)
    operations.append(CommitOperationAdd(
        path_in_repo=GLOBAL_META_REPO_PATH,
        path_or_fileobj="global_metadata.json"
    ))

    # --- 5. Update tracker ---
    # On bootstrap run, comb_all_shards / full_all_shards contain ALL existing shards
    # (since tracker was empty). This correctly seeds the tracker for future incremental runs.
    tracker["combined"] = list(set(comb_all_shards))
    tracker["full"] = list(set(full_all_shards))
    with open("merged_shards_tracker.json", "w") as f:
        json.dump(tracker, f, indent=4)
    operations.append(CommitOperationAdd(
        path_in_repo=TRACKER_REPO_PATH,
        path_or_fileobj="merged_shards_tracker.json"
    ))

    # --- 6. Upload sequentially to avoid XET 429 ---
    if any(op.path_in_repo != TRACKER_REPO_PATH and op.path_in_repo != GLOBAL_META_REPO_PATH
           for op in operations
           if op.path_in_repo not in [TRACKER_REPO_PATH, GLOBAL_META_REPO_PATH]):
        pass  # there are real merged files to upload

    if len(operations) <= 2 and not is_first_run:
        # Only tracker + metadata uploaded — no new shards at all (not a bootstrap)
        import sys
        sys.exit("Nothing new to merge. End of dataset or no new shards found. Stopping auto-loop.")
    elif len(operations) <= 2 and is_first_run:
        # Bootstrap run with no new shards somehow — still upload tracker
        print("[BOOTSTRAP] No new merged files but uploading tracker to seed future runs.")

    print(f"\nUploading {len(operations)} file(s) sequentially...")
    for idx, op in enumerate(operations):
        print(f"  [{idx+1}/{len(operations)}] {op.path_in_repo}")
        api.create_commit(
            repo_id=hf_repo_id,
            repo_type="dataset",
            operations=[op],
            commit_message=f"Incremental update: {os.path.basename(op.path_in_repo)}"
        )

    print("\nMerge complete!")
    print(f"  Total scanned : {updated_metadata['total_scanned']:,}")
    print(f"  Total live    : {updated_metadata['total_live']:,}")
    print(f"  Total fail    : {updated_metadata['total_fail']:,}")
    print(f"  CMS types     : {len(updated_metadata['cms_counts'])}")


if __name__ == "__main__":
    main()
