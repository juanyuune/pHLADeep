"""
deduplicate_splits.py
=====================
Removes exact duplicate peptides that appear in BOTH the training and test
sets of the same allele and label class.

Pipeline assumption:
  .esm2 files are named after the peptide sequence, e.g. ACDEFGHIL.esm2
  The filename stem IS the peptide sequence.

Directory structure expected:
  <emb_root>/
  └── HLA-A0101/
      ├── pos_train/   PEPTIDESEQ.esm2
      ├── neg_train/   PEPTIDESEQ.esm2
      ├── pos_test/    PEPTIDESEQ.esm2
      └── neg_test/    PEPTIDESEQ.esm2

What this script does per allele:
  1. For positive class:  find peptides in BOTH pos_train AND pos_test
     → remove duplicates from pos_test  (keep train intact)
  2. For negative class:  find peptides in BOTH neg_train AND neg_test
     → remove duplicates from neg_test  (keep train intact)
  3. Cross-label check:   find peptides that appear as both positive
     and negative in the SAME split → remove from the smaller class
  4. Prints a full report of what was removed

Usage:
  python deduplicate_splits.py \
      --emb_root /srv/jupyterlab/workspace/HLA1/data/expanded_emb \
      --dry_run          # preview only, nothing deleted

  python deduplicate_splits.py \
      --emb_root /srv/jupyterlab/workspace/HLA1/data/expanded_emb
      # actually deletes the duplicate files
"""

import os
import argparse


def get_peptides(folder):
    """Return dict: peptide_sequence -> full_file_path

    Filename format: HLA-A0101_ACDEFGHIL.esm2
    We extract only the peptide part after the underscore.
    """
    if not os.path.isdir(folder):
        return {}
    result = {}
    for fname in os.listdir(folder):
        if fname.endswith('.esm2'):
            stem = os.path.splitext(fname)[0]          # HLA-A0101_ACDEFGHIL
            # extract peptide: everything after the first underscore
            if '_' in stem:
                seq = stem.split('_', 1)[1]            # ACDEFGHIL
            else:
                seq = stem                             # fallback: use full stem
            result[seq] = os.path.join(folder, fname)
    return result


def remove_files(paths, dry_run, reason):
    for p in paths:
        if dry_run:
            print(f"  [DRY RUN] would delete: {os.path.basename(p)}  ({reason})")
        else:
            os.remove(p)
            print(f"  deleted: {os.path.basename(p)}  ({reason})")


def process_allele(allele_dir, dry_run):
    allele = os.path.basename(allele_dir)
    print(f"\n{'='*60}")
    print(f"Allele: {allele}")
    print(f"{'='*60}")

    splits = {
        'pos_train': get_peptides(os.path.join(allele_dir, 'pos_train')),
        'neg_train': get_peptides(os.path.join(allele_dir, 'neg_train')),
        'pos_test' : get_peptides(os.path.join(allele_dir, 'pos_test')),
        'neg_test' : get_peptides(os.path.join(allele_dir, 'neg_test')),
    }

    total_removed = 0

    # ── 1. Same-label train/test duplicates ──────────────────────────────
    for label in ('pos', 'neg'):
        train_peps = splits[f'{label}_train']
        test_peps  = splits[f'{label}_test']
        duplicates = set(train_peps.keys()) & set(test_peps.keys())

        if duplicates:
            print(f"\n  [{label}] train∩test duplicates: {len(duplicates)}")
            paths_to_remove = [test_peps[p] for p in duplicates]
            remove_files(paths_to_remove, dry_run,
                         reason=f"duplicate in {label}_train and {label}_test")
            total_removed += len(duplicates)
            # update in-memory view
            for p in duplicates:
                del splits[f'{label}_test'][p]
        else:
            print(f"  [{label}] train∩test duplicates: 0 ✓")

    # ── 2. Cross-label duplicates (same peptide = pos AND neg) ───────────
    # Check within train set
    cross_train = set(splits['pos_train'].keys()) & set(splits['neg_train'].keys())
    if cross_train:
        print(f"\n  [cross-label] pos_train∩neg_train conflicts: {len(cross_train)}")
        # Remove from neg_train (positive label takes priority)
        paths_to_remove = [splits['neg_train'][p] for p in cross_train]
        remove_files(paths_to_remove, dry_run,
                     reason="same peptide labeled positive AND negative in train")
        total_removed += len(cross_train)
        for p in cross_train:
            del splits['neg_train'][p]
    else:
        print(f"  [cross-label] pos_train∩neg_train conflicts: 0 ✓")

    # Check within test set
    cross_test = set(splits['pos_test'].keys()) & set(splits['neg_test'].keys())
    if cross_test:
        print(f"\n  [cross-label] pos_test∩neg_test conflicts: {len(cross_test)}")
        paths_to_remove = [splits['neg_test'][p] for p in cross_test]
        remove_files(paths_to_remove, dry_run,
                     reason="same peptide labeled positive AND negative in test")
        total_removed += len(cross_test)
        for p in cross_test:
            del splits['neg_test'][p]
    else:
        print(f"  [cross-label] pos_test∩neg_test conflicts: 0 ✓")

    # ── 3. Final counts ───────────────────────────────────────────────────
    print(f"\n  Final counts after deduplication:")
    grand = 0
    for split_name, peps in splits.items():
        n = len(peps)
        grand += n
        print(f"    {split_name:12s}: {n:6,}")
    print(f"    {'TOTAL':12s}: {grand:6,}")
    print(f"  Removed this allele: {total_removed}")

    return total_removed


def main():
    parser = argparse.ArgumentParser(
        description="Remove exact-duplicate peptides across train/test splits"
    )
    parser.add_argument(
        '--emb_root', required=True,
        help='Root directory containing HLA-XXXXX subdirectories'
    )
    parser.add_argument(
        '--dry_run', action='store_true',
        help='Preview what would be deleted without actually deleting'
    )
    args = parser.parse_args()

    if args.dry_run:
        print("=" * 60)
        print("DRY RUN MODE — nothing will be deleted")
        print("=" * 60)

    allele_dirs = sorted([
        os.path.join(args.emb_root, d)
        for d in os.listdir(args.emb_root)
        if d.startswith('HLA-') and
           os.path.isdir(os.path.join(args.emb_root, d))
    ])

    if not allele_dirs:
        print(f"No HLA-* directories found in {args.emb_root}")
        return

    grand_total_removed = 0
    for allele_dir in allele_dirs:
        removed = process_allele(allele_dir, args.dry_run)
        grand_total_removed += removed

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Alleles processed : {len(allele_dirs)}")
    print(f"Total files removed: {grand_total_removed}")
    if args.dry_run:
        print("(DRY RUN — nothing was actually deleted)")
    else:
        print("Deduplication complete.")
        print("\nNext step: rerun get_dataset.py on the cleaned expanded_emb/ folders")
        print("to regenerate the .npy files before training.")


if __name__ == '__main__':
    main()