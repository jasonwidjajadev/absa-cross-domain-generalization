## Pipeline overview

1. **Yelp sample (3,000 reviews)**  
   We **randomly sample 3,000 examples** from the Yelp dataset using `sample_yelp_reviews.py`.

   ```bash
   python sample_yelp_reviews.py --input-file yelp_reviews.csv --output-file sampled_3000_reviews.csv
   ```

2. **Two halves**  
   That sample is split **into two halves** (1,500 + 1,500) for downstream processing.

3. **Pure manual labeling**  
   All labels are produced manually. We split annotators into **two groups**, and each group has **two people**:
   - Group A: two annotators label each assigned review independently.
   - Group B: two annotators label each assigned review independently.

   In each group, the two annotators separately create two polarity result files for the same data.

4. **Compute kappa score first**  
   For each group, we first calculate Cohen's kappa from the two independent label files.

   ```bash
   python calculate_kappa.py --file-a <labels_a.csv> --file-b <labels_b.csv>
   ```

5. **Find disagreements and discuss**  
   After checking kappa, run `export_polarity_disagreements.py` to export rows where the two independent labels are different, then discuss every disagreement manually and decide the final label.

   ```bash
   python export_polarity_disagreements.py --polarities_a_csv <labels_a.csv> --polarities_b_json <labels_b.json> --output_csv <disagreements.csv>
   ```

6. **Merge finalized labels**  
   After discussion, we merge the adjudicated outputs from both groups into one finalized dataset.

7. **Build custom dataset (`build_custom_splits_from_halves.py`)**  
   After the two-half annotation/adjudication workflow, we merge the two half CSV files and split them into `train.csv` / `val.csv` / `test.csv` (80/10/10).  
   This script takes exactly **3 args**: two input files and one output directory.

   ```bash
   python build_custom_splits_from_halves.py --input-half1 <half1.csv> --input-half2 <half2.csv> --output-dir <output_dir>
   ```

8. **Combine two data dirs (`merge_split_dirs.py`)**  
   Merge two dataset directories into one output directory.  
   It automatically reads `train.csv`, `val.csv`, and `test.csv` from both input dirs and writes merged files with the same names to the output dir.

   ```bash
   python merge_split_dirs.py --data-dir-a <data_dir_a> --data-dir-b <data_dir_b> --output-dir <combined_output_dir>
   ```

