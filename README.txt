TE Major Project — prepared supervised dataset

Source: Baseline_file.xlsx

Files:
- labelled_comments_clean.csv: all rows with valid four-class labels and non-empty comment text.
- train.csv: 70% stratified training split.
- validation.csv: 15% stratified validation split.
- test.csv: 15% stratified test split.
- ambiguous_for_review.csv: AMBIGUOUS rows excluded from training.
- unlabelled_comments_not_used_for_training.csv: blank/other labels excluded from training.
- dataset_split_summary.json: row counts and class distributions.

Labels included: NOT, PRFN, OFFN, HATE.
Split seed: 42.
Split method: stratified random split.

Important limitations:
1. The workbook does not include video_id, so splitting by source video was not possible. For final evaluation, recover video_id from the raw comments file and split by video if possible.
2. The labelled subset appears to have been selected partly through lexicon matching. Its distribution may not represent all YouTube comments.
3. These labels are treated as human-verified according to the user's description. Keep annotation guidelines and reviewer records for the paper.
4. Do not use test.csv for training or model selection.
