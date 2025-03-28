# a blue start

 The ABlueStart dataset

python get_pair_co-occurrence.py --input_filepath "data/deidentified_starterpack_hif.json" --max_pack_size 4070 --num_workers 10

python s_line_count.py data/deidentified_starterpack_hif.json --smin 1 --smax 345 --output data/s_count.txt
