INPUT_CT=example/CTseg_1_raw.tif
SAVEDIR=example/output

python gigaheart_inference.py \
    --input_ct $INPUT_CT \
    --save_dir $SAVEDIR