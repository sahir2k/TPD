# VITONHD_release_input_person_combine_garment_240epochs_unpaired
CUDA_VISIBLE_DEVICES=0 python scripts/inference.py \
--config configs/inference/inference_VITONHD_unpaired.yaml \
--ckpt checkpoints/release/TPD_240epochs.ckpt \
--outdir inference_logs/VITONHD/VITONHD_release_input_person_combine_garment_240epochs_unpaired/ \
--seed 321 \
--batch_size 1 \
--C 5 \
--H 512 \
--W 768 \