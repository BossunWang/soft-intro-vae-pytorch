CUDA_VISIBLE_DEVICES=5 python main.py \
--dataset ms1m_v3_mask \
--device 0 \
--lr 2e-4 \
--num_epochs 5 \
--num_vae 0 \
--save_interval 1 \
--test_iter 5000 \
--log_dir "logs" \
--beta_kl 0.5 \
--beta_rec 0.5 \
--beta_neg 256 \
--z_dim 128 \
--batch_size 16