CUDA_VISIBLE_DEVICES=1,0 python main.py \
--dataset ms1m_v3_112 \
--device 0 \
--lr 2e-4 \
--num_epochs 150 \
--num_vae 0 \
--test_iter 1000 \
--save_interval 20 \
--log_dir "logs" \
--beta_kl 0.5 \
--beta_rec 1.0 \
--beta_neg 1024 \
--z_dim 256 \
--batch_size 16