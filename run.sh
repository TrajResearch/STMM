


python pretrain.py \
--tokenizer_path /root/xy/traj/t5/tokenizer/vocab_addpretrain \
--train_d ./dataset/t5_train_l2mm_mlm_bert_onlygps_mixtoken \
--val_d ./dataset/t5_val_l2mm_mlm_bert_onlygps_mixtoken \
--epoch_num 30 \
--encoder_path model2/encoder_path_l2mm_mlm_bert_onlygps_mixtoken \
--mlm_layer_path model2/mlm_layer_path_l2mm_mlm_bert_onlygps_mixtoken \
--t5_encoder_path model2/t5_encoder_path_l2mm_mlm_bert_onlygps_mixtoken \
--batch_size 128 > ./results2/l2mm_mlm_bert_onlygps_mlm_pretrain_mixtoken.txt



python train9_loadt5_trainbert_add2encoder.py \
--tokenizer_path ./tokenizer/vocab_addpretrain \
--train_d ./dataset/t5_train_len65_keep125_add_token_unk_41_addpretrain \
--val_d ./dataset/t5_val_len65_keep125_add_token_unk_41_addpretrain \
--test_d /root/xy/traj/t5/dataset/t5_test_len65_keep125_add_token_unk_41_addpretrain \
--t5_gan_model_path ./model2/testnmt2_loadt5_add2encoder_lr00001_0001_keep125 \
--use_generate \
--use_generate_num 0 \
--epoch_num 200 \
--gen_lr 0.0001 \
--g_r 1.0 \
--d_r 1.0 \
--max_length 41 \
--hidden_size 256 \
--d_kv 64 \
--d_ff 1024 \
--num_layers 4 \
--num_heads 4 \
--relative_attention_num_buckets 32 \
--num_decoder_layers 4 \
--t5_encoder_path ./model2/t5_encoder_path_l2mm_mlm_bert_onlygps_mixtoken \
--model_path /root/xy/traj/t5/pth/merge_xrx_clean_win25_maxlen25_addpretrain_pretrain \
--batch_size 128 > results3/testnmt2_loadt5_add2encoder_lr00001_0001_keep125.txt


