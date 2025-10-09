# watermark detail
msg_len = 16

# train setting
batch_size = 4
max_frame_len = 160          # 80, 160, 240
epoch = 30
train_data_path = "/workspace/unlab_chunk_6k"
log_step = 5
save_interval = (4625770 - 1) // batch_size      # ここはepoch単位で保存したいのでデータセットサイズごとに調整する
save_path = "/workspace/checkpoints/modelv2_3"
log_path = "/workspace/logs/modelv2/log3"