# watermark detail
msg_len = 10

# train setting
batch_size = 8
max_frame_len = 80
epoch = 1000
train_data_path = "/workspace/LibriTTS"
log_step = 5
save_interval = 1000 - 1      # ここはepoch単位で保存したいのでデータセットサイズごとに調整する
save_path = "/workspace/checkpoints"