import numpy as np
import torch

def chunked_msg_embedding(msg):
    chunk_list = []
    for i in range(4):
        # msg[:, i*nchunk_size : (i+1)*nchunk_size] => shape [b, nchunk_size]
        chunk_bits = msg[:, i * 4 : (i + 1) * 4]
        chunk_val = torch.zeros_like(chunk_bits[:, 0])  # shape [b]
        for bit_idx in range(4):
            # shift bits
            chunk_val += chunk_bits[:, bit_idx] << bit_idx
        #print(chunk_val)
        chunk_list.append(chunk_val)
    print(chunk_list)
    chunk_list = torch.stack(chunk_list, dim=1)  # shape [b, 4]
    print(chunk_list)


if __name__ == "__main__":
    msg = np.random.choice([0,1], [4, 16])
    msg = torch.tensor(msg)
    print(msg)
    chunked_msg_embedding(msg)