import torch
from deon6.model import DEON6, DEONConfig

def test_forward():
    c=DEONConfig(vocab_size=64,max_seq_len=16,n_layers=1,hidden_size=32,n_heads=4,n_kv_heads=2,ffn_size=88)
    m=DEON6(c)
    x=torch.randint(0,64,(1,8))
    y=m(x,x)
    assert y["logits"].shape == (1,8,64)
    assert y["loss"].ndim == 0
