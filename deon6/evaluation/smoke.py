import torch
from deon6.model import DEON6, DEONConfig

def run():
    c=DEONConfig(vocab_size=128,max_seq_len=32,n_layers=2,hidden_size=64,n_heads=4,n_kv_heads=2,ffn_size=176)
    m=DEON6(c)
    x=torch.randint(0,c.vocab_size,(2,16))
    out=m(x,x)
    assert out["logits"].shape==(2,16,c.vocab_size)
    assert torch.isfinite(out["loss"])
    return {"status":"ok","parameters":sum(p.numel() for p in m.parameters())}
