import argparse, json, math, random
from pathlib import Path
import torch
import yaml
from torch.utils.data import Dataset, DataLoader
from deon6.model import DEON6, DEONConfig

class CharDataset(Dataset):
    def __init__(self, path, seq_len, vocab_size):
        text = Path(path).read_text(encoding="utf-8")
        chars = sorted(set(text))
        ids = {c:i for i,c in enumerate(chars)}
        self.data = torch.tensor([ids[c] % vocab_size for c in text], dtype=torch.long)
        self.seq_len = seq_len
    def __len__(self):
        return max(1, len(self.data)-self.seq_len)
    def __getitem__(self, i):
        x = self.data[i:i+self.seq_len]
        y = self.data[i+1:i+self.seq_len+1]
        if len(x) < self.seq_len:
            x = torch.nn.functional.pad(x, (0,self.seq_len-len(x)))
            y = torch.nn.functional.pad(y, (0,self.seq_len-len(y)))
        return x, y

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args=ap.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text())
    mcfg=DEONConfig(**cfg["model"])
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["training"]["seed"])
    model=DEON6(mcfg).to(device)
    opt=torch.optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"], weight_decay=cfg["training"]["weight_decay"])
    ds=CharDataset(cfg["training"]["dataset"], mcfg.max_seq_len, mcfg.vocab_size)
    dl=DataLoader(ds, batch_size=cfg["training"]["batch_size"], shuffle=True, drop_last=False)
    out=Path(cfg["training"]["output_dir"]); out.mkdir(parents=True, exist_ok=True)
    steps=0
    model.train()
    for epoch in range(cfg["training"]["epochs"]):
        for x,y in dl:
            x,y=x.to(device),y.to(device)
            result=model(x,y)
            loss=result["loss"]/cfg["training"]["grad_accum_steps"]
            loss.backward()
            if (steps+1) % cfg["training"]["grad_accum_steps"] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad(set_to_none=True)
            steps += 1
            if steps % cfg["training"]["log_every"] == 0:
                print(json.dumps({"step":steps,"loss":float(result["loss"]),"device":str(device)}))
            if steps >= cfg["training"]["max_steps"]:
                break
        if steps >= cfg["training"]["max_steps"]:
            break
    torch.save({"model":model.state_dict(),"config":mcfg.__dict__}, out/"last.pt")
    print(f"saved {out/'last.pt'}")

if __name__=="__main__":
    main()
