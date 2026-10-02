import argparse
from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--vocab-size", type=int, default=65536)
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=["<pad>","<unk>","<bos>","<eos>","<system>","<user>","<assistant>","<tool>","<tool_result>"],
        show_progress=True,
    )
    tokenizer.train([args.input], trainer)
    tokenizer.save(str(out / "tokenizer.json"))
    print(f"saved {out / 'tokenizer.json'}")
if __name__ == "__main__":
    main()
