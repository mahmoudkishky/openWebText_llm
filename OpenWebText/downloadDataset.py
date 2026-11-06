from datasets import load_dataset

ds = load_dataset("./OpenWebText/openwebtext.py", "plain_text")
print(ds["train"][2])
