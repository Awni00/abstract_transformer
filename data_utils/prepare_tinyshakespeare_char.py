import numpy as np
import pickle

file_path = 'data/tiny_shakespeare.txt'
with open(file_path, 'r') as file:
    lines = file.readlines()

train_lines = int(len(lines) * 0.9)
val_lines = int(len(lines) * 0.025)
train_data = lines[:train_lines]
val_data = lines[train_lines:train_lines + val_lines]
test_data = lines[train_lines + val_lines:]

with open('data/tiny_shakespeare_train.txt', 'w') as file:
    file.writelines(train_data)

with open('data/tiny_shakespeare_val.txt', 'w') as file:
    file.writelines(val_data)

with open('data/tiny_shakespeare_test.txt', 'w') as file:
    file.writelines(test_data)

with open('data/tiny_shakespeare.txt', 'r') as file:
    text = file.read()

# get all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
print("all the unique characters:", ''.join(chars))
print(f"vocab size: {vocab_size:,}")

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }


def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string


for filename in ['data/tiny_shakespeare_train.txt', 'data/tiny_shakespeare_val.txt', 'data/tiny_shakespeare_test.txt']:
    with open(filename, 'r') as file:
        text = file.read()
    print('loading ', filename)
    print('num characters: ', len(text))
    print('tokenizing')
    tokens = encode(text)
    tokens = np.array(tokens, dtype=np.uint16)
    # save to bin file
    out_filename = filename.replace('shakespeare_', 'shakespeare_char_').replace('.txt', '.bin')
    print('saved to ', out_filename)
    tokens.tofile(out_filename)

    print('num tokens: ', len(tokens))
    print('first 10 tokens:')
    print(tokens[:10])
    print()

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}

with open('data/shakespeare_char_meta.pkl', 'wb') as f:
    pickle.dump(meta, f)
