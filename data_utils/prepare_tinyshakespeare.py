import numpy as np
import tiktoken

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

bytepair_enc = tiktoken.get_encoding("gpt2")


for filename in ['data/tiny_shakespeare_train.txt', 'data/tiny_shakespeare_val.txt', 'data/tiny_shakespeare_test.txt']:
    with open(filename, 'r') as file:
        text = file.read()
    print('loading ', filename)
    print('num characters: ', len(text))
    print('tokenizing')
    tokens = bytepair_enc.encode(text)
    tokens = np.array(tokens, dtype=np.uint16)
    # save to python file
    out_filename = filename.replace('.txt', '.bin')
    print('saved to ', out_filename)
    tokens.tofile(out_filename)

    print('num tokens: ', len(tokens))
    print('first 10 tokens:')
    print(tokens[:10])
    print()