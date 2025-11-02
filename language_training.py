# template

# import os
# import random
# from collections import Counter
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader, random_split
# from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
# from torchtext.data.utils import get_tokenizer

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # --------------------------------------------------
# # 1️⃣ TOKENIZER SETUP
# # --------------------------------------------------
# en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
# fr_tokenizer = get_tokenizer('spacy', language='fr_core_news_sm')

# # --------------------------------------------------
# # 2️⃣ LOAD RAW DATA
# # --------------------------------------------------
# def read_data(en_path, fr_path, max_len=50):
#     en_lines = open(en_path, encoding="utf-8").read().strip().split("\n")
#     fr_lines = open(fr_path, encoding="utf-8").read().strip().split("\n")

#     pairs = []
#     for en, fr in zip(en_lines, fr_lines):
#         src = en_tokenizer(en.lower())
#         trg = fr_tokenizer(fr.lower())
#         if len(src) <= max_len and len(trg) <= max_len:
#             pairs.append((src, trg))
#     return pairs

# DATA_PATH = "data/multi30k_en_fr/"
# train_pairs = read_data(os.path.join(DATA_PATH, "train.en"),
#                         os.path.join(DATA_PATH, "train.fr"))
# print(f"✅ Loaded {len(train_pairs)} sentence pairs.")
# print("👉 Example:", train_pairs[0])

# # --------------------------------------------------
# # 3️⃣ BUILD VOCABULARY
# # --------------------------------------------------
# def build_vocab(sentences, max_size=10000, min_freq=2):
#     counter = Counter([tok for sent in sentences for tok in sent])
#     vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
#     for token, freq in counter.most_common(max_size):
#         if freq >= min_freq and token not in vocab:
#             vocab[token] = len(vocab)
#     return vocab

# src_vocab = build_vocab([src for src, _ in train_pairs])
# trg_vocab = build_vocab([trg for _, trg in train_pairs])
# print(f"English vocab size: {len(src_vocab)}")
# print(f"French vocab size: {len(trg_vocab)}")

# # Reverse mapping for decode
# inv_trg_vocab = {v: k for k, v in trg_vocab.items()}

# PAD_IDX = src_vocab['<pad>']
# SOS_IDX = trg_vocab['<sos>']
# EOS_IDX = trg_vocab['<eos>']

# # --------------------------------------------------
# # 4️⃣ DATASET CLASS
# # --------------------------------------------------
# class TranslationDataset(Dataset):
#     def __init__(self, pairs, src_vocab, trg_vocab):
#         self.pairs = pairs
#         self.src_vocab = src_vocab
#         self.trg_vocab = trg_vocab

#     def encode_sentence(self, sentence, vocab):
#         return [vocab.get(tok, vocab['<unk>']) for tok in sentence]

#     def __getitem__(self, idx):
#         src, trg = self.pairs[idx]
#         src_ids = self.encode_sentence(src, self.src_vocab)
#         trg_ids = [SOS_IDX] + self.encode_sentence(trg, self.trg_vocab) + [EOS_IDX]
#         return torch.tensor(src_ids), torch.tensor(trg_ids)

#     def __len__(self):
#         return len(self.pairs)

# # --------------------------------------------------
# # 5️⃣ COLLATE FUNCTION
# # --------------------------------------------------
# def collate_fn(batch):
#     src_batch, trg_batch = zip(*batch)
#     src_lens = torch.tensor([len(s) for s in src_batch])
#     trg_lens = torch.tensor([len(t) for t in trg_batch])

#     src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
#     trg_batch = pad_sequence(trg_batch, padding_value=PAD_IDX)
#     return src_batch, trg_batch, src_lens, trg_lens

# dataset = TranslationDataset(train_pairs, src_vocab, trg_vocab)
# train_size = int(0.9 * len(dataset))
# val_size = len(dataset) - train_size
# train_ds, val_ds = random_split(dataset, [train_size, val_size])

# train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)
# val_loader = DataLoader(val_ds, batch_size=32, collate_fn=collate_fn)

# print("✅ DataLoader ready!")

# # --------------------------------------------------
# # 6️⃣ ENCODER–DECODER MODEL
# # --------------------------------------------------
# class Encoder(nn.Module):
#     def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
#         super().__init__()
#         self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=PAD_IDX)
#         self.lstm = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)

#     def forward(self, src, src_len):
#         embedded = self.embedding(src)
#         packed = pack_padded_sequence(embedded, src_len.cpu(), enforce_sorted=False)
#         outputs, (hidden, cell) = self.lstm(packed)
#         return hidden, cell


# class Decoder(nn.Module):
#     def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
#         super().__init__()
#         self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=PAD_IDX)
#         self.lstm = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
#         self.fc_out = nn.Linear(hid_dim, output_dim)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, input, hidden, cell):
#         input = input.unsqueeze(0)
#         embedded = self.dropout(self.embedding(input))
#         output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
#         prediction = self.fc_out(output.squeeze(0))
#         return prediction, hidden, cell


# class Seq2Seq(nn.Module):
#     def __init__(self, encoder, decoder, device):
#         super().__init__()
#         self.encoder = encoder
#         self.decoder = decoder
#         self.device = device

#     def forward(self, src, src_len, trg, teacher_forcing_ratio=0.5):
#         batch_size = trg.shape[1]
#         trg_len = trg.shape[0]
#         trg_vocab_size = self.decoder.fc_out.out_features

#         outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
#         hidden, cell = self.encoder(src, src_len)
#         input = trg[0, :]  # <sos>

#         for t in range(1, trg_len):
#             output, hidden, cell = self.decoder(input, hidden, cell)
#             outputs[t] = output
#             teacher_force = random.random() < teacher_forcing_ratio
#             top1 = output.argmax(1)
#             input = trg[t] if teacher_force else top1

#         return outputs

# # --------------------------------------------------
# # 7️⃣ MODEL INIT
# # --------------------------------------------------
# INPUT_DIM = len(src_vocab)
# OUTPUT_DIM = len(trg_vocab)
# ENC_EMB_DIM = 256
# DEC_EMB_DIM = 256
# HID_DIM = 512
# N_LAYERS = 2
# DROPOUT = 0.3

# enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, DROPOUT)
# dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DROPOUT)

# model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)
# print("✅ Model built!")

# # --------------------------------------------------
# # 8️⃣ TRAINING LOOP (Skeleton)
# # --------------------------------------------------
# criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# def train_epoch(model, loader, optimizer, criterion, clip=1):
#     model.train()
#     epoch_loss = 0
#     for src, trg, src_len, trg_len in loader:
#         src, trg = src.to(DEVICE), trg.to(DEVICE)
#         optimizer.zero_grad()
#         output = model(src, src_len, trg)
#         output_dim = output.shape[-1]
#         output = output[1:].reshape(-1, output_dim)
#         trg = trg[1:].reshape(-1)
#         loss = criterion(output, trg)
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
#         optimizer.step()
#         epoch_loss += loss.item()
#     return epoch_loss / len(loader)

# # Example run:
# for epoch in range(1):
#     loss = train_epoch(model, train_loader, optimizer, criterion)
#     print(f"Epoch {epoch+1}: Train Loss = {loss:.3f}")
