import os
import re
import string
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using system device: {device}")


def clean_text(text):
    text = text.lower()
    text = text.replace("<br />", " ")  # Clean HTML tags if any
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    return text.split()


# --- Classification Loader ---
def load_imdb_split(folder_path):
    texts, labels = [], []
    for sentiment in ['pos', 'neg']:
        sentiment_folder = os.path.join(folder_path, sentiment)
        label = 1 if sentiment == 'pos' else 0
        if os.path.exists(sentiment_folder):
            for filename in os.listdir(sentiment_folder):
                if filename.endswith(".txt"):
                    with open(os.path.join(sentiment_folder, filename), 'r', encoding='utf-8') as f:
                        texts.append(clean_text(f.read()))
                        labels.append(label)
    return texts, labels


# --- Generation Loader ---
def load_joke_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    tokenized_jokes = []
    for line in lines:
        if line.strip():
            tokens = clean_text(line)
            if len(tokens) >= 4:  # Must have enough words for 3 inputs + 1 target
                tokenized_jokes.append(tokens)
    return tokenized_jokes


# --- Vocabulary Builders ---
def build_vocab(data_list, max_vocab=10000):
    word_counts = {}
    for tokens in data_list:
        for word in tokens:
            word_counts[word] = word_counts.get(word, 0) + 1

    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:max_vocab]
    vocab = {word: i + 2 for i, (word, _) in enumerate(sorted_words)}
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = 1
    inv_vocab = {i: word for word, i in vocab.items()}
    return vocab, inv_vocab



class ClassificationDataset(Dataset):
    def __init__(self, data, labels, vocab, max_len=100):
        self.data = data
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.data[idx]
        indices = [self.vocab.get(w, 1) for w in tokens[:self.max_len]]
        indices += [0] * (self.max_len - len(indices))
        return torch.LongTensor(indices), torch.tensor(self.labels[idx])


class GenerationDataset(Dataset):
    def __init__(self, tokenized_jokes, vocab, seq_len=3):
        self.vocab = vocab
        self.seq_len = seq_len
        self.samples = []

        # Creating sliding window chunks: 3 words as input, 4th word as target
        for joke in tokenized_jokes:
            joke_indices = [vocab.get(w, 1) for w in joke]
            for i in range(len(joke_indices) - seq_len):
                x_tokens = joke_indices[i:i + seq_len]
                y_token = joke_indices[i + seq_len]
                self.samples.append((x_tokens, y_token))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.LongTensor(x), torch.tensor(y)



class UniversalRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, model_type="RNN"):
        super(UniversalRNN, self).__init__()
        self.model_type = model_type
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        if model_type == "RNN":
            self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        elif model_type == "LSTM":
            self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        elif model_type == "GRU":
            self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.rnn(x, hidden)

        if self.model_type == "LSTM":
            last_hidden = hidden[0][-1]
        else:
            last_hidden = hidden[-1]

        return self.fc(last_hidden), hidden



def generate_joke_story(model, start_phrase, vocab, inv_vocab, max_words=20):
    model.eval()
    words = clean_text(start_phrase)
    result = list(words)

    # Initialize with the 3 starting words
    input_indices = [vocab.get(w, 1) for w in words]

    with torch.no_grad():
        for _ in range(max_words):
            # Keep only the last 3 words as context window
            context = torch.LongTensor([input_indices[-3:]]).to(device)
            output, _ = model(context)
            next_idx = torch.argmax(output, dim=1).item()

            if next_idx == 0:  # Stop if it generates padding
                break

            result.append(inv_vocab.get(next_idx, "<UNK>"))
            input_indices.append(next_idx)

    return " ".join(result)


if __name__ == "__main__":

    # YOUR EXACT PATHWAYS
    train_path = r"C:\Users\layla\OneDrive\Desktop\stage_4_data\text_classification\train"
    test_path = r"C:\Users\layla\OneDrive\Desktop\stage_4_data\text_classification\test"
    joke_path = r"C:\Users\layla\OneDrive\Desktop\stage_4_data\text_generation\data"

    # CRITICAL SPEED FIX: Downsample data so it doesn't take hours
    print("\n--- Loading Movie Sentiment Data ---")
    tr_txt, tr_lbl = load_imdb_split(train_path)
    te_txt, te_lbl = load_imdb_split(test_path)

    # Keep only 2,000 samples for training and 1,000 for testing to make it fast!
    tr_txt, tr_lbl = tr_txt[:2000], tr_lbl[:2000]
    te_txt, te_lbl = te_txt[:1000], te_lbl[:1000]
    print(f"Loaded {len(tr_txt)} train reviews and {len(te_txt)} test reviews.")

    imdb_vocab, _ = build_vocab(tr_txt, max_vocab=10000)
    train_cls_dataset = ClassificationDataset(tr_txt, tr_lbl, imdb_vocab, max_len=50)  # Reduced max_len to 50 for speed
    train_cls_loader = DataLoader(train_cls_dataset, batch_size=64, shuffle=True)

    current_arch = "GRU"

    # 1. Train Classification
    print(f"\nTraining Classification Model: {current_arch}")
    cls_model = UniversalRNN(len(imdb_vocab), 128, 256, 2, current_arch).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cls_model.parameters(), lr=0.002)

    for epoch in range(2):
        cls_model.train()
        loss_total = 0
        for bx, by in train_cls_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            pred, _ = cls_model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            loss_total += loss.item()
        print(f"Epoch {epoch + 1} Classification Loss: {loss_total / len(train_cls_loader):.4f}")

    # 2. Train Generation
    print("\n--- Loading Joke Generation Data ---")
    joke_tokens = load_joke_data(joke_path)

    joke_vocab, joke_inv_vocab = build_vocab(joke_tokens, max_vocab=5000)
    train_gen_dataset = GenerationDataset(joke_tokens, joke_vocab, seq_len=3)
    train_gen_loader = DataLoader(train_gen_dataset, batch_size=32, shuffle=True)

    print(f"\nTraining Generation Model: {current_arch}")
    gen_model = UniversalRNN(len(joke_vocab), 128, 256, len(joke_vocab), current_arch).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(gen_model.parameters(), lr=0.002)

    for epoch in range(3):  # Lowered to 3 epochs for testing speed
        gen_model.train()
        loss_total = 0
        for bx, by in train_gen_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            pred, _ = gen_model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            loss_total += loss.item()
        print(f"Epoch {epoch + 1} Generation Loss: {loss_total / len(train_gen_loader):.4f}")

    print(f"\nGenerated output ({current_arch}):")
    story = generate_joke_story(gen_model, "what did the", joke_vocab, joke_inv_vocab)
    print(f'"{story}"')