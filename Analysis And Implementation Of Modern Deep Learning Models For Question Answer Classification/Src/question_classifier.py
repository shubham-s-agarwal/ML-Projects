
import argparse
import configparser
from typing import List
from sklearn.metrics import f1_score
import pandas as pd
import re
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

SENTENCE = "Sentence"
OUTPUT = "Output"
PADDED_SENTENCES = "padded_sentences"
LABELS_ = "labels_"
SENTENCE_LENGTHS = "sentence_lengths"

SEED = 1
torch.manual_seed(SEED)


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help='Configuration file')
parser.add_argument('--train', action='store_true', help='Training mode - model is saved')


parser.add_argument('--test', action='store_true', help='Testing mode - needs a model to load')
args = parser.parse_args()

config = configparser.ConfigParser()
config.read(args.config)


data_path = config["PATH"]["data"]
path_train = config["PATH"]["path_train"]
path_dev = config["PATH"]["path_dev"]
path_test = config["PATH"]["path_test"]
model_name = config["MODEL_PARAMETERS"]["model"]
epochs = int(config["MODEL_PARAMETERS"]["epoch"])
pre_trained_emb_path = config["MODEL_PARAMETERS"]["path_pre_emb"]


hidden_neurons =  int(config["MODEL_PARAMETERS"]["hidden_neurons"])
lr =  float(config["MODEL_PARAMETERS"]["lr_param"])
hidden_neurons =  int(config["MODEL_PARAMETERS"]["hidden_neurons"])
hidden_layers =  int(config["MODEL_PARAMETERS"]["hidden_layers"])
batch_size_val = int(config["MODEL_PARAMETERS"]["batch_size"])


word_embedding_dim =  int(config["MODEL_PARAMETERS"]["word_embedding_dim"])
path_eval_result =  config["MODEL_PARAMETERS"]["path_eval_result"]

path_weights =  config["MODEL_PARAMETERS"]["path_weights"]

freeze_option_  =  config["MODEL_PARAMETERS"]["freeze"]
freeze_option = (freeze_option_=='True')

random_initialization_embeddings_  =  config["MODEL_PARAMETERS"]["random_intialization"]
random_initialization_embeddings = (random_initialization_embeddings_ =='True')
    

""" UTILITY FUNCTIONS """

def read_entire_dataset():
    df = pd.read_csv(data_path, encoding="latin-1", sep="\n", header=None)
    df = df[0].str.split(" ", n=1, expand=True)
    df.rename({0: OUTPUT, 1: SENTENCE}, axis="columns", inplace=True)

    return df


def word_extraction(sentence):
    words = re.sub("[^\w]", " ", sentence).split()
    ignore = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
              "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its",
              "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this",
              "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has",
              "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or",
              "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between",
              "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down",
              "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there",
              "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other",
              "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t",
              "can", "will", "just", "don", "should", "now"]
    cleaned_text = [w.lower() for w in words if w not in ignore]

    return cleaned_text


def clean_sentences(sentences):
    extracted_sentences = []
    for sent in sentences:
        extracted_sentences.append(word_extraction(sent))
    return extracted_sentences


def build_label_dict():
    """Give each label a unique number using the training dataset to keep label numbering consistent"""
    df = read_entire_dataset()
    label_to_id = {}
    count = 0
    training_labels = df[OUTPUT]
    for label in training_labels:
        label_exist = label_to_id.get(label) is not None
        if not label_exist:
            label_to_id[label] = count
            count += 1

    return label_to_id


def load_embeddings():
    embedding_dict = {}
    with open(pre_trained_emb_path, "rt") as fi:
        full_content = fi.read().strip().split("\n")

    for i in full_content:
        temp_embeddings = i.split(" ")
        first_embedding_item = temp_embeddings[0].split("\t")
        word = first_embedding_item[0]
        first_embedding_digit = first_embedding_item[1]
        temp_embeddings[0] = first_embedding_digit

        embeddings = [float(val) for val in temp_embeddings]
        embedding_dict[word.lower()] = embeddings
        embedding_dict[word.lower()] = embeddings

    return embedding_dict


def create_embedding_matrix(embedding_dict, vocab_dict):
    # Let the first row 0 of the matrix be the padding
    embedding_length = 300

    first_row = [0] * embedding_length
    embedding_matrix = [first_row]
    for word, _id in vocab_dict.items():
        if embedding_dict.get(word):
            embedding_matrix.append(embedding_dict.get(word))

        else:
            embedding_matrix.append(embedding_dict.get("#unk#"))

    return np.array(embedding_matrix)


def make_bow_vector(sentence, embedding_matrix):
    sentence_embedding = sum([embedding_matrix[_id] for _id in sentence]) / len(
        sentence
    )
    vec = torch.tensor(sentence_embedding, dtype=torch.float)
    return vec.view(1, -1).float()


""" DATA PREPROCESSING CLASSES """


class PerformDataSplit:
    def __init__(self):
        self.df = read_entire_dataset()
        self.perform_split_and_write()

    # Train Test Split Function
    def _split_train_val(self, test_size=0.1, shuffle_state=False):
        X_train, X_val, Y_train, Y_val = train_test_split(
            self.df[[SENTENCE]],
            self.df[[OUTPUT]],
            shuffle=shuffle_state,
            test_size=test_size,
            random_state=15,
        )

        X_train.reset_index(inplace=True)
        X_train = X_train[[SENTENCE]]
        X_val.reset_index(inplace=True)
        X_val = X_val[[SENTENCE]]

        Y_train.reset_index(inplace=True)
        Y_train = Y_train[[OUTPUT]]
        Y_val.reset_index(inplace=True)
        Y_val = Y_val[[OUTPUT]]

        return X_train, X_val, Y_train, Y_val

    def perform_split_and_write(self):
        # Call the train_test_split
        X_train, X_val, Y_train, Y_val = self._split_train_val()

        Train = X_train.merge(Y_train, left_on=X_train.index, right_on=Y_train.index)
        Train = Train[[SENTENCE, OUTPUT]]

        Val = X_val.merge(Y_val, left_on=X_val.index, right_on=Y_val.index)
        Val = Val[[SENTENCE, OUTPUT]]

        Train.to_csv(path_train, index=None, sep=" ")
        Val.to_csv(path_dev, index=None, sep=" ")

    def load_splited_datasets(self, train_path, val_path, test_path):
        df_train = pd.read_csv(train_path, sep=" ")
        df_val = pd.read_csv(val_path, sep=" ")
        df_test = pd.read_csv(test_path, encoding="latin-1", sep="\n", header=None)
        df_test = df_test[0].str.split(" ", n=1, expand=True)
        df_test.rename({0: OUTPUT, 1: SENTENCE}, axis="columns", inplace=True)

        return df_train, df_val, df_test


class Preprocess:
    def __init__(self, dataset_sentence: List[str], dataset_labels: List[str], batch_size: int):
        self.embedding_dict = load_embeddings()
        self.labels_dict = build_label_dict()
        self.vocab_dict = self._get_vocab_dict
        self.dataset_sentence = dataset_sentence
        self.dataset_labels = dataset_labels
        self.sentences_ = self.sentences_as_id
        self.labels = self.labels_as_id
        self.batch_size = batch_size
        self.batch_datasets = {}

    @property
    def _get_clean_sentences(self):
        """Removes stop words etc from sentences in the dataset"""
        return clean_sentences(self.dataset_sentence)

    @property
    def _get_vocab_dict(self):
        # Give each vocab a unique number
        vocab_to_id = {}
        for index, (word, _) in enumerate(self.embedding_dict.items()):
            vocab_to_id[word] = index + 1
        return vocab_to_id

    @property
    def get_vocab_size(self):
        return len(self.vocab_dict.keys())

    @property
    def sentences_as_id(self) -> List[List[int]]:
        cleaned_sentences_ = self._get_clean_sentences
        sentence_repr = []
        for sentence in cleaned_sentences_:
            sentence_as_id = [self.vocab_dict.get(word) or 0 for word in sentence]
            # handle empty sentences
            if len(sentence_as_id) == 0:
                sentence_as_id = [0]
            sentence_repr.append(sentence_as_id)
        return sentence_repr

    @property
    def labels_as_id(self):
        labels_as_id = []
        for label_name in self.dataset_labels:
            label_id = self.labels_dict.get(label_name)
            labels_as_id.append(label_id)
        return labels_as_id

    @property
    def get_embedding_matrix(self):
        return create_embedding_matrix(self.embedding_dict, self.vocab_dict)

    @property
    def _sentences_with_labels(self):
        return zip(self.sentences_, self.labels)

    @property
    def _sorted_sentences(self):
        """Sort sentences in a batch in descending order based on the number of tokens"""
        return sorted(
            self._sentences_with_labels, key=lambda x: len(x[0]), reverse=True
        )

    @property
    def _batch_dataset(self):
        data_set_size = len(self._sorted_sentences)
        remainder = data_set_size % self.batch_size
        divisible_dataset = data_set_size - remainder

        batches_ = int(divisible_dataset / self.batch_size)
        batches = [self.batch_size] * batches_
        if remainder > 0:
            batches.append(remainder)

        return batches

    def pad_sentence(self, batched_dataset: List):
        """
        Make sentences same length by padding with 0 and return
        the length of each sentence in the batch and it label
        """
        sentence_lengths = []
        padded_sentences = []
        labels_ = []

        max_sentence = len(batched_dataset[0][0])
        for sentence, label in batched_dataset:
            sentence_length = len(sentence)
            padding_length = max_sentence - sentence_length
            sentence_lengths.append(sentence_length)

            padding = [0] * padding_length
            padded_sentences.append(sentence + padding)
            labels_.append(label)

        return padded_sentences, labels_, sentence_lengths

    def create_batched_dataset(self):
        batches = self._batch_dataset
        index = 0
        for i, x in enumerate(batches):
            index += x
            from_ = i * batches[i - 1]
            to = index
            batched_dataset = self._sorted_sentences[from_:to]
            padded_sentences, labels_, sentence_lengths = self.pad_sentence(
                batched_dataset
            )

            self.batch_datasets[i + 1] = {
                PADDED_SENTENCES: torch.LongTensor(padded_sentences),
                LABELS_: torch.LongTensor(labels_),
                "sentence_lengths": torch.LongTensor(sentence_lengths),
            }

        return self.batch_datasets


""" MODELS CLASSES """

class BILSTMModel(nn.Module):
    def __init__(
            self,
            vocab_size,
            embed_dim,
            hidden,
            n_label,
            n_layers,
            embedding_matrix,
            random_init,
            freeze,
    ):
        super().__init__()
        if not random_init:
            self.embedding = nn.Embedding.from_pretrained(
                torch.tensor(embedding_matrix, dtype=torch.float32), freeze=freeze
            )

        # embedding layer
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim)

        # lstm layer
        self.lstm = nn.LSTM(
            embed_dim,
            hidden,
            num_layers=n_layers,
            bidirectional=True,
            dropout=0.2,
            batch_first=True,
        )

        # dense layer
        self.fc = nn.Linear(hidden * 2, n_label)

        # activation function
        self.act = nn.ReLU()

    def forward(self, sentences, sentence_lengths):
        # embedded = [batch size, sent_len, emb dim]
        embedded = self.embedding(sentences)

        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, sentence_lengths, batch_first=True
        )

        # hidden = [batch size, num layers * num directions,hid dim]
        packed_output, (hidden, cell) = self.lstm(packed_embedded)

        # concat the final forward and backward hidden state
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)

        # hidden = [batch size, hid dim * num directions]
        dense_outputs = self.fc(hidden)

        # Final activation function
        outputs = self.act(dense_outputs)

        # return outputs
        return F.softmax(outputs, dim=1)


class FeedforwardNeuralNetModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedforwardNeuralNetModel, self).__init__()

        # Linear function 1: vocab_size --> 500
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # Non-linearity 1
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Non-linearity 2
        self.relu2 = nn.ReLU()

        # Linear function 3 (readout): 500 --> 3
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Linear function 1
        out = self.fc1(x)
        # Non-linearity 1
        out = self.relu1(out)

        # Linear function 2
        out = self.fc2(out)
        # Non-linearity 2
        out = self.relu2(out)

        # Linear function 3 (readout)
        out = self.fc3(out)

        return F.softmax(out, dim=1)


class CNNModel(nn.Module):
    def __init__(
            self,
            vocab_size,
            embed_dim,
            hidden,
            n_label,
            n_layers,
            embedding_matrix,
            random_init,
            freeze,
    ):
        super(CNNModel, self).__init__()
        filters = [2, 3, 4]
        dropout = 0.5
        n_filters = 100

        # embedding layer
        if not random_init:
            self.embedding = nn.Embedding.from_pretrained(
                torch.tensor(embedding_matrix, dtype=torch.float32), freeze=freeze
            )
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=embed_dim,
                    out_channels=n_filters,
                    kernel_size=fs,
                    padding=fs // 2,
                )
                for fs in filters
            ]
        )

        self.fc = nn.Linear(len(filters) * n_filters, n_label)

        self.dropout = nn.Dropout(dropout)

    def forward(self, sentences, sentence_lengths=None):
        # sentences = [batch size, sent len]

        embedded = self.embedding(sentences)

        # embedded = [batch size, sent len, emb dim]

        embedded = embedded.permute(0, 2, 1)

        # embedded = [batch size, emb dim, sent len]

        conved = [F.relu(conv(embedded)) for conv in self.convs]

        # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        # pooled_n = [batch size, n_filters]

        cat = self.dropout(torch.cat(pooled, dim=1))

        # cat = [batch size, n_filters * len(filter_sizes)]

        return self.fc(cat)


""" TRAINING TESTING AND EVALUATION HELPER FUNCTIONS """


def model_accuracy(predict, y):
    true_predict = (predict == y).float()
    acc = true_predict.sum() / len(true_predict)
    return acc


def train(model, train_dataset, optimizer, criterion):
    total_loss = 0.0
    total_acc = 0.0

    for batch_number, data in train_dataset.items():
        padded_sentences = data[PADDED_SENTENCES]
        labels = data[LABELS_]
        sentence_lengths = data[SENTENCE_LENGTHS]

        optimizer.zero_grad()
        output = model(padded_sentences, sentence_lengths)
        #print("outputs ", output.argmax(axis=1))
        #print("labels ", labels)

        loss = criterion(output, labels)
        acc = model_accuracy(output.argmax(axis=1), labels)
        #print("acc ", acc)

        loss.backward()

        # update the weights
        optimizer.step()

        total_loss += loss.item()
        total_acc += acc.item()

    avg_total_loss = total_loss / len(train_dataset.items())
    avg_total_acc = total_acc / len(train_dataset.items())

    return avg_total_loss, avg_total_acc


def evaluate(model, val_dataset, criterion):
    total_loss = 0.0
    total_acc = 0.0

    # deactivating dropout layers
    model.eval()

    # deactivates autograd
    with torch.no_grad():
        for batch_number, data in val_dataset.items():
            padded_sentences = data[PADDED_SENTENCES]
            labels = data[LABELS_]
            sentence_lengths = data[SENTENCE_LENGTHS]

            output = model(padded_sentences, sentence_lengths)
            loss = criterion(output, labels)
            acc = model_accuracy(output.argmax(axis=1), labels)

            total_loss += loss.item()
            total_acc += acc.item()

    avg_total_loss = total_loss / len(val_dataset.items())
    avg_total_acc = total_acc / len(val_dataset.items())

    return avg_total_loss, avg_total_acc


def train_bow(model, train_dataset, optimizer, criterion, embedding_matrix, dataset_size):
    total_loss = 0.0
    total_acc = 0.0

    for batch_number, data in train_dataset.items():
        padded_sentences = data[PADDED_SENTENCES]
        labels = data[LABELS_]
        sentence_lengths = data[SENTENCE_LENGTHS]

        optimizer.zero_grad()

        # Iterate for each batch
        for sentence, label, length in zip(padded_sentences, labels, sentence_lengths):
            # Make the bag of words vector for stemmed tokens
            bow_vec = make_bow_vector(sentence[:length], embedding_matrix)

            # Get the target label
            target = torch.LongTensor([label])

            # Forward pass to get output
            probs = model(bow_vec)

            # Calculate Loss: softmax --> cross entropy loss
            loss = criterion(probs, target)

            acc = model_accuracy(probs.argmax(axis=1), target)

            # Getting gradients w.r.t. parameters
            loss.backward()

            # Accumulating the loss over time
            total_loss += loss.item()
            total_acc += acc.item()

        # update the weights
        optimizer.step()

    avg_total_loss = total_loss / dataset_size
    avg_total_acc = total_acc / dataset_size

    return avg_total_loss, avg_total_acc


def evaluate_bow(model, val_dataset, criterion, embedding_matrix, dataset_size):
    total_loss = 0.0
    total_acc = 0.0

    # deactivating dropout layers
    model.eval()

    # deactivates autograd
    with torch.no_grad():
        for batch_number, data in val_dataset.items():
            padded_sentences = data[PADDED_SENTENCES]
            labels = data[LABELS_]
            sentence_lengths = data[SENTENCE_LENGTHS]

            for sentence, label, length in zip(
                    padded_sentences, labels, sentence_lengths
            ):
                # Make the bag of words vector for stemmed tokens
                bow_vec = make_bow_vector(sentence[:length], embedding_matrix)

                # Get the target label
                target = torch.LongTensor([label])

                # Forward pass to get output
                probs = model(bow_vec)

                # Calculate Loss: softmax --> cross entropy loss
                loss = criterion(probs, target)

                acc = model_accuracy(probs.argmax(axis=1), target)

                # Accumulating the loss over time
                total_loss += loss.item()
                total_acc += acc.item()

    avg_total_loss = total_loss / dataset_size
    avg_total_acc = total_acc / dataset_size

    return avg_total_loss, avg_total_acc


def load_test_dataset():
    df_test = pd.read_csv(path_test, encoding="latin-1", sep="\013", header=None)
    df_test = df_test[0].str.split(" ", n=1, expand=True)
    df_test.rename({0: OUTPUT, 1: SENTENCE}, axis="columns", inplace=True)

    sentences = df_test[SENTENCE]
    labels = df_test[OUTPUT]
    BATCH_SIZE = 1

    # Pass in the unprocessed datasets sentences and labels
    cleaned_dataset = Preprocess(sentences, labels, BATCH_SIZE)
    test_dataset = cleaned_dataset.create_batched_dataset()

    return test_dataset, cleaned_dataset.vocab_dict, cleaned_dataset.labels_dict


def test_model(model, embedding_matrix, bow_model=False):
    test_dataset, VOCAB_DICT, LABEL_DICT = load_test_dataset()
    total_dataset = len(test_dataset)

    # load weights for best performing model

    #****************************************************************
    path = path_weights
    model.load_state_dict(torch.load(path))
    model.eval()

    reverse_vocab_dict = {val: key for key, val in VOCAB_DICT.items()}
    reverse_label_dict = {val: key for key, val in LABEL_DICT.items()}
    correctly_predicted = 0

    to_write_file = "Sentence\tPredicted\tCorrect"
    
    pred_ls = []
    true_ls = []
    for _, data in test_dataset.items():
        padded_sentences = data[PADDED_SENTENCES]
        labels = data[LABELS_]
        sentence_lengths = data[SENTENCE_LENGTHS]

        for sentence, sentence_length, label in zip(
                padded_sentences, sentence_lengths, labels
        ):
            sentence_ = sentence.tolist()
            sentence_length_ = int(sentence_length)
            label_ = int(label)

            if bow_model:
                bow_vec = make_bow_vector(sentence[:sentence_length_], embedding_matrix)
                output = model(bow_vec).argmax(axis=1)
                prediction = int(output)

            else:
                output = model(
                    torch.LongTensor([sentence_]), torch.LongTensor([sentence_length_])
                ).argmax(axis=1)
                prediction = int(output)

            s= ".".join([reverse_vocab_dict.get(_id) or "" for _id in sentence_])

            to_write_file = to_write_file + (f"\n{s}\t{reverse_label_dict.get(prediction)}\t{reverse_label_dict.get(label_)}")
    
    
            if prediction == label_:
                correctly_predicted += 1

            true_ls.append(label_)
            pred_ls.append(prediction)

    print("Testing Completed\n\nResults:")
    print("\nCorrectly Predicted:", correctly_predicted)

    print("Incorrectly Predicted:", total_dataset - correctly_predicted)

    accuracy = (correctly_predicted / total_dataset) * 100
    print("\nAccuracy:", accuracy)


    print("\nF1 Score:",f1_score(true_ls, pred_ls, average='weighted'))

    with open(path_eval_result, "w") as text_file:
        print(to_write_file, file=text_file)

    print(f"\nTesting Results written successfully at: {path_eval_result}")


if __name__ == "__main__":
    print("************************************************")
    if (args.train):
        instruction = "Train"
    else:
        instruction = "Test"
    if random_initialization_embeddings ==True:
        emb_ins = "Random Initialized Embeddings"
    else:
        emb_ins = "Pre-trained Embeddings"

    print(f"\nPARAMETERS:\n\nModel: {model_name}\nFreeze: {freeze_option}\nWord_Embeddings: {emb_ins}\nInstruction: {instruction}")


    split = PerformDataSplit()
    df_train, df_val, df_test = split.load_splited_datasets(
        path_train, path_dev, path_test
    )

    # Set Hyper parameters
    BATCH_SIZE = batch_size_val
    EMBEDDING_DIM = word_embedding_dim
    HIDDEN = hidden_neurons
    NUM_LABEL = 50
    NUM_LAYERS = hidden_layers

    # Pass in the unprocessed datasets sentences and labels
    preprocessed_training_dataset = Preprocess(df_train[SENTENCE], df_train[OUTPUT], BATCH_SIZE)
    preprocessed_val_dataset = Preprocess(df_val[SENTENCE], df_val[OUTPUT], BATCH_SIZE)

    # Set needed constant parameters
    EMBEDDING_MATRIX = preprocessed_training_dataset.get_embedding_matrix
    VOCAB_SIZE = preprocessed_training_dataset.get_vocab_size
    LABEL_DICT = preprocessed_training_dataset.labels_dict
    RANDOM_EMBEDDING_MATRIX = np.array(nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM).weight.detach().numpy())

    # Batch datasets
    train_dataset = preprocessed_training_dataset.create_batched_dataset()
    valid_dataset = preprocessed_val_dataset.create_batched_dataset()

    best_valid_loss = float("inf")
    if model_name=='bow':
    
        model = FeedforwardNeuralNetModel(EMBEDDING_DIM, HIDDEN, NUM_LABEL)
        optimizer = optim.Adam(model.parameters(),lr=lr)
        criterion = nn.CrossEntropyLoss(reduction="sum")



    elif model_name=='bilstm':
        model = BILSTMModel(
            vocab_size=VOCAB_SIZE,
            embed_dim=EMBEDDING_DIM,
            hidden=HIDDEN,
            n_label=NUM_LABEL,
            n_layers=NUM_LAYERS,
            embedding_matrix=EMBEDDING_MATRIX,
            random_init=random_initialization_embeddings,
            freeze=freeze_option,)
        optimizer = optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss(reduction="sum")

    elif model_name=='cnn':
        model = CNNModel(
            vocab_size=VOCAB_SIZE,
            embed_dim=EMBEDDING_DIM,
            hidden=HIDDEN,
            n_label=NUM_LABEL,
            n_layers=NUM_LAYERS,
            embedding_matrix=EMBEDDING_MATRIX,
            random_init=random_initialization_embeddings,
            freeze=freeze_option,)
        optimizer = optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss(reduction="sum")

    if args.train:
        print("\n\nModel Training Beginning!!!")
        
        for epoch in range(epochs):
            if model_name!='bow':
                #train the model
                train_loss, train_acc = train(model, train_dataset, optimizer, criterion)
                # evaluate the model
                valid_loss, valid_acc = evaluate(model, valid_dataset, criterion)
            
            elif random_initialization_embeddings==False:
                train_loss, train_acc = train_bow(
                    model,
                    train_dataset,
                    optimizer,
                    criterion,
                    EMBEDDING_MATRIX,
                    dataset_size=len(df_train),
                )
                valid_loss, valid_acc = evaluate_bow(
                    model,
                    valid_dataset,
                    criterion,
                    EMBEDDING_MATRIX,
                    dataset_size=len(df_val),
                )

            else:
                train_loss, train_acc = train_bow(
                    model,
                    train_dataset,
                    optimizer,
                    criterion,
                    RANDOM_EMBEDDING_MATRIX,
                    dataset_size=len(df_train),
                )

                valid_loss, valid_acc = evaluate_bow(
                    model,
                    valid_dataset,
                    criterion,
                    RANDOM_EMBEDDING_MATRIX,
                    dataset_size=len(df_val),
                )

            # save the best model
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), path_weights)

            print(
                f"\n\tEpoch {epoch + 1}: Training Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%"
            )
            print(
                f"\t\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%"
            )
    if args.test:

        if model_name=='bow' and random_initialization_embeddings==True:
            EMBEDDING_MATRIX = RANDOM_EMBEDDING_MATRIX
        
        print("\n\nTesting Model!!!")
        test_model(model, EMBEDDING_MATRIX, bow_model=(model_name=='bow'))


