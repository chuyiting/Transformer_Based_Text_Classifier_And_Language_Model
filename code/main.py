import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import os
import time
import argparse

from tokenizer import SimpleTokenizer
from dataset import SpeechesClassificationDataset, LanguageModelingDataset
from transformer import CLSModel, LanguageModel, ALiBi, DeBERTa
import matplotlib.pyplot as plt 


seed = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" Hyperparameters to use for training to roughly match 
the numbers mentioned in the assignment description """
batch_size = 16  # Number of independent sequences  we will process in parallel
block_size = 32  # Maximum context length for predictions
learning_rate = 1e-3  # Learning rate for the optimizer
n_embd = 64  # Embedding dimension
n_head = 2  # Number of attention heads
n_layer = 4  # Number of transformer layers


eval_interval = 100  # How often to evaluate train and test perplexity during training
max_iters = 700 # For language modeling, we can process all the batches for the entire dataset, but that takes a while, so we'll limit it to 500 iterations. For batch size of 16 and block size of  32, this is roughly, this is  500 * 16 * 32 = 256000 tokens, SOTA LMs are trained on trillions of tokens, so this is a very small dataset.
eval_iters = 200  # Number of iterations to evaluate perplexity on the test set


## classifier training hyperparameters. It is a simple 1 hidden layer feedforward network, with input 
## size of 64, hidden size of 50 and output size of 3.

n_input = 64  # Input size for the classifier, should match the embedding size of the transformer
n_hidden = 100  # Hidden size for the classifier
n_output = 3  # Output size for the classifier, we have 3 classes
epochs_CLS = 15 # epochs for classifier training

LM_map = {
    'LM': LanguageModel, # the default Model
    'ALiBi': ALiBi,
    'DeBERTa': DeBERTa
}

def load_texts(directory):
    """
    This function loads all texts from the specified directory, ignoring any files with "test" in their name. The text is used for "training" the tokenizer. Since our tokenizer is simple, we don't need to do any training, but we still need to ignore the test data. 
    """

    texts = []
    files = os.listdir(directory)
    for filename in files: 
        if "test" in filename:  ## don't "read test files"
            continue
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            texts.append(file.read()) # train_CLS.tsv, train_LM.txt
    return texts



def collate_batch(batch):
    """ Collate a batch of data into a single tensor with padding."""
    '''
    return: (Tensor): (B), (B, block_size)
    '''
    data, labels = zip(*batch)  # Separate the data and labels
    # Pad sequences to the fixed length
    padded_sequences = pad_sequence(data, batch_first=True, padding_value=0)
    padded_sequences = padded_sequences[:, :block_size]  # Truncate if longer
    # Add padding if shorter
    padded_sequences = torch.nn.functional.pad(padded_sequences, (0, max(0, block_size - padded_sequences.shape[1])), "constant", value=0)
    labels = torch.stack(labels)  
    return padded_sequences, labels

def compute_classifier_accuracy(classifier, data_loader):
    """ Compute the accuracy of the classifier on the data in data_loader."""
    classifier.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)
            outputs = classifier(X)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == Y).sum().item()
            total_samples += Y.size(0)
        accuracy = (100 * total_correct / total_samples)
        classifier.train()
        return accuracy


def compute_perplexity(decoderLMmodel, data_loader, eval_iters=100):
    """ Compute the perplexity of the decoderLMmodel on the data in data_loader.
    Make sure to use the cross entropy loss for the decoderLMmodel.
    """
    decoderLMmodel.eval()
    losses= []
    total_loss = 0
    for X, Y in data_loader:
        X, Y = X.to(device), Y.to(device)
        loss = decoderLMmodel(X, Y) # your model should be computing the cross entropy loss
        losses.append(loss.item())
        total_loss += loss.item()
        if len(losses) >= eval_iters: break


    losses = torch.tensor(losses)
    mean_loss = losses.mean()
    perplexity = torch.exp(mean_loss).item()  # Calculate perplexity as exp(mean loss)

    decoderLMmodel.train()
    return perplexity

def load_LM_data(path, tokenizer):
    with open(path, 'r', encoding='utf-8') as f:
        lmtrainText = f.read()
    LM_dataset = LanguageModelingDataset(tokenizer, lmtrainText,  block_size)
    LM_loader = DataLoader(LM_dataset, batch_size=batch_size, shuffle=True)
    return LM_dataset, LM_loader

def save_checkpoint(model, filename):
    checkpoint = {
        'model': model.state_dict()
    }

    torch.save(checkpoint, filename)

def plot(value, labels, plt_y_label, title, filename, x_values=None):
     # Plot the training accuracy
    plt.figure(figsize=(8, 6))
    for val, label in zip(value, labels):
        if x_values is not None:
            plt.plot(x_values, val, label=label)
        else:
            plt.plot(val, label=label)


    plt.xlabel('Epochs')
    plt.ylabel(plt_y_label)
    plt.title(title)
    plt.legend()
    plt.grid()

    # Save the  figure
    plt.savefig(filename)
    print(f"\n\n{plt_y_label} plot saved as {filename}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run model training based on specified model type')
    parser.add_argument('--model', type=str, required=True, help='Model type to train (e.g., CLS, LM)')

    args = parser.parse_args()
    assert args.model in LM_map.keys(), "Invalid model type. Choose either 'CLS' or 'LM'."
    
    print("Loading data and creating tokenizer ...")
    texts = load_texts('speechesdataset')
    tokenizer = SimpleTokenizer(' '.join(texts)) # create a tokenizer from the data
    print("Vocabulary size is", tokenizer.vocab_size)

    if args.model == 'CLS':
        print('Load CLS training dataset...')
        train_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/train_CLS.tsv")
        test_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/test_CLS.tsv")
        train_CLS_loader = DataLoader(train_CLS_dataset, batch_size=batch_size, collate_fn=collate_batch,shuffle=True)
        test_CLS_loader = DataLoader(test_CLS_dataset, batch_size=batch_size, collate_fn=collate_batch)
        print('Finish loading CLS training dataset...')

   
        # for the classification  task, you will train for a fixed number of epochs like this:
        classifier = CLSModel(vocab_size=tokenizer.vocab_size, n_embd=n_embd, n_head=n_head, block_size=block_size, n_layer=n_layer, n_classes=n_output, n_hidden=n_hidden)
        optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)

        start_time = time.time()
        print('start training...')
        for epoch in range(epochs_CLS):
            # the data has already been truncated into (B, block_size), (B)
            for xb, yb in train_CLS_loader:
                xb, yb = xb.to(device), yb.to(device)

                # CLS training code here
                pred = classifier(xb) # (B, n_classes)
                criterion = nn.CrossEntropyLoss()
                loss = criterion(pred, yb)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            train_accuracy = compute_classifier_accuracy(classifier, train_CLS_loader)
            test_accuracy = compute_classifier_accuracy(classifier, test_CLS_loader)
            print(f'E[{epoch}] Loss: {loss.item()}, Train Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}')
        end_time = time.time()
        print(f'Finish training, total training time: {end_time-start_time} seconds\n')
        
        accuracy = compute_classifier_accuracy(classifier, test_CLS_loader)
        print(f'CLS model accuracy on test dataset: {accuracy}')

        print('saving check points for analysis...')
        save_checkpoint(classifier, 'CLS_model.pth')
        print('done saving checkpoints for CLS!')
    
    else:
        print('Load LM datasets...')
        _, train_LM_loader = load_LM_data("speechesdataset/train_LM.txt", tokenizer)
        _, test_LM_hbush_loader = load_LM_data("speechesdataset/test_LM_hbush.txt", tokenizer)
        _, test_LM_obama_loader = load_LM_data("speechesdataset/test_LM_obama.txt", tokenizer)
        _, test_LM_wbush_loader = load_LM_data("speechesdataset/test_LM_wbush.txt", tokenizer)

        print('Finish loading LM training dataset...')

        model_class = LM_map[args.model]
        lm = model_class(tokenizer.vocab_size, n_embd=n_embd, n_head=n_head, block_size=block_size, n_layer=n_layer, n_hidden=n_hidden)
        optimizer = torch.optim.Adam(lm.parameters(), lr=learning_rate)
        
        start_time = time.time()
        print('start training...')
        all_loss = []
        perplexity_tr = []
        perplexity_hb = []
        perplexity_wb = []
        perplexity_ob = []
        eval_epochs = []
        # for the language modeling task, you will iterate over the training data for a fixed number of iterations like this:
        for i, (xb, yb) in enumerate(train_LM_loader):
            if i >= max_iters:
                break
            xb, yb = xb.to(device), yb.to(device)
            # LM training code here
            loss = lm(xb, yb)
            all_loss.append(loss.item())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if (i+1) % eval_interval == 0:
                perplexity_train = compute_perplexity(lm, train_LM_loader)
                perplexity_wbush = compute_perplexity(lm, test_LM_wbush_loader)
                perplexity_obama = compute_perplexity(lm, test_LM_obama_loader)
                perplexity_hbush = compute_perplexity(lm, test_LM_hbush_loader)
                perplexity_tr.append(perplexity_train)
                perplexity_hb.append(perplexity_hbush)
                perplexity_wb.append(perplexity_wbush)
                perplexity_ob.append(perplexity_obama)
                eval_epochs.append(i+1)
                print(f'B[{i+1}] Loss: {loss.item()}, Perplexity Train: {perplexity_train}, Perplexity WBush: {perplexity_wbush}, Perplexity Obama: {perplexity_obama}, Perplexity HBush: {perplexity_hbush}')
        
        end_time = time.time()
        print(f'Finish training, total training time: {end_time-start_time} seconds\n')
        
        plot([all_loss], [f'{args.model}'], 'training loss', title=f'Training Loss for {args.model}', filename=f'{args.model}_loss.png')
        plot([perplexity_hb, perplexity_wb, perplexity_ob, perplexity_tr], ['hbush', 'wbush', 'obama', 'train'], 'Perplexity', title=f'Perplexity for {args.model}', filename=f'{args.model}_perplexity.png', x_values=eval_epochs)
        print('saving check points for analysis...')
        save_checkpoint(lm, f'{args.model}_model.pth')
        print('done saving check points for LM!')

if __name__ == "__main__":
    main()
