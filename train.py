import argparse
import os
import tqdm
import torch
from sklearn.metrics import accuracy_score

from eval_utils import downstream_validation
import utils
import data_utils
import numpy as np
import model
from torch.utils.data import TensorDataset, DataLoader  # pytorch
import matplotlib.pyplot as plt  # plotting


def setup_dataloader(args):
    """
    return:
        - train_loader: torch.utils.data.Dataloader
        - val_loader: torch.utils.data.Dataloader
    """

    # read in training data from books dataset
    sentences = data_utils.process_book_dir(args.data_dir)

    # build one hot maps for input and output
    (
        vocab_to_index,
        index_to_vocab,
        suggested_padding_len,
    ) = data_utils.build_tokenizer_table(sentences, vocab_size=args.vocab_size)

    # create encoded input and output numpy matrices for the entire dataset and then put them into tensors
    encoded_sentences, lens = data_utils.encode_data(
        sentences,
        vocab_to_index,
        suggested_padding_len,
    )

    # ================== TODO: CODE HERE ================== #
    # Task: Given the tokenized and encoded text, you need to
    # create inputs to the LM model you want to train.
    # E.g., could be target word in -> context out or
    # context in -> target word out.
    # You can build up that input/output table across all
    # encoded sentences in the dataset!
    # Then, split the data into train set and validation set
    # (you can use utils functions) and create respective
    # dataloaders.
    # ===================================================== #

    context_size = 2
    number_of_entries = lens.size * suggested_padding_len
    # print("Total data size =")
    # print(lens.size*suggested_padding_len)

    # Set of words (input) associated with one target word (output)
    input_table = np.zeros((number_of_entries, context_size*2), dtype=np.int32)
    output_table = np.zeros((number_of_entries), dtype=np.int32)

    entry_idx = 0
    # Step 1: lets loop through the sentences and create pairs
    for sentence in encoded_sentences:
        # Within each sentence, we will create sentence length number of input/output combinations
        for target in range(len(sentence)):
            context_words = set()
            target_word = sentence[target]
            for context_word_id in range(target-context_size, target+context_size+1):
                if (context_word_id < 0 or context_word_id >= len(sentence) or context_word_id == target):
                    continue
                context_words.add(sentence[context_word_id])

            # Make context_words size 2*context
            context_words = list(context_words)
            while len(context_words) < (context_size*2):
                context_words.append(0)
            input_table[entry_idx] = list(context_words)
            output_table[entry_idx] = target_word
            entry_idx += 1

    # Examine the shape of our input and ouput tables
    # print("Example of input/output table")
    # print(input_table.shape)
    # print(output_table.shape)

    # Create input/output tensors
    x_tensor = torch.from_numpy(
        input_table)
    y_tensor = torch.from_numpy(output_table)

    # # Test the creation of tensors
    # print("X Tensor Created (Shape):")
    # print(x_tensor.shape)

    # print("X Tensor Created (Stride):")
    # print(x_tensor.stride())

    # print("Y Tensor Created (Shape):")
    # print(y_tensor.shape)

    # print("Y Tensor Created (Stride):")
    # print(y_tensor.stride())

    dataset = TensorDataset(x_tensor, y_tensor)

    # https://stackoverflow.com/questions/50544730/how-do-i-split-a-custom-dataset-into-training-and-test-datasets
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size])

    minibatch_size = args.batch_size

    train_loader = DataLoader(
        train_dataset, shuffle=True, batch_size=minibatch_size)
    val_loader = DataLoader(
        val_dataset, shuffle=True, batch_size=minibatch_size)

    return train_loader, val_loader, index_to_vocab


def setup_model(args):
    """
    return:
        - model: YourOwnModelClass
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Initialize your CBOW or Skip-Gram model.
    # ===================================================== #

    EMBEDDING_DIM = 100
    VOCAB_SIZE = args.vocab_size

    CBOW_model = model.CBOW(VOCAB_SIZE, EMBEDDING_DIM)
    return CBOW_model


def setup_optimizer(args, model):
    """
    return:
        - criterion: loss_fn
        - optimizer: torch.optim
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Initialize the loss function for predictions.
    # Also initialize your optimizer.
    # ===================================================== #
    LEARNING_RATE = 0.005
    # able to do this because i'm not doing skip gram
    criterion = torch.nn.CrossEntropyLoss()

    # Compare both optimizers and pick better one
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    # optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    return criterion, optimizer


def train_epoch(
    args,
    model,
    loader,
    optimizer,
    criterion,
    device,
    training=True,


):
    model.train()
    epoch_loss = 0.0

    # keep track of the model predictions for computing accuracy
    pred_labels = []
    target_labels = []

    # iterate over each batch in the dataloader
    # NOTE: you may have additional outputs from the loader __getitem__, you can modify this
    for (inputs, labels) in tqdm.tqdm(loader):
        # put model inputs to device
        inputs, labels = inputs.to(device).long(), labels.to(device).long()

        # calculate the loss and train accuracy and perform backprop
        # NOTE: feel free to change the parameters to the model forward pass here + outputs
        pred_logits = model(inputs, labels)

        # calculate prediction loss
        loss = criterion(pred_logits.squeeze(), labels)

        # step optimizer and compute gradients during training
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # logging
        epoch_loss += loss.item()

        # compute metrics
        preds = pred_logits.argmax(-1)
        pred_labels.extend(preds.cpu().numpy())
        target_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(pred_labels, target_labels)
    epoch_loss /= len(loader)

    return epoch_loss, acc


def validate(args, model, loader, optimizer, criterion, device):
    # set model to eval mode
    model.eval()

    # don't compute gradients
    with torch.no_grad():
        val_loss, val_acc = train_epoch(
            args,
            model,
            loader,
            optimizer,
            criterion,
            device,
            training=False,
        )

    return val_loss, val_acc


def main(args):
    device = data_utils.get_device(args.force_cpu)

    # load analogies for downstream eval
    external_val_analogies = data_utils.read_analogies(args.analogies_fn)

    if args.downstream_eval:
        word_vec_file = os.path.join(args.outputs_dir, args.word_vector_fn)
        assert os.path.exists(
            word_vec_file), "need to train the word vecs first!"
        downstream_validation(word_vec_file, external_val_analogies)
        return

    # get dataloaders
    train_loader, val_loader, i2v = setup_dataloader(args)
    loaders = {"train": train_loader, "val": val_loader}

    # build model
    model = setup_model(args)
    print(model)

    # get optimizer
    criterion, optimizer = setup_optimizer(args, model)

    # Set up tables to collect train/val loss/acc
    train_loss_summary = []
    val_loss_summary = []
    train_acc_summary = []
    val_acc_summary = []

    for epoch in range(args.num_epochs):
        # train model for a single epoch
        print(f"Epoch {epoch}")
        train_loss, train_acc = train_epoch(
            args,
            model,
            loaders["train"],
            optimizer,
            criterion,
            device,
        )

        print(f"train loss : {train_loss} | train acc: {train_acc}")
        # Within outer for loop, keep track of train loss/acc data
        train_loss_summary.extend([train_loss])
        train_acc_summary.extend([train_acc])

        if epoch % args.val_every == 0 or epoch == (args.num_epochs-1):
            val_loss, val_acc = validate(
                args,
                model,
                loaders["val"],
                optimizer,
                criterion,
                device,
            )
            print(f"val loss : {val_loss} | val acc: {val_acc}")
            # within inner for loop, keep track of val loss/acc data
            val_loss_summary.extend([val_loss])
            val_acc_summary.extend([val_acc])

        # ======================= NOTE ======================== #
        # Saving the word vectors to disk and running the eval
        # can be costly when you do it multiple times. You could
        # change this to run only when your training has concluded.
        # However, incremental saving means if something crashes
        # later or you get bored and kill the process you'll still
        # have a word vector file and some results.
        # ===================================================== #

        # save word vectors
        word_vec_file = os.path.join(args.outputs_dir, args.word_vector_fn)
        print("saving word vec to ", word_vec_file)
        data_utils.save_word2vec_format(word_vec_file, model, i2v)

        # evaluate learned embeddings on a downstream task
        downstream_validation(word_vec_file, external_val_analogies)

        if epoch % args.save_every == 0 or epoch == (args.num_epochs-1):
            ckpt_file = os.path.join(args.outputs_dir, "model.ckpt")
            print("saving model to ", ckpt_file)
            torch.save(model, ckpt_file)

    # outside the for loop, print out the summary tables
    # https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    #
    ax1.set_title("train loss")
    ax1.plot(train_loss_summary, label="train")
    #
    ax2.set_title("val loss")
    ax2.plot(val_loss_summary, label="val")
    #
    ax3.set_title("train accuracy")
    ax3.plot(train_acc_summary, label="train")
    #
    ax4.set_title("val accuracy")
    ax4.plot(val_acc_summary, label="val")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs_dir", default="", type=str,
                        help="where to save training outputs")
    parser.add_argument("--data_dir", type=str, default="books/",
                        help="where the book dataset is stored")
    parser.add_argument(
        "--downstream_eval",
        action="store_true",
        help="run downstream eval on trained word vecs",
    )
    # ======================= NOTE ======================== #
    # If you adjust the vocab_size down below 3000, there
    # may be analogies in the downstream evaluation that have
    # words that are not in your vocabulary, resulting in
    # automatic (currently) zero score for an ABCD where one
    # of A, B, C, or D is not in the vocab. A visible warning
    # will be generated by the evaluation loop for these examples.
    # ===================================================== #
    parser.add_argument(
        "--vocab_size", type=int, default=3000, help="size of vocabulary"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="size of each batch in loader"
    )
    parser.add_argument("--force_cpu", action="store_true", help="debug mode")
    parser.add_argument(
        "--analogies_fn", type=str, default="analogies_v3000_1309.json", help="filepath to the analogies json file"
    )
    parser.add_argument(
        "--word_vector_fn", type=str, help="filepath to store the learned word vectors",
        default='learned_word_vectors.txt'
    )
    parser.add_argument(
        "--num_epochs", default=30, type=int, help="number of training epochs"
    )
    parser.add_argument(
        "--val_every",
        default=5,
        type=int,
        help="number of epochs between every eval loop",
    )
    parser.add_argument(
        "--save_every",
        default=5,
        type=int,
        help="number of epochs between saving model checkpoint",
    )
    # ================== TODO: CODE HERE ================== #
    # Task (optional): Add any additional command line
    # parameters you may need here
    # ===================================================== #

    args = parser.parse_args()
    main(args)
