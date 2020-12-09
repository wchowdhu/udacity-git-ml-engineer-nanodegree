import argparse
import json
import os
import pickle
import sys
import sagemaker_containers
import pandas as pd
import torch
import torch.optim as optim
import torch.utils.data

from model import LSTMClassifier

def model_fn(model_dir):
    """Load the PyTorch model from the `model_dir` directory."""
    print("Loading model.")

    # First, load the parameters used to create the model.
    model_info = {}
    model_info_path = os.path.join(model_dir, 'model_info.pth')
    with open(model_info_path, 'rb') as f:
        model_info = torch.load(f)

    print("model_info: {}".format(model_info))

    # Determine the device and construct the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMClassifier(model_info['embedding_dim'], model_info['hidden_dim'], model_info['vocab_size'], num_layers=model_info['num_layers'], bidirectional=model_info['bidirectional'], drop_prob=model_info['drop_prob'])

    # Load the stored model parameters.
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    # Load the saved word_dict.
    word_dict_path = os.path.join(model_dir, 'word_dict.pkl')
    with open(word_dict_path, 'rb') as f:
        model.word_dict = pickle.load(f)

    model.to(device).eval()

    print("Done loading model.")
    return model


def _get_train_data_loader(batch_size, training_dir, train_file):
    print("Get train data loader.")

    train_data = pd.read_csv(os.path.join(training_dir, train_file), header=None, names=None) #[15000 reviews, 502 features]

    train_y = torch.from_numpy(train_data[[0]].values).float().squeeze() #Shape [15000 rows] includes the sentiment labels of all reviews
    train_X = torch.from_numpy(train_data.drop([0], axis=1).values).long() #Shape [15000 rows, 501 columns]. Columns include the length (first column) and 500 numerical entries of the examples

    train_ds = torch.utils.data.TensorDataset(train_X, train_y) # Build the dataset. Each sample will be retrieved by indexing tensors along the first dimension.
    
    print("Train Data: {}".format(train_data.shape))

    return torch.utils.data.DataLoader(train_ds, batch_size=batch_size) # Build the dataloader. Dataloader: Combines a dataset and a sampler, and provides an iterable over the given dataset.


def _get_validation_data_loader(batch_size, validation_dir, validation_file):
    print("Get validation data loader.")

    validation_data = pd.read_csv(os.path.join(validation_dir, validation_file), header=None, names=None) #[10000 reviews, 502 features]

    validation_y = torch.from_numpy(validation_data[[0]].values).float().squeeze() #Shape [10000 rows] includes the sentiment labels of all reviews
    validation_X = torch.from_numpy(validation_data.drop([0], axis=1).values).long() #Shape [10000 rows, 501 columns]. Columns include the length (first column) and 500 numerical entries of the examples

    validation_ds = torch.utils.data.TensorDataset(validation_X, validation_y) # Build the dataset. Each sample will be retrieved by indexing tensors along the first dimension.
    
    print("Validation Data: {}".format(validation_data.shape))

    return torch.utils.data.DataLoader(validation_ds, batch_size=batch_size) # Build the dataloader. Dataloader: Combines a dataset and a sampler, and provides an iterable over the given dataset.


def validate(model, validation_loader, device, loss_fn, epoch):
    model.eval()
    validation_loss = 0
    with torch.no_grad():
        for batch in validation_loader:
            # get the inputs; batch is a list of [inputs, labels]
            batch_X, batch_y = batch
            
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            predictions = model(batch_X) #forward propagate the inputs to get the model predictions. shape: [batch_size]
            loss = loss_fn(predictions, batch_y) #compute the loss to check how far off the model predictions are from the correct outputs
            
            validation_loss += loss.data.item() #sum up batch loss

    average_loss = validation_loss / len(validation_loader)
    print("Epoch: {}, BCELoss Validation: {:.4f}".format(epoch, average_loss))


def train(model, train_loader, validation_loader, epochs, optimizer, loss_fn, device):
    """
    This is the training method that is called by the PyTorch training script. The parameters
    passed are as follows:
    model        - The PyTorch model that we wish to train.
    train_loader - The PyTorch DataLoader that should be used during training.
    validation_loader - The PyTorch DataLoader that should be used during validation for hyperparameter tuning.
    epochs       - The total number of epochs to train for.
    optimizer    - The optimizer to use during training.
    loss_fn      - The loss function used for training.
    device       - Where the model and data should be loaded (gpu or cpu).
    """
    
    # TODO: Paste the train() method developed in the notebook here.
    for epoch in range(1, epochs + 1):
        model.train() #set the model in training phase
        total_loss = 0
        for batch in train_loader:  
            # get the inputs; batch is a list of [inputs, labels]
            batch_X, batch_y = batch
            
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            # TODO: Complete this train method to train the model provided.
            optimizer.zero_grad() #zero the parameter gradients. This resets the gradients after every batch. 
            predictions = model(batch_X) #forward propagate the inputs to get the model predictions. shape: [batch_size]
            loss = loss_fn(predictions, batch_y) #compute the loss to check how far off the model predictions are from the correct outputs
            loss.backward() #back propagate the loss and compute the gradients
            optimizer.step() #update the weights 
            
            total_loss += loss.data.item() #Compute the total loss
        average_loss = total_loss / len(train_loader)
        print("Epoch: {}, BCELoss Train: {:.4f}".format(epoch, average_loss))
        if validation_loader:
            validate(model, validation_loader, device, loss_fn, epoch)


if __name__ == '__main__':
    # All of the model parameters and training parameters are sent as arguments when the script
    # is executed. Here we set up an argument parser to easily access the parameters.

    parser = argparse.ArgumentParser()

    # Training Parameters
    parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                        help='input batch size for training (default: 512)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    # Model Parameters
    parser.add_argument('--embedding_dim', type=int, default=32, metavar='N',
                        help='size of the word embeddings (default: 32)')
    parser.add_argument('--hidden_dim', type=int, default=100, metavar='N',
                        help='size of the hidden dimension (default: 100)')
    parser.add_argument('--vocab_size', type=int, default=5000, metavar='N',
                        help='size of the vocabulary (default: 5000)')
    parser.add_argument('--num_layers', type=int, default=1, metavar='N',
                        help='number of hidden layers (default: 1)')
    parser.add_argument('--bidirectional', type=bool, default=False, metavar='N',
                        help='bidirectional LSTM (default: False)')
    parser.add_argument('--drop_prob', type=float, default=0.0, metavar='N',
                        help='dropout probability (default: 0.0)')
    parser.add_argument('--hyperparameter_tuning', type=bool, default=False, metavar='N',
                        help='process the validation data for hyperparameter tuning (default: False)')
    parser.add_argument('--train_file', type=str, default='train.csv')
    parser.add_argument('--validation_file', type=str, default='validation_hpt.csv')

    # SageMaker Parameters
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device {}.".format(device))
    print("Data-dir {}.".format(args.data_dir))
    print("Model-dir {}.".format(args.model_dir))

    torch.manual_seed(args.seed)

    # Load the training data.
    train_loader = _get_train_data_loader(args.batch_size, args.data_dir, args.train_file)
    
    # Load the validation data.
    if args.hyperparameter_tuning == True:
        print ("Hyperparameter Tuning: {}".format(args.hyperparameter_tuning))
        validation_loader = _get_validation_data_loader(args.batch_size, args.data_dir, args.validation_file)
    else: 
        validation_loader = None

    # Build the model.
    model = LSTMClassifier(args.embedding_dim, args.hidden_dim, args.vocab_size, num_layers=args.num_layers, bidirectional=args.bidirectional, drop_prob=args.drop_prob).to(device)

    with open(os.path.join(args.data_dir, "word_dict.pkl"), "rb") as f:
        model.word_dict = pickle.load(f)

    print("Model loaded with embedding_dim {}, hidden_dim {}, vocab_size {}, num_layers {}, bidirectional {}, drop_prob {}.".format(
        args.embedding_dim, args.hidden_dim, args.vocab_size, args.num_layers, args.bidirectional, args.drop_prob
    ))

    # Train the model.
    optimizer = optim.Adam(model.parameters())
    loss_fn = torch.nn.BCELoss()

    train(model, train_loader, validation_loader, args.epochs, optimizer, loss_fn, device)

    # Save the parameters used to construct the model
    model_info_path = os.path.join(args.model_dir, 'model_info.pth')
    with open(model_info_path, 'wb') as f:
        model_info = {
            'embedding_dim': args.embedding_dim,
            'hidden_dim': args.hidden_dim,
            'vocab_size': args.vocab_size,
            'num_layers': args.num_layers,
            'bidirectional': args.bidirectional,
            'drop_prob': args.drop_prob,
        }
        torch.save(model_info, f)

	# Save the word_dict
    word_dict_path = os.path.join(args.model_dir, 'word_dict.pkl')
    with open(word_dict_path, 'wb') as f:
        pickle.dump(model.word_dict, f)

	# Save the model parameters
    model_path = os.path.join(args.model_dir, 'model.pth')
    with open(model_path, 'wb') as f:
        torch.save(model.cpu().state_dict(), f)
    
 	# Print model summary
    print ("---MODEL SUMMARY---")
    print (model)
#     print ("---MODEL PARAMETERS SUMMARY---")
#     for name, param in model.named_parameters():
#         if param.requires_grad:
#             print (name, param.data)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print ("TOTAL LEARNABLE PARAMETERS: {}".format(pytorch_total_params))
   
