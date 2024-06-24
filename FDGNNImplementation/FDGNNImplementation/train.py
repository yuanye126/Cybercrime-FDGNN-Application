import copy
from tqdm import tqdm
import torch.nn.functional as F
from torchmetrics.classification import F1Score, Precision, Recall, Accuracy
from torch.optim import Adam

def train(model, n_epoch, dataloader, weights, test_dataloader=None, h=None, o=None, nl=None, batch_size=None, weight_decay=0.0):
    best_f1 = float('-inf')  # Initialize best F1-score to negative infinity
    counter = 0
    best_model_state = None  # Initialize best_model_state to store the best model's state

    f1_score = F1Score(num_classes=2, task='binary', average='weighted')
    precision_score = Precision(num_classes=2, task='binary', average='weighted')
    recall_score = Recall(num_classes=2, task='binary', average='weighted')
    accuracy_score = Accuracy(num_classes=2, task='binary', average='weighted')

    # Define the optimizer with weight decay
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=weight_decay)

    for epoch in tqdm(range(n_epoch), desc="{} {} {} {}".format(h, o, nl, batch_size)):
        for batched_graph, labels in dataloader:
            model.train()
            feat = batched_graph.ndata['feature'].float()

            predd = model(batched_graph, feat)

            loss = F.cross_entropy(predd, labels, weight=weights)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Perform evaluation on the test set
        _, _, acc, pr, re, f1, auc_roc = pred(model, 'weighted', test_dataloader)
        print('Accuracy: {}, Precision: {}, Recall: {}, F1-score: {}, AUCROC: {}'.format(acc, pr, re, f1, auc_roc))

        # Check if the current model is the best based on F1-score
        if f1 > best_f1:
            print("New best model found with F1-score:", f1)
            best_f1 = f1
            counter = 0  # Reset counter
            best_model_state = copy.deepcopy(model.state_dict())  # Save a deep copy of the best model state
            print("Best model state checksum:", get_model_checksum(best_model_state))
        else:
            counter += 1  # Increment counter

        # Check early stopping criterion
        # Add your early stopping code here

    # Load the best model state
    if best_model_state is not None:
        print("Loading the best model state.")
        print("Best model state checksum before loading:", get_model_checksum(best_model_state))
        model.load_state_dict(best_model_state)
        print("Model state checksum after loading:", get_model_checksum(model.state_dict()))

    return model, best_f1
