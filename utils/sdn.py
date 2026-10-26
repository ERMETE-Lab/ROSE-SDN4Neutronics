# Author: Stefano Riva, PhD Student, NRG, Politecnico di Milano
# Latest Code Update: 09 July 2025
# Latest Doc  Update: 09 July 2025

import torch
import torch.nn as nn
from copy import deepcopy

mre = lambda datatrue, datapred: ((datatrue - datapred).pow(2).sum(axis = -1).sqrt() / (datatrue).pow(2).sum(axis = -1).sqrt()).mean()
num2p = lambda prob : ("%.2f" % (100*prob)) + "%"

class ShallowDecoderNetwork(nn.Module):
    """
    A shallow feedforward neural network for regression or reduced-order modeling, 
    with optional projection onto a spatial basis (e.g., POD modes).

    This network maps a low-dimensional input (e.g., sensor measurements or control parameters)
    to a set of coefficients, which can optionally be projected back to the high-dimensional 
    physical space using provided POD modes.

    Parameters
    ----------
    in_dim : int
        Dimensionality of the input features.
    out_dim : int
        Dimensionality of the output coefficients (e.g., number of POD modes).
    decoder_sizes : list of int, optional
        Sizes of the hidden layers in the decoder. The full architecture will include
        layers from `in_dim` to `decoder_sizes[0]`, ..., to `out_dim`. Default is [350, 400].
    dropout : float, optional
        Dropout rate applied after each hidden layer (excluding the last). Default is 0.0.
    pod_modes : torch.Tensor or None, optional
        Tensor of shape (n_points, out_dim) representing spatial basis (e.g., POD modes).
        If provided, the output is projected back to the spatial domain via matrix multiplication.
        Default is None.
    """

    def __init__(self, in_dim, out_dim, decoder_sizes=[350, 400], dropout=0.0, pod_modes=None):
        super().__init__()

        self.decoder = torch.nn.ModuleList()
        decoder_sizes.insert(0, in_dim)
        decoder_sizes.append(out_dim)

        for i in range(len(decoder_sizes) - 1):
            self.decoder.append(torch.nn.Linear(decoder_sizes[i], decoder_sizes[i + 1]))
            if i != len(decoder_sizes) - 2:
                self.decoder.append(torch.nn.Dropout(dropout))
                self.decoder.append(torch.nn.ReLU())

        self.pod_modes = pod_modes
        if pod_modes is not None:
            assert pod_modes.shape[1] == out_dim

    def forward_pod(self, x):
        """
        Forward pass through the decoder to produce reduced-order coefficients.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_dim).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, out_dim), representing reduced-order coefficients.
        """
        output = x
        for layer in self.decoder:
            output = layer(output)
        return output

    def forward(self, x):
        """
        Full forward pass through the network, including optional projection
        to the physical space using POD modes.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_dim).

        Returns
        -------
        torch.Tensor
            If `pod_modes` is None, returns the decoded coefficients of shape (batch_size, out_dim).
            Otherwise, returns the full field reconstruction of shape (batch_size, n_points).
        """
        output = self.forward_pod(x)

        if self.pod_modes is not None:
            output = torch.matmul(output, self.pod_modes.T)

        return output

    def freeze(self):
        """
        Disables training by setting the module to evaluation mode
        and freezing all parameters (no gradients will be computed).
        """
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """
        Enables training by setting the module to training mode
        and unfreezing all parameters (gradients will be computed).
        """
        self.train()
        for param in self.parameters():
            param.requires_grad = True

class MyDataset(torch.utils.data.Dataset):
    '''
    Input: sequence of input measurements with shape (ntrajectories, ntimes, ninput) and corresponding measurements of high-dimensional state with shape (ntrajectories, ntimes, noutput)
    Output: Torch dataset
    '''

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.len = X.shape[0]
        
    def __getitem__(self, index):
        return self.X[index], self.Y[index]
    
    def __len__(self):
        return self.len
    
def fit(model, train_dataset, valid_dataset,
        batch_size=64, epochs=4000, 
        optim=torch.optim.Adam, 
        criterion=nn.MSELoss(),
        lr=0.001, verbose=True, patience=5):
    """
    Trains a PyTorch model using a closure-based optimizer with early stopping.

    The function performs iterative training over a specified number of epochs 
    and tracks mean relative error (MRE) on training and validation sets. 
    Early stopping is triggered if validation performance does not improve 
    for `patience` consecutive epochs.

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model to train.
    train_dataset : torch.utils.data.Dataset
        Dataset containing training inputs (`X`) and targets (`Y`).
        It must expose `X` and `Y` attributes for full-batch evaluation.
    valid_dataset : torch.utils.data.Dataset
        Dataset containing validation inputs (`X`) and targets (`Y`).
        It must expose `X` and `Y` attributes for full-batch evaluation.
    batch_size : int, optional
        Number of samples per batch in training. Default is 64.
    epochs : int, optional
        Maximum number of training epochs. Default is 4000.
    optim : torch.optim.Optimizer, optional
        Optimizer class to use (not instance). Default is `torch.optim.Adam`.
    criterion : callable, optional
        Loss function used for backpropagation. Default is `torch.nn.MSELoss()`.
    lr : float, optional
        Initial learning rate for the optimizer. Default is 0.001.
    verbose : bool, optional
        Whether to print loss updates during training. Default is True.
    patience : int, optional
        Number of epochs to wait for validation improvement before early stopping. Default is 5.

    Returns
    -------
    train_error_list : np.ndarray
        Array of training errors (MRE) recorded after each epoch.
    valid_error_list : np.ndarray
        Array of validation errors (MRE) recorded after each epoch.

    Notes
    -----
    - Learning rate decays by a factor of 0.5 every 100 epochs.
    - The function assumes the dataset exposes `.X` and `.Y` attributes for evaluation.
    - The model will revert to the best-performing parameters (in terms of validation loss) 
      if early stopping is triggered.
    """
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim(model.parameters(), lr=lr)

    train_error_list = []
    valid_error_list = []

    patience_counter = 0
    best_params = model.state_dict()

    for epoch in range(1, epochs+1):
        for k, data in enumerate(train_loader):
            model.train()
            def closure():
                outputs = model(data[0])
                optimizer.zero_grad()
                loss = criterion(outputs, data[1])
                loss.backward()
                return loss
            optimizer.step(closure)
        model.eval()
        with torch.no_grad():
            train_error = mre(train_dataset.Y, model(train_dataset.X))
            valid_error = mre(valid_dataset.Y, model(valid_dataset.X))
            train_error_list.append(train_error)
            valid_error_list.append(valid_error)
        if verbose:
            print("Epoch "+ str(epoch) + ": Training MRE = " + num2p(train_error_list[-1].detach()) + " \t Validation MRE = " + num2p(valid_error_list[-1].detach()) + " "*10, end = "\r")

        if valid_error == torch.min(torch.tensor(valid_error_list)):
            patience_counter = 0
            best_params = deepcopy(model.state_dict())
        else:
            patience_counter += 1

        if patience_counter == patience:
            model.load_state_dict(best_params)
            train_error = mre(train_dataset.Y, model(train_dataset.X)).detach()
            valid_error = mre(valid_dataset.Y, model(valid_dataset.X)).detach()
            if verbose:
                print("Training done: Training MRE = " + num2p(train_error) + " \t Validation MRE = " + num2p(valid_error))
            break
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = 1e-3 * (0.5 ** (epoch // 100))
   
    return torch.tensor(train_error_list).detach().cpu().numpy(), torch.tensor(valid_error_list).detach().cpu().numpy()
 