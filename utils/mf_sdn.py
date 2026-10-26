import torch
import torch.nn as nn

from copy import deepcopy

class CorrectionDataset(torch.utils.data.Dataset):
    """
    Custom PyTorch Dataset for training correction models in multi-fidelity settings.

    This dataset provides input-output tuples where the goal is to learn the discrepancy
    between low-fidelity (LF) and high-fidelity (HF) solutions, optionally conditioned 
    on a reduced representation of the LF input.

    Parameters
    ----------
    v_LF : torch.Tensor
        Reduced low-fidelity representation of shape (num_samples, r), 
        e.g., POD coefficients or encoded inputs.
    u_LF : torch.Tensor
        Full-field low-fidelity solutions of shape (num_samples, Nh).
    u_HF : torch.Tensor
        Full-field high-fidelity solutions of shape (num_samples, Nh).
    """

    def __init__(self, v_LF, u_LF, u_HF):
        self.v_LF = v_LF
        self.u_LF = u_LF
        self.u_HF = u_HF

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return self.v_LF.shape[0]

    def __getitem__(self, idx):
        """
        Retrieves the input and target data at a given index.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.

        Returns
        -------
        v_lf : torch.Tensor
            Reduced representation of the LF input at index `idx`.
        u_lf : torch.Tensor
            Full-field LF solution at index `idx`.
        u_hf : torch.Tensor
            Full-field HF solution at index `idx`.
        """
        v_lf = self.v_LF[idx]
        u_lf = self.u_LF[idx]
        u_hf = self.u_HF[idx]
        return v_lf, u_lf, u_hf
    
class ResidualSDNetwork(nn.Module):
    """
    Residual shallow decoder network to model the correction (residual) between 
    low-fidelity and high-fidelity fields in a multi-fidelity learning setup.

    The network maps a reduced representation (e.g., low-fidelity POD coefficients)
    to a correction field, which can be added to a low-fidelity prediction.

    Parameters
    ----------
    in_dim : int
        Dimensionality of the input (e.g., reduced coefficients).
    out_dim : int
        Dimensionality of the output (e.g., number of spatial points in the field).
    decoder_sizes : list of int, optional
        Sizes of the hidden layers in the decoder. The full network architecture is 
        constructed as [in_dim, *decoder_sizes, out_dim]. Default is [350, 400].
    dropout : float, optional
        Dropout rate applied after each hidden layer (excluding the last). Default is 0.0.
    """

    def __init__(self, in_dim, out_dim, decoder_sizes=[350, 400], dropout=0.0):
        super().__init__()

        self.decoder = torch.nn.ModuleList()
        decoder_sizes.insert(0, in_dim)
        decoder_sizes.append(out_dim)

        for i in range(len(decoder_sizes) - 1):
            self.decoder.append(torch.nn.Linear(decoder_sizes[i], decoder_sizes[i + 1]))
            if i != len(decoder_sizes) - 2:
                self.decoder.append(torch.nn.Dropout(dropout))
                self.decoder.append(torch.nn.ReLU())

    def forward(self, x):
        """
        Forward pass through the residual decoder.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_dim).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, out_dim), representing the predicted residual.
        """
        output = x
        for layer in self.decoder:
            output = layer(output)
        return output

    def freeze(self):
        """
        Freezes the network parameters (no gradients computed) and sets it to evaluation mode.
        """
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """
        Unfreezes the network parameters (gradients will be computed) and sets it to training mode.
        """
        self.train()
        for param in self.parameters():
            param.requires_grad = True

def fit_res(model, train_dataset: CorrectionDataset, valid_dataset: CorrectionDataset,
            batch_size=64, epochs=4000, 
            optim=torch.optim.Adam, 
            criterion=torch.nn.MSELoss(),
            lr=0.001, verbose=False, patience=5):
    """
    Train a residual correction neural network using a modified additive correction scheme.

    This function trains a PyTorch model to predict the correction :math:`\delta` such that:
    .. math::
        u_{HF} = u_{LF} + \delta(v_{LF})

    Training is performed using a custom closure-based optimizer loop and early stopping
    based on the validation loss. The correction is applied additively to the LF prediction.

    Parameters
    ----------
    model : torch.nn.Module
        Neural network model that predicts the correction from v_LF.
    
    train_dataset : CorrectionDataset
        Training dataset containing tuples (v_LF, u_LF, u_HF).
    
    valid_dataset : CorrectionDataset
        Validation dataset used for early stopping.
    
    batch_size : int, optional
        Number of samples per batch (default is 64).
    
    epochs : int, optional
        Maximum number of training epochs (default is 4000).
    
    optim : torch.optim.Optimizer, optional
        Optimizer class to use for training (default is `torch.optim.Adam`).
    
    criterion : callable, optional
        Loss function to use (default is `torch.nn.MSELoss()`).
    
    lr : float, optional
        Initial learning rate (default is 0.001).
    
    verbose : bool, optional
        If True, prints progress at each epoch (default is False).
    
    patience : int, optional
        Number of epochs to wait for validation improvement before early stopping (default is 5).

    Returns
    -------
    train_error_list : np.ndarray
        List of training errors (per epoch), converted to CPU and detached from the computation graph.
    
    valid_error_list : np.ndarray
        List of validation errors (per epoch), converted to CPU and detached from the computation graph.

    Notes
    -----
    - The model is trained to minimize the error between u_HF and the sum of model(v_LF) and u_LF.
    - Early stopping is triggered when validation loss does not improve for `patience` epochs.
    - The learning rate decays by a factor of 0.5 every 100 epochs.
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
                outputs = model(data[0]) + data[1]  # Adding the LF output to the SDN output
                optimizer.zero_grad()
                loss = criterion(outputs, data[2])
                loss.backward()
                return loss
            optimizer.step(closure)
        model.eval()
        with torch.no_grad():
            train_error = criterion(train_dataset.u_HF, model(train_dataset.v_LF) + train_dataset.u_LF)
            valid_error = criterion(valid_dataset.u_HF, model(valid_dataset.v_LF) + valid_dataset.u_LF)
            train_error_list.append(train_error)
            valid_error_list.append(valid_error)
        if verbose:
            print("Epoch "+ str(epoch) + f": Training loss = {float(train_error_list[-1]):.4e} Validation Loss {float(valid_error_list[-1]):.4e}", end = "\r")

        if valid_error == torch.min(torch.tensor(valid_error_list)):
            patience_counter = 0
            best_params = deepcopy(model.state_dict())
        else:
            patience_counter += 1

        if patience_counter == patience:
            model.load_state_dict(best_params)
            train_error = criterion(train_dataset.u_HF, model(train_dataset.v_LF) + train_dataset.u_LF)
            valid_error = criterion(valid_dataset.u_HF, model(valid_dataset.v_LF) + valid_dataset.u_LF)
            if verbose:      
                print("Training done: " + f": Training loss = {float(train_error_list[-1]):.4e} Validation Loss {float(valid_error_list[-1]):.4e}", end = "\r")
            break
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = 1e-3 * (0.5 ** (epoch // 100))
   
    return torch.tensor(train_error_list).detach().cpu().numpy(), torch.tensor(valid_error_list).detach().cpu().numpy()
 