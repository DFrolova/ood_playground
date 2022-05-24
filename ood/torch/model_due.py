import numpy as np



def train_step(*inputs: np.ndarray, architecture: Module, criterion: Callable, optimizer: Optimizer,
               n_targets: int = 1, loss_key: str = None, **optimizer_params) -> np.ndarray:
    """
    Performs a forward-backward pass, and make a gradient step, according to the given ``inputs``.
    Parameters
    ----------
    inputs
        inputs batches. The last ``n_targets`` batches are passed to ``criterion``.
        The remaining batches are fed into the ``architecture``.
    architecture
        the neural network architecture.
    criterion
        the loss function. Returns either a scalar or a dictionary of scalars.
        In the latter case ``loss_key`` must be provided.
    optimizer
    n_targets
        how many values from ``inputs`` to be considered as targets.
    loss_key
        in case ``criterion`` returns a dictionary of scalars,
        indicates which key should be used for gradient computation.
    optimizer_params
        additional parameters that will override the optimizer's current parameters (e.g. lr).
    Notes
    -----
    Note that both input and output are **not** of type ``torch.Tensor`` - the conversion
    to and from ``torch.Tensor`` is made inside this function.
    References
    ----------
    `optimizer_step`
    """
    architecture.train()
    if n_targets >= 0:
        n_inputs = len(inputs) - n_targets
    else:
        n_inputs = -n_targets

    assert 0 <= n_inputs <= len(inputs)
    inputs = sequence_to_var(*inputs, device=architecture)
    inputs, targets = inputs[:n_inputs], inputs[n_inputs:]

    loss = criterion(architecture(*inputs), *targets)

    if loss_key is not None:
        optimizer_step(optimizer, loss[loss_key], **optimizer_params)
        return dmap(to_np, loss)

    optimizer_step(optimizer, loss, **optimizer_params)
    return to_np(loss)