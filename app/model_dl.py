"""
Deep Learning model placeholders.

This file defines very simple "dummy" classes for LSTM, TCN, and Transformer.
They do NOT train or predict real values yet.

Why do we add these?
- The Streamlit app can import these classes now without errors.
- The project structure is complete.
- Later we can replace each placeholder with a real PyTorch or Keras model.
"""


class DummyLSTM:
    """
    Placeholder for LSTM model.
    Later this will become a real PyTorch/Keras LSTM network.
    """

    def __init__(self):
        # You can add parameters here later if needed.
        pass

    def predict(self, x):
        """
        Fake predict function.
        For now, it only returns x, so the pipeline does not break.
        """
        return x


class DummyTCN:
    """
    Placeholder for TCN model.
    Later this will become a real Temporal Convolutional Network.
    """

    def __init__(self):
        pass

    def predict(self, x):
        return x


class DummyTransformer:
    """
    Placeholder for Transformer model.
    Later this will become a small attention-based model.
    """

    def __init__(self):
        pass

    def predict(self, x):
        return x


def get_model(model_name: str):
    """
    A small helper function that returns the right placeholder model
    depending on its name.

    Example:
    model = get_model("LSTM")
    """

    model_name = model_name.lower()

    if model_name == "lstm":
        return DummyLSTM()
    elif model_name == "tcn":
        return DummyTCN()
    elif model_name == "transformer":
        return DummyTransformer()
    else:
        raise ValueError(f"Unknown model name: {model_name}")
