import torch
from huggingface_hub import PyTorchModelHubMixin
from language_models import DualAttnTransformerLM, TransformerLM


class DualAttnTransformerLM_HFHub(PyTorchModelHubMixin, DualAttnTransformerLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class TransformerLM_HFHub(PyTorchModelHubMixin, TransformerLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)