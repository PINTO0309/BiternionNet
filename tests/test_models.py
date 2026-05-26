import torch
from torch import nn

from biternionnet.models import ModelConfig, build_model


def test_backbone_activation_switches_to_swish():
    model = build_model(ModelConfig(output_dim=2, head="biternion", backbone_activation="swish"))
    assert any(isinstance(module, nn.SiLU) for module in model.backbone.modules())
    assert not any(isinstance(module, nn.ReLU) for module in model.backbone.modules())
    assert any(isinstance(module, nn.ReLU) for module in model.head.modules())


def test_backbone_activation_relu_remains_default():
    model = build_model(ModelConfig(output_dim=2, head="biternion"))
    assert any(isinstance(module, nn.ReLU) for module in model.backbone.modules())
    assert not any(isinstance(module, nn.SiLU) for module in model.backbone.modules())


def test_swish_model_forward_shape():
    model = build_model(ModelConfig(output_dim=2, head="biternion", backbone_activation="swish"))
    output = model(torch.zeros(2, 3, 46, 46))
    assert tuple(output.shape) == (2, 2)
