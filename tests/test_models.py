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


def test_standard_backbone_matches_notebook_feature_map():
    # Notebook: 46 -> 44 -> 42 -> 21 -> 19 -> 17 -> 9 -> 7 -> 5 (Theano rounds pooling up).
    model = build_model(ModelConfig(output_dim=2, head="biternion"))
    assert tuple(model.backbone(torch.zeros(1, 3, 46, 46)).shape) == (1, 64, 5, 5)
    assert model.head[2].in_features == 64 * 5 * 5


def test_floor_pooling_kept_for_legacy_checkpoints():
    model = build_model(ModelConfig(output_dim=2, head="biternion", pool_ceil_mode=False))
    assert tuple(model.backbone(torch.zeros(1, 3, 46, 46)).shape) == (1, 64, 4, 4)


def test_idiap_and_hoc_feature_maps():
    idiap = build_model(ModelConfig(output_dim=3, head="linear", variant="idiap", input_size=(68, 68)))
    assert tuple(idiap.backbone(torch.zeros(1, 3, 68, 68)).shape) == (1, 64, 5, 5)
    hoc = build_model(ModelConfig(output_dim=4, head="classification", variant="hoc", input_size=(123, 54)))
    assert tuple(hoc.backbone(torch.zeros(1, 3, 123, 54)).shape) == (1, 64, 7, 2)
