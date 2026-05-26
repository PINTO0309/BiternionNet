import onnx
import torch

from biternionnet.experiments import get_experiment
from biternionnet.models import ModelConfig, build_model
from biternionnet.onnx_export import export_checkpoint_to_onnx


def test_export_checkpoint_to_static_and_dynamic_batch_onnx(tmp_path):
    config = get_experiment("smoke-biternion")
    model = build_model(
        ModelConfig(
            output_dim=2,
            head=config.model_head,
            variant=config.model_variant,
            input_size=config.input_size,
            backbone_activation=config.backbone_activation,
        )
    )
    checkpoint = tmp_path / "best.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {},
            "experiment": config.__dict__,
            "class_to_idx": {},
            "history": [],
        },
        checkpoint,
    )

    outputs = export_checkpoint_to_onnx(checkpoint, tmp_path / "onnx", prefix="smoke", opset=17)

    for path in outputs.values():
        onnx.checker.check_model(onnx.load(path))

    static_dims = [dim.dim_param or dim.dim_value for dim in onnx.load(outputs["static_simplified"]).graph.input[0].type.tensor_type.shape.dim]
    dynamic_dims = [dim.dim_param or dim.dim_value for dim in onnx.load(outputs["dynamic_batch_simplified"]).graph.input[0].type.tensor_type.shape.dim]
    assert static_dims == [1, 3, 46, 46]
    assert dynamic_dims == ["batch", 3, 46, 46]
