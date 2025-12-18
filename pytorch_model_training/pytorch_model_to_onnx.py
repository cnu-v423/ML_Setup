import torch
import torchvision.models as models
from pytorch_backbone_model_v2 import build_unet_resnet50
import yaml
import os
import traceback




def convert_model_to_onnx(config):

    try:

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = build_unet_resnet50(
            num_classes=1, input_size=config['data']['input_size'], freeze_backbone=True
        ).to(device)


        # weights_path = '/workspace/input/triton_models/building_models/unet_resnet50_final.pt'
        weights_path = '/workspace/input/ML_training/models/pakka_house_256_model_v3/unet_resnet50_final.pt'

        if weights_path and os.path.exists(weights_path):
            print(f"✅ Loading weights from: {weights_path}")
            state_dict = torch.load(weights_path, map_location=device)
            model.load_state_dict(state_dict)
        else:
            print(f"❌ Weights file not found: {weights_path}")
            return


        model.eval()

        dummy_input = torch.randn(1, 3, 256, 256, device=device)

        torch.onnx.export(
            model,
            dummy_input,
            "/workspace/input/ML_training/models/pakka_house_256_model_v3/model.onnx",
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"}
            },
            verbose=True
        )

        print("Model to Onnx is converted.")

    except Exception as e:
        print(f"❌ ONNX export failed: {e}")
        traceback.print_exc()


if __name__ == '__main__':
    with open('../config/config_v1.yaml', 'r') as f:
        config = yaml.safe_load(f)

    convert_model_to_onnx(config)
    