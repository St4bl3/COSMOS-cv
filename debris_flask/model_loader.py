import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import os

# --- USER: THIS IS YOUR EXACT MODEL PATH ---
ABSOLUTE_MODEL_PATH = r"C:\college\CV\COSMOS\debris_models\debris_detector_epoch_15.pth"
# --- ---

NUM_CLASSES = 2  # 1 class (asteroid/debris) + background
detection_model = None
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("--- MODEL LOADER DEBUG START (Attempting V1 Architecture) ---")
print(f"Attempting to load model from hardcoded path: {ABSOLUTE_MODEL_PATH}")
print(f"Device for model: {DEVICE}")

if not os.path.exists(ABSOLUTE_MODEL_PATH):
    print(f"CRITICAL ERROR: File does NOT exist at the specified path: {ABSOLUTE_MODEL_PATH}")
    print("Please verify the path is absolutely correct and the file is there.")
    detection_model = None
else:
    print(f"SUCCESS: File FOUND at: {ABSOLUTE_MODEL_PATH}")
    try:
        print("Defining model architecture (Attempting FasterRCNN_ResNet50_FPN - V1)...")

        # --- KEY CHANGE HERE: Explicitly use the V1 model ---
        try:
            # For newer torchvision, weights are specified this way for V1
            weights_v1 = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            model_instance = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights_v1)
            print("Using FasterRCNN_ResNet50_FPN (V1) architecture with new weights enum.")
        except AttributeError:
            # Fallback for older torchvision versions that use pretrained=True for V1
            model_instance = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
            print("Using FasterRCNN_ResNet50_FPN (V1) architecture with pretrained=True.")
        # --- END KEY CHANGE ---

        in_features = model_instance.roi_heads.box_predictor.cls_score.in_features
        model_instance.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
        print(f"Model head replaced for {NUM_CLASSES} classes.")

        print(f"Attempting to load checkpoint (weights) from: {ABSOLUTE_MODEL_PATH}")
        # Load checkpoint with strict=False first to see all mismatches if any persist
        # We can change to strict=True if it loads cleanly
        checkpoint = torch.load(ABSOLUTE_MODEL_PATH, map_location=DEVICE)
        print("Checkpoint file loaded into memory.")

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            print("Checkpoint is a dictionary with 'model_state_dict'. Loading it...")
            # Try with strict=False to get more info if there are still minor mismatches
            model_instance.load_state_dict(checkpoint['model_state_dict'], strict=True) # Change to strict=True
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint: 
            print("Checkpoint is a dictionary with 'state_dict'. Loading it...")
            model_instance.load_state_dict(checkpoint['state_dict'], strict=True) # Change to strict=True
        elif isinstance(checkpoint, dict):
            print("Checkpoint is a dictionary, attempting to load directly as state_dict...")
            model_instance.load_state_dict(checkpoint, strict=True) # Change to strict=True
        else:
            print("Checkpoint is not a dictionary, assuming it's a direct state_dict. Loading it...")
            model_instance.load_state_dict(checkpoint, strict=True) # Change to strict=True

        print("SUCCESS: Model weights loaded into architecture.")
        model_instance.eval()
        print("Model set to evaluation mode.")
        detection_model = model_instance 
        detection_model.to(DEVICE) 

    except FileNotFoundError: 
        print(f"CRITICAL FileNotFoundError (secondary check): Could not load {ABSOLUTE_MODEL_PATH}")
        detection_model = None
    except RuntimeError as e:
        print(f"CRITICAL RuntimeError during PyTorch model loading or state_dict assignment: {e}")
        print("This can happen if there's still an architecture mismatch or the file is corrupted.")
        import traceback
        traceback.print_exc()
        detection_model = None
    except Exception as e:
        print(f"CRITICAL UNEXPECTED ERROR during model loading: {e}")
        import traceback
        traceback.print_exc()
        detection_model = None

if detection_model is not None:
    print("--- MODEL LOADER DEBUG END: Model Loaded Successfully! ---")
else:
    print("--- MODEL LOADER DEBUG END: Model FAILED To Load. Check CRITICAL errors above. ---")