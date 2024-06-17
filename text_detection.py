import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2

# グローバル変数としてモデルを定義
recognition_model = None

def load_recognition_model():
    global recognition_model
    if recognition_model is None:
        # ここにモデルの定義を記述
        class MyModelClass(torch.nn.Module):
            def __init__(self):
                super(MyModelClass, self).__init__()
                # モデルの構造を定義
                self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
                self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
                self.fc1 = torch.nn.Linear(64*32*32, 128)
                self.fc2 = torch.nn.Linear(128, 10)  # クラス数を指定

            def forward(self, x):
                x = torch.nn.functional.relu(self.conv1(x))
                x = torch.nn.functional.relu(self.conv2(x))
                x = x.view(-1, 64*32*32)
                x = torch.nn.functional.relu(self.fc1(x))
                x = self.fc2(x)
                return x
        
        recognition_model = MyModelClass()
        recognition_model.load_state_dict(torch.load("path_to_model_weights.pth", map_location='cpu'))
        recognition_model.eval()
    return recognition_model

# 文字検出関数
def detect_text(image_np):
    # ここに文字検出の処理を記述
    # 例としてダミーのボックスと処理後の画像を返す
    height, width, _ = image_np.shape
    boxes = [
        [[int(width*0.1), int(height*0.1)], [int(width*0.2), int(height*0.1)], [int(width*0.2), int(height*0.2)], [int(width*0.1), int(height*0.2)]],
        [[int(width*0.3), int(height*0.3)], [int(width*0.4), int(height*0.3)], [int(width*0.4), int(height*0.4)], [int(width*0.3), int(height*0.4)]]
    ]
    processed_image = image_np
    return boxes, processed_image

# バウンディングボックスを描画する関数
def draw_boxes(image_np, boxes):
    image_with_boxes = image_np.copy()
    for box in boxes:
        cv2.polylines(image_with_boxes, [np.array(box)], isClosed=True, color=(0, 255, 0), thickness=2)
    return image_with_boxes

# バウンディングボックスから文字を抽出する関数
def extract_text_from_boxes(image_np, boxes, model, transform):
    texts = []
    for box in boxes:
        x_min = min(box, key=lambda x: x[0])[0]
        y_min = min(box, key=lambda x: x[1])[1]
        x_max = max(box, key=lambda x: x[0])[0]
        y_max = max(box, key=lambda x: x[1])[1]
        char_image = image_np[y_min:y_max, x_min:x_max]
        char_image_pil = Image.fromarray(char_image).convert('L')
        char_image_tensor = transform(char_image_pil).unsqueeze(0)
        with torch.no_grad():
            output = model(char_image_tensor)
        _, predicted = torch.max(output, 1)
        text = str(predicted.item())  # クラス番号を文字に変換（必要に応じてマッピング）
        texts.append(text)
    return texts

# データ変換の定義
data_transforms = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
