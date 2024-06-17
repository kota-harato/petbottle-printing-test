import torch
import numpy as np
import cv2
from utils.craft import CRAFT
from utils.craft_utils import adjustResultCoordinates, normalizeMeanVariance
from utils.imgproc import loadImage, resize_aspect_ratio, cvt2HeatmapImg
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn

# CRAFTモデルのロード関数
def load_craft_model(weight_path='craft_mlt_25k.pth'):
    net = CRAFT()  # initialize
    state_dict = torch.load(weight_path, map_location='cpu')

    # プレフィックス 'module.' を削除
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k  # remove 'module.' of DataParallel
        new_state_dict[name] = v

    net.load_state_dict(new_state_dict)
    net.eval()
    return net

net = load_craft_model()

# クラス名をファイルから読み込み
with open('class_names.txt', 'r', encoding='utf-8') as f:
    class_names = [line.strip() for line in f]

# クラス名とインデックスのマッピングを読み込み
class_to_idx = {}
with open('class_to_idx.txt', 'r', encoding='utf-8') as f:
    for line in f:
        class_name, idx = line.strip().split('\t')
        class_to_idx[class_name] = int(idx)

# インデックスからクラス名へのマッピングを作成
idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}

# 文字認識モデルのロード関数
def load_recognition_model(weight_path='character_recognition_model.pth', num_classes=len(class_names)):
    model = models.resnet50(weights=None)  # 事前学習済みモデルを使わない
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)  # クラス数に合わせて変更

    state_dict = torch.load(weight_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    return model

recognition_model = load_recognition_model()

# 変換
data_transforms = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def detect_text(image, text_threshold=0.7, box_expansion=20):
    orig_height, orig_width = image.shape[:2]

    # リサイズ
    img_resized, target_ratio, _ = resize_aspect_ratio(image, 1280, interpolation=cv2.INTER_LINEAR, mag_ratio=1.5)

    # 前処理
    x = normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)
    x = torch.autograd.Variable(x.unsqueeze(0))

    # テキスト領域の検出
    with torch.no_grad():
        y, _ = net(x)

    # テキストスコアマップを取得
    score_text = y[0, :, :, 0].cpu().data.numpy()

    # スコアマップの閾値処理
    binary_map = score_text > text_threshold

    # ヒートマップを元の画像サイズにリサイズ
    binary_map_resized = cv2.resize(binary_map.astype(np.uint8), (orig_width, orig_height))

    # 輪郭の検出
    contours, _ = cv2.findContours(binary_map_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # ボックスの取得
    boxes = []
    for contour in contours:
        rect = cv2.boundingRect(contour)
        # ボックスを広げる
        x_min = max(rect[0] - box_expansion, 0)
        y_min = max(rect[1] - box_expansion, 0)
        x_max = min(rect[0] + rect[2] + box_expansion, orig_width)
        y_max = min(rect[1] + rect[3] + box_expansion, orig_height)
        box = [
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max]
        ]
        boxes.append(box)

    # ボックスをx座標の昇順に並べ替え
    boxes.sort(key=lambda box: box[0][0])

    # 最初の検出ボックスのy座標を使用
    if boxes:
        y_min = min([box[0][1] for box in boxes])
        y_max = max([box[2][1] for box in boxes])

    # ボックスのy座標を調整して上部と下部に余裕を持たせる
    y_min = max(y_min - box_expansion, 0)
    y_max = min(y_max + box_expansion, orig_height)

    # 新しい切り出し領域を作成
    new_boxes = []
    for i in range(len(boxes)):
        if i == 0:
            x_min = boxes[i][0][0]
        else:
            x_min = (boxes[i][0][0] + boxes[i-1][1][0]) // 2
        if i == len(boxes) - 1:
            x_max = boxes[i][1][0]
        else:
            x_max = (boxes[i][1][0] + boxes[i+1][0][0]) // 2
        
        new_boxes.append([[int(x_min), int(y_min)], [int(x_max), int(y_min)], [int(x_max), int(y_max)], [int(x_min), int(y_max)]])

    return new_boxes, image

def draw_boxes(image, boxes):
    img_with_boxes = image.copy()
    for box in boxes:
        x_min, y_min = box[0]
        x_max, y_max = box[2]
        cv2.rectangle(img_with_boxes, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
    return img_with_boxes

def classify_character(image, model, transform):
    image = Image.fromarray(image).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)  # バッチ次元を追加
    with torch.no_grad():
        output = model(image)
    _, predicted = torch.max(output, 1)
    return predicted.item()

def extract_text_from_boxes(image, boxes, model, transform):
    texts = []
    for box in boxes:
        x_min, y_min = box[0]
        x_max, y_max = box[2]
        char_image = image[y_min:y_max, x_min:x_max]
        char_image = cv2.resize(char_image, (32, 32))  # リサイズを追加
        label_idx = classify_character(char_image, model, transform)
        if 0 <= label_idx < len(idx_to_class):
            label = idx_to_class[label_idx]  # インデックスからクラス名に変換
        else:
            label = '?'
        texts.append(label)
    return texts
