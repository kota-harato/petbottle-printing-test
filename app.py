import streamlit as st
import numpy as np
from PIL import Image
import base64
from io import BytesIO
import cv2
from text_detection import detect_text, draw_boxes, extract_text_from_boxes, recognition_model, data_transforms

# ページ設定を「wide」に設定
st.set_page_config(layout="wide")

# スタイルを定義
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f0f2f6;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .title {
        color: #333;
        text-align: center;
        font-size: 36px;
        font-weight: bold;
    }
    .subheader {
        color: #333;
        font-size: 24px;
        font-weight: bold;
        margin-top: 20px;
    }
    .highlight {
        color: #ff6347;
        font-weight: bold;
        font-size: 24px;
        text-align: center;
    }
    .ok {
        background-color: #28a745;
        color: white;
        font-weight: bold;
        font-size: 32px;
        text-align: center;
        padding: 20px;
        border-radius: 10px;
        border: 5px solid #28a745;
        margin-top: 20px;
    }
    .ng {
        background-color: #dc3545;
        color: white;
        font-weight: bold;
        font-size: 32px;
        text-align: center;
        padding: 20px;
        border-radius: 10px;
        border: 5px solid #dc3545;
        margin-top: 20px;
    }
    .uploaded-image {
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
    .character-container {
        display: flex;
        justify-content: center;
        flex-wrap: wrap;
        margin-top: 20px;
    }
    .character-box {
        display: inline-block;
        text-align: center;
        margin: 10px;
    }
    .character-image {
        height: 50px;
    }
    #videoElement {
        width: 100%;
        height: auto;
    }
    #overlay {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
    }
    #overlay canvas {
        width: 100%;
        height: auto;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="title">文字検出とOCR</div>', unsafe_allow_html=True)

# JavaScriptとHTMLの埋め込み
components.html(
    """
    <div style="position: relative;">
        <video id="videoElement" autoplay></video>
        <div id="overlay">
            <canvas id="guideCanvas"></canvas>
        </div>
    </div>
    <script>
        var video = document.querySelector("#videoElement");

        if (navigator.mediaDevices.getUserMedia) {
            navigator.mediaDevices.getUserMedia({ video: true })
            .then(function (stream) {
                video.srcObject = stream;
            })
            .catch(function (err0r) {
                console.log("Something went wrong!");
            });
        }

        video.addEventListener('loadedmetadata', function() {
            var canvas = document.getElementById('guideCanvas');
            var context = canvas.getContext('2d');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;

            // 赤い枠線の描画
            var rectWidth = canvas.width * 0.4;
            var rectHeight = rectWidth * (7 / 5);
            var left = (canvas.width - rectWidth) / 2;
            var top = (canvas.height - rectHeight) / 2;
            context.strokeStyle = 'red';
            context.lineWidth = 5;
            context.strokeRect(left, top, rectWidth, rectHeight);
        });
    </script>
    """,
    height=600,
)

# マスターデータの入力
master_data = st.text_input("マスターデータを入力してください", "")

uploaded_files = st.file_uploader("画像を選択してください...", type=["jpg", "png"], accept_multiple_files=True)

def get_image_base64(image_array):
    img = Image.fromarray(image_array)
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return encoded

if uploaded_files and master_data:
    for uploaded_file in uploaded_files:
        st.markdown(f'<div class="subheader">処理中: {uploaded_file.name}</div>', unsafe_allow_html=True)
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)
        boxes, processed_image = detect_text(image_np)
        img_with_boxes = draw_boxes(processed_image, boxes)

        col1, col2 = st.columns(2)

        with col1:
            st.image(img_with_boxes, caption='検出結果', use_column_width=True, output_format='PNG')

        with col2:
            st.markdown('<div class="subheader">検出された文字とOCR結果:</div>', unsafe_allow_html=True)

            # 表形式で検出結果を表示
            texts = extract_text_from_boxes(processed_image, boxes, recognition_model, data_transforms)
            combined_text = ''.join(texts)  # 予測結果を結合して1文にする
            results = []
            for i, (box, text) in enumerate(zip(boxes, texts)):
                x_min, y_min = box[0]
                x_max, y_max = box[2]
                char_image = processed_image[y_min:y_max, x_min:x_max]
                char_image_resized = cv2.resize(char_image, (50, 50))  # リサイズ
                char_image_base64 = get_image_base64(char_image_resized)
                results.append(f'<div class="character-box"><img src="data:image/png;base64,{char_image_base64}" class="character-image"><br>{text}</div>')

            # HTMLの生成
            html_content = ''.join(results)

            # HTMLの表示
            st.markdown(f'<div class="character-container">{html_content}</div>', unsafe_allow_html=True)

            # 予測結果を結合して表示
            st.markdown('<div class="subheader">結合されたOCR結果:</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="highlight">{combined_text}</div>', unsafe_allow_html=True)

            # マスターデータと比較
            if combined_text == master_data:
                st.markdown('<div class="ok">OK</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="ng">NG</div>', unsafe_allow_html=True)
