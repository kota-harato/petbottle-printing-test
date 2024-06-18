import streamlit as st
import json
import os
import numpy as np
from PIL import Image
import base64
from io import BytesIO
import cv2
from text_detection import detect_text, draw_boxes, extract_text_from_boxes, recognition_model, data_transforms, net

# ページ設定を「wide」に設定
st.set_page_config(layout="wide")

# グローバル変数
MASTER_DATA_FILE = 'master_data.json'

# マスターデータの読み込み
def load_master_data():
    if os.path.exists(MASTER_DATA_FILE):
        with open(MASTER_DATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

# マスターデータの保存
def save_master_data(data):
    with open(MASTER_DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# メインメニュー
menu = ["OCR", "マスターデータ登録"]
choice = st.sidebar.selectbox("メニュー", menu)

if choice == "OCR":
    st.markdown('<div class="title">文字検出とOCR</div>', unsafe_allow_html=True)

    # マスターデータの選択
    master_data = load_master_data()
    if not master_data:
        st.warning("マスターデータが登録されていません。マスターデータを登録してください。")
    else:
        master_choice = st.selectbox("マスターデータを選択してください", list(master_data.keys()))

        uploaded_files = st.file_uploader("画像を選択してください...", type=["jpg", "png"], accept_multiple_files=True)

        def get_image_base64(image_array):
            img = Image.fromarray(image_array)
            buffer = BytesIO()
            img.save(buffer, format="PNG")
            encoded = base64.b64encode(buffer.getvalue()).decode()
            return encoded

        if uploaded_files and master_choice:
            for uploaded_file in uploaded_files:
                st.markdown(f'<div class="subheader">処理中: {uploaded_file.name}</div>', unsafe_allow_html=True)
                image = Image.open(uploaded_file).convert("RGB")
                image_np = np.array(image)
                boxes, processed_image = detect_text(image_np, net)  # netを渡す
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
                    if combined_text == master_data[master_choice]:
                        st.markdown('<div class="ok">OK</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="ng">NG</div>', unsafe_allow_html=True)

elif choice == "マスターデータ登録":
    st.markdown('<div class="title">マスターデータ登録</div>', unsafe_allow_html=True)
    
    master_data = load_master_data()
    new_master_name = st.text_input("新しいマスターデータ名を入力してください", "")
    new_master_value = st.text_area("マスターデータの値を入力してください", "")

    if st.button("登録"):
        if new_master_name and new_master_value:
            master_data[new_master_name] = new_master_value
            save_master_data(master_data)
            st.success(f"マスターデータ '{new_master_name}' を登録しました。")
        else:
            st.error("マスターデータ名と値の両方を入力してください。")

    st.markdown('<div class="subheader">登録済みのマスターデータ</div>', unsafe_allow_html=True)
    for name, value in master_data.items():
        st.write(f"**{name}**: {value}")
