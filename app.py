import streamlit as st
import json
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from PIL import Image
import base64
from io import BytesIO
import cv2
from text_detection import detect_text, draw_boxes, extract_text_from_boxes, recognition_model, data_transforms, net

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
        flex-wrap: wrap;
        justify-content: center;
        margin-top: 20px;
    }
    .character-box {
        display: flex;
        flex-direction: column;
        align-items: center;
        margin: 10px;
    }
    .character-image {
        height: 50px;
    }
    .stTextInput, .stTextArea {
        background-color: #e6f2ff;  /* 入力欄の背景色を変更 */
        border: 1px solid #a1c2e8;
        border-radius: 5px;
        padding: 10px;
    }
    .stSelectbox {
        background-color: #e6f2ff;  /* プルダウンリストの背景色を変更 */
        border: 1px solid #a1c2e8;
        border-radius: 5px;
        padding: 10px;
    }
    .stButton {
        margin-top: 20px;
    }
    .expiry-highlight {
        font-size: 24px;
        font-weight: bold;
        color: #007BFF;
        background-color: #e6f2ff;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        margin-top: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

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

# マスターデータの初期化
def initialize_master_data():
    if os.path.exists(MASTER_DATA_FILE):
        os.remove(MASTER_DATA_FILE)

# 賞味期限の計算
def calculate_expiry_date(manufacture_date, rule):
    manufacture_date_obj = datetime.strptime(manufacture_date, "%Y-%m")
    if rule == "ルール1: 製造日から1年後":
        expiry_date_obj = manufacture_date_obj + timedelta(days=365)
    elif rule == "ルール2: 製造日から6か月後":
        expiry_date_obj = manufacture_date_obj + timedelta(days=6*30)
    elif rule == "ルール3: 製造日から3か月後":
        expiry_date_obj = manufacture_date_obj + timedelta(days=3*30)
    return expiry_date_obj.strftime("%Y年%m月") + "+HP"

# メインメニュー
menu = ["OCR", "マスターデータ登録"]
choice = st.sidebar.selectbox("メニュー", menu)

def get_image_base64(image_array):
    img = Image.fromarray(image_array)
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return encoded

if choice == "OCR":
    st.markdown('<div class="title">文字検出とOCR</div>', unsafe_allow_html=True)

    # マスターデータの選択
    master_data = load_master_data()
    if not master_data:
        st.warning("マスターデータが登録されていません。マスターデータを登録してください。")
    else:
        master_choice = st.selectbox("マスターデータを選択してください", list(master_data.keys()))

        uploaded_files = st.file_uploader("画像を選択してください...", type=["jpg", "png"], accept_multiple_files=True)

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
                    if combined_text == master_data[master_choice]["expiry_date"]:
                        st.markdown('<div class="ok">OK</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="ng">NG</div>', unsafe_allow_html=True)

elif choice == "マスターデータ登録":
    st.markdown('<div class="title">マスターデータ登録</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown('<div class="subheader">登録済みのマスターデータ</div>', unsafe_allow_html=True)

        master_data = load_master_data()
        if master_data:
            df = pd.DataFrame(master_data).T.reset_index()
            df.columns = ['キー', '品目名', '製造年月', '賞味期限']
            st.dataframe(df)

        for key in master_data.keys():
            if st.button(f"削除 {key}"):
                del master_data[key]
                save_master_data(master_data)
                st.success(f"マスターデータ '{key}' を削除しました。")
                st.experimental_rerun()  # 削除後に再描画するために追加

        if st.button("マスターデータを初期化"):
            initialize_master_data()
            st.success("マスターデータを初期化しました。")
            st.experimental_rerun()

    with col2:
        st.markdown('<div class="subheader">新しいマスターデータの登録</div>', unsafe_allow_html=True)

        product_name = st.text_input("品目名を入力してください")
        manufacture_date = st.text_input("製造年月を入力してください (YYYY-MM)", max_chars=7)
        expiry_rule = st.selectbox("賞味期限のルールを選択してください", ["ルール1: 製造日から1年後", "ルール2: 製造日から6か月後", "ルール3: 製造日から3か月後"], key='expiry_rule_selectbox', help='賞味期限の計算ルールを選択してください')

        if manufacture_date:
            expiry_date = calculate_expiry_date(manufacture_date, expiry_rule)
            st.markdown(f'<div class="expiry-highlight">賞味期限: {expiry_date}</div>', unsafe_allow_html=True)
        else:
            expiry_date = None

        if st.button("登録"):
            if product_name and manufacture_date:
                key = f"{product_name}_{manufacture_date}"
                master_data[key] = {
                    "product_name": product_name,
                    "manufacture_date": manufacture_date,
                    "expiry_date": expiry_date
                }
                save_master_data(master_data)
                st.success(f"品目名 '{product_name}'、製造年月 '{manufacture_date}' と賞味期限 '{expiry_date}' を登録しました。")
            else:
                st.error("品目名、製造年月を入力してください。")
