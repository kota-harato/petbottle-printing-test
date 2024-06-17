import os
import streamlit.components.v1 as components

# Streamlitコンポーネントの宣言
_component_func = components.declare_component(
    "webcam_overlay",
    path=os.path.dirname(os.path.abspath(__file__)),
)

# カスタムコンポーネント関数
def webcam_overlay():
    component_value = _component_func()
    return component_value
