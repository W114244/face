import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import cv2
import mediapipe as mp
import numpy as np
import math

# --- 1. 核心資料庫 (完整 17 字) ---
TARGETS = {
    '大': {"h_range": (0.20, 0.40), "hint": "下巴放鬆垂直下沉"},
    '嗚': {"h_range": (0.05, 0.15), "hint": "雙唇極度向中心縮圓"},
    '一': {"h_range": (0.02, 0.12), "hint": "嘴角用力向耳根拉平"},
    '啊': {"h_range": (0.35, 0.60), "hint": "垂直張力最大化"},
    '喔': {"h_range": (0.25, 0.45), "hint": "呈垂直長橢圓形"},
    '屋': {"h_range": (0.02, 0.12), "hint": "最緊湊的縮小圓孔"},
    '誒': {"h_range": (0.12, 0.25), "hint": "嘴角微張並橫向拉開"},
    '七': {"h_range": (0.05, 0.15), "hint": "橫向拉力極限，露牙"},
    '咪': {"h_range": (0.00, 0.08), "hint": "抿嘴延展，測試肌肉耐力"},
    '咕': {"h_range": (0.10, 0.20), "hint": "後舌根發力，嘴微圓"},
    '咖': {"h_range": (0.30, 0.50), "hint": "舌根下沉，大張口"},
    '唏': {"h_range": (0.05, 0.15), "hint": "牙齒微合，嘴角拉開"},
    '蘇': {"h_range": (0.08, 0.18), "hint": "唇部微突，小圓口"},
    '特': {"h_range": (0.15, 0.25), "hint": "舌尖抵齒齦，瞬間彈開"},
    '勒': {"h_range": (0.10, 0.20), "hint": "舌尖彈擊，口型自然"},
    '配': {"h_range": (0.15, 0.30), "hint": "雙唇爆發力訓練"},
    '美': {"h_range": (0.05, 0.15), "hint": "抿嘴後放鬆，唇肌訓練"}
}

st.set_page_config(page_title="AI Speech Coach", layout="centered")
st.title("🗣️ AI 語言教練 (17字完全體)")

sel_word = st.sidebar.selectbox("🎯 練習目標", list(TARGETS.keys()))

class FaceProcessor(VideoTransformerBase):
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1) # 鏡像
        h_img, w_img, _ = img.shape
        results = self.face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        if results.multi_face_landmarks:
            for flm in results.multi_face_landmarks:
                lm = flm.landmark
                # 繪製追蹤點幫助對準
                cv2.circle(img, (int(lm[13].x*w_img), int(lm[13].y*h_img)), 3, (0, 255, 0), -1)
                cv2.circle(img, (int(lm[14].x*w_img), int(lm[14].y*h_img)), 3, (0, 255, 0), -1)
        
        # --- 補上這行靈魂 return ---
        return img 

webrtc_streamer(
    key="speech-coach-v2", 
    video_transformer_factory=FaceProcessor,
    rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
    media_stream_constraints={"video": True, "audio": False}
)

st.info(f"💡 發音指引：{TARGETS[sel_word]['hint']}")