import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import cv2
import mediapipe as mp
import numpy as np
import time
import math

# --- 1. 核心資料庫 (手機版優化比例) ---
TARGETS = {
    '大': {"h_range": (0.20, 0.40), "w_range": (0.35, 0.55), "hint": "下巴放鬆垂直下沉", "muscle": "顳肌 (Temporal)"},
    '嗚': {"h_range": (0.05, 0.15), "w_range": (0.15, 0.30), "hint": "雙唇極度向中心縮圓", "muscle": "口輪匝肌 (Orbicularis)"},
    '一': {"h_range": (0.02, 0.12), "w_range": (0.65, 0.85), "hint": "嘴角用力向耳根拉平", "muscle": "笑肌 (Risorius)"},
    '啊': {"h_range": (0.35, 0.60), "w_range": (0.40, 0.65), "hint": "垂直張力最大化", "muscle": "降口角肌 (Depressor)"},
    '喔': {"h_range": (0.25, 0.45), "w_range": (0.30, 0.50), "hint": "呈垂直長橢圓形", "muscle": "口輪匝肌上層"}
}

st.set_page_config(page_title="AI Speech Coach", layout="centered")
st.title("🗣️ AI 語言教練")

# 介面設定
sel_word = st.sidebar.selectbox("🎯 選擇練習字", list(TARGETS.keys()))
diff_lv = st.sidebar.slider("🔥 難度 (1-5)", 1, 5, 3)

class FaceProcessor(VideoTransformerBase):
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h_img, w_img, _ = img.shape
        results = self.face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        if results.multi_face_landmarks:
            for flm in results.multi_face_landmarks:
                lm = flm.landmark
                # 計算歸一化基準
                f_w = math.sqrt((lm[454].x - lm[234].x)**2 + (lm[454].y - lm[234].y)**2)
                # 繪製關鍵點協助對準
                cv2.circle(img, (int(lm[13].x*w_img), int(lm[13].y*h_img)), 3, (0, 255, 0), -1)
                cv2.circle(img, (int(lm[14].x*w_img), int(lm[14].y*h_img)), 3, (0, 255, 0), -1)
        return img

# WebRTC 啟動 (手機相機關鍵)
webrtc_streamer(
    key="speech-coach", 
    video_transformer_factory=FaceProcessor,
    rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
    media_stream_constraints={"video": True, "audio": False}
)

st.info(f"💡 指引：{TARGETS[sel_word]['hint']}")
st.warning(f"💪 訓練肌肉：{TARGETS[sel_word]['muscle']}")