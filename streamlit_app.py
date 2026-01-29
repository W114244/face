import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import cv2
import mediapipe as mp
import numpy as np
import math

# --- 1. 核心資料庫 (確保這段在最前面) ---
TARGETS = {
    '大': {"h_range": (0.20, 0.40), "hint": "下巴放鬆垂直下沉", "muscle": "顳肌"},
    '嗚': {"h_range": (0.05, 0.15), "hint": "雙唇極度向中心縮圓", "muscle": "口輪匝肌"},
    '一': {"h_range": (0.02, 0.12), "hint": "嘴角用力向耳根拉平", "muscle": "笑肌"},
    '啊': {"h_range": (0.35, 0.60), "hint": "垂直張力最大化", "muscle": "降口角肌"},
    '喔': {"h_range": (0.25, 0.45), "hint": "呈垂直長橢圓形", "muscle": "口輪匝肌上層"}
}

st.set_page_config(page_title="AI Speech Coach")
st.title("🗣️ AI 語言教練")

sel_word = st.sidebar.selectbox("🎯 選擇練習字", list(TARGETS.keys()))

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
                f_w = math.sqrt((lm[454].x - lm[234].x)**2 + (lm[454].y - lm[234].y)**2)
                cv2.circle(img, (int(lm[13].x*w_img), int(lm[13].y*h_img)), 3, (0, 255, 0), -1)
                cv2.circle(img, (int(lm[14].x*w_img), int(lm[14].y*h_img)), 3, (0, 255, 0), -1)
        return img

webrtc_streamer(
    key="speech-coach", 
    video_transformer_factory=FaceProcessor,
    rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
    media_stream_constraints={"video": True, "audio": False}
)

st.info(f"💡 指引：{TARGETS[sel_word]['hint']}")
st.warning(f"💪 訓練肌肉：{TARGETS[sel_word]['muscle']}")