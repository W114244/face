import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import cv2
import mediapipe as mp
import numpy as np
import time
import math
from PIL import Image, ImageDraw, ImageFont

# --- 1. 核心資料庫 (沿用你的 17 字雙軸比例) ---
TARGETS = {
    '大': {"h_range": (0.20, 0.40), "w_range": (0.35, 0.55), "hint": "下巴放鬆垂直下沉", "muscle": "顳肌 (Temporal)"},
    '嗚': {"h_range": (0.05, 0.15), "w_range": (0.15, 0.30), "hint": "雙唇極度向中心縮圓", "muscle": "口輪匝肌 (Orbicularis)"},
    '一': {"h_range": (0.02, 0.12), "w_range": (0.65, 0.85), "hint": "嘴角用力向耳根拉平", "muscle": "笑肌 (Risorius)"},
    '啊': {"h_range": (0.35, 0.60), "w_range": (0.40, 0.65), "hint": "垂直張力最大化", "muscle": "降口角肌 (Depressor)"},
    '喔': {"h_range": (0.25, 0.45), "w_range": (0.30, 0.50), "hint": "呈垂直長橢圓形", "muscle": "口輪匝肌上層"},
    '屋': {"h_range": (0.02, 0.12), "w_range": (0.15, 0.28), "hint": "最緊湊的縮小圓孔", "muscle": "口輪匝肌核心"},
    '哼': {"h_range": (0.00, 0.08), "w_range": (0.45, 0.65), "hint": "閉唇用力抿緊", "muscle": "頦肌 (Mentalis)"},
    '七': {"h_range": (0.05, 0.15), "w_range": (0.70, 0.95), "hint": "橫向拉力極限，露牙", "muscle": "笑肌+頰肌極限"},
    '咪': {"h_range": (0.00, 0.08), "w_range": (0.60, 0.85), "hint": "抿嘴延展，測試肌肉耐力", "muscle": "口輪匝肌邊緣"}
}

# --- 2. 介面設定 ---
st.set_page_config(page_title="AI Speech Coach", layout="centered")
st.title("🗣️ AI 語言教練 (手機版)")

# 側邊選單取代鍵盤與 Tkinter
sel_word = st.sidebar.selectbox("🎯 選擇練習字", list(TARGETS.keys()))
diff_lv = st.sidebar.slider("🔥 難度等級 (1-5)", 1, 5, 3)
tol_map = {1: 0.12, 2: 0.08, 3: 0.05, 4: 0.03, 5: 0.01}
TOLERANCE = tol_map[diff_lv]

# --- 3. 核心處理類別 ---
class FaceProcessor(VideoTransformerBase):
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
        self.hold_start = None
        self.count = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h_img, w_img, _ = img.shape
        
        results = self.face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        is_ok = False
        
        if results.multi_face_landmarks:
            for flm in results.multi_face_landmarks:
                lm = flm.landmark
                # 臉寬基準計算 (你的歸一化邏輯)
                f_w = math.sqrt((lm[454].x - lm[234].x)**2 + (lm[454].y - lm[234].y)**2)
                curr_h = abs(lm[13].y - lm[14].y) / f_w
                curr_w = abs(lm[78].x - lm[308].x) / f_w
                
                # 判定邏輯
                t = TARGETS[sel_word]
                W_TOL = TOLERANCE * (12.0 if diff_lv <= 2 else 6.0 if diff_lv <= 4 else 3.0)
                h_ok = (t["h_range"][0]-TOLERANCE <= curr_h <= t["h_range"][1]+TOLERANCE)
                w_ok = (t["w_range"][0]-W_TOL <= curr_w <= t["w_range"][1]+W_TOL)
                is_ok = h_ok and w_ok

                # 繪製你的視覺箭頭導引
                tp, bp = (int(lm[13].x*w_img), int(lm[13].y*h_img)), (int(lm[14].x*w_img), int(lm[14].y*h_img))
                lp, rp = (int(lm[78].x*w_img), int(lm[78].y*h_img)), (int(lm[308].x*w_img), int(lm[308].y*h_img))
                
                def draw_arrow(p1, p2, color):
                    cv2.arrowedLine(img, p1, p2, color, 3, tipLength=0.3)

                # 垂直引導 (太小則往外拉，太大則往內縮)
                if curr_h < t["h_range"][0]-TOLERANCE: 
                    draw_arrow(tp, (tp[0], tp[1]-40), (0, 255, 0))
                    draw_arrow(bp, (bp[0], bp[1]+40), (0, 255, 0))
                elif curr_h > t["h_range"][1]+TOLERANCE:
                    draw_arrow(tp, (tp[0], tp[1]+40), (0, 0, 255))
                    draw_arrow(bp, (bp[0], bp[1]-40), (0, 0, 255))

        # 成功計時器
        if is_ok:
            if self.hold_start is None: self.hold_start = time.time()
            if time.time() - self.hold_start >= 2.0:
                self.count += 1
                self.hold_start = None
        else:
            self.hold_start = None

        return img

# --- 4. 啟動 Web 串流 ---
webrtc_streamer(
    key="speech-coach",
    video_transformer_factory=FaceProcessor,
    rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
    media_stream_constraints={"video": True, "audio": False}
)

# 顯示動作指引
st.info(f"💡 動作指引：{TARGETS[sel_word]['hint']}")
st.warning(f"💪 訓練肌肉：{TARGETS[sel_word]['muscle']}")