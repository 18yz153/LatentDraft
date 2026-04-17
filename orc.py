import mss
import numpy as np
import cv2
import requests
from pathlib import Path
from src.utils import load_heroes
import time
import matplotlib.pyplot as plt

class HeroDetector:
    def __init__(self):
        
        # 初始化 ORB 和 匹配器
        self.orb = cv2.ORB_create(
            nfeatures=1000, 
            scaleFactor=1.2, 
            nlevels=4, # 小图不需要太多层采样
            edgeThreshold=15, 
            patchSize=15, # 默认 31 太大，小头像边缘会被切掉
            fastThreshold=10 # 降低阈值以提取更多特征点
        )
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        

    def get_auto_hero_regions(self, dection_area):
        # --- 1. 你的核心逻辑计算 ---
        gray = cv2.cvtColor(dection_area, cv2.COLOR_BGR2GRAY)
        h_strip, w_strip = gray.shape

        
        max_color = np.max(dection_area, axis=2) 
        combined = np.max(max_color, axis=0).astype(float)
        # 权重参数 1: 0.4 和 0.6
        combined = combined
        


        # 计算并画出阈值线
        # 参数 3: 0.5 乘数
        threshold = combined[10]*2

        # 执行识别逻辑
        binary = (combined > threshold).astype(np.uint8)
        binary = np.tile(binary, (10, 1))
        # kernel = np.ones((1, 10), np.uint8)
        # binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)

        candidates = []
        for i in range(1, num_labels):
            x, w = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_WIDTH]
            if w > w_strip * 0.03: # 参数 4: 0.03 最小宽度
                candidates.append((x, w))
                # 在信号图上标出候选块

        candidates.sort()

        return candidates

    def detection(self, hero_blocks, strip_bgr):
        team = [0,0,0,0,0]
        
        # 3. 遍历检测到的块进行识别
        for i, (x, w) in enumerate(hero_blocks):
            crop_bgr = strip_bgr[:, x:x+w]
            crop_gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
            
            
            # 特征点检测
            kp_crop, des_crop = self.orb.detectAndCompute(crop_gray, None)
            
            if des_crop is None or len(kp_crop) < 60: # 稍微降低门槛，因为 crop 变精准了
                continue

            best_id, max_matches = 0, 0
            for h_id, des_temp in self.templates_des.items():
                matches = self.bf.match(des_crop, des_temp)
                good_matches = [m for m in matches if m.distance < 45]
                
                if len(good_matches) > max_matches:
                    max_matches = len(good_matches)
                    best_id = h_id
            
            # 判定门槛
            if max_matches >= 2:
                team[i] = best_id
        return team
    
    def initialize(self,w, h):
        self.templates_des = {}
        
        temp_dir = Path("hero_templates")
        temp_dir.mkdir(exist_ok=True)
        heroes = load_heroes() 

        for h_id, data in heroes.items():
            local_path = temp_dir / f"{h_id}.png"
            if not local_path.exists(): 
                response = requests.get(f"https://cdn.cloudflare.steamstatic.com{data['img']}")
                with open(str(local_path), "wb") as f:
                    f.write(response.content)

            img = cv2.imread(str(local_path))
            if img is not None:
                img = cv2.resize(img, (w, h))
                gray_temp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # 预提取特征点和描述符
                _, des = self.orb.detectAndCompute(gray_temp, None)
                if des is not None:
                    self.templates_des[int(h_id)] = des


    def get_id_list(self, pasted_image):
        img_array = np.array(pasted_image.image_data.convert("RGB"))
        # PIL 是 RGB，OpenCV 是 BGR，必须转换，否则颜色识别会乱
        full_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        sh, sw, _ = full_img.shape
        
            
        strip_cfg = {
            "left": int(sw * 0.05), 
            "top": 0, 
            "width": int(sw * 0.9), 
            "height": int(sh * 0.07)
        }
        if sw/sh > 10:
            strip_bgr = full_img
        else:
            strip_bgr = full_img[0:int(sh*0.07),:]
        left_dection_area = full_img[0:8,:strip_cfg["width"]//2]
        right_dection_area = full_img[0:8,strip_cfg["width"]//2:]
        left_strip = strip_bgr[:, :strip_cfg["width"]//2]
        right_strip = strip_bgr[:, strip_cfg["width"]//2:]


        # 2. 自动获取 10 个英雄的坐标块
        rad_hero_blocks = self.get_auto_hero_regions(left_dection_area)
        dire_hero_blocks = self.get_auto_hero_regions(right_dection_area)
        w = rad_hero_blocks[0][1]
        h = strip_bgr.shape[0]
        self.initialize(w, h)

        rad_team = self.detection(rad_hero_blocks[:5], left_strip)
        dire_team = self.detection(dire_hero_blocks[:5], right_strip)

        all_ids = rad_team + dire_team     
        return all_ids


if __name__ == "__main__":
    detector = HeroDetector()
    print(f"识别结果: {detector.get_id_list(cv2.imread('test.png'))}")
    