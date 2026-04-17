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
        # self.sct = mss.mss()
        
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
        
        # 存储模板的特征描述符：{int_id: descriptors}
        self.templates_des = {}
        
        temp_dir = Path("hero_templates")
        temp_dir.mkdir(exist_ok=True)
        heroes = load_heroes()
        with mss.mss() as sct:
            monitor = sct.monitors[1]
            self.sw, self.sh = monitor["width"], monitor["height"]
        
        target_w = int(0.077 * self.sw) # 与 rw 一致
        target_h = int(0.07 * self.sh)  # 与 rh 一致

        for h_id, data in heroes.items():
            local_path = temp_dir / f"{h_id}.png"
            # [下载逻辑省略，保持你原有的即可]
            if not local_path.exists(): continue
            
            img = cv2.imread(str(local_path))
            if img is not None:
                img = cv2.resize(img, (target_w, target_h))
                gray_temp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # 预提取特征点和描述符
                _, des = self.orb.detectAndCompute(gray_temp, None)
                if des is not None:
                    self.templates_des[int(h_id)] = des

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

    def detection(self, hero_blocks, strip_bgr, debug_mode=False):
        team = [0,0,0,0,0]
        
        # 3. 遍历检测到的块进行识别
        for i, (x, w) in enumerate(hero_blocks):
            crop_bgr = strip_bgr[:, x:x+w]
            crop_gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
            
            # 记录识别的小图方便调试
            if debug_mode:
                cv2.imwrite(f"debug_auto_slot_{i}.png", crop_gray)
            
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

    def get_id_list(self, debug_mode=False):
        ids = []
        with mss.mss() as sct:
            # 1. 一次性截取整条 (覆盖全屏宽度的 5% 到 95% 以确保包含所有英雄)
            # 假设你的 strip 还是 0.07 高度
            full_monitor = sct.monitors[1]
            sw, sh = full_monitor["width"], full_monitor["height"]
            
            strip_cfg = {
                "left": int(sw * 0.05), 
                "top": 0, 
                "width": int(sw * 0.9), 
                "height": int(sh * 0.07)
            }
            
            # 如果是 debug 模式且有 test.png，则模拟截取
            if debug_mode:
                full_img = cv2.imread("test.png")
                strip_bgr = full_img[0:int(sh*0.07),:]
            else:
                strip_bgr = np.array(sct.grab(strip_cfg))[:,:,:3]
            left_dection_area = full_img[0:int(sh*0.008),:strip_cfg["width"]//2]
            right_dection_area = full_img[0:int(sh*0.008),strip_cfg["width"]//2:]
            left_strip = strip_bgr[:, :strip_cfg["width"]//2]
            right_strip = strip_bgr[:, strip_cfg["width"]//2:]


            # 2. 自动获取 10 个英雄的坐标块
            rad_hero_blocks = self.get_auto_hero_regions(left_dection_area)
            dire_hero_blocks = self.get_auto_hero_regions(right_dection_area)
            rad_team = self.detection(rad_hero_blocks, left_strip, debug_mode)
            dire_team = self.detection(dire_hero_blocks, right_strip, debug_mode)
            
        all_ids = rad_team + dire_team     
        return all_ids


if __name__ == "__main__":
    detector = HeroDetector()
    print(f"识别结果: {detector.get_id_list(debug_mode=True)}")
    