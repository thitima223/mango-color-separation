import cv2
import numpy as np
import urllib.request
from PIL import Image, ImageDraw, ImageFont

url = 'http://192.168.1.116/cam-lo.jpg'
cv2.namedWindow("live transmission", cv2.WINDOW_AUTOSIZE)

try:
    font_path = "C:/Windows/Fonts/tahoma.ttf"
    font = ImageFont.truetype(font_path, 24)
except:
    font = ImageFont.load_default()

def put_thai_text(img, text, position, color=(0, 255, 0)):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    draw.text(position, text, font=font, fill=color)
    
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

COLOR_RANGES = [
    {
        "name": "ยังไม่สุก รออีก 4 วัน",
        "lower": np.array([35, 40, 100]),
        "upper": np.array([85, 255, 255]),
        "color_bgr": (0, 255, 0) 
    },
    {
        "name": "สุกแล้ว",
        "lower": np.array([15, 80, 100]),
        "upper": np.array([34, 255, 255]),
        "color_bgr": (0, 255, 255) 
    },
    {
        "name": "ยังไม่สุก รออีก 2 วัน",
        "lower": np.array([15, 10, 130]),
        "upper": np.array([34, 79, 255]),
        "color_bgr": (0, 165, 255) 
    }
]

while True:
    try:
        img_resp = urllib.request.urlopen(url, timeout=2)
        imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        frame = cv2.imdecode(imgnp, -1)

        if frame is None:
            continue

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        best_color_name = "none"
        best_contour = None
        best_area = 0
        best_bgr = (255, 255, 255)
        best_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        for color_info in COLOR_RANGES:
            mask = cv2.inRange(hsv, color_info["lower"], color_info["upper"])
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for c in cnts:
                area = cv2.contourArea(c)
                if area > 2000 and area > best_area:
                    best_area = area
                    best_contour = c
                    best_color_name = color_info["name"]
                    best_bgr = color_info["color_bgr"]
                    best_mask = mask

        if best_contour is not None:
            cv2.drawContours(frame, [best_contour], -1, best_bgr, 3)
            M = cv2.moments(best_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                cv2.circle(frame, (cx, cy), 7, (255, 255, 255), -1)
                
                rgb_color = (best_bgr[2], best_bgr[1], best_bgr[0])
                frame = put_thai_text(frame, best_color_name, (cx - 30, cy - 35), rgb_color)

        res = cv2.bitwise_and(frame, frame, mask=best_mask)

        cv2.imshow("live transmission", frame)
        cv2.imshow("mask", best_mask)
        cv2.imshow("res", res)

    except Exception as e:
        print(f"Error: {e}")

    key = cv2.waitKey(5)
    if key == ord('q'):
        break

cv2.destroyAllWindows()
