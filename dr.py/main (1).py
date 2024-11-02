import math
import cv2
from cvzone.HandTrackingModule import HandDetector

# Initialize the hand detector
detector = HandDetector(maxHands=2, detectionCon=0.8)

# Initialize video capture for webcam and shield video
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
shield_path = "shield.mp4"
shield = cv2.VideoCapture(shield_path)

if not shield.isOpened():
    print(f"Failed to open shield video file: {shield_path}")

def mapFromTo(x, a, b, c, d):
    return (x - a) / (b - a) * (d - c) + c

def Overlay(background, overlay, x, y, size):
    background_h, background_w, c = background.shape
    imgScale = mapFromTo(size, 200, 20, 1.5, 0.2)
    overlay = cv2.resize(overlay, (0, 0), fx=imgScale, fy=imgScale)
    h, w, c = overlay.shape
    try:
        if x + w / 2 >= background_w or y + h / 2 >= background_h or x - w / 2 <= 0 or y - h / 2 <= 0:
            return background
        else:
            overlayImage = overlay[..., :3]
            mask = overlay[..., :3] / 255.0
            background[int(y - h / 2):int(y + h / 2), int(x - w / 2):int(x + w / 2)] = (1 - mask) * background[int(y - h / 2):int(y + h / 2), int(x - w / 2):int(x + w / 2)] + overlay
            return background
    except Exception as e:
        print(f"Error in overlay: {e}")
        return background

def findDistance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

showShield = True
changeTimer = 0

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image from webcam")
        break
    
    # Detect hands in the image
    hands, img = detector.findHands(img, flipType=False, draw=False)
    final = img.copy()

    if hands:
        success, shieldImage = shield.read()
        if not success or shieldImage is None or shieldImage.size == 0:
            print("Failed to read shield image from video")
            shield.set(cv2.CAP_PROP_POS_FRAMES, 0)
            success, shieldImage = shield.read()
            if not success or shieldImage is None or shieldImage.size == 0:
                print("Failed to reset and read shield image from video")
                continue  # Skip this iteration if shield image is not available

        if len(hands) == 2:
            changeTimer += 1
            try:
                if findDistance(hands[0]["lmList"][9], hands[1]["lmList"][9]) < 30:
                    if changeTimer > 100:
                        showShield = not showShield
                        changeTimer = 0
            except Exception as e:
                print(f"Error in distance calculation: {e}")

            if showShield:
                for hand in hands:
                    bbox = hand["bbox"]
                    handSize = bbox[2]
                    cx, cy = hand["center"]
                    if detector.fingersUp(hand)[1]:  # Check if the index finger is up
                        final = Overlay(img, shieldImage, cx, cy, handSize)

        elif len(hands) == 1:
            hand = hands[0]
            bbox = hand["bbox"]
            handSize = bbox[2]
            cx, cy = hand["center"]
            if detector.fingersUp(hand)[1]:  # Check if the index finger is up
                final = Overlay(img, shieldImage, cx, cy, handSize)

    cv2.imshow("Doctor Strange", cv2.flip(final, 1))
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
