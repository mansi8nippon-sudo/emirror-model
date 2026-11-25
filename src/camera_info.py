
import cv2

# Open laptop built-in camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera. Try camera index 1.")
    exit()

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)

print("Camera opened successfully!")
print("Resolution:", int(width), "x", int(height))
print("FPS:", fps)

ret, frame = cap.read()

if ret:
    print("Actual frame shape:", frame.shape)
    cv2.imshow("Camera Test", frame)
    cv2.waitKey(3000)
else:
    print("Could not capture frame.")

cap.release()
cv2.destroyAllWindows()
