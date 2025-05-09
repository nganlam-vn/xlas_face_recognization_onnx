import argparse
import numpy as np
import cv2 as cv
import onnxruntime as ort
import os
import time

def str2bool(v):
    if v.lower() in ['on', 'yes', 'true', 'y', 't']:
        return True
    elif v.lower() in ['off', 'no', 'false', 'n', 'f']:
        return False
    else:
        raise NotImplementedError

parser = argparse.ArgumentParser()
parser.add_argument('--image1', '-i1', type=str, help='Path to the input image1. Omit for detecting on default camera.')
parser.add_argument('--image2', '-i2', type=str, help='Path to the input image2. When image1 and image2 parameters given then the program try to find a face on both images and runs face recognition algorithm.')
parser.add_argument('--video', '-v', type=str, help='Path to the input video.')
parser.add_argument('--scale', '-sc', type=float, default=1.0, help='Scale factor used to resize input video frames.')
parser.add_argument('--face_detection_model', '-fd', type=str, default='D:/baitap/Nam3_HK2/XLAS/NhanDangKhuonMatOnnx/model/face_detection_yunet_2023mar.onnx', help='Path to the face detection model.')
parser.add_argument('--face_recognition_model', '-fr', type=str, default='D:/baitap/Nam3_HK2/XLAS/NhanDangKhuonMatOnnx/model/face_recognition_sface_2021dec.onnx', help='Path to the face recognition model.')
parser.add_argument('--score_threshold', type=float, default=0.6, help='Filtering out faces of score < score_threshold.')
parser.add_argument('--nms_threshold', type=float, default=0.3, help='Suppress bounding boxes of iou >= nms_threshold.')
parser.add_argument('--top_k', type=int, default=5000, help='Keep top_k bounding boxes before NMS.')
parser.add_argument('--save', '-s', type=str2bool, default=False, help='Set true to save results. This flag is invalid when using camera.')
parser.add_argument('--person', '-p', type=str, default='trunghieu', help='Person name folder to save images to.')
parser.add_argument('--num_images', '-n', type=int, default=100, help='Number of images to capture when pressing S.')
args = parser.parse_args()

# Function to visualize faces and FPS on image
def visualize(input, faces, fps, thickness=2, saved_count=0, total_to_capture=40, capture_mode=False):
    if faces is not None and len(faces) > 0:
        for idx, face in enumerate(faces):
            print(f'Face {idx}, top-left coordinates: ({face[0]}, {face[1]}), width: {face[2]}, height: {face[3]}, score: {face[4]:.2f}')
            coords = face[:4].astype(np.int32)
            cv.rectangle(input, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 255, 0), thickness)
    
    cv.putText(input, f'FPS: {fps:.2f}', (1, 16), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Add instruction text for saving faces
    if capture_mode:
        cv.putText(input, f"CAPTURING: {saved_count}/{total_to_capture}", (10, frameHeight - 40), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv.putText(input, f"Press 'S' to capture {total_to_capture} face images", (10, frameHeight - 40), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
    cv.putText(input, f"Faces saved: {saved_count}", (10, frameHeight - 15), 
              cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv.putText(input, "Press 'ESC' to exit", (frameWidth - 180, frameHeight - 15), 
              cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

# Create a face detector using OpenCV's DNN face detector
def create_opencv_face_detector():
    # Use OpenCV's built-in face detection model
    print("Creating OpenCV DNN face detector...")
    modelFile = "D:/baitap/Nam3_HK2/XLAS/NhanDangKhuonMatOnnx/model/opencv_face_detector_uint8.pb"
    configFile = "D:/baitap/Nam3_HK2/XLAS/NhanDangKhuonMatOnnx/model/opencv_face_detector.pbtxt"
    
    # Check if files exist, if not, use Haar Cascade as fallback
    if not os.path.isfile(modelFile) or not os.path.isfile(configFile):
        print("DNN face detection model not found, using Haar Cascade instead")
        face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
        return face_cascade, "haar"
    
    # Create DNN face detector
    net = cv.dnn.readNetFromTensorflow(modelFile, configFile)
    return net, "dnn"

# Detect faces using OpenCV
def detect_faces(frame, detector):
    height, width, _ = frame.shape
    faces = []
    
    detector_type = detector[1]  # Get the detector type (dnn or haar)
    detector = detector[0]  # Get the actual detector
    
    if detector_type == "dnn":
        # DNN-based detection
        blob = cv.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
        detector.setInput(blob)
        detections = detector.forward()
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > args.score_threshold:
                x1 = int(detections[0, 0, i, 3] * width)
                y1 = int(detections[0, 0, i, 4] * height)
                x2 = int(detections[0, 0, i, 5] * width)
                y2 = int(detections[0, 0, i, 6] * height)
                
                # Convert to x, y, w, h format
                x = max(0, x1)
                y = max(0, y1)
                w = min(width - x, x2 - x1)
                h = min(height - y, y2 - y1)
                
                faces.append([x, y, w, h, confidence])
    else:
        # Haar cascade detection
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        detected_faces = detector.detectMultiScale(gray, 1.1, 4)
        
        for (x, y, w, h) in detected_faces:
            # Haar cascade doesn't provide confidence, so we use a default value
            faces.append([x, y, w, h, 0.9])
    
    print(f"Found {len(faces)} faces")
    return np.array(faces) if faces else np.zeros((0, 5))

# Inference function for face recognition
def recognize_face(frame, face, recognizer_session):
    face_align = frame[face[1]:face[1] + face[3], face[0]:face[0] + face[2]]
    input_tensor = cv.resize(face_align, (112, 112)).astype(np.float32)
    input_tensor = np.expand_dims(input_tensor, axis=0)
    inputs = {recognizer_session.get_inputs()[0].name: input_tensor}

    # Run recognition
    outputs = recognizer_session.run(None, inputs)
    return outputs[0]  # Return recognized face embedding

if __name__ == '__main__':
    # Create OpenCV face detector
    face_detector = create_opencv_face_detector()
    
    # Load face recognition model using ONNX Runtime
    recognizer_session = ort.InferenceSession(args.face_recognition_model)

    # Create a directory to save images if it doesn't exist
    base_save_dir = r"D:/baitap/Nam3_HK2/XLAS/NhanDangKhuonMatOnnx/images"
    person_dir = os.path.join(base_save_dir, args.person)
    if not os.path.exists(person_dir):
        os.makedirs(person_dir)
        print(f"Created directory: {person_dir}")

    cap = cv.VideoCapture(0)
    frameWidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    # Variables for face capturing
    total_to_capture = args.num_images
    images_captured = 0
    capture_mode = False
    capture_interval = 5  # Frames to wait between captures
    frame_counter = 0
    
    # FPS calculation
    fps_start_time = time.time()
    frame_count = 0
    
    while True:
        hasFrame, frame = cap.read()
        if not hasFrame:
            print('No frames grabbed!')
            break

        # Inference for face detection
        faces = detect_faces(frame, face_detector)
        
        # Update FPS calculation
        frame_count += 1
        if frame_count >= 10:
            fps = frame_count / (time.time() - fps_start_time)
            fps_start_time = time.time()
            frame_count = 0
        else:
            fps = 30  # Default initial value

        # Handle face capturing mode
        frame_to_display = frame.copy()
        
        if capture_mode and len(faces) > 0:
            frame_counter += 1
            
            # Capture a face every few frames
            if frame_counter >= capture_interval:
                frame_counter = 0
                
                # Get the first detected face with some margin
                x, y, w, h = map(int, faces[0][:4])
                
                # Add margin around the face (20% of width/height)
                margin_w = int(w * 0.2)
                margin_h = int(h * 0.2)
                
                # Make sure we don't go out of frame bounds
                x_with_margin = max(0, x - margin_w)
                y_with_margin = max(0, y - margin_h)
                w_with_margin = min(frameWidth - x_with_margin, w + 2*margin_w)
                h_with_margin = min(frameHeight - y_with_margin, h + 2*margin_h)
                
                face_align = frame[y_with_margin:y_with_margin+h_with_margin, 
                                  x_with_margin:x_with_margin+w_with_margin]
                
                # Save the face image to the person's directory
                file_name = os.path.join(person_dir, f"{args.person}_{images_captured+1:04d}.jpg")
                cv.imwrite(file_name, face_align)
                print(f"Saved face image to {file_name}")
                
                images_captured += 1
                
                # Exit capture mode if we've captured enough images
                if images_captured >= total_to_capture:
                    capture_mode = False
                    print(f"Finished capturing {total_to_capture} images!")
        
        # Visualize results
        visualize(frame_to_display, faces, fps, 
                 saved_count=images_captured, 
                 total_to_capture=total_to_capture,
                 capture_mode=capture_mode)

        # Display the frame
        cv.imshow('Live', frame_to_display)

        key = cv.waitKey(1)
        if key == 27:  # ESC key to break
            break
            
        # Start capturing when pressing 'S'
        if (key == ord('s') or key == ord('S')) and not capture_mode and len(faces) > 0:
            capture_mode = True
            images_captured = 0
            frame_counter = 0
            print(f"Starting to capture {total_to_capture} images for {args.person}...")

    cap.release()
    cv.destroyAllWindows()
