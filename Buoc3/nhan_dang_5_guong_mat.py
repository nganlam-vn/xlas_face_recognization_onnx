import streamlit as st
import cv2 as cv
import numpy as np
import os
import joblib
import time
from PIL import Image
import onnxruntime as ort

# Define the base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Setup page configuration
st.set_page_config(
    page_title="Nhận diện 5 gương mặt",
    page_icon="😀",
    layout="wide"
)

# Title and description
st.title("Nhận diện 5 gương mặt")
st.write("Hệ thống nhận diện gương mặt sử dụng ONNX")

# Model paths
face_detection_model = os.path.join(BASE_DIR, 'model/face_detection_yunet_2023mar.onnx')
face_recognition_model = os.path.join(BASE_DIR, 'model/face_recognition_sface_2021dec.onnx')

# Load the models
@st.cache_resource
def load_models():
    try:
        # Try to load the 5 people model first, if fails, try the regular model
        try:
            svc = joblib.load(os.path.join(BASE_DIR, 'model/svc.pkl'))
            st.sidebar.success("SVC model for 5 people loaded successfully")
        except:
            svc = joblib.load(os.path.join(BASE_DIR, 'model/svc.pkl'))
            st.sidebar.success("Regular SVC model loaded successfully")
        
        # Define our expected people names for all 5 people
        people_names = ["duythien", "kimngan", "thienphuc", "thuhang", "trunghieu"]
        
        # Adjust people_names to match the number of classes in the model
        if hasattr(svc, 'classes_'):
            num_classes = len(svc.classes_)
            if num_classes < len(people_names):
                people_names = people_names[:num_classes]
        
        # Create the face detector
        detector = cv.FaceDetectorYN.create(
            face_detection_model,
            "",
            (320, 320),
            0.9,  # score threshold
            0.3,  # nms threshold
            5000  # top k
        )
        
        # Create the face recognizer
        recognizer = cv.FaceRecognizerSF.create(
            face_recognition_model,
            ""
        )
        
        # Create ONNX sessions as fallback
        detector_session = ort.InferenceSession(face_detection_model)
        recognizer_session = ort.InferenceSession(face_recognition_model)
        
        return svc, people_names, detector, recognizer, detector_session, recognizer_session
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None, None, None

# Load the models
svc, people_names, detector, recognizer, detector_session, recognizer_session = load_models()

# Fallback detection function using Haar cascade
def detect_faces_haar(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
    detected_faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    
    faces = []
    for (x, y, w, h) in detected_faces:
        # Add basic face info with placeholder confidence
        face_info = [x, y, w, h, 0.9]  # No landmarks, just basic detection
        faces.append(face_info)
    
    if faces:
        return [len(faces), np.array(faces)]
    else:
        return [0, None]

# Enhanced feature extraction
def extract_features(face_img):
    if face_img is None or face_img.size == 0:
        return None
    
    try:
        # Resize to the model's expected input size
        resized = cv.resize(face_img, (112, 112))
        
        # Convert BGR to RGB as the model expects RGB input
        resized = cv.cvtColor(resized, cv.COLOR_BGR2RGB)
        
        # Normalize pixel values to [0,1]
        resized = resized / 255.0
        
        # Apply histogram equalization for better contrast
        for i in range(3):
            # Convert to uint8 for histogram equalization
            temp_channel = (resized[:,:,i] * 255).astype(np.uint8)
            # Apply CLAHE for better contrast
            clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(temp_channel)
            # Convert back to float [0,1]
            resized[:,:,i] = enhanced / 255.0
        
        # Convert to expected layout and add batch dimension
        input_data = np.transpose(resized, (2, 0, 1))  # HWC to CHW
        input_data = np.expand_dims(input_data, axis=0).astype(np.float32)
        
        input_name = recognizer_session.get_inputs()[0].name
        output_name = recognizer_session.get_outputs()[0].name
        
        features = recognizer_session.run([output_name], {input_name: input_data})
        
        # Normalize the feature vector for better matching
        feature_vec = features[0][0]
        norm = np.linalg.norm(feature_vec)
        if norm > 0:
            feature_vec = feature_vec / norm
        
        return features[0]
    except Exception as e:
        st.error(f"Error in feature extraction: {e}")
        return None

# Better face alignment
def align_face(img, face):
    if face is None or len(face) < 4:
        return None
    
    x, y, w, h = face[0:4].astype(int)
    
    # Add a margin around the face for better recognition
    margin_x = int(w * 0.2)
    margin_y = int(h * 0.2)
    
    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(img.shape[1], x + w + margin_x)
    y2 = min(img.shape[0], y + h + margin_y)
    
    face_region = img[y1:y2, x1:x2]
    
    # Apply histogram equalization for better contrast
    if len(img.shape) == 3:  # Color image
        gray_face = cv.cvtColor(face_region, cv.COLOR_BGR2GRAY)
        equalized_face = cv.equalizeHist(gray_face)
        equalized_face_bgr = cv.cvtColor(equalized_face, cv.COLOR_GRAY2BGR)
        
        # Blend with original for natural look but improved contrast
        alpha = 0.7
        enhanced_face = cv.addWeighted(face_region, alpha, equalized_face_bgr, 1-alpha, 0)
        return enhanced_face
    else:
        return face_region

# Face recognition function
def recognize_face(frame):
    # Resize frame for display
    frame_h, frame_w = frame.shape[:2]
    
    # Set detector input size
    detector.setInputSize((frame_w, frame_h))
    
    # Detect faces
    try:
        faces = detector.detect(frame)
    except:
        # Fallback to Haar cascade if YuNet fails
        faces = detect_faces_haar(frame)
    
    # Create a copy of the frame for drawing
    result_frame = frame.copy()
    
    # Process faces if any are detected
    if faces[1] is not None:
        for face in faces[1]:
            # Try using the recognizer's alignCrop first
            try:
                face_align = recognizer.alignCrop(frame, face)
            except:
                # Fallback to our custom alignment function
                face_align = align_face(frame, face)
            
            if face_align is not None:
                # Try using the recognizer's feature extraction
                try:
                    face_feature = recognizer.feature(face_align)
                except:
                    # Fallback to our custom feature extraction
                    face_feature = extract_features(face_align)
                
                if face_feature is not None:
                    try:
                        # Predict identity
                        if hasattr(svc, 'predict_proba'):
                            # SVC with probability
                            probabilities = svc.predict_proba(face_feature)[0]
                            class_idx = np.argmax(probabilities)
                            confidence = probabilities[class_idx]
                            result = people_names[class_idx]
                        else:
                            # Linear SVC
                            test_predict = svc.predict(face_feature)
                            decision_values = svc.decision_function(face_feature)
                            result = people_names[test_predict[0]]
                            
                            # Convert decision values to confidence score
                            if decision_values.ndim > 1:
                                confidence = 1 / (1 + np.exp(-np.max(decision_values)))
                            else:
                                confidence = 1 / (1 + np.exp(-np.abs(decision_values[0])))
                        
                        # Apply confidence boost
                        confidence = min(1.0, confidence * 1.3)
                        
                        # Label as unknown if confidence is too low
                        if confidence < 0.3:
                            result = "Unknown"
                        
                        # Draw bounding box and label
                        coords = face[:-1].astype(np.int32)
                        cv.rectangle(result_frame, (coords[0], coords[1]), 
                                    (coords[0]+coords[2], coords[1]+coords[3]), 
                                    (0, 255, 0), 2)
                        
                        # Draw label with confidence score
                        label = f"{result} ({confidence:.2f})"
                        cv.putText(result_frame, label, 
                                (coords[0], coords[1] - 10), 
                                cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # Draw facial landmarks if available (for YuNet detection)
                        if len(coords) > 4:
                            for i in range(4, len(coords), 2):
                                if i+1 < len(coords):
                                    cv.circle(result_frame, (coords[i], coords[i+1]), 2, 
                                            (255, 0, 0), 2)
                    except Exception as e:
                        st.error(f"Error in recognition: {e}")
    
    return result_frame

# Sidebar options
st.sidebar.title("Options")
input_option = st.sidebar.radio("Select input source:", ["Webcam", "Upload Image"])

if input_option == "Webcam":
    st.header("Live Face Recognition")
    
    # Create a placeholder for the webcam feed
    video_placeholder = st.empty()
    
    # Start/stop button
    start_button = st.button("Start/Stop Camera")
    
    # Session state to track if the webcam is active
    if 'webcam_active' not in st.session_state:
        st.session_state.webcam_active = False
    
    # Toggle webcam state
    if start_button:
        st.session_state.webcam_active = not st.session_state.webcam_active
    
    if st.session_state.webcam_active:
        # Open webcam
        cap = cv.VideoCapture(0)
        
        # Check if webcam is opened correctly
        if not cap.isOpened():
            st.error("Failed to open webcam")
        else:
            while st.session_state.webcam_active:
                # Read a frame from the webcam
                ret, frame = cap.read()
                
                if not ret:
                    st.error("Failed to read frame from webcam")
                    break
                
                # Process the frame for face recognition
                result_frame = recognize_face(frame)
                
                # Convert BGR to RGB for display
                result_frame = cv.cvtColor(result_frame, cv.COLOR_BGR2RGB)
                
                # Display the frame in the Streamlit app
                video_placeholder.image(result_frame, channels="RGB", use_column_width=True)
                
                # Check if we should stop the webcam
                if not st.session_state.webcam_active:
                    break
                
                # Short pause to reduce CPU usage
                time.sleep(0.01)
            
            # Release the webcam
            cap.release()
    else:
        # Display a message when the webcam is not active
        video_placeholder.info("Click 'Start/Stop Camera' to activate the webcam")
        
else:  # Upload Image
    st.header("Upload an Image for Face Recognition")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png", "bmp"])
    
    if uploaded_file is not None:
        # Convert the uploaded file to an OpenCV image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv.imdecode(file_bytes, cv.IMREAD_COLOR)
        
        # Process the image for face recognition
        result_image = recognize_face(image)
        
        # Convert BGR to RGB for display
        result_image = cv.cvtColor(result_image, cv.COLOR_BGR2RGB)
        
        # Display the original and processed images side by side
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(cv.cvtColor(image, cv.COLOR_BGR2RGB), use_column_width=True)
            
        with col2:
            st.subheader("Detected Faces")
            st.image(result_image, use_column_width=True)

# Footer
st.markdown("---")
st.caption("Face Recognition System using ONNX Runtime and OpenCV")