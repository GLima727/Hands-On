import cv2
import numpy as np
import time
import os
import yaml

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from concurrent.futures import ThreadPoolExecutor

from dataclasses import dataclass
from typing import Optional, List, Dict, Any

class MediaPipeHandsExplorer:
    def __init__(self, config_path="config.yaml"):
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize MediaPipe
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands
        
        # Initialize from config
        self.gesture_interval = self.config['gesture']['interval']
        self.current_gesture = "unknown"
        
        # Initialize gesture recognizer
        self._init_gesture_recognizer()
        
        # Initialize hands detector
        self.hands = self.mp_hands.Hands(**self.config['mediapipe'])
        
        # Debug modes
        self.debug_modes = self.config['debug']

    def _load_config(self, config_path):
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
                print(f"Configuration loaded from {config_path}")
                return config
        except FileNotFoundError:
            print(f"Config file {config_path} not found. Using default settings.")
            return self._default_config()
        except yaml.YAMLError as e:
            print(f"Error parsing config file: {e}. Using default settings.")
            return self._default_config()
        
    def _init_gesture_recognizer(self):
        
        model_path = os.path.abspath(self.config['gesture']['model_path'])
        
        if not os.path.exists(model_path):
            print(f"Warning: Gesture model not found at {model_path}")
            self.gesture_recognizer = None
            return
        
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.GestureRecognizerOptions(base_options=base_options)
        self.gesture_recognizer = vision.GestureRecognizer.create_from_options(options)
        print("Gesture recognizer initialized successfully")


    def process_frame(self, image):
        """Process a single frame for hand landmarks and gesture recognition"""
        # Flip the image for a later selfie-view display
        #image = cv2.flip(image, 1)

        # Process the image
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image)
        
        # Convert back to BGR for display
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return results, image
    
    def draw_enhanced_landmarks(self, image, results):
        """Draw landmarks with additional debug information"""
        if not results.multi_hand_landmarks:
            return image
        
        height, width, _ = image.shape
        
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Draw standard landmarks and connections
            if self.debug_modes['show_landmarks']:
                self.mp_drawing.draw_landmarks(
                    image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style())
            
            # Draw bounding box
            if self.debug_modes['show_bounding_box']:
                self._draw_bounding_box(image, hand_landmarks, width, height)
        
        return image
    
    def _draw_bounding_box(self, image, landmarks, width, height):
        """Draw bounding box around the hand"""
        x_coords = [lm.x * width for lm in landmarks.landmark]
        y_coords = [lm.y * height for lm in landmarks.landmark]
        
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))
        
        cv2.rectangle(image, (x_min - 10, y_min - 10), (x_max + 10, y_max + 10), (0, 255, 0), 2)
   
    def recognize_gesture(self, frame_bgr) -> str:
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Wrap into MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Run gesture recognition
        recognition_result = self.gesture_recognizer.recognize(mp_image)
        if not recognition_result.gestures:
            return None

        # Top gesture
        top_gesture = recognition_result.gestures[0][0]
        print(top_gesture)
        if self.debug_modes['print_gesture_recognition']:
            print(f"Gesture recognized: {top_gesture.category_name} ({top_gesture.score:.2f})")

        return top_gesture.category_name

    def should_process_gesture(self, results) -> bool:
        """
        Smart decision on whether to process gesture this frame based on MediaPipe results
        """        
        has_hands = bool(results.multi_hand_landmarks)
        
        return has_hands


    def run(self):
        """Main execution loop"""    
        cap = cv2.VideoCapture(0)


        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            results, image = self.process_frame(image)

            image = self.draw_enhanced_landmarks(image, results)
            
            #if self.should_process_gesture(results):

            self.current_gesture = self.recognize_gesture(image)            
                
            # Display the image
            cv2.imshow('MediaPipe Hands Explorer', image)
            
            # Check for ESC key press (key code 27)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                break
        
        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()