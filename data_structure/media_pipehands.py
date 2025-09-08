import cv2
import mediapipe as mp
import numpy as np
import time
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

class MediaPipeHandsExplorer:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands
        
        # Default parameters - you can experiment with these
        self.params = {
            'static_image_mode': False,
            'max_num_hands': 2,
            'model_complexity': 1,
            'min_detection_confidence': 0.5,
            'min_tracking_confidence': 0.5
        }
        
        # Initialize hands detector
        self.hands = self.mp_hands.Hands(**self.params)
        
        # Debug modes
        self.debug_modes = {
            'show_landmarks': True,
            'show_bounding_box': True,
            }
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
    def print_all_parameters(self):
        """Print all current parameters and their meanings"""
        print("\n=== CURRENT MEDIAPIPE HANDS PARAMETERS ===")
        for key, value in self.params.items():
            print(f"{key}: {value}")
            self._explain_parameter(key)
        print("=" * 50)
    
    def toggle_debug_mode(self, mode_name):
        """Toggle a debug visualization mode"""
        if mode_name in self.debug_modes:
            self.debug_modes[mode_name] = not self.debug_modes[mode_name]
            print(f"Toggled {mode_name}: {self.debug_modes[mode_name]}")
        else:
            print(f"Debug mode {mode_name} not found!")
    
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
    
    def calculate_fps(self):
        """Calculate and update FPS"""
        self.fps_counter += 1
        if time.time() - self.fps_start_time > 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_start_time = time.time()
    
    def draw_debug_info(self, image):
        """Draw debug information on the image"""
        y_offset = 30
        
        # FPS
        if self.debug_modes['show_fps']:
            cv2.putText(image, f"FPS: {self.current_fps}", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 30
        
        # Current parameters
        param_text = f"Model: {self.params['model_complexity']} | Max Hands: {self.params['max_num_hands']}"
        cv2.putText(image, param_text, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 25
        
        confidence_text = f"Det: {self.params['min_detection_confidence']:.2f} | Track: {self.params['min_tracking_confidence']:.2f}"
        cv2.putText(image, confidence_text, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def run(self):
        """Main execution loop"""    
        cap = cv2.VideoCapture(0)
        
        print("MediaPipe Hands Explorer started!")
        print("Press ESC to exit...")
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Flip the image for a later selfie-view display
            image = cv2.flip(image, 1)

            # Process the image
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(image)
            
            # Convert back to BGR for display
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Draw enhanced landmarks
            image = self.draw_enhanced_landmarks(image, results)
            
            # Display the image
            cv2.imshow('MediaPipe Hands Explorer', image)
            
            # Check for ESC key press (key code 27)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                break
        
        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()