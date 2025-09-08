import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any
import json

class HandAnalysisUtils:
    """Utility class for detailed hand analysis and data extraction"""
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        
    def analyze_single_frame(self, image_path: str, save_analysis: bool = True):
        """Analyze a single image file and extract all possible information"""
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load image: {image_path}")
            return None
        
        # Process with MediaPipe
        with self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.5) as hands:
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)
            
            analysis = self._extract_all_data(results, image.shape)
            
            if save_analysis:
                self._save_analysis_to_file(analysis, image_path)
                self._create_visualization(image, results, image_path)
            
            return analysis
    
    def _extract_all_data(self, results, image_shape) -> Dict[str, Any]:
        """Extract all available data from MediaPipe results"""
        
        analysis = {
            'image_dimensions': {
                'height': image_shape[0],
                'width': image_shape[1],
                'channels': image_shape[2]
            },
            'detection_results': {
                'hands_detected': 0,
                'hands_data': []
            }
        }
        
        if not results.multi_hand_landmarks:
            return analysis
        
        analysis['detection_results']['hands_detected'] = len(results.multi_hand_landmarks)
        
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            hand_data = {
                'hand_index': hand_idx,
                'landmarks': [],
                'landmark_statistics': {},
                'geometric_features': {},
                'handedness': None
            }
            
            # Extract landmark data
            for idx, landmark in enumerate(hand_landmarks.landmark):
                landmark_info = {
                    'id': idx,
                    'name': self._get_landmark_name(idx),
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'visibility': landmark.visibility,
                    'pixel_x': int(landmark.x * image_shape[1]),
                    'pixel_y': int(landmark.y * image_shape[0])
                }
                hand_data['landmarks'].append(landmark_info)
            
            # Extract handedness information
            if results.multi_handedness and hand_idx < len(results.multi_handedness):
                handedness = results.multi_handedness[hand_idx]
                hand_data['handedness'] = {
                    'label': handedness.classification[0].label,
                    'score': handedness.classification[0].score
                }
            
            # Calculate statistics
            hand_data['landmark_statistics'] = self._calculate_landmark_statistics(hand_data['landmarks'])
            
            # Calculate geometric features
            hand_data['geometric_features'] = self._calculate_geometric_features(hand_data['landmarks'])
            
            analysis['detection_results']['hands_data'].append(hand_data)
        
        return analysis
    
    def _get_landmark_name(self, landmark_id: int) -> str:
        """Get landmark name by ID"""
        landmark_names = [
            "WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
            "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
            "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
            "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
            "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"
        ]
        return landmark_names[landmark_id] if landmark_id < len(landmark_names) else f"UNKNOWN_{landmark_id}"
    
    def _calculate_landmark_statistics(self, landmarks: List[Dict]) -> Dict[str, Any]:
        """Calculate statistical information about landmarks"""
        
        x_coords = [lm['x'] for lm in landmarks]
        y_coords = [lm['y'] for lm in landmarks]
        z_coords = [lm['z'] for lm in landmarks]
        visibilities = [lm['visibility'] for lm in landmarks]
        
        return {
            'x_range': {'min': min(x_coords), 'max': max(x_coords), 'mean': np.mean(x_coords)},
            'y_range': {'min': min(y_coords), 'max': max(y_coords), 'mean': np.mean(y_coords)},
            'z_range': {'min': min(z_coords), 'max': max(z_coords), 'mean': np.mean(z_coords)},
            'visibility_stats': {
                'min': min(visibilities), 
                'max': max(visibilities), 
                'mean': np.mean(visibilities),
                'low_visibility_count': sum(1 for v in visibilities if v < 0.5)
            },
            'bounding_box': {
                'width': max(x_coords) - min(x_coords),
                'height': max(y_coords) - min(y_coords),
                'area': (max(x_coords) - min(x_coords)) * (max(y_coords) - min(y_coords))
            }
        }
    
    def _calculate_geometric_features(self, landmarks: List[Dict]) -> Dict[str, Any]:
        """Calculate geometric features of the hand"""
        
        # Get key landmarks
        wrist = landmarks[0]
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
        
        # Calculate distances
        features = {
            'finger_tip_distances_from_wrist': {
                'thumb': self._euclidean_distance(wrist, thumb_tip),
                'index': self._euclidean_distance(wrist, index_tip),
                'middle': self._euclidean_distance(wrist, middle_tip),
                'ring': self._euclidean_distance(wrist, ring_tip),
                'pinky': self._euclidean_distance(wrist, pinky_tip)
            },
            'inter_finger_distances': {
                'thumb_index': self._euclidean_distance(thumb_tip, index_tip),
                'index_middle': self._euclidean_distance(index_tip, middle_tip),
                'middle_ring': self._euclidean_distance(middle_tip, ring_tip),
                'ring_pinky': self._euclidean_distance(ring_tip, pinky_tip)
            }
        }
        
        # Calculate finger angles (simplified)
        features['finger_angles'] = self._calculate_finger_angles(landmarks)
        
        # Hand span (thumb to pinky)
        features['hand_span'] = self._euclidean_distance(thumb_tip, pinky_tip)
        
        return features
    
    def _euclidean_distance(self, point1: Dict, point2: Dict) -> float:
        """Calculate 3D Euclidean distance between two landmarks"""
        return np.sqrt(
            (point1['x'] - point2['x'])**2 + 
            (point1['y'] - point2['y'])**2 + 
            (point1['z'] - point2['z'])**2
        )
    
    def _calculate_finger_angles(self, landmarks: List[Dict]) -> Dict[str, float]:
        """Calculate simplified finger angles"""
        angles = {}
        
        # Define finger landmark groups (MCP, PIP, DIP, TIP)
        fingers = {
            'thumb': [2, 3, 4],  # CMC, MCP, IP, TIP (simplified)
            'index': [5, 6, 7, 8],
            'middle': [9, 10, 11, 12],
            'ring': [13, 14, 15, 16],
            'pinky': [17, 18, 19, 20]
        }
        
        wrist = landmarks[0]
        
        for finger_name, finger_landmarks in fingers.items():
            if len(finger_landmarks) >= 3:
                # Calculate angle at middle joint (simplified)
                p1 = landmarks[finger_landmarks[0]]
                p2 = landmarks[finger_landmarks[1]]
                p3 = landmarks[finger_landmarks[-1]]  # tip
                
                # Vector from wrist to tip
                tip_vector = np.array([p3['x'] - wrist['x'], p3['y'] - wrist['y']])
                # Calculate angle with respect to horizontal
                angle = np.arctan2(tip_vector[1], tip_vector[0]) * 180 / np.pi
                angles[finger_name] = angle
        
        return angles
    
    def _save_analysis_to_file(self, analysis: Dict, original_image_path: str):
        """Save analysis results to JSON file"""
        output_path = original_image_path.replace('.', '_analysis.')
        if output_path.endswith('_analysis.jpg') or output_path.endswith('_analysis.png'):
            output_path = output_path.rsplit('.', 1)[0] + '.json'
        
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        print(f"Analysis saved to: {output_path}")
    
    def _create_visualization(self, image, results, original_image_path: str):
        """Create comprehensive visualization"""
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Original image with landmarks
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        annotated_image = image_rgb.copy()
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    annotated_image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        
        axes[0, 0].imshow(annotated_image)
        axes[0, 0].set_title('Original Image with Landmarks')
        axes[0, 0].axis('off')
        
        # Landmark coordinate plot
        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [1 - lm.y for lm in hand_landmarks.landmark]  # Flip Y for proper orientation
                
                axes[0, 1].scatter(x_coords, y_coords, alpha=0.7, s=50, label=f'Hand {hand_idx + 1}')
                
                # Add landmark numbers
                for idx, (x, y) in enumerate(zip(x_coords, y_coords)):
                    axes[0, 1].annotate(str(idx), (x, y), xytext=(5, 5), 
                                      textcoords='offset points', fontsize=8)
        
        axes[0, 1].set_title('Landmark Coordinates (Normalized)')
        axes[0, 1].set_xlabel('X Coordinate')
        axes[0, 1].set_ylabel('Y Coordinate (Flipped)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Z-coordinate depth plot
        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                z_coords = [lm.z for lm in hand_landmarks.landmark]
                landmark_ids = list(range(len(z_coords)))
                
                axes[1, 0].plot(landmark_ids, z_coords, 'o-', alpha=0.7, label=f'Hand {hand_idx + 1}')
        
        axes[1, 0].set_title('Z-Coordinates (Relative Depth)')
        axes[1, 0].set_xlabel('Landmark ID')
        axes[1, 0].set_ylabel('Z Coordinate')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Visibility scores
        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                visibility_scores = [lm.visibility for lm in hand_landmarks.landmark]
                landmark_ids = list(range(len(visibility_scores)))
                
                axes[1, 1].bar([x + hand_idx * 0.4 for x in landmark_ids], visibility_scores, 
                              width=0.4, alpha=0.7, label=f'Hand {hand_idx + 1}')
        
        axes[1, 1].set_title('Landmark Visibility Scores')
        axes[1, 1].set_xlabel('Landmark ID')
        axes[1, 1].set_ylabel('Visibility Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save visualization
        output_path = original_image_path.replace('.', '_visualization.')
        if output_path.endswith('_visualization.jpg') or output_path.endswith('_visualization.png'):
            output_path = output_path.rsplit('.', 1)[0] + '_plot.png'
        
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {output_path}")
        plt.show()
    
    def compare_parameters(self, image_path: str, parameter_sets: List[Dict]):
        """Compare different parameter configurations on the same image"""
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load image: {image_path}")
            return
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results_list = []