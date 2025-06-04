import numpy as np
from collections import defaultdict, deque
import cv2

class TrackingService:
    def __init__(self):
        """Initialize tracking service with optimized thresholds for earlier counting"""
        self.tracks = {}
        self.next_track_id = 1
        self.track_history = defaultdict(list)
        self.lost_tracks = {}
        self.max_lost_frames = 15  # Increased for better tracking
        self.min_track_length = 3   # Reduced from 5 to 3 for earlier counting
        self.unique_objects_seen = defaultdict(set)
        self.object_type_counts = defaultdict(int)
        
        # ROI-specific tracking with adjusted thresholds
        self.roi_track_history = defaultdict(int)  # Track how many frames each track was in ROI
        self.roi_qualified_tracks = defaultdict(set)  # Tracks that spent enough time in ROI
        self.min_roi_frames = 2  # Reduced from 3 to 2 for earlier ROI qualification
        
        # Progressive counting system for smoother results
        self.progressive_unique_count = defaultdict(int)  # Track counts that build up over time
        
    def update_tracks(self, detections, frame, roi_enabled=False):
        """Update tracks with fixed ROI logic"""
        current_tracks = {}
        
        # If we have existing tracks, try to match them
        if self.tracks:
            matched_tracks, unmatched_detections = self._match_detections_to_tracks(detections)
            
            # Update matched tracks
            for track_id, detection in matched_tracks.items():
                detection['track_id'] = track_id
                detection['track_length'] = len(self.track_history[track_id]) + 1
                current_tracks[track_id] = detection
                
                # Store comprehensive track data
                track_data = {
                    'center': detection['center'],
                    'class_name': detection['class_name'],
                    'object_type': detection.get('object_type', 'other'),
                    'in_roi': detection.get('in_roi', True),
                    'confidence': detection.get('confidence', 0.0),
                    'bbox': detection['bbox']
                }
                self.track_history[track_id].append(track_data)
                
                # Update ROI tracking
                if roi_enabled and detection.get('in_roi', False):
                    self.roi_track_history[track_id] += 1
                    
                    # Mark as ROI-qualified if spent enough time in ROI
                    if self.roi_track_history[track_id] >= self.min_roi_frames:
                        object_type = detection.get('object_type', 'other')
                        self.roi_qualified_tracks[object_type].add(track_id)
                
                # Keep track history manageable
                if len(self.track_history[track_id]) > 50:
                    self.track_history[track_id] = self.track_history[track_id][-30:]
            
            # Create new tracks for unmatched detections
            for detection in unmatched_detections:
                track_id = self.next_track_id
                self.next_track_id += 1
                
                detection['track_id'] = track_id
                detection['track_length'] = 1
                current_tracks[track_id] = detection
                
                track_data = {
                    'center': detection['center'],
                    'class_name': detection['class_name'],
                    'object_type': detection.get('object_type', 'other'),
                    'in_roi': detection.get('in_roi', True),
                    'confidence': detection.get('confidence', 0.0),
                    'bbox': detection['bbox']
                }
                self.track_history[track_id] = [track_data]
                
                # Initialize ROI tracking for new tracks
                if roi_enabled and detection.get('in_roi', False):
                    self.roi_track_history[track_id] = 1
        else:
            # First frame - create tracks for all detections
            for detection in detections:
                track_id = self.next_track_id
                self.next_track_id += 1
                
                detection['track_id'] = track_id
                detection['track_length'] = 1
                current_tracks[track_id] = detection
                
                track_data = {
                    'center': detection['center'],
                    'class_name': detection['class_name'],
                    'object_type': detection.get('object_type', 'other'),
                    'in_roi': detection.get('in_roi', True),
                    'confidence': detection.get('confidence', 0.0),
                    'bbox': detection['bbox']
                }
                self.track_history[track_id] = [track_data]
                
                # Initialize ROI tracking
                if roi_enabled and detection.get('in_roi', False):
                    self.roi_track_history[track_id] = 1
        
        # Update lost tracks
        for track_id in self.tracks:
            if track_id not in current_tracks:
                if track_id not in self.lost_tracks:
                    self.lost_tracks[track_id] = 0
                self.lost_tracks[track_id] += 1
        
        # Remove tracks that have been lost too long
        tracks_to_remove = []
        for track_id, lost_frames in self.lost_tracks.items():
            if lost_frames > self.max_lost_frames:
                tracks_to_remove.append(track_id)
                # Add to unique objects if track was long enough
                if len(self.track_history[track_id]) >= self.min_track_length:
                    track_data = self.track_history[track_id]
                    if track_data:
                        object_type = track_data[-1].get('object_type', 'other')
                        class_name = track_data[-1].get('class_name', 'unknown')
                        
                        # For ROI tracking, only count if qualified
                        if roi_enabled:
                            if track_id in self.roi_qualified_tracks[object_type]:
                                self.unique_objects_seen[object_type].add(track_id)
                                self.object_type_counts[class_name] += 1
                        else:
                            # For non-ROI tracking, count normally
                            self.unique_objects_seen[object_type].add(track_id)
                            self.object_type_counts[class_name] += 1
        
        for track_id in tracks_to_remove:
            del self.lost_tracks[track_id]
            if track_id in self.track_history:
                del self.track_history[track_id]
            if track_id in self.roi_track_history:
                del self.roi_track_history[track_id]
        
        # Clean up lost tracks that are now active again
        for track_id in current_tracks:
            if track_id in self.lost_tracks:
                del self.lost_tracks[track_id]
        
        self.tracks = current_tracks
        return list(current_tracks.values())
    
    def _match_detections_to_tracks(self, detections):
        """Enhanced matching with better distance thresholds"""
        matched_tracks = {}
        unmatched_detections = []
        
        track_ids = list(self.tracks.keys())
        detection_indices = list(range(len(detections)))
        
        if not track_ids or not detection_indices:
            return matched_tracks, detections
        
        # Create distance matrix with improved object type consideration
        distances = np.zeros((len(track_ids), len(detections)))
        
        for i, track_id in enumerate(track_ids):
            track_center = self.tracks[track_id]['center']
            track_class = self.tracks[track_id].get('class_name', '')
            
            for j, detection in enumerate(detections):
                det_center = detection['center']
                det_class = detection.get('class_name', '')
                
                # Calculate spatial distance
                spatial_distance = np.sqrt((track_center['x'] - det_center['x'])**2 + 
                                         (track_center['y'] - det_center['y'])**2)
                
                # Enhanced class similarity bonus/penalty
                class_bonus = 0
                if track_class == det_class:
                    class_bonus = -30  # Strong bonus for exact class match
                elif self._same_object_type(track_class, det_class):
                    class_bonus = -15  # Medium bonus for same object type
                else:
                    class_bonus = 100  # Strong penalty for different types
                
                distances[i, j] = spatial_distance + class_bonus
        
        # Improved greedy matching with dynamic thresholds
        base_max_distance = 80
        used_detections = set()
        
        # Sort tracks by confidence/length for better matching priority
        track_priorities = [(i, track_ids[i]) for i in range(len(track_ids))]
        track_priorities.sort(key=lambda x: len(self.track_history[x[1]]), reverse=True)
        
        for i, track_id in track_priorities:
            best_detection_idx = None
            best_distance = float('inf')
            
            # Dynamic threshold based on track history
            track_length = len(self.track_history[track_id])
            max_distance = base_max_distance + (track_length * 5)  # More lenient for longer tracks
            
            for j in detection_indices:
                if j in used_detections:
                    continue
                    
                if distances[i, j] < best_distance and distances[i, j] < max_distance:
                    best_distance = distances[i, j]
                    best_detection_idx = j
            
            if best_detection_idx is not None:
                matched_tracks[track_id] = detections[best_detection_idx]
                used_detections.add(best_detection_idx)
        
        # Collect unmatched detections
        for j, detection in enumerate(detections):
            if j not in used_detections:
                unmatched_detections.append(detection)
        
        return matched_tracks, unmatched_detections
    
    def _same_object_type(self, class1, class2):
        """Enhanced object type matching"""
        type_groups = {
            'vehicle': ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'train'],
            'person': ['person'],
            'animal': ['cat', 'dog', 'horse', 'cow', 'sheep', 'bird', 'elephant', 'bear', 'zebra', 'giraffe'],
            'furniture': ['chair', 'couch', 'bed', 'dining table', 'bench'],
            'electronics': ['tv', 'laptop', 'cell phone', 'remote', 'keyboard', 'mouse'],
            'sports': ['sports ball', 'baseball bat', 'tennis racket', 'frisbee', 'skateboard', 'surfboard']
        }
        
        for group_name, objects in type_groups.items():
            if class1.lower() in objects and class2.lower() in objects:
                return True
        return False
    
    def format_tracks_for_display(self, tracks):
        """Format tracks for display"""
        return tracks
    
    def get_track_statistics(self, roi_enabled=False):
        """Get comprehensive tracking statistics with immediate and progressive counting"""
        # Count current long tracks by type
        current_object_counts = defaultdict(int)
        immediate_counts = defaultdict(int)  # Tracks with shorter requirements for immediate feedback
        
        for track_id, history in self.track_history.items():
            if track_id in self.tracks and history:
                object_type = history[-1].get('object_type', 'other')
                
                # Immediate counting (tracks with 2+ frames for faster feedback)
                if len(history) >= 2:
                    immediate_counts[object_type] += 1
                
                # Standard counting with full requirements
                if len(history) >= self.min_track_length:
                    # For ROI mode, only count if qualified
                    if roi_enabled:
                        if track_id in self.roi_qualified_tracks[object_type]:
                            current_object_counts[object_type] += 1
                    else:
                        current_object_counts[object_type] += 1
        
        # Calculate total unique objects by type (final counts)
        total_unique_by_type = {}
        for object_type, unique_tracks in self.unique_objects_seen.items():
            total_unique_by_type[object_type] = len(unique_tracks) + current_object_counts[object_type]
        
        # Add current active types not yet in unique_objects_seen
        for object_type, count in current_object_counts.items():
            if object_type not in total_unique_by_type:
                total_unique_by_type[object_type] = count
        
        # Calculate immediate unique counts (for early frames feedback)
        immediate_unique_by_type = {}
        for object_type, unique_tracks in self.unique_objects_seen.items():
            immediate_unique_by_type[object_type] = len(unique_tracks) + immediate_counts[object_type]
        
        # Add immediate active types
        for object_type, count in immediate_counts.items():
            if object_type not in immediate_unique_by_type:
                immediate_unique_by_type[object_type] = count
        
        # Calculate total unique objects
        total_unique = sum(total_unique_by_type.values())
        immediate_unique = sum(immediate_unique_by_type.values())
        
        return {
            'unique_objects_total': total_unique,
            'unique_people': total_unique_by_type.get('person', 0),
            'unique_by_type': total_unique_by_type,
            'active_tracks': len(self.tracks),
            'total_tracks': self.next_track_id - 1,
            'lost_tracks': len(self.lost_tracks),
            'object_type_counts': dict(self.object_type_counts),
            'roi_qualified_tracks': {k: len(v) for k, v in self.roi_qualified_tracks.items()} if roi_enabled else {},
            # New immediate feedback counts
            'immediate_unique_total': immediate_unique,
            'immediate_unique_people': immediate_unique_by_type.get('person', 0),
            'immediate_unique_by_type': immediate_unique_by_type,
            'tracking_explanation': self._get_tracking_explanation(roi_enabled)
        }
    
    def _get_tracking_explanation(self, roi_enabled):
        """Provide explanation of counting methodology"""
        if roi_enabled:
            return f"Objects need {self.min_roi_frames}+ frames in ROI and {self.min_track_length}+ total frames to count as unique"
        else:
            return f"Objects need {self.min_track_length}+ frames to count as unique"
    
    def get_unique_object_count(self, object_type=None, immediate=False):
        """Get unique object count for specific type or total with immediate option"""
        stats = self.get_track_statistics()
        
        if immediate:
            # Return immediate counts for early feedback
            if object_type:
                return stats['immediate_unique_by_type'].get(object_type, 0)
            return stats['immediate_unique_total']
        else:
            # Return final qualified counts
            if object_type:
                return stats['unique_by_type'].get(object_type, 0)
            return stats['unique_objects_total'] 