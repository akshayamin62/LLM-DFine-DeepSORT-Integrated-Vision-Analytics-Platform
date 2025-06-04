import os
import cv2
import numpy as np
import torch
from transformers import AutoModelForObjectDetection, AutoImageProcessor, pipeline
from PIL import Image
from config import Config
from datetime import datetime
from tracking_service import TrackingService

class DFineService:
    def __init__(self):
        # Initialize D-Fine model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 D-Fine initialized on {self.device}")
        
        # Use D-Fine medium model - exactly as in notebook
        self.checkpoint = "ustc-community/dfine-medium-obj365"
        
        self.model = None
        self.image_processor = None
        self.pipe = None
        
        try:
            # Load D-Fine model and processor - following notebook pattern
            self.image_processor = AutoImageProcessor.from_pretrained(self.checkpoint)
            self.model = AutoModelForObjectDetection.from_pretrained(self.checkpoint).to(self.device)
            print(f"✅ D-Fine model ready!")
        except Exception as e:
            try:
                self.pipe = pipeline("object-detection", model=self.checkpoint, device=self.device)
                print("✅ D-Fine pipeline ready!")
            except Exception as e2:
                print(f"❌ Failed to load D-Fine model: {e2}")
                raise e2
    
    def process_media(self, file_path, parameters):
        """Process image or video based on detection mode and parameters"""
        # Determine file type
        file_ext = os.path.splitext(file_path)[1].lower()
        is_video = file_ext in ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
        
        if is_video:
            return self._process_video(file_path, parameters)
        else:
            return self._process_image(file_path, parameters)
    
    def _process_image(self, image_path, parameters):
        """Process a single image with D-Fine detection"""
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        original_image = image.copy()
        height, width = image.shape[:2]
        
        # Convert BGR to RGB for D-Fine
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        
        # Run D-Fine detection
        detections = self._run_dfine_detection(pil_image, parameters)
        
        # Handle different detection modes
        count_only = parameters.get('count_only', False)
        roi_enabled = parameters.get('roi_detection', False)
        
        if count_only:
            # Count-only mode: create count display but also save detection frame
            count_result = self._generate_count_only_result(detections, parameters, original_image)
            
            # Also create and save the detection frame for viewing
            detection_frame = self._draw_detections_enhanced(original_image.copy(), detections, parameters)
            detection_filename = f"image_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            detection_path = os.path.join('outputs', detection_filename)
            cv2.imwrite(detection_path, detection_frame)
            
            # Add detection frame info to results
            count_result['detection_frame'] = detection_filename
            print(f"📁 Saved detection frame: {detection_filename}")
            
            return count_result
        else:
            # Standard detection mode: full visual output with enhanced drawing
            output_image = self._draw_detections_enhanced(original_image, detections, parameters)
            
            # Add summary overlay for images
            self._add_image_summary_overlay(output_image, detections, parameters)
            
            result = self._save_and_return_results(output_image, detections, parameters, image_path)
            print(f"📁 Saved image result: {result['filename']}")
            
            return result
    
    def _run_dfine_detection(self, pil_image, parameters):
        """Run D-Fine detection on PIL image with ROI support and any object type"""
        detections = []
        
        try:
            confidence_threshold = parameters.get('confidence_threshold', 0.3)
            target_classes = parameters.get('classes_to_detect', 'all')
            roi_detection = parameters.get('roi_detection', False)
            roi_type = parameters.get('roi_type', 'rectangle')
            
            # Get image dimensions for ROI calculations
            img_width, img_height = pil_image.size
            roi_mask = None
            
            # Create ROI mask if ROI detection is enabled
            if roi_detection:
                roi_mask = self._create_roi_mask(img_width, img_height, roi_type, parameters.get('roi_coordinates'))
            
            if self.model is not None and self.image_processor is not None:
                # Use transformers approach - exactly like notebook
                
                # Preprocess image
                inputs = self.image_processor(pil_image, return_tensors="pt")
                inputs = inputs.to(self.device)
                
                # Run inference
                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                # Post-process results - exactly like notebook
                target_sizes = [(pil_image.height, pil_image.width)]
                postprocessed_outputs = self.image_processor.post_process_object_detection(
                    outputs,
                    target_sizes=target_sizes,
                    threshold=confidence_threshold
                )
                
                image_detections = postprocessed_outputs[0]
                
                # Convert to our standard format with ROI filtering
                for i in range(len(image_detections['scores'])):
                    score = float(image_detections['scores'][i])
                    label_id = int(image_detections['labels'][i])
                    box = image_detections['boxes'][i].tolist()
                    
                    # Get class name from model config
                    class_name = self.model.config.id2label.get(label_id, f"class_{label_id}")
                    
                    # Filter by target classes (if not "all")
                    if target_classes != "all" and target_classes != ["all"]:
                        if isinstance(target_classes, list):
                            if class_name.lower() not in [cls.lower() for cls in target_classes]:
                                continue
                        else:
                            if class_name.lower() != target_classes.lower():
                                continue
                    
                    x1, y1, x2, y2 = box
                    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                    
                    # Check if detection is in ROI (if ROI is enabled)
                    in_roi = True
                    if roi_detection and roi_mask is not None:
                        in_roi = self._is_detection_in_roi(center_x, center_y, roi_mask)
                        
                        # Skip detections outside ROI if ROI filtering is strict
                        if not in_roi and parameters.get('roi_strict_filtering', True):
                            continue
                    
                    detection = {
                        'class_name': class_name,
                        'confidence': score,
                        'bbox': [x1, y1, x2, y2],
                        'center': {'x': center_x, 'y': center_y},
                        'size': {'width': x2 - x1, 'height': y2 - y1},
                        'in_roi': in_roi,
                        'object_type': self._classify_object_type(class_name)
                    }
                    detections.append(detection)
            
            elif self.pipe is not None:
                # Use pipeline approach as fallback
                results = self.pipe(pil_image, threshold=confidence_threshold)
                
                for result in results:
                    class_name = result['label']
                    
                    # Filter by target classes
                    if target_classes != "all" and target_classes != ["all"]:
                        if isinstance(target_classes, list):
                            if class_name.lower() not in [cls.lower() for cls in target_classes]:
                                continue
                        else:
                            if class_name.lower() != target_classes.lower():
                                continue
                    
                    box = result['box']
                    x1, y1, x2, y2 = box['xmin'], box['ymin'], box['xmax'], box['ymax']
                    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                    
                    # Check ROI
                    in_roi = True
                    if roi_detection and roi_mask is not None:
                        in_roi = self._is_detection_in_roi(center_x, center_y, roi_mask)
                        
                        if not in_roi and parameters.get('roi_strict_filtering', True):
                            continue
                    
                    detection = {
                        'class_name': class_name,
                        'confidence': result['score'],
                        'bbox': [x1, y1, x2, y2],
                        'center': {'x': center_x, 'y': center_y},
                        'size': {'width': x2 - x1, 'height': y2 - y1},
                        'in_roi': in_roi,
                        'object_type': self._classify_object_type(class_name)
                    }
                    detections.append(detection)
        
        except Exception as e:
            print(f"❌ Detection error: {e}")
            detections = []
        
        return detections
    
    def _process_video(self, video_path, parameters):
        """Process video with high FPS tracking for accurate DeepSORT performance"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"🎬 Processing video: {duration:.1f}s, {total_frames} frames, {fps:.1f} FPS")
        print(f"📏 Video dimensions: {width}x{height}")
        
        # Clean up old frames to keep outputs folder manageable
        self._cleanup_old_frames()
        
        # Initialize tracking service with adjusted thresholds for earlier counting
        tracker = TrackingService()
        
        # Process at 20 frames per second for high accuracy DeepSORT tracking
        target_processing_fps = 5
        frame_interval = max(1, int(fps // target_processing_fps)) if fps > 0 else 1
        
        # Calculate output FPS to match original video duration
        output_fps = target_processing_fps if fps >= target_processing_fps else fps
        
        frame_count = 0
        processed_frames = 0
        all_detections = []
        processed_frame_info = []
        processed_frame_paths = []  # Store paths for video creation
        
        # Check if ROI is enabled
        roi_enabled = parameters.get('roi_detection', False)
        create_output_video = parameters.get('create_output_video', True)  # New parameter
        
        print(f"📊 Analyzing at {target_processing_fps} FPS (every {frame_interval} frames) for accurate tracking")
        print(f"🎥 Output video will be created at {output_fps} FPS to match original duration")
        if roi_enabled:
            print(f"🎯 ROI enabled: {parameters.get('roi_type', 'rectangle')}")
        if create_output_video:
            print(f"🎥 Output video will preserve original duration ({duration:.1f}s)")
        
        # Setup video writer for output video
        output_video_path = None
        video_writer = None
        if create_output_video:
            output_video_filename = f"processed_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            output_video_path = os.path.join('outputs', output_video_filename)
            
            # Use XVID codec with calculated FPS to maintain original duration
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_writer = cv2.VideoWriter(output_video_path, fourcc, output_fps, (width, height))
            print(f"🎥 Setting up output video: {output_video_filename} at {output_fps} FPS")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every nth frame for high-frequency tracking
            if frame_count % frame_interval == 0:
                processed_frames += 1
                
                # Convert frame to PIL for D-Fine
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(rgb_frame)
                
                # Run D-Fine detection
                frame_detections = self._run_dfine_detection(pil_frame, parameters)
                
                # Update tracker with detections and ROI information
                tracks = tracker.update_tracks(frame_detections, frame, roi_enabled=roi_enabled)
                tracked_detections = tracker.format_tracks_for_display(tracks)
                
                all_detections.extend(tracked_detections)
                
                # Draw detections on frame with improved colors
                frame_with_detections = self._draw_detections_enhanced(frame.copy(), tracked_detections, parameters)
                
                # Add frame information overlay with updated FPS info
                frame_with_info = self._add_frame_info_overlay_enhanced(frame_with_detections, processed_frames, 
                                                             len(tracked_detections), tracker.get_track_statistics(roi_enabled),
                                                             target_processing_fps, output_fps)
                
                # Save frame to outputs folder
                frame_filename = f"frame_{processed_frames:04d}_{datetime.now().strftime('%H%M%S')}.jpg"
                frame_path = os.path.join('outputs', frame_filename)
                cv2.imwrite(frame_path, frame_with_info)
                processed_frame_paths.append(frame_path)
                
                # Write frame to output video
                if video_writer is not None:
                    video_writer.write(frame_with_info)
                
                # Store frame information
                stats = tracker.get_track_statistics(roi_enabled)
                frame_info = {
                    'frame_number': processed_frames,
                    'filename': frame_filename,
                    'detections_count': len(tracked_detections),
                    'unique_people_so_far': stats['unique_people'],
                    'active_tracks': stats['active_tracks'],
                    'timestamp': frame_count / fps if fps > 0 else frame_count,
                    'detections': tracked_detections[:5],
                    'original_frame_index': frame_count,
                    'video_fps': fps,
                    'processing_fps': target_processing_fps,
                    'output_fps': output_fps,
                    'roi_stats': stats.get('roi_qualified_tracks', {}) if roi_enabled else {}
                }
                processed_frame_info.append(frame_info)
                
                # Only print frame number (clean output)
                print(f"Frame {processed_frames}")
                
                # Show progress every 50 processed frames (more frequent due to higher FPS)
                if processed_frames % 50 == 0:
                    if roi_enabled:
                        print(f"  ✅ Progress: {processed_frames} frames, {stats['unique_people']} unique people in ROI")
                    else:
                        print(f"  ✅ Progress: {processed_frames} frames, {stats['unique_people']} unique people")
            
            frame_count += 1
        
        cap.release()
        
        # Finalize output video
        if video_writer is not None:
            video_writer.release()
            
            # Verify output video duration
            if os.path.exists(output_video_path):
                verify_cap = cv2.VideoCapture(output_video_path)
                if verify_cap.isOpened():
                    output_frame_count = int(verify_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    output_fps_actual = verify_cap.get(cv2.CAP_PROP_FPS)
                    output_duration = output_frame_count / output_fps_actual if output_fps_actual > 0 else 0
                    verify_cap.release()
                    
                    print(f"🎥 Output video created: {output_video_filename}")
                    print(f"📏 Original duration: {duration:.1f}s | Output duration: {output_duration:.1f}s")
                    print(f"🎯 Duration match: {'✅ Perfect' if abs(duration - output_duration) < 0.5 else '⚠️ Slight difference'}")
                else:
                    print(f"🎥 Output video created: {output_video_filename}")
            else:
                print(f"❌ Failed to create output video")
        
        # Get final tracking statistics
        track_stats = tracker.get_track_statistics(roi_enabled)
        unique_person_count = track_stats['unique_people']
        
        print(f"🎯 Final Results: {unique_person_count} unique people detected")
        if roi_enabled:
            print(f"🎯 ROI Qualified Tracks: {track_stats.get('roi_qualified_tracks', {})}")
        print(f"📁 Saved {len(processed_frame_info)} frames to outputs folder")
        print(f"⚡ Processing FPS: {target_processing_fps} | Output FPS: {output_fps}")
        
        # Create result summary
        result_image = self._create_video_summary_enhanced(unique_person_count, processed_frames, duration, track_stats, target_processing_fps)
        
        results = {
            'total_objects': len(all_detections),
            'unique_people_count': unique_person_count,
            'detections': all_detections[-20:] if len(all_detections) > 20 else all_detections,
            'filename': result_image,
            'processed_frames': processed_frame_info,
            'video_analysis': {
                'duration_seconds': duration,
                'frames_processed': processed_frames,
                'unique_people_detected': unique_person_count,
                'total_detections': len(all_detections),
                'tracking_stats': track_stats,
                'fps': fps,
                'processing_fps': target_processing_fps,
                'output_fps': output_fps,
                'roi_enabled': roi_enabled,
                'roi_type': parameters.get('roi_type', '') if roi_enabled else '',
                'output_video': output_video_filename if create_output_video else None,
                'output_video_path': output_video_path if create_output_video else None
            }
        }
        
        return self._convert_numpy_types(results)
    
    def _generate_count_only_result(self, detections, parameters, original_image):
        """Generate count-only result without visual detection boxes"""
        # Create simple count display image
        result_image = np.ones((400, 600, 3), dtype=np.uint8) * 255
        
        # Count information
        total_count = len(detections)
        
        # Draw count information
        cv2.putText(result_image, "D-FINE DETECTION RESULTS", (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(result_image, f"Total Objects Found: {total_count}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 0), 2)
        
        # Class breakdown
        class_counts = {}
        for det in detections:
            class_name = det['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        y_pos = 220
        cv2.putText(result_image, "Breakdown:", (50, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        y_pos += 30
        
        for class_name, count in class_counts.items():
            cv2.putText(result_image, f"  {class_name}: {count}", (70, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 2)
            y_pos += 25
        
        # Add D-Fine branding
        cv2.putText(result_image, "Powered by D-Fine AI", (50, y_pos + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 2)
        
        # Save count display
        output_filename = f"dfine_count_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        output_path = os.path.join('outputs', output_filename)
        cv2.imwrite(output_path, result_image)
        
        results = {
            'total_objects': total_count,
            'detections': detections,
            'filename': output_filename
        }
        
        return self._convert_numpy_types(results)
    
    def _create_video_summary_enhanced(self, unique_count, processed_frames, duration, track_stats, processing_fps):
        """Create an enhanced visual summary for video results with FPS information"""
        result_image = np.ones((450, 700, 3), dtype=np.uint8) * 255
        
        # Title
        cv2.putText(result_image, "D-FINE HIGH-FPS VIDEO ANALYSIS", (120, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Results
        cv2.putText(result_image, f"Duration: {duration:.1f}s", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 100), 2)
        cv2.putText(result_image, f"Frames Processed: {processed_frames}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 100), 2)
        cv2.putText(result_image, f"Processing FPS: {processing_fps}", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)
        cv2.putText(result_image, f"Unique People: {unique_count}", (50, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 150, 0), 2)
        
        # Tracking statistics
        cv2.putText(result_image, "Enhanced Tracking Statistics:", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(result_image, f"Active Tracks: {track_stats['active_tracks']}", (50, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 100), 2)
        cv2.putText(result_image, f"Total Tracks Created: {track_stats['total_tracks']}", (50, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 100), 2)
        
        # Performance information
        cv2.putText(result_image, "Performance Benefits:", (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(result_image, f"• High-frequency tracking at {processing_fps} FPS", (70, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 0), 1)
        cv2.putText(result_image, "• Improved DeepSORT accuracy", (70, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 0), 1)
        cv2.putText(result_image, "• Original video duration preserved", (70, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 0), 1)
        
        # Save
        output_filename = f"dfine_video_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        output_path = os.path.join('outputs', output_filename)
        cv2.imwrite(output_path, result_image)
        
        return output_filename
    
    def _save_and_return_results(self, output_image, detections, parameters, original_path):
        """Save output image and return results"""
        output_filename = f"dfine_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        output_path = os.path.join('outputs', output_filename)
        cv2.imwrite(output_path, output_image)
        
        results = {
            'total_objects': len(detections),
            'detections': detections,
            'filename': output_filename
        }
        
        return self._convert_numpy_types(results)
    
    def _draw_detections_enhanced(self, image, detections, parameters):
        """Enhanced drawing with better colors and track ID visibility"""
        output_image = image.copy()
        
        line_thickness = parameters.get('line_thickness', 3)
        show_labels = parameters.get('show_labels', True)
        show_confidence = parameters.get('show_confidence', True)
        roi_detection = parameters.get('roi_detection', False)
        roi_type = parameters.get('roi_type', 'rectangle')
        
        # Draw ROI overlay if enabled
        if roi_detection:
            self._draw_roi_overlay(output_image, roi_type, parameters.get('roi_coordinates'))
        
        # Enhanced color scheme with bright, contrasting colors
        type_colors = {
            'person': (0, 255, 0),      # Bright Green for people
            'vehicle': (0, 0, 255),     # Bright Red for vehicles
            'animal': (255, 255, 0),    # Bright Yellow for animals
            'furniture': (255, 0, 255), # Bright Magenta for furniture
            'electronics': (0, 165, 255), # Bright Orange for electronics
            'food': (255, 20, 147),     # Deep Pink for food
            'sports': (30, 144, 255),   # Dodger Blue for sports
            'household': (50, 205, 50), # Lime Green for household
            'other': (128, 128, 128)    # Gray for other
        }
        
        # Bright fallback colors for track IDs
        track_colors = [
            (0, 255, 255), (255, 0, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0),
            (0, 128, 255), (255, 128, 0), (128, 0, 255), (0, 255, 128), (255, 165, 0),
            (173, 255, 47), (255, 20, 147), (0, 191, 255), (255, 69, 0), (124, 252, 0)
        ]
        
        for i, detection in enumerate(detections):
            x1, y1, x2, y2 = detection['bbox']
            object_type = detection.get('object_type', 'other')
            in_roi = detection.get('in_roi', True)
            track_id = detection.get('track_id', None)
            
            # Choose color based on track ID if available, otherwise object type
            if track_id is not None:
                color = track_colors[track_id % len(track_colors)]
            else:
                color = type_colors.get(object_type, (128, 128, 128))
            
            # Adjust color and thickness if outside ROI
            if roi_detection and not in_roi:
                # Dim color for objects outside ROI but keep them visible
                color = tuple(int(c * 0.6) for c in color)
                line_thickness_adj = max(2, line_thickness - 1)
            else:
                line_thickness_adj = line_thickness
            
            # Draw bounding box with enhanced visibility
            cv2.rectangle(output_image, (int(x1), int(y1)), (int(x2), int(y2)), color, line_thickness_adj)
            
            # Draw filled corner markers for better visibility
            corner_size = 8
            cv2.rectangle(output_image, (int(x1), int(y1)), (int(x1 + corner_size), int(y1 + corner_size)), color, -1)
            cv2.rectangle(output_image, (int(x2 - corner_size), int(y1)), (int(x2), int(y1 + corner_size)), color, -1)
            cv2.rectangle(output_image, (int(x1), int(y2 - corner_size)), (int(x1 + corner_size), int(y2)), color, -1)
            cv2.rectangle(output_image, (int(x2 - corner_size), int(y2 - corner_size)), (int(x2), int(y2)), color, -1)
            
            if show_labels:
                label_parts = [detection['class_name']]
                
                # Add track ID prominently if available
                if track_id is not None:
                    label_parts.insert(0, f"ID:{track_id}")
                
                if show_confidence:
                    label_parts.append(f"{detection['confidence']:.2f}")
                
                if roi_detection and in_roi:
                    label_parts.append("ROI")
                
                label = " ".join(label_parts)
                
                # Enhanced label background with better contrast
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                thickness = 2
                
                label_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
                label_y = max(int(y1) - 10, label_size[1] + 15)
                
                # Create high-contrast background
                bg_color = (0, 0, 0) if sum(color) > 400 else (255, 255, 255)
                text_color = (255, 255, 255) if bg_color == (0, 0, 0) else (0, 0, 0)
                
                # Draw background rectangle with padding
                padding = 5
                cv2.rectangle(output_image, 
                             (int(x1) - padding, label_y - label_size[1] - padding),
                             (int(x1) + label_size[0] + padding, label_y + padding), 
                             bg_color, -1)
                
                # Draw colored border around label
                cv2.rectangle(output_image, 
                             (int(x1) - padding, label_y - label_size[1] - padding),
                             (int(x1) + label_size[0] + padding, label_y + padding), 
                             color, 2)
                
                # Draw text
                cv2.putText(output_image, label, (int(x1), label_y - 3),
                           font, font_scale, text_color, thickness)
        
        # Add detection summary overlay
        self._add_detection_summary_overlay(output_image, detections, parameters)
        
        return output_image
    
    def _draw_roi_overlay(self, image, roi_type, roi_coordinates=None):
        """Draw ROI visualization on image"""
        height, width = image.shape[:2]
        overlay = image.copy()
        
        # Create ROI mask
        roi_mask = self._create_roi_mask(width, height, roi_type, roi_coordinates)
        
        # Draw ROI border
        roi_color = (0, 255, 255)  # Yellow for ROI
        contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            cv2.drawContours(overlay, [contour], -1, roi_color, 3)
        
        # Add ROI semi-transparent overlay
        roi_colored = np.zeros_like(image)
        roi_colored[roi_mask == 1] = roi_color
        cv2.addWeighted(image, 0.9, roi_colored, 0.1, 0, image)
        
        # Add ROI label
        cv2.putText(image, f"ROI: {roi_type}", (10, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, roi_color, 2)
    
    def _add_detection_summary_overlay(self, image, detections, parameters):
        """Add detection summary information overlay"""
        height, width = image.shape[:2]
        
        # Count objects by type
        type_counts = {}
        roi_counts = {'in_roi': 0, 'total': len(detections)}
        
        for detection in detections:
            obj_type = detection.get('object_type', 'other')
            type_counts[obj_type] = type_counts.get(obj_type, 0) + 1
            
            if detection.get('in_roi', True):
                roi_counts['in_roi'] += 1
        
        # Create summary text
        summary_lines = [f"Total: {len(detections)}"]
        
        if parameters.get('roi_detection', False):
            summary_lines.append(f"In ROI: {roi_counts['in_roi']}")
        
        # Add top object types
        for obj_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True)[:3]:
            if count > 0:
                summary_lines.append(f"{obj_type.title()}: {count}")
        
        # Draw summary box
        box_height = len(summary_lines) * 25 + 20
        box_width = 200
        
        overlay = image.copy()
        cv2.rectangle(overlay, (width - box_width - 10, 10), 
                     (width - 10, 10 + box_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        # Draw summary text
        for i, line in enumerate(summary_lines):
            y_pos = 35 + i * 25
            cv2.putText(image, line, (width - box_width + 5, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    def _add_frame_info_overlay_enhanced(self, frame, frame_number, detection_count, track_stats, processing_fps, output_fps):
        """Add enhanced frame information overlay with FPS details"""
        height, width = frame.shape[:2]
        
        # Create overlay background
        overlay = frame.copy()
        
        # Define larger overlay area for more information
        overlay_height = 170
        cv2.rectangle(overlay, (0, 0), (400, overlay_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Add text information
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        text_color = (255, 255, 255)
        thickness = 1
        
        # Frame info
        y_offset = 25
        cv2.putText(frame, f"Frame: {frame_number}", (10, y_offset), font, font_scale, text_color, thickness)
        
        y_offset += 25
        cv2.putText(frame, f"Detections: {detection_count}", (10, y_offset), font, font_scale, text_color, thickness)
        
        # Show both immediate and final counts
        immediate_people = track_stats.get('immediate_unique_people', 0)
        final_people = track_stats.get('unique_people', 0)
        
        y_offset += 25
        if immediate_people > 0:
            cv2.putText(frame, f"People (immediate): {immediate_people}", (10, y_offset), font, font_scale, (0, 255, 255), thickness)
        else:
            cv2.putText(frame, f"People (tracking...)", (10, y_offset), font, font_scale, (128, 128, 128), thickness)
        
        y_offset += 25
        cv2.putText(frame, f"People (qualified): {final_people}", (10, y_offset), font, font_scale, (0, 255, 0), thickness)
        
        y_offset += 25
        cv2.putText(frame, f"Active Tracks: {track_stats['active_tracks']}", (10, y_offset), font, font_scale, text_color, thickness)
        
        # Add FPS information
        y_offset += 25
        cv2.putText(frame, f"Processing: {processing_fps} FPS", (10, y_offset), font, 0.5, (255, 165, 0), thickness)
        
        # Add explanation
        explanation = track_stats.get('tracking_explanation', '')
        if explanation and frame_number <= 10:  # Show explanation for first 10 frames
            # Split explanation into multiple lines if too long
            words = explanation.split()
            lines = []
            current_line = ""
            for word in words:
                test_line = current_line + " " + word if current_line else word
                if len(test_line) > 40:  # Roughly 40 characters per line
                    if current_line:
                        lines.append(current_line)
                    current_line = word
                else:
                    current_line = test_line
            if current_line:
                lines.append(current_line)
            
            # Draw explanation lines
            for i, line in enumerate(lines):
                if i < 2:  # Max 2 lines
                    cv2.putText(frame, line, (10, height - 50 + (i * 20)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        # Add enhanced branding with FPS info
        cv2.putText(frame, f"D-Fine + High-FPS Tracking ({processing_fps} FPS)", (width - 350, height - 20), font, 0.5, (0, 255, 0), 1)
        
        return frame
    
    def _add_image_summary_overlay(self, image, detections, parameters):
        """Add summary overlay specifically for single images"""
        height, width = image.shape[:2]
        
        # Count objects by type
        type_counts = {}
        roi_counts = {'in_roi': 0, 'total': len(detections)}
        
        for detection in detections:
            obj_type = detection.get('object_type', 'other')
            class_name = detection.get('class_name', 'unknown')
            type_counts[class_name] = type_counts.get(class_name, 0) + 1
            
            if detection.get('in_roi', True):
                roi_counts['in_roi'] += 1
        
        # Create summary text
        summary_lines = [f"Total Objects: {len(detections)}"]
        
        if parameters.get('roi_detection', False):
            summary_lines.append(f"In ROI: {roi_counts['in_roi']}")
            summary_lines.append(f"ROI Type: {parameters.get('roi_type', 'rectangle')}")
        
        # Add object counts
        for class_name, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True)[:4]:
            if count > 0:
                summary_lines.append(f"{class_name}: {count}")
        
        # Draw summary box in bottom-left corner
        box_height = len(summary_lines) * 30 + 20
        box_width = max(200, max(len(line) for line in summary_lines) * 12)
        
        # Semi-transparent background
        overlay = image.copy()
        cv2.rectangle(overlay, (10, height - box_height - 10), 
                     (box_width + 10, height - 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, image, 0.2, 0, image)
        
        # Draw border
        cv2.rectangle(image, (10, height - box_height - 10), 
                     (box_width + 10, height - 10), (0, 255, 255), 2)
        
        # Draw summary text
        for i, line in enumerate(summary_lines):
            y_pos = height - box_height + 5 + (i + 1) * 30
            cv2.putText(image, line, (20, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to native Python types for JSON serialization"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    def _cleanup_old_frames(self, keep_recent=50):
        """Clean up old frame files to prevent outputs folder from getting too large"""
        try:
            outputs_dir = 'outputs'
            if not os.path.exists(outputs_dir):
                return
            
            # Get all frame files (start with 'frame_')
            frame_files = [f for f in os.listdir(outputs_dir) if f.startswith('frame_') and f.endswith('.jpg')]
            
            # Sort by modification time (newest first)
            frame_files.sort(key=lambda f: os.path.getmtime(os.path.join(outputs_dir, f)), reverse=True)
            
            # Remove old files if we have more than the limit
            if len(frame_files) > keep_recent:
                files_to_remove = frame_files[keep_recent:]
                removed_count = 0
                for file_to_remove in files_to_remove:
                    try:
                        os.remove(os.path.join(outputs_dir, file_to_remove))
                        removed_count += 1
                    except Exception:
                        pass  # Ignore individual file removal errors
                
                if removed_count > 0:
                    print(f"🧹 Cleaned up {removed_count} old frame files")
        except Exception:
            pass  # Ignore cleanup errors to not interrupt main processing
    
    def _create_roi_mask(self, width, height, roi_type, roi_coordinates=None):
        """Create ROI mask based on type and coordinates"""
        mask = np.zeros((height, width), dtype=np.uint8)
        
        if roi_coordinates:
            # Use specific coordinates if provided
            if roi_type == "rectangle" and len(roi_coordinates) >= 4:
                x1, y1, x2, y2 = roi_coordinates[:4]
                mask[int(y1):int(y2), int(x1):int(x2)] = 1
            # Add more coordinate-based ROI types here
        else:
            # Use predefined ROI regions
            if roi_type == "center":
                # Center 50% of image
                x1, y1 = int(width * 0.25), int(height * 0.25)
                x2, y2 = int(width * 0.75), int(height * 0.75)
                mask[y1:y2, x1:x2] = 1
                
            elif roi_type == "left":
                # Left 40% of image
                mask[:, :int(width * 0.4)] = 1
                
            elif roi_type == "right":
                # Right 40% of image
                mask[:, int(width * 0.6):] = 1
                
            elif roi_type == "top":
                # Top 40% of image
                mask[:int(height * 0.4), :] = 1
                
            elif roi_type == "bottom":
                # Bottom 40% of image
                mask[int(height * 0.6):, :] = 1
                
            elif roi_type == "background":
                # Outer 20% border
                border = min(int(width * 0.2), int(height * 0.2))
                mask[:border, :] = 1  # top
                mask[-border:, :] = 1  # bottom
                mask[:, :border] = 1  # left
                mask[:, -border:] = 1  # right
                
            elif roi_type == "foreground":
                # Inner 60% of image
                x1, y1 = int(width * 0.2), int(height * 0.2)
                x2, y2 = int(width * 0.8), int(height * 0.8)
                mask[y1:y2, x1:x2] = 1
                
            else:  # default rectangle - center area
                x1, y1 = int(width * 0.2), int(height * 0.2)
                x2, y2 = int(width * 0.8), int(height * 0.8)
                mask[y1:y2, x1:x2] = 1
        
        return mask
    
    def _is_detection_in_roi(self, center_x, center_y, roi_mask):
        """Check if detection center is within ROI"""
        try:
            height, width = roi_mask.shape
            x, y = int(center_x), int(center_y)
            
            # Ensure coordinates are within image bounds
            if 0 <= x < width and 0 <= y < height:
                return roi_mask[y, x] == 1
            return False
        except:
            return True  # Default to True if error
    
    def _classify_object_type(self, class_name):
        """Classify object into categories for better processing"""
        categories = {
            'person': ['person'],
            'vehicle': ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'airplane', 'boat', 'train'],
            'animal': ['cat', 'dog', 'horse', 'cow', 'sheep', 'elephant', 'bear', 'zebra', 'giraffe', 'bird'],
            'furniture': ['chair', 'couch', 'bed', 'dining table', 'bench'],
            'electronics': ['tv', 'laptop', 'cell phone', 'remote', 'keyboard', 'mouse'],
            'food': ['apple', 'banana', 'sandwich', 'pizza', 'cake', 'orange', 'hot dog'],
            'sports': ['sports ball', 'baseball bat', 'tennis racket', 'frisbee', 'skateboard', 'surfboard'],
            'household': ['bottle', 'cup', 'bowl', 'knife', 'fork', 'spoon', 'microwave', 'oven', 'refrigerator'],
            'other': []
        }
        
        for category, objects in categories.items():
            if class_name.lower() in [obj.lower() for obj in objects]:
                return category
        
        return 'other'
    
    def _draw_detections(self, image, detections, parameters):
        """Compatibility method - redirects to enhanced version"""
        return self._draw_detections_enhanced(image, detections, parameters)
    
    def _add_frame_info_overlay(self, frame, frame_number, detection_count, track_stats):
        """Compatibility method - redirects to enhanced version with default FPS values"""
        return self._add_frame_info_overlay_enhanced(frame, frame_number, detection_count, track_stats, 20, 20) 