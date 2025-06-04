import json
from groq import Groq
from config import Config

class LLMService:
    def __init__(self):
        self.client = Groq(api_key=Config.GROQ_API_KEY)
        
        # Complete list of D-Fine detectable objects
        self.detectable_objects = [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", 
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", 
            "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", 
            "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", 
            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", 
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", 
            "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", 
            "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", 
            "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
        ]
        
    def parse_user_prompt(self, user_prompt):
        """
        Advanced prompt parsing to identify objects, ROI, and detection parameters
        """
        system_prompt = f"""
        You are an expert computer vision AI assistant. Analyze the user's request and generate optimal D-Fine detection parameters.
        You can detect any of these objects: {', '.join(self.detectable_objects)}

        Return your response as a JSON object with this structure:
        {{
            "confidence_threshold": float (0.1-0.9, optimal for requested objects),
            "classes_to_detect": list of class names or "all",
            "roi_detection": boolean (true if user mentions specific areas/regions),
            "roi_type": string ("rectangle", "polygon", "center", "edges", "top", "bottom", "left", "right"),
            "roi_coordinates": object (coordinates if specific region mentioned),
            "tracking_enabled": boolean (true for videos or multiple object tracking),
            "count_only": boolean (true if user only wants numbers),
            "show_labels": boolean,
            "show_confidence": boolean, 
            "line_thickness": int (1-5),
            "detection_focus": string ("counting", "identification", "tracking", "analysis"),
            "area_filter": string ("small", "medium", "large", "all"),
            "reasoning": "explanation of choices"
        }}

        OBJECT IDENTIFICATION RULES:
        1. Extract specific objects mentioned (car, person, dog, etc.)
        2. Use synonyms: "people/humans" = person, "vehicle" = car, "animal" = depends on context
        3. If no specific objects mentioned, use "all"
        4. For counting tasks, focus on the main object being counted

        ROI DETECTION RULES:
        1. Look for spatial references: "in the center", "on the left", "background", "foreground"
        2. Look for area descriptions: "parking lot", "sidewalk", "entrance", "specific area"
        3. Set roi_type based on description:
           - "rectangle" for "box area", "specific region"
           - "center" for "middle", "center area"
           - "edges" for "borders", "edges"
           - "top/bottom/left/right" for directional areas

        EXAMPLES:
        - "count cars in the parking lot" → cars + ROI rectangle
        - "detect people in the center of image" → person + ROI center  
        - "find all animals on the left side" → [cat, dog, horse, etc.] + ROI left
        - "track vehicles entering from the right" → [car, truck, bus] + ROI right + tracking
        - "count bottles on the table" → bottle + ROI center/rectangle
        """
        
        try:
            completion = self.client.chat.completions.create(
                model=Config.GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Analyze this request: {user_prompt}"}
                ],
                temperature=0.3,
                max_completion_tokens=1024,
                top_p=0.9,
                stream=False,
                stop=None,
            )
            
            response_content = completion.choices[0].message.content
            
            # Extract JSON from response
            json_start = response_content.find('{')
            json_end = response_content.rfind('}') + 1
            
            if json_start != -1 and json_end != -1:
                json_str = response_content[json_start:json_end]
                parsed_result = json.loads(json_str)
                
                # Post-process and validate
                return self._validate_and_enhance_parameters(parsed_result, user_prompt)
            else:
                return self._get_intelligent_fallback(user_prompt)
                
        except Exception as e:
            print(f"LLM Error: {e}")
            return self._get_intelligent_fallback(user_prompt)
    
    def _validate_and_enhance_parameters(self, params, user_prompt):
        """Validate and enhance LLM-generated parameters"""
        prompt_lower = user_prompt.lower()
        
        # Ensure required fields
        defaults = {
            "confidence_threshold": 0.3,
            "classes_to_detect": "all",
            "roi_detection": False,
            "roi_type": "rectangle",
            "roi_coordinates": None,
            "tracking_enabled": True,
            "count_only": False,
            "show_labels": True,
            "show_confidence": True,
            "line_thickness": 3,
            "detection_focus": "identification",
            "area_filter": "all",
            "reasoning": "Default parameters"
        }
        
        for key, default_value in defaults.items():
            if key not in params:
                params[key] = default_value
        
        # Smart object detection
        params["classes_to_detect"] = self._extract_objects_from_prompt(prompt_lower, params.get("classes_to_detect", "all"))
        
        # Smart ROI detection
        roi_info = self._detect_roi_from_prompt(prompt_lower)
        if roi_info["has_roi"]:
            params["roi_detection"] = True
            params["roi_type"] = roi_info["type"]
            params["roi_coordinates"] = roi_info["coordinates"]
        
        # Smart detection focus
        params["detection_focus"] = self._determine_detection_focus(prompt_lower)
        
        # Count-only detection
        count_keywords = ["count", "how many", "number of", "total"]
        if any(keyword in prompt_lower for keyword in count_keywords):
            params["count_only"] = True
            params["detection_focus"] = "counting"
        
        # Confidence optimization based on object type
        if isinstance(params["classes_to_detect"], list):
            params["confidence_threshold"] = self._optimize_confidence_for_objects(params["classes_to_detect"])
        
        return params
    
    def _extract_objects_from_prompt(self, prompt_lower, llm_suggestion):
        """Extract specific objects mentioned in prompt"""
        mentioned_objects = []
        
        # Direct object mentions
        for obj in self.detectable_objects:
            if obj in prompt_lower:
                mentioned_objects.append(obj)
        
        # Handle synonyms and variations
        synonyms = {
            "people": "person", "humans": "person", "human": "person",
            "vehicle": ["car", "truck", "bus"], "vehicles": ["car", "truck", "bus"],
            "animal": ["cat", "dog", "horse", "cow", "sheep"], "animals": ["cat", "dog", "horse", "cow", "sheep"],
            "furniture": ["chair", "couch", "bed", "dining table"],
            "electronics": ["tv", "laptop", "cell phone", "remote"],
            "food": ["apple", "banana", "sandwich", "pizza", "cake"]
        }
        
        for synonym, objects in synonyms.items():
            if synonym in prompt_lower:
                if isinstance(objects, list):
                    mentioned_objects.extend(objects)
                else:
                    mentioned_objects.append(objects)
        
        # Remove duplicates
        mentioned_objects = list(set(mentioned_objects))
        
        # If LLM suggested specific objects and we found some, combine them
        if isinstance(llm_suggestion, list) and mentioned_objects:
            combined = list(set(mentioned_objects + llm_suggestion))
            return combined
        elif mentioned_objects:
            return mentioned_objects
        elif isinstance(llm_suggestion, list):
            return llm_suggestion
        else:
            return "all"
    
    def _detect_roi_from_prompt(self, prompt_lower):
        """Enhanced ROI detection from prompt with better spatial recognition"""
        roi_info = {"has_roi": False, "type": "rectangle", "coordinates": None}
        
        # Enhanced spatial keywords with context
        spatial_keywords = {
            "center": ["center", "middle", "central", "in the center", "at the center"],
            "left": ["left", "left side", "on the left", "to the left", "left part", "left area"],
            "right": ["right", "right side", "on the right", "to the right", "right part", "right area"], 
            "top": ["top", "upper", "above", "top part", "upper area", "overhead"],
            "bottom": ["bottom", "lower", "below", "bottom part", "lower area", "ground level"],
            "background": ["background", "back", "far", "distant", "behind", "backdrop"],
            "foreground": ["foreground", "front", "near", "close", "in front", "immediate area"]
        }
        
        # Area-specific keywords with better matching
        area_keywords = {
            "rectangle": [
                "parking lot", "parking area", "road", "street", "sidewalk", "crosswalk", 
                "zebra crossing", "intersection", "entrance", "exit", "door", "doorway",
                "window", "table", "desk", "field", "ground", "floor area", "specific area",
                "designated area", "marked area", "section", "zone", "region"
            ]
        }
        
        # Check for spatial keywords
        for roi_type, keywords in spatial_keywords.items():
            if any(keyword in prompt_lower for keyword in keywords):
                roi_info["has_roi"] = True
                roi_info["type"] = roi_type
                print(f"🎯 ROI detected: {roi_type} (keyword match)")
                break
        
        # Check for area-specific keywords
        if not roi_info["has_roi"]:
            for roi_type, keywords in area_keywords.items():
                if any(keyword in prompt_lower for keyword in keywords):
                    roi_info["has_roi"] = True
                    roi_info["type"] = roi_type
                    print(f"🎯 ROI detected: {roi_type} (area keyword match)")
                    break
        
        # Enhanced context detection
        context_patterns = [
            ("in the", "on the", "at the", "near the", "around the"),
            ("specific", "particular", "designated", "marked"),
            ("crossing", "intersection", "junction", "corner"),
            ("standing", "walking", "moving", "located")
        ]
        
        for patterns in context_patterns:
            if any(pattern in prompt_lower for pattern in patterns):
                if not roi_info["has_roi"]:
                    roi_info["has_roi"] = True
                    roi_info["type"] = "rectangle"
                    print(f"🎯 ROI detected: rectangle (context pattern match)")
                break
        
        # Special handling for zebra crossing and similar concepts
        zebra_patterns = ["zebra crossing", "crosswalk", "pedestrian crossing", "crossing"]
        if any(pattern in prompt_lower for pattern in zebra_patterns):
            roi_info["has_roi"] = True
            roi_info["type"] = "center"  # Crossings are typically center-focused
            print(f"🎯 ROI detected: center (zebra crossing/crosswalk)")
        
        if roi_info["has_roi"]:
            print(f"📍 Final ROI configuration: Type={roi_info['type']}, Has_ROI={roi_info['has_roi']}")
        
        return roi_info
    
    def _determine_detection_focus(self, prompt_lower):
        """Determine the main focus of detection"""
        if any(word in prompt_lower for word in ["count", "how many", "number"]):
            return "counting"
        elif any(word in prompt_lower for word in ["track", "follow", "movement", "moving"]):
            return "tracking"
        elif any(word in prompt_lower for word in ["identify", "what", "recognize", "classify"]):
            return "identification"
        elif any(word in prompt_lower for word in ["analyze", "analysis", "study", "examine"]):
            return "analysis"
        else:
            return "identification"
    
    def _optimize_confidence_for_objects(self, objects):
        """Optimize confidence threshold based on object types"""
        # Different objects have different detection difficulty
        easy_objects = ["person", "car", "truck", "bus", "chair", "tv", "laptop"]
        medium_objects = ["bicycle", "motorcycle", "bottle", "cup", "book"]
        hard_objects = ["fork", "knife", "spoon", "remote", "cell phone"]
        
        if any(obj in hard_objects for obj in objects):
            return 0.25  # Lower threshold for hard objects
        elif any(obj in easy_objects for obj in objects):
            return 0.4   # Higher threshold for easy objects
        else:
            return 0.3   # Medium threshold
    
    def _get_intelligent_fallback(self, user_prompt):
        """Intelligent fallback when LLM fails"""
        prompt_lower = user_prompt.lower()
        
        # Extract objects using simple keyword matching
        detected_objects = []
        for obj in self.detectable_objects:
            if obj in prompt_lower:
                detected_objects.append(obj)
        
        if not detected_objects:
            detected_objects = "all"
        
        # Detect basic ROI
        has_roi = any(word in prompt_lower for word in ["center", "left", "right", "top", "bottom", "area", "region"])
        
        # Detect counting intent
        count_only = any(word in prompt_lower for word in ["count", "how many", "number"])
        
        return {
            "confidence_threshold": 0.3,
            "classes_to_detect": detected_objects,
            "roi_detection": has_roi,
            "roi_type": "center" if "center" in prompt_lower else "rectangle",
            "roi_coordinates": None,
            "tracking_enabled": True,
            "count_only": count_only,
            "show_labels": True,
            "show_confidence": True,
            "line_thickness": 3,
            "detection_focus": "counting" if count_only else "identification",
            "area_filter": "all",
            "reasoning": "Fallback parameters using keyword analysis"
        } 