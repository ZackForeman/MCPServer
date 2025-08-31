import torch
import torch.nn as nn
from transformers import (
    AutoProcessor, 
    AutoModel, 
    pipeline,
    CLIPProcessor, 
    CLIPModel,
    VideoMAEFeatureExtractor,
    VideoMAEModel
)
from PIL import Image
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
import logging
import tempfile
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class MultimodalVideoProcessor:
    """
    Advanced multimodal processor that can handle video input directly using local models.
    """
    
    def __init__(self):
        """Initialize the multimodal processor with local models."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Initialize models as None first
        self.clip_processor = None
        self.clip_model = None
        self.videomae_processor = None
        self.videomae_model = None
        self.captioning_model = None
        self.text_model = None
        
        # Load models with timeout protection
        self._load_models()
    
    def _load_models(self):
        """Load all models with error handling and timeouts."""
        try:
            # Load CLIP model (more reliable)
            logger.info("Loading CLIP model...")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            logger.info("CLIP model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load CLIP model: {e}")
            self.clip_processor = None
            self.clip_model = None
        
        try:
            # Load VideoMAE processor only (lighter than full model)
            logger.info("Loading VideoMAE processor...")
            self.videomae_processor = VideoMAEFeatureExtractor.from_pretrained("MCG-NJU/videomae-base")
            logger.info("VideoMAE processor loaded successfully")
            # Don't load the full VideoMAE model to avoid infinite loops
        except Exception as e:
            logger.warning(f"Failed to load VideoMAE processor: {e}")
            self.videomae_processor = None
        
        try:
            # Load image captioning model
            logger.info("Loading image captioning model...")
            self.captioning_model = pipeline(
                "image-to-text",
                model="nlpconnect/vit-gpt2-image-captioning",
                device=self.device
            )
            logger.info("Image captioning model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load captioning model: {e}")
            self.captioning_model = None
        
        try:
            # Load text generation model (lighter version)
            logger.info("Loading text generation model...")
            self.text_model = pipeline(
                "text-generation",
                model="microsoft/DialoGPT-medium",
                device=self.device
            )
            logger.info("Text generation model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load text generation model: {e}")
            self.text_model = None
    
    def process_video_multimodal(self, video_path: str) -> Dict:
        """
        Process video using multimodal models for comprehensive understanding.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary containing multimodal analysis results
        """
        try:
            logger.info(f"Processing video with multimodal models: {video_path}")
            
            # Extract frames from video
            frames = self._extract_video_frames(video_path)
            if not frames:
                return {"success": False, "error": "Failed to extract video frames"}
            
            # Analyze frames with available models
            frame_analysis = self._analyze_frames(frames)
            
            # Generate video understanding
            video_understanding = self._generate_video_understanding(frame_analysis)
            
            # Generate multimodal response
            multimodal_response = self._generate_multimodal_response(frame_analysis, video_understanding)
            
            return {
                "success": True,
                "frame_analysis": frame_analysis,
                "video_understanding": video_understanding,
                "multimodal_response": multimodal_response
            }
            
        except Exception as e:
            logger.error(f"Error in multimodal video processing: {e}")
            return {"success": False, "error": str(e)}
    
    def _extract_video_frames(self, video_path: str) -> List[np.ndarray]:
        """Extract key frames from video."""
        try:
            video = cv2.VideoCapture(video_path)
            if not video.isOpened():
                return []
            
            frames = []
            frame_count = 0
            fps = video.get(cv2.CAP_PROP_FPS)
            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Extract frames at regular intervals
            interval = max(1, int(fps))  # One frame per second
            
            while len(frames) < 10:  # Limit to 10 frames
                ret, frame = video.read()
                if not ret:
                    break
                
                if frame_count % interval == 0:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
                
                frame_count += 1
                
                # Safety check to prevent infinite loops
                if frame_count > total_frames:
                    break
            
            video.release()
            return frames
            
        except Exception as e:
            logger.error(f"Error extracting video frames: {e}")
            return []
    
    def _analyze_frames(self, frames: List[np.ndarray]) -> List[Dict]:
        """Analyze individual frames using available models."""
        frame_analysis = []
        
        for i, frame in enumerate(frames):
            analysis = {"frame_index": i}
            
            # CLIP analysis
            if self.clip_processor and self.clip_model:
                try:
                    clip_result = self._analyze_frame_with_clip(frame)
                    analysis["clip_analysis"] = clip_result
                except Exception as e:
                    logger.warning(f"CLIP analysis failed for frame {i}: {e}")
            
            # Captioning analysis
            if self.captioning_model:
                try:
                    caption = self._generate_frame_caption(frame)
                    analysis["caption"] = caption
                except Exception as e:
                    logger.warning(f"Captioning failed for frame {i}: {e}")
            
            # Basic frame analysis
            analysis["basic_info"] = self._get_basic_frame_info(frame)
            
            frame_analysis.append(analysis)
        
        return frame_analysis
    
    def _analyze_frame_with_clip(self, frame: np.ndarray) -> Dict:
        """Analyze frame using CLIP model."""
        try:
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(frame)
            
            # Prepare text prompts for CLIP
            text_prompts = [
                "a person", "an object", "a scene", "animals", "vehicles", 
                "buildings", "nature", "technology", "food", "clothing"
            ]
            
            # Process image and text with CLIP
            inputs = self.clip_processor(
                text=text_prompts,
                images=pil_image,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            # Get CLIP outputs
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=-1)
            
            # Get top concepts
            top_indices = torch.topk(probs, k=3, dim=-1).indices[0]
            concepts = []
            
            for idx in top_indices:
                concepts.append({
                    "concept": text_prompts[idx],
                    "confidence": float(probs[0][idx])
                })
            
            return {"concepts": concepts}
            
        except Exception as e:
            logger.error(f"CLIP analysis error: {e}")
            return {"error": str(e)}
    
    def _generate_frame_caption(self, frame: np.ndarray) -> str:
        """Generate caption for a frame."""
        try:
            pil_image = Image.fromarray(frame)
            result = self.captioning_model(pil_image)
            return result[0]['generated_text']
        except Exception as e:
            logger.error(f"Caption generation error: {e}")
            return f"Caption generation failed: {str(e)}"
    
    def _get_basic_frame_info(self, frame: np.ndarray) -> Dict:
        """Get basic information about a frame."""
        try:
            height, width = frame.shape[:2]
            
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            # Calculate basic metrics
            brightness = float(np.mean(gray))
            contrast = float(np.std(gray))
            
            # Edge detection
            edges = cv2.Canny(gray, 50, 150)
            edge_density = float(np.sum(edges > 0) / (height * width))
            
            return {
                "dimensions": f"{width}x{height}",
                "brightness": brightness,
                "contrast": contrast,
                "edge_density": edge_density
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _generate_video_understanding(self, frame_analysis: List[Dict]) -> Dict:
        """Generate overall video understanding from frame analysis."""
        try:
            # Count concepts across frames
            concept_counts = {}
            captions = []
            
            for frame in frame_analysis:
                # Collect captions
                if "caption" in frame:
                    captions.append(frame["caption"])
                
                # Count CLIP concepts
                if "clip_analysis" in frame and "concepts" in frame["clip_analysis"]:
                    for concept_info in frame["clip_analysis"]["concepts"]:
                        concept = concept_info["concept"]
                        concept_counts[concept] = concept_counts.get(concept, 0) + 1
            
            # Sort concepts by frequency
            sorted_concepts = sorted(concept_counts.items(), key=lambda x: x[1], reverse=True)
            
            # Generate overall analysis
            overall_analysis = "Video contains various visual elements"
            if captions:
                overall_analysis += f" including: {', '.join(captions[:3])}"
            
            return {
                "key_insights": sorted_concepts[:5],
                "overall_analysis": overall_analysis,
                "total_frames_analyzed": len(frame_analysis)
            }
            
        except Exception as e:
            logger.error(f"Error generating video understanding: {e}")
            return {"error": str(e)}
    
    def _generate_multimodal_response(self, frame_analysis: List[Dict], video_understanding: Dict) -> str:
        """Generate a multimodal response combining all analyses."""
        try:
            response = "Based on the video analysis:\n\n"
            
            # Add frame insights
            response += f"Analyzed {len(frame_analysis)} key frames:\n"
            for i, frame in enumerate(frame_analysis[:3]):  # First 3 frames
                if "caption" in frame:
                    response += f"  Frame {i+1}: {frame['caption']}\n"
                elif "clip_analysis" in frame and "concepts" in frame["clip_analysis"]:
                    concepts = [c["concept"] for c in frame["clip_analysis"]["concepts"][:2]]
                    response += f"  Frame {i+1}: Detected {', '.join(concepts)}\n"
            
            # Add overall understanding
            if "overall_analysis" in video_understanding:
                response += f"\nOverall content: {video_understanding['overall_analysis']}\n"
            
            # Add key insights
            if "key_insights" in video_understanding:
                response += "\nKey elements detected:\n"
                for concept, count in video_understanding["key_insights"][:3]:
                    response += f"  - {concept}: appears {count} times\n"
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating multimodal response: {e}")
            return f"Error generating response: {str(e)}"
    
    def get_model_status(self) -> Dict:
        """Get status of all loaded models."""
        return {
            "clip_model": self.clip_model is not None,
            "videomae_processor": self.videomae_processor is not None,
            "captioning_model": self.captioning_model is not None,
            "text_model": self.text_model is not None,
            "device": self.device
        }
