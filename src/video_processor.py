import cv2
import numpy as np
from moviepy import VideoFileClip
import speech_recognition as sr
from pydub import AudioSegment
import tempfile
import os
from PIL import Image
import torch
from transformers import pipeline
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoProcessor:
    def __init__(self):
        """Initialize the video processor with necessary models and components."""
        self.recognizer = sr.Recognizer()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Initialize multimodal model for video understanding - use compatible model
        try:
            # Use a simpler, compatible model for video classification
            self.video_understanding_model = pipeline(
                "image-classification",  # Use image classification instead of video
                model="microsoft/resnet-50",  # More compatible model
                device=self.device
            )
            logger.info("Loaded video understanding model (ResNet-50)")
        except Exception as e:
            logger.warning(f"Could not load video understanding model: {e}")
            self.video_understanding_model = None
            
        # Initialize image captioning model
        try:
            self.image_captioning_model = pipeline(
                "image-to-text",
                model="nlpconnect/vit-gpt2-image-captioning",
                device=self.device
            )
            logger.info("Loaded image captioning model")
        except Exception as e:
            logger.warning(f"Could not load image captioning model: {e}")
            self.image_captioning_model = None

    def process_video(self, video_path: str, audio_path: str = None) -> Dict:
        """
        Process a video file and extract audio, visual, and combined information.
        
        Args:
            video_path: Path to the video file
            audio_path: Optional path to separate audio file (WAV format)
            
        Returns:
            Dictionary containing extracted information
        """
        try:
            logger.info(f"Processing video: {video_path}")
            if audio_path:
                logger.info(f"Using separate audio file: {audio_path}")
            
            # Extract audio and transcribe
            if audio_path and os.path.exists(audio_path):
                # Use separate audio file
                audio_text = self._transcribe_audio_file(audio_path)
            else:
                # Extract audio from video (original method)
                audio_text = self._extract_audio_and_transcribe(video_path)
            
            # Extract visual information
            visual_info = self._extract_visual_information(video_path)
            
            # Combine information
            combined_info = self._combine_information(audio_text, visual_info)
            
            return {
                "audio_transcription": audio_text,
                "visual_information": visual_info,
                "combined_analysis": combined_info,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def _extract_audio_and_transcribe(self, video_path: str) -> str:
        """Extract audio from video and transcribe speech to text."""
        try:
            # Load video and extract audio
            video = VideoFileClip(video_path)
            
            # Check if video has audio
            if video.audio is None:
                logger.info("Video has no audio track")
                video.close()
                return "No audio track found in video"
            
            audio = video.audio
            
            # Create temporary audio file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                temp_audio_path = temp_audio.name
            
            # Export audio as WAV
            audio.write_audiofile(temp_audio_path, verbose=False, logger=None)
            audio.close()
            video.close()
            
            # Convert to format suitable for speech recognition
            audio_segment = AudioSegment.from_wav(temp_audio_path)
            
            # Convert to 16kHz mono (required for speech recognition)
            audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)
            
            # Export as WAV again
            audio_segment.export(temp_audio_path, format="wav")
            
            # Transcribe using speech recognition
            with sr.AudioFile(temp_audio_path) as source:
                audio_data = self.recognizer.record(source)
                
            # Try multiple recognition engines
            transcription = ""
            
            # Try Google Speech Recognition
            try:
                transcription = self.recognizer.recognize_google(audio_data)
                logger.info("Used Google Speech Recognition")
            except sr.UnknownValueError:
                logger.warning("Google Speech Recognition could not understand audio")
            except sr.RequestError:
                logger.warning("Google Speech Recognition service unavailable")
            
            # If Google fails, try Sphinx (offline)
            if not transcription:
                try:
                    transcription = self.recognizer.recognize_sphinx(audio_data)
                    logger.info("Used Sphinx (offline) Speech Recognition")
                except sr.UnknownValueError:
                    logger.warning("Sphinx could not understand audio")
                except Exception as e:
                    logger.warning(f"Sphinx error: {e}")
            
            # Clean up temporary file
            os.unlink(temp_audio_path)
            
            return transcription if transcription else "No speech detected"
            
        except Exception as e:
            logger.error(f"Error in audio extraction/transcription: {e}")
            return f"Error processing audio: {str(e)}"

    def _transcribe_audio_file(self, audio_path: str) -> str:
        """Transcribe speech from a separate audio file (WAV format)."""
        try:
            logger.info(f"Transcribing audio file: {audio_path}")
            
            # Convert to format suitable for speech recognition
            audio_segment = AudioSegment.from_wav(audio_path)
            
            # Convert to 16kHz mono (required for speech recognition)
            audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)
            
            # Create temporary file for converted audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                temp_audio_path = temp_audio.name
            
            # Export as WAV
            audio_segment.export(temp_audio_path, format="wav")
            
            # Transcribe using speech recognition
            with sr.AudioFile(temp_audio_path) as source:
                audio_data = self.recognizer.record(source)
                
            # Try multiple recognition engines
            transcription = ""
            
            # Try Google Speech Recognition
            try:
                transcription = self.recognizer.recognize_google(audio_data)
                logger.info("Used Google Speech Recognition")
            except sr.UnknownValueError:
                logger.warning("Google Speech Recognition could not understand audio")
            except sr.RequestError:
                logger.warning("Google Speech Recognition service unavailable")
            
            # If Google fails, try Sphinx (offline)
            if not transcription:
                try:
                    transcription = self.recognizer.recognize_sphinx(audio_data)
                    logger.info("Used Sphinx (offline) Speech Recognition")
                except sr.UnknownValueError:
                    logger.warning("Sphinx could not understand audio")
                except Exception as e:
                    logger.warning(f"Sphinx error: {e}")
            
            # Clean up temporary file
            os.unlink(temp_audio_path)
            
            return transcription if transcription else "No speech detected"
            
        except Exception as e:
            logger.error(f"Error transcribing audio file: {e}")
            return f"Error transcribing audio: {str(e)}"

    def _extract_visual_information(self, video_path: str) -> Dict:
        """Extract visual information from video frames."""
        try:
            video = cv2.VideoCapture(video_path)
            
            if not video.isOpened():
                raise ValueError("Could not open video file")
            
            # Get video properties
            fps = video.get(cv2.CAP_PROP_FPS)
            frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            # Extract key frames (every 1 second or every 30 frames)
            frame_interval = max(1, int(fps))
            frames = []
            frame_descriptions = []
            
            frame_idx = 0
            while True:
                ret, frame = video.read()
                if not ret:
                    break
                    
                if frame_idx % frame_interval == 0:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
                    
                    # Generate description for this frame
                    description = self._describe_frame(frame_rgb)
                    frame_descriptions.append(description)
                    
                    # Limit to reasonable number of frames
                    if len(frames) >= 10:
                        break
                        
                frame_idx += 1
            
            video.release()
            
            # Analyze video content if model is available
            video_analysis = ""
            if self.video_understanding_model and frames:
                try:
                    # Use first frame for analysis
                    first_frame = frames[0]
                    # Convert to PIL Image
                    pil_image = Image.fromarray(first_frame)
                    
                    # Get video classification
                    results = self.video_understanding_model(pil_image)
                    video_analysis = f"Video content analysis: {results[0]['label']} (confidence: {results[0]['score']:.2f})"
                    
                except Exception as e:
                    logger.warning(f"Video analysis failed: {e}")
            
            return {
                "frame_count": frame_count,
                "fps": fps,
                "duration": duration,
                "key_frames_extracted": len(frames),
                "frame_descriptions": frame_descriptions,
                "video_analysis": video_analysis
            }
            
        except Exception as e:
            logger.error(f"Error in visual information extraction: {e}")
            return {"error": str(e)}

    def _describe_frame(self, frame: np.ndarray) -> str:
        """Generate a description for a single frame."""
        try:
            if self.image_captioning_model:
                # Convert numpy array to PIL Image
                pil_image = Image.fromarray(frame)
                
                # Generate caption
                caption = self.image_captioning_model(pil_image)[0]['generated_text']
                return caption
            else:
                # Basic frame analysis using OpenCV
                height, width = frame.shape[:2]
                # Calculate average brightness
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                brightness = np.mean(gray)
                
                # Detect edges
                edges = cv2.Canny(gray, 50, 150)
                edge_density = np.sum(edges > 0) / (height * width)
                
                return f"Frame: {width}x{height}, brightness: {brightness:.1f}, edge density: {edge_density:.3f}"
                
        except Exception as e:
            logger.warning(f"Frame description failed: {e}")
            return f"Frame analysis error: {str(e)}"

    def _combine_information(self, audio_text: str, visual_info: Dict) -> str:
        """Combine audio and visual information into a comprehensive analysis."""
        try:
            combined = f"Audio Transcription: {audio_text}\n\n"
            combined += "Visual Information:\n"
            
            if "error" in visual_info:
                combined += f"Visual analysis error: {visual_info['error']}\n"
            else:
                combined += f"Duration: {visual_info.get('duration', 'Unknown'):.2f} seconds\n"
                combined += f"FPS: {visual_info.get('fps', 'Unknown'):.1f}\n"
                combined += f"Frames analyzed: {visual_info.get('key_frames_extracted', 0)}\n"
                
                if visual_info.get('video_analysis'):
                    combined += f"Content: {visual_info['video_analysis']}\n"
                
                if visual_info.get('frame_descriptions'):
                    combined += "\nKey frame descriptions:\n"
                    for i, desc in enumerate(visual_info['frame_descriptions'][:3]):  # Limit to first 3
                        combined += f"  Frame {i+1}: {desc}\n"
            
            return combined
            
        except Exception as e:
            logger.error(f"Error combining information: {e}")
            return f"Error combining information: {str(e)}"

    def process_video_command(self, video_path: str, audio_path: str = None) -> str:
        """
        Process video and return a formatted command string for the AI model.
        
        Args:
            video_path: Path to the video file
            audio_path: Optional path to separate audio file (WAV format)
            
        Returns:
            Formatted string containing video analysis for AI processing
        """
        result = self.process_video(video_path, audio_path)
        
        if not result["success"]:
            return f"Error processing video: {result['error']}"
        
        # Format for AI model consumption
        command_text = f"""
VIDEO INPUT ANALYSIS:
===================

SPOKEN COMMAND: {result['audio_transcription']}

VISUAL CONTEXT: {result['combined_analysis']}

INSTRUCTIONS: Please analyze the above video input and respond to the spoken command while considering the visual context provided. The user has provided a video instead of text input.
"""
        
        return command_text
