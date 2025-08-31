#!/usr/bin/env python3
"""
Final Enhanced Interactive Video Recording and Processing Demo
This demo uses the simple audio-video recorder to capture both video and audio simultaneously.
"""

import cv2
import numpy as np
import os
import sys
import time
import tempfile
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FinalEnhancedDemo:
    def __init__(self):
        """Initialize the final enhanced interactive demo."""
        self.recorder = None
        self.video_path = None
        self.audio_path = None
        
        # Video processing imports
        try:
            from simple_audio_video_recorder import SimpleAudioVideoRecorder
            from video_processor import VideoProcessor
            from multimodal_model import MultimodalVideoProcessor
            from enhanced_video_client import EnhancedVideoClient
            
            self.recorder = SimpleAudioVideoRecorder()
            self.video_processor = VideoProcessor()
            self.multimodal_processor = MultimodalVideoProcessor()
            self.enhanced_client = EnhancedVideoClient()
            self.modules_loaded = True
            logger.info("All video processing modules loaded successfully")
        except ImportError as e:
            logger.error(f"Failed to load video processing modules: {e}")
            self.modules_loaded = False
    
    def run_demo(self):
        """Run the complete final enhanced interactive demo."""
        print("=== Final Enhanced Interactive Video Recording and Processing Demo ===")
        print("This demo will:")
        print("1. Open a camera window for video AND audio recording")
        print("2. Allow you to record video with audio using the 'R' key")
        print("3. Save audio and video separately for reliability")
        print("4. Process the recorded video through all modules")
        print("5. Display results and analysis")
        print("\nRequirements:")
        print("- Working camera")
        print("- Working microphone")
        print("- All video processing modules installed")
        print("- Sufficient disk space for temporary video files")
        print("=" * 60)
        
        # Check if modules are loaded
        if not self.modules_loaded:
            print("\n❌ Video processing modules not available!")
            print("Please ensure all dependencies are installed:")
            print("- opencv-python")
            print("- pyaudio")
            print("- moviepy")
            print("- speechrecognition")
            print("- transformers")
            print("- torch")
            print("- And other required packages")
            return
        
        # Check camera and audio availability
        if not self.recorder.start_camera():
            print("❌ Failed to start camera. Please check your camera connection.")
            return
        
        if not self.recorder.start_audio_stream():
            print("❌ Failed to start audio stream. Please check your microphone.")
            return
        
        try:
            # Run recording interface
            self._run_recording_interface()
            
            # Process video if available
            if self.video_path and os.path.exists(self.video_path):
                self._process_recorded_video()
            else:
                print("\nNo video was recorded or saved.")
                
        except KeyboardInterrupt:
            print("\nDemo interrupted by user.")
        except Exception as e:
            logger.error(f"Demo error: {e}")
        finally:
            # Cleanup
            if self.recorder:
                self.recorder.cleanup()
            print("\nDemo completed. Camera and audio released.")
    
    def _run_recording_interface(self):
        """Run the recording interface."""
        print("\n=== Audio-Video Recording Started ===")
        print("Press 'R' to start/stop recording")
        print("Press 'S' to save and process video")
        print("Press 'Q' to quit")
        print("=" * 30)
        
        while True:
            # Display camera feed
            ret, frame = self.recorder.video_cap.read()
            if not ret:
                logger.error("Failed to read frame")
                break
            
            # Create display frame
            display_frame = frame.copy()
            
            # Add recording indicator
            if self.recorder.recording:
                # Red circle when recording
                cv2.circle(display_frame, (50, 50), 20, (0, 0, 255), -1)
                cv2.putText(display_frame, "REC", (70, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Show recording info
                info = self.recorder.get_recording_info()
                cv2.putText(display_frame, f"Video: {info['video_frames']} frames", (10, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(display_frame, f"Audio: {info['audio_frames']} chunks", (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(display_frame, f"Duration: {info['video_duration']:.1f}s", (10, 140), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Add instructions
            cv2.putText(display_frame, "R: Record/Stop, S: Save & Process, Q: Quit", (10, 450), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Add status
            status = "Recording..." if self.recorder.recording else "Ready to record"
            cv2.putText(display_frame, status, (10, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                       (0, 255, 0) if not self.recorder.recording else (0, 0, 255), 2)
            
            # Show frame
            cv2.imshow('Final Enhanced Video Demo - Press R to Record', display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                break
            elif key == ord('r') or key == ord('R'):
                if not self.recorder.recording:
                    # Start recording
                    self.recorder.start_recording()
                    print("Recording started...")
                else:
                    # Stop recording
                    self.recorder.stop_recording()
                    print(f"Recording stopped. Captured {self.recorder.get_recording_info()['video_frames']} frames")
            elif key == ord('s') or key == ord('S'):
                if self.recorder.audio_frames and self.recorder.video_frames:
                    if self._save_and_process_recording():
                        print("Video saved and processed successfully!")
                    else:
                        print("Failed to save or process video")
                else:
                    print("No recorded frames to save")
    
    def _save_and_process_recording(self):
        """Save recorded audio-video and prepare for processing."""
        try:
            # Save the recording (separate audio and video files)
            video_path, audio_path = self.recorder.save_recording()
            if video_path and audio_path:
                self.video_path = video_path
                self.audio_path = audio_path
                print(f"\nVideo saved to: {video_path}")
                print(f"Audio saved to: {audio_path}")
                
                # Get recording info
                info = self.recorder.get_recording_info()
                print(f"Duration: {info['video_duration']:.2f} seconds")
                print(f"Video frames: {info['video_frames']}")
                print(f"Audio chunks: {info['audio_frames']}")
                
                return True
            else:
                print("Failed to save recording")
                return False
                
        except Exception as e:
            logger.error(f"Error saving recording: {e}")
            return False
    
    def _process_recorded_video(self):
        """Process the recorded video using all available modules."""
        if not self.video_path or not os.path.exists(self.video_path):
            print("No recorded video available")
            return
        
        if not self.modules_loaded:
            print("Video processing modules not available")
            return
        
        print("\n" + "="*60)
        print("PROCESSING RECORDED AUDIO-VIDEO")
        print("="*60)
        
        # Test 1: Basic Video Processing (now with audio!)
        print("\n1. Testing Basic Video Processing...")
        try:
            basic_result = self.video_processor.process_video(self.video_path, self.audio_path)
            if basic_result.get("success"):
                print("✅ Basic processing successful!")
                print(f"Audio transcription: {basic_result['audio_transcription'][:100]}...")
                print(f"Visual info: {basic_result['visual_information'].get('frame_count', 'N/A')} frames")
            else:
                print(f"❌ Basic processing failed: {basic_result.get('error', 'Unknown error')}")
        except Exception as e:
            print(f"❌ Basic processing error: {e}")
        
        # Test 2: Multimodal Processing
        print("\n2. Testing Multimodal Processing...")
        try:
            multimodal_result = self.multimodal_processor.process_video_multimodal(self.video_path)
            if multimodal_result.get("success"):
                print("✅ Multimodal processing successful!")
                if "frame_analysis" in multimodal_result:
                    print(f"Frames analyzed: {len(multimodal_result['frame_analysis'])}")
                if "video_understanding" in multimodal_result:
                    print(f"Overall analysis: {multimodal_result['video_understanding'].get('overall_analysis', 'N/A')[:100]}...")
            else:
                print(f"❌ Multimodal processing failed: {multimodal_result.get('error', 'Unknown error')}")
        except Exception as e:
            print(f"❌ Multimodal processing error: {e}")
        
        # Test 3: Enhanced Client Processing...
        print("\n3. Testing Enhanced Client Processing...")
        try:
            enhanced_result = self.enhanced_client.process_video_enhanced(self.video_path, self.audio_path, use_multimodal=True)
            if "error" not in enhanced_result:
                print("✅ Enhanced processing successful!")
                print(f"Processing type: {enhanced_result.get('processing_type', 'Unknown')}")
                if "command_text" in enhanced_result:
                    print("Command text generated successfully")
            else:
                print(f"❌ Enhanced processing failed: {enhanced_result['error']}")
        except Exception as e:
            print(f"❌ Enhanced processing error: {e}")
        
        # Test 4: Video Command Processing
        print("\n4. Testing Video Command Processing...")
        try:
            command_text = self.video_processor.process_video_command(self.video_path, self.audio_path)
            if command_text and not command_text.startswith("Error"):
                print("✅ Video command processing successful!")
                print("Command text preview:")
                print(command_text[:200] + "..." if len(command_text) > 200 else command_text)
            else:
                print(f"❌ Video command processing failed: {command_text}")
        except Exception as e:
            print(f"❌ Video command processing error: {e}")
        
        print("\n" + "="*60)
        print("AUDIO-VIDEO PROCESSING COMPLETE")
        print("=" * 60)
        
        # Show file locations
        print(f"\nFiles saved:")
        print(f"  Video: {self.video_path}")
        print(f"  Audio: {self.audio_path}")
        print("\nYou can now:")
        print("  1. Use these files for further processing")
        print("  2. Upload them to other systems")
        print("  3. Analyze them with other tools")
        
        # Clean up temporary files
        try:
            if self.video_path and os.path.exists(self.video_path):
                os.unlink(self.video_path)
                print(f"Cleaned up video file: {self.video_path}")
            if self.audio_path and os.path.exists(self.audio_path):
                os.unlink(self.audio_path)
                print(f"Cleaned up audio file: {self.audio_path}")
        except Exception as e:
            logger.warning(f"Could not clean up temporary files: {e}")

def main():
    """Main function to run the final enhanced demo."""
    demo = FinalEnhancedDemo()
    
    try:
        demo.run_demo()
    except KeyboardInterrupt:
        print("\nSession interrupted by user.")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
