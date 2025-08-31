#!/usr/bin/env python3
"""
Simple Audio-Video Recording System
Captures both video and audio simultaneously and saves them separately.
"""

import cv2
import numpy as np
import pyaudio
import wave
import threading
import time
import tempfile
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleAudioVideoRecorder:
    def __init__(self):
        """Initialize the simple audio-video recorder."""
        self.video_cap = None
        self.audio_stream = None
        self.audio_frames = []
        self.video_frames = []
        self.recording = False
        self.fps = 30.0
        self.sample_rate = 44100
        self.channels = 1
        self.chunk = 1024
        self.audio_format = pyaudio.paInt16
        
        # Audio recording thread
        self.audio_thread = None
        self.audio_stop_event = threading.Event()
        
        # Video recording thread
        self.video_thread = None
        self.video_stop_event = threading.Event()
        
        # Recording paths
        self.video_path = None
        self.audio_path = None
        
    def start_camera(self):
        """Start the camera capture."""
        try:
            self.video_cap = cv2.VideoCapture(0)
            if not self.video_cap.isOpened():
                logger.error("Could not open camera")
                return False
            
            # Set camera properties
            self.video_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.video_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.video_cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            logger.info("Camera started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error starting camera: {e}")
            return False
    
    def start_audio_stream(self):
        """Start the audio stream."""
        try:
            self.audio = pyaudio.PyAudio()
            self.audio_stream = self.audio.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk
            )
            
            logger.info("Audio stream started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error starting audio stream: {e}")
            return False
    
    def record_audio(self):
        """Record audio in a separate thread."""
        try:
            while not self.audio_stop_event.is_set():
                try:
                    data = self.audio_stream.read(self.chunk, exception_on_overflow=False)
                    self.audio_frames.append(data)
                except Exception as e:
                    logger.warning(f"Audio read error: {e}")
                    break
        except Exception as e:
            logger.error(f"Audio recording error: {e}")
    
    def record_video(self):
        """Record video in a separate thread."""
        try:
            while not self.video_stop_event.is_set():
                ret, frame = self.video_cap.read()
                if not ret:
                    break
                
                # Store frame
                self.video_frames.append(frame.copy())
                
                # Control frame rate
                time.sleep(1.0 / self.fps)
                
        except Exception as e:
            logger.error(f"Video recording error: {e}")
    
    def start_recording(self):
        """Start recording both audio and video."""
        if self.recording:
            logger.warning("Recording already in progress")
            return False
        
        # Clear previous recordings
        self.audio_frames = []
        self.video_frames = []
        
        # Reset stop events
        self.audio_stop_event.clear()
        self.video_stop_event.clear()
        
        # Start audio recording thread
        self.audio_thread = threading.Thread(target=self.record_audio)
        self.audio_thread.start()
        
        # Start video recording thread
        self.video_thread = threading.Thread(target=self.record_video)
        self.video_thread.start()
        
        self.recording = True
        logger.info("Recording started")
        return True
    
    def stop_recording(self):
        """Stop recording both audio and video."""
        if not self.recording:
            logger.warning("No recording in progress")
            return False
        
        # Signal threads to stop
        self.audio_stop_event.set()
        self.video_stop_event.set()
        
        # Wait for threads to finish
        if self.audio_thread:
            self.audio_thread.join(timeout=2.0)
        if self.video_thread:
            self.video_thread.join(timeout=2.0)
        
        self.recording = False
        logger.info("Recording stopped")
        return True
    
    def save_recording(self, output_dir=None):
        """Save the recorded audio and video to separate files."""
        if not self.audio_frames or not self.video_frames:
            logger.error("No audio or video frames to save")
            return None, None
        
        try:
            # Create output directory
            if output_dir:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
            else:
                # Create temporary directory
                temp_dir = tempfile.mkdtemp()
                output_path = Path(temp_dir)
            
            # Save audio as WAV
            audio_path = output_path / "recording_audio.wav"
            self._save_audio(str(audio_path))
            self.audio_path = str(audio_path)
            
            # Save video as MP4
            video_path = output_path / "recording_video.mp4"
            self._save_video(str(video_path))
            self.video_path = str(video_path)
            
            logger.info(f"Audio saved to: {audio_path}")
            logger.info(f"Video saved to: {video_path}")
            
            return str(video_path), str(audio_path)
            
        except Exception as e:
            logger.error(f"Error saving recording: {e}")
            return None, None
    
    def _save_audio(self, audio_path):
        """Save audio frames to WAV file."""
        try:
            with wave.open(audio_path, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.audio.get_sample_size(self.audio_format))
                wf.setframerate(self.sample_rate)
                wf.writeframes(b''.join(self.audio_frames))
            
            logger.info(f"Audio saved to: {audio_path}")
            
        except Exception as e:
            logger.error(f"Error saving audio: {e}")
            raise
    
    def _save_video(self, video_path):
        """Save video frames to MP4 file."""
        try:
            if not self.video_frames:
                raise ValueError("No video frames to save")
            
            # Get frame dimensions
            height, width = self.video_frames[0].shape[:2]
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, self.fps, (width, height))
            
            # Write frames
            for frame in self.video_frames:
                out.write(frame)
            
            out.release()
            logger.info(f"Video saved to: {video_path}")
            
        except Exception as e:
            logger.error(f"Error saving video: {e}")
            raise
    
    def get_recording_info(self):
        """Get information about the current recording."""
        info = {
            "recording": self.recording,
            "video_frames": len(self.video_frames),
            "audio_frames": len(self.audio_frames),
            "video_duration": len(self.video_frames) / self.fps if self.video_frames else 0,
            "audio_duration": len(self.audio_frames) * self.chunk / self.sample_rate if self.audio_frames else 0
        }
        return info
    
    def cleanup(self):
        """Clean up resources."""
        try:
            # Stop recording if active
            if self.recording:
                self.stop_recording()
            
            # Close audio stream
            if self.audio_stream:
                self.audio_stream.stop_stream()
                self.audio_stream.close()
            
            # Close audio
            if hasattr(self, 'audio'):
                self.audio.terminate()
            
            # Release camera
            if self.video_cap:
                self.video_cap.release()
            
            # Close all OpenCV windows
            cv2.destroyAllWindows()
            
            logger.info("Cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

def main():
    """Test the simple audio-video recorder."""
    recorder = SimpleAudioVideoRecorder()
    
    try:
        print("=== Simple Audio-Video Recorder Test ===")
        
        # Start camera
        if not recorder.start_camera():
            print("❌ Failed to start camera")
            return
        
        # Start audio stream
        if not recorder.start_audio_stream():
            print("❌ Failed to start audio stream")
            return
        
        print("✅ Camera and audio ready")
        print("Press 'R' to start/stop recording")
        print("Press 'S' to save recording")
        print("Press 'Q' to quit")
        
        # Main loop
        while True:
            # Display camera feed
            ret, frame = recorder.video_cap.read()
            if not ret:
                break
            
            # Add recording indicator
            if recorder.recording:
                cv2.circle(frame, (50, 50), 20, (0, 0, 255), -1)
                cv2.putText(frame, "REC", (70, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Show recording info
                info = recorder.get_recording_info()
                cv2.putText(frame, f"Video: {info['video_frames']} frames", (10, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"Audio: {info['audio_frames']} chunks", (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Add instructions
            cv2.putText(frame, "R: Record/Stop, S: Save, Q: Quit", (10, 450), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show frame
            cv2.imshow('Simple Audio-Video Recorder', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                break
            elif key == ord('r') or key == ord('R'):
                if not recorder.recording:
                    recorder.start_recording()
                    print("Recording started...")
                else:
                    recorder.stop_recording()
                    print("Recording stopped...")
            elif key == ord('s') or key == ord('S'):
                if recorder.audio_frames and recorder.video_frames:
                    video_path, audio_path = recorder.save_recording()
                    if video_path and audio_path:
                        print(f"Video saved to: {video_path}")
                        print(f"Audio saved to: {audio_path}")
                    else:
                        print("Failed to save recording")
                else:
                    print("No recording to save")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        recorder.cleanup()
        print("Recorder cleaned up")

if __name__ == "__main__":
    main()
