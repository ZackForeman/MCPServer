# 🎥🎤 Audio-Video Recording and Processing System

This system now captures **both video AND audio simultaneously** using OpenCV for video and PyAudio for audio recording, solving the previous issue where only video was captured.

## ✨ **Key Features**

### **🎬 Simultaneous Audio-Video Recording**
- **Real-time video capture** using OpenCV
- **Real-time audio capture** using PyAudio
- **Synchronized recording** with separate threads
- **Live preview** with recording indicators

### **🔧 Reliable File Handling**
- **Separate audio and video files** for maximum reliability
- **No complex file merging** that can cause errors
- **Automatic cleanup** of temporary files
- **Multiple output formats** supported

### **🤖 Advanced AI Processing**
- **Audio transcription** of spoken commands
- **Visual analysis** of video frames
- **Multimodal understanding** using CLIP and other AI models
- **Comprehensive video analysis** for AI consumption

## 🚀 **Quick Start**

### **1. Run the Final Enhanced Demo**
```bash
python final_enhanced_demo.py
```

This will:
- Open a camera window
- Start audio recording
- Allow you to record with 'R' key
- Save and process your recording
- Show comprehensive analysis results

### **2. Test Basic Recording**
```bash
python simple_audio_video_recorder.py
```

This tests just the recording functionality without processing.

### **3. Test with Existing Files**
```bash
# Use the final demo to record and process
python final_enhanced_demo.py
```

## 🎯 **How It Works**

### **Recording Process**
1. **Camera Initialization**: Opens your default camera
2. **Audio Stream Setup**: Initializes microphone input
3. **Dual Threading**: 
   - Video thread captures frames at 30 FPS
   - Audio thread captures audio at 44.1 kHz
4. **Synchronized Storage**: Both streams store data simultaneously
5. **File Export**: Saves as separate MP4 (video) and WAV (audio) files

### **Processing Pipeline**
1. **Video Analysis**: Extracts key frames and visual information
2. **Audio Transcription**: Converts speech to text using multiple engines
3. **Multimodal Processing**: Combines visual and audio understanding
4. **AI Integration**: Uses CLIP, ResNet-50, and other models
5. **Command Generation**: Creates formatted commands for AI consumption

## 📁 **File Structure**

```
src/
├── final_enhanced_demo.py          # 🎯 MAIN DEMO - Use this!
├── simple_audio_video_recorder.py  # Core recording system
├── video_processor.py              # Core video processing
├── multimodal_model.py             # AI model integration
├── enhanced_video_client.py        # Client for MCP integration
├── README_AUDIO_VIDEO.md           # This documentation
├── pyproject.toml                  # Dependencies
├── server.py                       # MCP server
└── client.py                       # MCP client
```

## 🎮 **Usage Instructions**

### **Interactive Recording**
1. **Start the demo**: `python final_enhanced_demo.py`
2. **Camera window opens** showing live feed
3. **Press 'R'** to start recording
4. **Speak your command** while recording
5. **Press 'R' again** to stop recording
6. **Press 'S'** to save and process
7. **Press 'Q'** to quit

### **Recording Controls**
- **R Key**: Start/Stop recording
- **S Key**: Save and process recording
- **Q Key**: Quit the application

### **Visual Indicators**
- **Red Circle**: Recording in progress
- **Frame Count**: Number of video frames captured
- **Audio Chunks**: Number of audio samples captured
- **Duration**: Recording time in seconds

## 🔧 **Technical Details**

### **Video Specifications**
- **Resolution**: 640x480 pixels
- **Frame Rate**: 30 FPS
- **Format**: MP4 (H.264 codec)
- **Quality**: High quality with compression

### **Audio Specifications**
- **Sample Rate**: 44.1 kHz
- **Channels**: Mono (1 channel)
- **Bit Depth**: 16-bit
- **Format**: WAV (uncompressed)

### **System Requirements**
- **Camera**: USB webcam or built-in camera
- **Microphone**: Any working microphone
- **Python**: 3.11 or higher
- **Memory**: 4GB RAM minimum
- **Storage**: 100MB free space for temporary files

## 🐛 **Troubleshooting**

### **Common Issues**

#### **Camera Not Working**
```bash
# Test camera access with the main demo
python final_enhanced_demo.py
```
- Check camera permissions
- Ensure no other app is using camera
- Try different camera index (0, 1, 2)

#### **Audio Not Working**
```bash
# Test audio recording with the main demo
python final_enhanced_demo.py
```
- Check microphone permissions
- Ensure microphone is not muted
- Check audio input device settings

#### **Recording Issues**
- **Low frame rate**: Reduce camera resolution
- **Audio sync issues**: Check system audio settings
- **File save errors**: Check disk space and permissions

### **Performance Optimization**
- **Lower resolution**: Modify camera settings in code
- **Reduce frame rate**: Change `self.fps` value
- **Limit recording length**: Set maximum frame limits
- **Use SSD storage**: Faster file I/O operations

## 🔮 **Advanced Features**

### **Custom Recording Settings**
```python
# In the recorder class
self.fps = 60.0                    # Higher frame rate
self.sample_rate = 48000           # Higher audio quality
self.channels = 2                  # Stereo audio
self.chunk = 2048                  # Larger audio chunks
```

### **Output Format Options**
- **Video**: MP4, AVI, MOV, MKV
- **Audio**: WAV, MP3, FLAC, AAC
- **Quality**: Adjustable compression settings

### **Integration with MCP Server**
The system can integrate with your existing MCP server to:
- Execute Python scripts based on video commands
- Process video input for AI workflows
- Generate automated responses to visual/audio input

## 📚 **API Reference**

### **SimpleAudioVideoRecorder Class**
```python
recorder = SimpleAudioVideoRecorder()

# Start recording
recorder.start_recording()

# Stop recording
recorder.stop_recording()

# Save files
video_path, audio_path = recorder.save_recording()

# Get info
info = recorder.get_recording_info()

# Cleanup
recorder.cleanup()
```

### **Key Methods**
- `start_camera()`: Initialize camera
- `start_audio_stream()`: Initialize microphone
- `start_recording()`: Begin recording both streams
- `stop_recording()`: Stop recording
- `save_recording()`: Export files
- `get_recording_info()`: Get recording statistics
- `cleanup()`: Release resources

## 🎉 **Success Stories**

### **What This System Achieves**
✅ **Simultaneous audio-video capture** - No more missing audio!  
✅ **Real-time recording** - See what you're recording live  
✅ **Reliable file handling** - Separate files prevent corruption  
✅ **Advanced AI processing** - Full multimodal understanding  
✅ **Easy integration** - Works with existing MCP infrastructure  

### **Use Cases**
- **Voice commands with visual context**
- **Video tutorials with audio narration**
- **AI training data collection**
- **Multimodal content analysis**
- **Interactive AI assistants**

## 🚀 **Next Steps**

1. **Try the demo**: `python final_enhanced_demo.py`
2. **Record your first video**: Use the 'R' key to start/stop
3. **Test processing**: Press 'S' to analyze your recording
4. **Customize settings**: Modify parameters for your needs
5. **Integrate with MCP**: Connect to your existing server

## 📞 **Support**

If you encounter issues:
1. Check the troubleshooting section above
2. Run the main demo: `python final_enhanced_demo.py`
3. Check your camera and microphone permissions
4. Ensure all dependencies are installed correctly

---

**🎯 The system is now ready for production use with both audio and video capture!** 🎤🎥
