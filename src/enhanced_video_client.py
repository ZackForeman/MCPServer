import asyncio
import configparser
import os
import sys
from pathlib import Path
from typing import Dict
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec
from llama_index.core.agent.workflow import (
    FunctionAgent, 
    ToolCallResult, 
    ToolCall)
from llama_index.core.workflow import Context
from multimodal_model import MultimodalVideoProcessor
from video_processor import VideoProcessor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedVideoClient:
    """
    Enhanced video client that can process video input using multimodal models
    and integrate with the MCP server for script execution.
    """
    
    def __init__(self):
        """Initialize the enhanced video client."""
        self.llm = Ollama(model="gpt-oss:20b", request_timeout=1200.0)
        Settings.llm = self.llm
        
        # Initialize processors
        self.video_processor = VideoProcessor()
        self.multimodal_processor = MultimodalVideoProcessor()
        
        # MCP client setup
        self.mcp_client = BasicMCPClient("http://127.0.0.1:8000/sse")
        self.mcp_tool = McpToolSpec(client=self.mcp_client)
        
        self.system_prompt = """\
You are an advanced AI assistant that can process both text and video inputs using multimodal understanding. 
When given video input, you will receive:
1. A transcription of the spoken words
2. Visual context from the video frames
3. Advanced multimodal analysis using CLIP, VideoMAE, and other models
4. Combined analysis of the video content

Use this comprehensive information to understand the user's request and respond appropriately. 
You can also execute Python scripts using the available tools to fulfill user requests.
"""

    async def get_agent(self):
        """Get the AI agent with tools."""
        tools = await self.mcp_tool.to_tool_list_async()
        agent = FunctionAgent(
            name="EnhancedVideoAgent",
            description="An advanced agent that can work with both text and video inputs using multimodal models, and execute Python scripts.",
            tools=tools,
            llm=self.llm,
            system_prompt=self.system_prompt,
        )
        return agent

    async def handle_user_input(self, message_content: str, agent: FunctionAgent, agent_context: Context, verbose: bool = False):
        """Handle user input (text or video path)."""
        handler = agent.run(message_content, ctx=agent_context)
        response = await handler
        return str(response)

    def process_video_enhanced(self, video_path: str, audio_path: str = None, use_multimodal: bool = True) -> Dict:
        """
        Process video using enhanced multimodal processing.
        
        Args:
            video_path: Path to the video file
            audio_path: Optional path to separate audio file (WAV format)
            use_multimodal: Whether to use advanced multimodal models
            
        Returns:
            Dictionary containing enhanced video analysis
        """
        try:
            if not os.path.exists(video_path):
                return {"error": f"Video file not found at {video_path}"}
            
            logger.info(f"Processing video with enhanced analysis: {video_path}")
            if audio_path:
                logger.info(f"Using separate audio file: {audio_path}")
            
            if use_multimodal and self.multimodal_processor:
                # Use advanced multimodal processing
                result = self.multimodal_processor.process_video_multimodal(video_path)
                if result.get("success"):
                    return self._format_multimodal_result(result)
                else:
                    logger.warning("Multimodal processing failed, falling back to basic processing")
            
            # Fall back to basic video processing
            basic_result = self.video_processor.process_video(video_path, audio_path)
            return self._format_basic_result(basic_result)
            
        except Exception as e:
            logger.error(f"Error in enhanced video processing: {e}")
            return {"error": str(e)}

    def _format_multimodal_result(self, result: Dict) -> Dict:
        """Format multimodal processing result for AI consumption."""
        try:
            formatted = {
                "processing_type": "multimodal",
                "success": True,
                "command_text": self._generate_multimodal_command(result)
            }
            
            # Add detailed analysis
            if "frame_analysis" in result:
                formatted["frame_count"] = len(result["frame_analysis"])
            
            if "video_understanding" in result:
                formatted["key_insights"] = result["video_understanding"].get("key_insights", [])
                formatted["overall_analysis"] = result["video_understanding"].get("overall_analysis", "")
            
            if "multimodal_response" in result:
                formatted["ai_response"] = result["multimodal_response"]
            
            return formatted
            
        except Exception as e:
            logger.error(f"Error formatting multimodal result: {e}")
            return {"error": str(e)}

    def _format_basic_result(self, result: Dict) -> Dict:
        """Format basic video processing result."""
        try:
            if not result.get("success"):
                return {"error": result.get("error", "Unknown error")}
            
            formatted = {
                "processing_type": "basic",
                "success": True,
                "command_text": self._generate_basic_command(result)
            }
            
            # Add basic analysis
            if "audio_transcription" in result:
                formatted["transcription"] = result["audio_transcription"]
            
            if "visual_information" in result:
                formatted["visual_summary"] = result["visual_information"]
            
            return formatted
            
        except Exception as e:
            logger.error(f"Error formatting basic result: {e}")
            return {"error": str(e)}

    def _generate_multimodal_command(self, result: Dict) -> str:
        """Generate command text from multimodal analysis."""
        command = "VIDEO INPUT ANALYSIS (MULTIMODAL):\n"
        command += "=" * 50 + "\n\n"
        
        # Add frame analysis summary
        if "frame_analysis" in result:
            command += f"FRAMES ANALYZED: {len(result['frame_analysis'])}\n\n"
            
            # Add sample frame descriptions
            for i, frame in enumerate(result["frame_analysis"][:3]):  # First 3 frames
                if "caption" in frame:
                    command += f"Frame {i+1}: {frame['caption']}\n"
                elif "clip_analysis" in frame and "concepts" in frame["clip_analysis"]:
                    concepts = [c["concept"] for c in frame["clip_analysis"]["concepts"][:2]]
                    command += f"Frame {i+1}: Detected concepts: {', '.join(concepts)}\n"
        
        # Add video understanding
        if "video_understanding" in result:
            command += f"\nOVERALL ANALYSIS: {result['video_understanding'].get('overall_analysis', 'N/A')}\n"
            
            if "key_insights" in result["video_understanding"]:
                command += "\nKEY INSIGHTS:\n"
                for concept, count in result["video_understanding"]["key_insights"][:5]:
                    command += f"  - {concept}: {count} occurrences\n"
        
        # Add AI response if available
        if "multimodal_response" in result:
            command += f"\nAI INTERPRETATION:\n{result['multimodal_response']}\n"
        
        command += "\nINSTRUCTIONS: Please analyze this video input and respond to any spoken commands while considering the visual context provided."
        
        return command

    def _generate_basic_command(self, result: Dict) -> str:
        """Generate command text from basic analysis."""
        command = "VIDEO INPUT ANALYSIS (BASIC):\n"
        command += "=" * 50 + "\n\n"
        
        if "audio_transcription" in result:
            command += f"SPOKEN COMMAND: {result['audio_transcription']}\n\n"
        
        if "visual_information" in result:
            command += f"VISUAL CONTEXT: {result['visual_information']}\n\n"
        
        command += "INSTRUCTIONS: Please analyze this video input and respond to the spoken command while considering the visual context provided."
        
        return command

    def get_supported_video_formats(self) -> list:
        """Get list of supported video formats."""
        return ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v']

    def is_video_file(self, file_path: str) -> bool:
        """Check if the given file is a supported video format."""
        return Path(file_path).suffix.lower() in self.get_supported_video_formats()

    def get_processing_status(self) -> Dict:
        """Get status of all processing capabilities."""
        status = {
            "video_processor": True,
            "multimodal_processor": self.multimodal_processor is not None,
            "supported_formats": self.get_supported_video_formats()
        }
        
        if self.multimodal_processor:
            status.update(self.multimodal_processor.get_model_status())
        
        return status

    async def run_interactive_session(self):
        """Run the interactive session with enhanced video support."""
        try:
            agent = await self.get_agent()
            agent_context = Context(agent)
            
            # Display status
            status = self.get_processing_status()
            
            print("=== Enhanced Video-Enabled AI Assistant ===")
            print("Processing Capabilities:")
            print(f"  - Basic video processing: {'✓' if status['video_processor'] else '✗'}")
            print(f"  - Multimodal processing: {'✓' if status['multimodal_processor'] else '✗'}")
            print(f"  - Device: {status.get('device', 'Unknown')}")
            print(f"  - Supported formats: {', '.join(status['supported_formats'])}")
            print("\nCommands:")
            print("  - Enter text directly for text-based requests")
            print("  - Enter video file path to process video input")
            print("  - Type 'status' to see processing capabilities")
            print("  - Type 'exit' to quit")
            print("=" * 50)
            
            while True:
                user_input = input("\nEnter your message or video file path: ").strip()
                
                if user_input.lower() == "exit":
                    print("Goodbye!")
                    break
                
                if user_input.lower() == "status":
                    self._display_status(status)
                    continue
                
                if not user_input:
                    continue
                
                # Check if input is a video file
                if self.is_video_file(user_input):
                    await self._handle_video_input(user_input, agent, agent_context)
                else:
                    await self._handle_text_input(user_input, agent, agent_context)
                        
        except Exception as e:
            print(f"Error in interactive session: {e}")

    def _display_status(self, status: Dict):
        """Display processing capabilities status."""
        print("\n" + "=" * 50)
        print("PROCESSING CAPABILITIES STATUS")
        print("=" * 50)
        
        print(f"Basic Video Processing: {'✓' if status['video_processor'] else '✗'}")
        print(f"Multimodal Processing: {'✓' if status['multimodal_processor'] else '✗'}")
        
        if status['multimodal_processor']:
            print(f"\nMultimodal Models:")
            print(f"  - CLIP Model: {'✓' if status.get('clip_model') else '✗'}")
            print(f"  - VideoMAE Model: {'✓' if status.get('videomae_model') else '✗'}")
            print(f"  - Captioning Model: {'✓' if status.get('captioning_model') else '✗'}")
            print(f"  - Text Generation: {'✓' if status.get('text_model') else '✗'}")
        
        print(f"\nDevice: {status.get('device', 'Unknown')}")
        print(f"Supported Formats: {', '.join(status['supported_formats'])}")
        print("=" * 50)

    async def _handle_video_input(self, video_path: str, agent: FunctionAgent, agent_context: Context):
        """Handle video input processing."""
        print(f"\nProcessing video: {video_path}")
        print("This may take a moment...")
        
        # Process video with enhanced analysis
        result = self.process_video_enhanced(video_path, use_multimodal=True)
        
        if "error" in result:
            print(f"Video processing failed: {result['error']}")
            return
        
        print(f"\nVideo analysis complete! ({result['processing_type']} processing)")
        print("=" * 50)
        print(result['command_text'])
        print("=" * 50)
        
        # Send to AI agent
        wrapped_input = f"""write a python script to:
        {result['command_text']}
        give ONLY THE CODE to the run script tool to execute the python code and add install commands for the dependencies as comments at the top of the file without extra text. Then, using the response from the tool, give a response to the user.
        """
        
        try:
            response = await self.handle_user_input(wrapped_input, agent, agent_context, verbose=True)
            print("\nAgent Response:")
            print(response)
        except Exception as e:
            print(f"Error getting agent response: {e}")

    async def _handle_text_input(self, text_input: str, agent: FunctionAgent, agent_context: Context):
        """Handle text input processing."""
        wrapped_input = f"""write a python script to:
        {text_input}
        give ONLY THE CODE to the run script tool to execute the python code and add install commands for the dependencies as comments at the top of the file without extra text. Then, using the response from the tool, give a response to the user.
        """
        
        try:
            response = await self.handle_user_input(wrapped_input, agent, agent_context, verbose=True)
            print("\nAgent Response:")
            print(response)
        except Exception as e:
            print(f"Error getting agent response: {e}")

if __name__ == "__main__":
    client = EnhancedVideoClient()
    
    try:
        asyncio.run(client.run_interactive_session())
    except KeyboardInterrupt:
        print("\nSession interrupted by user.")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)
