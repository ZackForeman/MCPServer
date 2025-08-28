import asyncio
import configparser
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec
from llama_index.core.agent.workflow import (
    FunctionAgent, 
    ToolCallResult, 
    ToolCall)
from llama_index.core.workflow import Context

llm = Ollama(model="gpt-oss:20b", request_timeout=1200.0)
Settings.llm = llm

SYSTEM_PROMPT = """\
You are an AI assistant for creating and running scripts to execute user requests.
"""

async def get_agent(tools: McpToolSpec):
    tools = await tools.to_tool_list_async()
    agent = FunctionAgent(
        name="Agent",
        description="An agent that can work with Python Scripts.",
        tools=tools,
        llm=llm,
        system_prompt=SYSTEM_PROMPT,
    )
    return agent

async def handle_user_message(
    message_content: str,
    agent: FunctionAgent,
    agent_context: Context,
    verbose: bool = False,
):
    handler = agent.run(message_content, ctx=agent_context)
    #async for event in handler.stream_events():
    #    if verbose and type(event) == ToolCall:
    #        print(f"Calling tool {event.tool_name} with kwargs {event.tool_kwargs}")
    #    elif verbose and type(event) == ToolCallResult:
    #        print(f"Tool {event.tool_name} returned {event.tool_output}")

    response = await handler
    return str(response)

if __name__ == "__main__":
    mcp_client = BasicMCPClient("http://127.0.0.1:8000/sse")
    mcp_tool = McpToolSpec(client=mcp_client) 
    agent = asyncio.get_event_loop().run_until_complete(get_agent(mcp_tool))
    agent_context = Context(agent)

    while True:
        user_input = input("Enter your message: ")
        wrapped_input =  f"""write a python script to:
        {user_input}
        give ONLY THE CODE to the run script tool to execute the python code and add install commands for the dependencies as comments at the top of the file without extra text. Then, using the response from the tool, give a response to the user. 
"""
        if user_input == "exit":
            break
        # print("User: ", wrapped_input)
        response = asyncio.get_event_loop().run_until_complete(handle_user_message(wrapped_input, agent, agent_context, verbose=True))
        print("Agent: ", response)
