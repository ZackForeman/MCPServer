from fastmcp import FastMCP
import subprocess
import os
import configparser

mcp = FastMCP("Run Script", dependencies=["subprocess", "os"])

config = configparser.ConfigParser()
wd = os.path.dirname(__file__)

mcp_environment = os.path.join(wd, "..\\MCPEnvironment")
config.read(os.path.join(wd, "..\\config.ini"))

uv_install = config.get('SERVER', 'UV_INSTALL') # "C:\Users\zackf\.local\bin\uv.exe"


@mcp.tool()
def run_script(python_code: str):
    try:
        if (not uv_install):
            return "uv install not detected, add to config.ini file"

        subprocess.check_output([
            "powershell.exe", 
            "-Command", 
            f"Set-Location {mcp_environment}; {uv_install} venv"
            ], shell=True)


        for line in python_code.splitlines():
            line = line.strip()
            if line.startswith("#"):
                import_proc = subprocess.run([
                    "powershell.exe",
                    "-Command",
                    f"Set-Location {mcp_environment}; .venv\\Scripts\\activate; {uv_install} {line[2:]}"
                    ], shell=True)

        
        with open(mcp_environment + "\\script.py", "w") as f:
            f.write(python_code)
            f.flush()
        
        output = subprocess.check_output([
            "powershell.exe", 
            "-Command", 
            f"Set-Location {mcp_environment}; .venv\\Scripts\\activate; {uv_install} run script.py"
            ]) 
        
        if not output: 
            output = "Tool executed successfully. Tool does not need to be run again"
        
        return output
    except Exception as e:
        return f"Error running tool: {str(e)}"

if __name__ == "__main__":
    print("Starting Server ... ")
    mcp.run(transport="sse", host="127.0.0.1", port=8000)

