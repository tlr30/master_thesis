"""
Open WebUI Tool: VS Code Script Executor

This module defines a custom tool intended for integration with Open WebUI. It provides a method
to remotely execute Python or shell commands in a VS Code environment via a Flask server
(e.g., inside a Docker container).

It streams the output live to the WebUI interface using an asynchronous event emitter.

🔧 To implement this tool in Open WebUI:
1. Add this script to the `tools/` directory of your WebUI instance.
2. Register `Tools` as an available tool in the system configuration.
3. Ensure the corresponding Flask server is running and listening at HOST_SERVER_URL.

Requirements:
- A Flask backend at `http://host.docker.internal:5000` with `/run-script` endpoint.
- Open WebUI environment with support for async tools and streaming.

Author: Tim Riekeles
Date: 2025-06-05
"""
import json
import aiohttp
from typing import Callable, Awaitable, Any
from pydantic import BaseModel, Field


class Tools:
    """
    A utility class for executing terminal commands on a VS Code backend via an HTTP server.

    This is intended to be used as a **custom tool in Open WebUI**, allowing Python scripts or shell
    commands to be executed in a controlled development environment, such as VS Code running inside
    a Docker container.

    The tool streams live output back to the WebUI via an event emitter interface.
    """
    class UserValves(BaseModel):
        """
        Configuration model for host server communication.

        Attributes:
            HOST_SERVER_URL (str): The base URL of the HTTP server on the host machine.
        """
        HOST_SERVER_URL: str = Field(
            default="http://host.docker.internal:5000",
            description="URL of the HTTP server running on the host machine.",
        )

    def __init__(self):
        self.valves = self.UserValves()

    async def run_command_in_vscode(
        self,
        command: str,
        __event_emitter__: Callable[[Any], Awaitable[None]] = None,
    ) -> str:
        """
        Sends a command to a Flask-based script executor running in VS Code and streams output.

        This function is meant to be used from Open WebUI as a tool. It posts a command to an HTTP
        endpoint and listens for streaming output from the script, which can be streamed line-by-line
        to the user via the provided `__event_emitter__`.

        Args:
            command (str): The shell command or Python command to run on the host server.
            __event_emitter__ (Callable): Optional async callback for real-time output streaming
                                          to the WebUI.

        Returns:
            str: A JSON string containing either the full output or an error message.
        """
        try:
            async with aiohttp.ClientSession() as session:
                ## send command to the host server script runner
                async with session.post(
                    f"{self.valves.HOST_SERVER_URL}/run-script",
                    json={"command": command},
                ) as resp:
                    output = ""
                    
                    ## stream output line by line as it's received from the Flask server
                    async for line_bytes in resp.content:
                        line = line_bytes.decode("utf-8").strip()
                        output += line + "\n"

                        ## send the line back to the UI (if emitter provided)
                        if __event_emitter__:
                            await __event_emitter__(
                                {
                                    "type": "status",
                                    "data": {"line": line},
                                }
                            )

                    ## notify WebUI that script execution has completed
                    if __event_emitter__:
                        await __event_emitter__(
                            {
                                "type": "status",
                                "data": {
                                    "description": "✅ Script finished.",
                                    "done": True,
                                },
                            }
                        )

                    return json.dumps({"status": "OK", "output": output})

        except Exception as e:
            ## handle any connection or execution errors
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"❌ Error: {str(e)}",
                            "done": True,
                        },
                    }
                )

            return json.dumps(
                {
                    "status": "ERROR",
                    "output": str(e),
                }
            )
