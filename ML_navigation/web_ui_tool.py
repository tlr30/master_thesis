"""
HTTP Client for Running Commands on a Host Server

This module provides a class `Tools` that interacts with an HTTP server running on the host
machine.
It sends commands to the server and handles responses, including error handling and status updates.

Classes:
    Tools:
        - Manages communication with the host server.
        - Provides a method to send commands and handle responses.

Dependencies:
    - json: For encoding and decoding JSON data.
    - requests: For making HTTP requests.
    - pydantic: For defining and validating configuration models.
    - typing: For type hints.
"""

import json
import requests
from typing import Callable, Awaitable, Any
from pydantic import BaseModel, Field


class Tools:
    """
    A class to interact with an HTTP server running on the host machine.

    Attributes:
        valves (UserValves): Configuration for the host server URL.

    Methods:
        run_command_in_vscode(command, __event_emitter__):
            Sends a command to the host server and handles the response.
    """

    class UserValves(BaseModel):
        """
        Configuration model for the host server URL.

        Attributes:
            HOST_SERVER_URL (str): The URL of the HTTP server running on the host machine.
                                  Defaults to "http://host.docker.internal:5000".
        """

        HOST_SERVER_URL: str = Field(
            default="http://host.docker.internal:5000",
            description="The URL of the HTTP server running on the host machine.",
        )

    def __init__(self):
        self.valves = self.UserValves()

    async def run_command_in_vscode(
        self,
        command: str,
        __event_emitter__: Callable[[Any], Awaitable[None]] = None,
    ) -> str:
        """
        Sends a command to the HTTP server on the host machine and handles the response.

        Args:
            command (str): The command to pass to the Python script.
            __event_emitter__ (Callable[[Any], Awaitable[None]]): Optional event emitter for
                OpenWebUI status updates. Defaults to None.

        Returns:
            str: A JSON-encoded string containing the status and output of the command execution.

        Raises:
            RuntimeError: If the server returns a non-200 status code.
        """
        try:
            ## send a POST request to the HTTP server
            response = requests.post(
                f"{self.valves.HOST_SERVER_URL}/run-script",
                json={"command": command},
            )

            ## handle non-200 status codes
            if response.status_code != 200:
                raise RuntimeError(f"Server returned error: {response.text}")

            ## emit a status update if an event emitter is provided
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"Executed on host: {command}",
                            "done": True,
                        },
                    }
                )

            ## return the server's response as a JSON-encoded string
            return json.dumps(response.json())

        except Exception as e:
            ## emit an error status update if an event emitter is provided
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"Error: {str(e)}",
                            "done": True,
                        },
                    }
                )

            ## return an error response as a JSON-encoded string
            return json.dumps(
                {
                    "status": "ERROR",
                    "output": str(e),
                }
            )