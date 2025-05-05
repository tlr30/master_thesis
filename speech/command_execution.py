"""
AGV Command Execution Script

This script processes natural language commands to control an Automated Guided Vehicle (AGV).
It normalizes the command to extract the intent (e.g., "go to" or "return to") and the target
(e.g., colour and object). Based on the extracted intent and target, it executes A* search to find
the shortest path from its current position to the target object.

Usage:
    python script.py "go to the red block"

Dependencies:
    - argparse: For parsing command-line arguments.
    - subprocess: For executing external scripts.

Functions:
    normalize_command(command):
        - Normalizes a natural language command to extract intent, colour, and object.

    execute_agv_command(intent, colour, obj):
        - Executes the AGV command based on the extracted intent, colour, and object.
"""
import argparse
import subprocess

def normalize_command(command):
    """
    Normalizes a natural language command to extract intent, colour, and object.

    Args:
        command (str): The natural language command to normalize.

    Returns:
        tuple: A tuple containing:
            - intent (str): The extracted intent ("go_to" or "return_to").
            - colour (str): The extracted colour (e.g., "red", "blue").
            - obj (str): The extracted object (e.g., "block", "pallet").
    """

    ## german and english intents and keywords
    ## define keywords for intents
    go_keywords = ["go to", "move to", "navigate to", "head to", "geh", "fahr", "bewege"]
    return_keywords = ["return to", "go back to", "come back to", "geh zurück", "fahr zurück"]

    ## define target keywords
    colours = ["blue", "red", "green", "yellow", "blau", "rot", "grün", "gelb"]
    objects = ["block", "box", "pallet", "palette"]

    ## normalize the command to lowercase
    command = command.lower()

    ## determine intent
    intent = None
    if any(keyword in command for keyword in go_keywords):
        intent = "go_to"
    elif any(keyword in command for keyword in return_keywords):
        intent = "return_to"

    ## extract colour and object from the command
    colour = None
    obj = None
    for c in colours:
        if c in command:
            colour = c
            break

    for o in objects:
        if o in command:
            obj = o
            break

    return intent, colour, obj

def execute_agv_command(intent, colour, obj):
    """
    Executes the AGV command based on the extracted intent, colour, and object.

    Args:
        intent (str): The intent extracted from the command ("go_to" or "return_to").
        colour (str): The colour of the target object (e.g., "red", "blue").
        obj (str): The type of object (e.g., "block", "cube").

    Returns:
        None
    """
    ## validate the extracted intent, colour, and object
    if not intent or not colour or not obj:
        print("Could not understand the command.")
        return

    ## handle the intent
    if intent == "go_to":
        print(f"AGV is moving to the {colour} {obj}.")

    elif intent == "return_to":
        print(f"AGV is returning to the {colour} {obj}.")
        
    else:
        print("Unknown intent.")
        
    ## execute the external A* search script that finds the shortest path to the target colour
    subprocess.run(["python", "ML_navigation/Astar.py", "--target_colour", colour, "--target_object", obj])

if __name__ == "__main__":

    import sys


    ## set up argument parsing
    parser = argparse.ArgumentParser(description="Execute AGV commands based on voice input.")
    parser.add_argument("command", type=str, help="The command to execute.")

    ## parse the arguments
    args = parser.parse_args()

    ## normalize and execute the command
    intent, colour, obj = normalize_command(args.command)
    execute_agv_command(intent, colour, obj)