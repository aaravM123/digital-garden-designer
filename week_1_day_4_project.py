readme = """
# 🌿 GPT-4o Digital Garden Designer

This intelligent agent uses OpenAI’s function calling to help users design personalized gardens based on any natural language prompt. It extracts preferences, recommends plants, builds a layout, and generates a shopping list — all dynamically through tool-based AI.

## 🧠 What It Can Do
- Parse your goals into structured garden preferences
- Suggest ideal plants based on zone, sun, and maintenance
- Create a themed garden layout
- Generate a care-friendly plant shopping list

## 💡 Example Prompts
- “I want a colorful, low-maintenance garden in Austin.”
- “Design a peaceful shaded garden for my backyard in Seattle.”
- “I’d like a tropical setup that attracts butterflies in Zone 11a.”

## 🚀 Features
- 🔧 Auto-chained GPT-4o function calls
- 🧱 Modular tools for goal parsing, plant selection, and layout
- 💬 GPT streaming responses with memory of past turns
- ✅ Works with any custom garden description

## 🛠️ Run Instructions
```bash
pip install openai
"""
with open("README.md", "w") as f:
  f.write(readme)
!cat README.md

!pip install --quiet openai
import openai
import json
from getpass import getpass


api_key = getpass("Enter you OpenAI API Key: ")
client = openai.OpenAI(api_key = api_key)

# Implement Memory Log To Store Past Prompts and Response
memory_log = []

# Formatted Message History For GPT Based on memory_log
def build_message_history(memory_log):
  messages = []
  for turn in memory_log:
    messages.append({"role":"user", "content": turn["prompt"]})
    messages.append({"role": "assistant", "content": turn["response"]})
  return messages

# Python Implementation for Each Tool
def parse_garden_goals(description):
  """Uses GPT to extract the goals from the user's description"""
  system_prompt = "You are a garden design assistant. Extract the user's preferences and return them as a structured dictionary with keys: theme, sun, maintenance, location, and USDA zone (zone). Please return your output as a valid JSON object with keys: theme, sun, maintenance, location, and zone. Only return JSON."

  response = client.chat.completions.create(
      model = "gpt-4o",
      messages = [
          {"role": "system", "content": system_prompt},
          {"role": "user", "content": description}
      ]
  )

  content = response.choices[0].message.content

  try:
    parsed = json.loads(content)

  except:
    parsed = {"error": "Could not parse GPT output", "raw_output": content}

  return parsed


def suggest_plants(zone, maintenance, sun):
  prompt = f"""
  Suggest 4 plants that thrive in USDA Zone {zone}, with {maintenance} maintenance and {sun} sun exposure.
  Return only a Python list of plant names like:
  ["Lavender", "Aganpthus", "Succulent", "Rosemary"]
  """
  response = client.chat.completions.create(
      model = "gpt-4o",
      messages = [{"role": "user", "content": prompt}]
  )

  try:
    return json.loads(response.choices[0].message.content)
  except:
    return ["Plant 1", "Plant 2", "Plant 3", "Plant 4"]



def design_layout(plants, theme):
  layout = f"🌿 {theme.title()} Garden Layout:\n"
  sections = ["Front", "Middle", "Back", "Accent", "Corner", "Path Border"]

  for i, plant in enumerate(plants):
    section = sections[i % len(sections)]
    layout += f"- {section}: {plant}\n"

  return layout


def generate_shopping_list(plants):
  prompt = f"Give a shopping list with quantity and care notes for: {', '.join(plants)}."
  response = client.chat.completions.create(
      model = "gpt-4o",
      messages = [{"role": "user", "content": prompt}]
  )

  try:
    return json.loads(response.choices[0].message.content)
  except:
    return {plant: {"Quantity": 3, "Care Notes": "Generic care."} for plant in plants}

# Define Tool Schemas for GPT Function Calling
functions = [
    {
        "name": "parse_garden_goals",
        "description": "Parses a garden description into structured goals.",
        "parameters": {
            "type": "object",
            "properties": {
                "description": {"type": "string"}
            },
            "required": ["description"]
        }
    },
    {
        "name": "suggest_plants",
        "description": "Suggests plants for a zone and maintenance level.",
        "parameters": {
            "type": "object",
            "properties": {
                "zone": {"type": "string"},
                "maintenance": {"type": "string"},
                "sun": {"type": "string"}
            },
            "required": ["zone", "maintenance", "sun"]
        }
    },
    {
        "name": "design_layout",
        "description": "Creates a garden layout using selected plants and theme.",
        "parameters": {
            "type": "object",
            "properties": {
                "plants": {"type": "array", "items": {"type": "string"}},
                "theme": {"type": "string"}
            },
            "required": ["plants", "theme"]
        }
    },
    {
        "name": "generate_shopping_list",
        "description": "Generates a plant shopping list.",
        "parameters": {
            "type": "object",
            "properties": {
                "plants": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["plants"]
        }
    }
]

# Stream GPT + Tool Execution + Memory Saving
def stream_with_memory(prompt):
  messages = build_message_history(memory_log)
  messages.append({"role": "user", "content": prompt})

  # First GPT Call to decide What Function To Call
  response = client.chat.completions.create(
      model = "gpt-4o",
      messages = messages,
      functions = functions,
      function_call = "auto",
  )

  message = response.choices[0].message
  full_response = ""

  if message.function_call:
    function_name = message.function_call.name
    arguments = json.loads(message.function_call.arguments)

    if function_name == "parse_garden_goals":
      result = parse_garden_goals(**arguments)

    elif function_name == "suggest_plants":
      result = suggest_plants(**arguments)

    elif function_name == "design_layout":
      result = design_layout(**arguments)

    elif function_name == "generate_shopping_list":
      result = generate_shopping_list(**arguments)

    else:
      result = {"error": "Unknown function"}


  # Send Result Back to GPT
  second_response = client.chat.completions.create(
      model = "gpt-4o",
      messages = [
          *messages,
          message.model_dump(),
          {
              "role": "function",
              "name": function_name,
              "content": json.dumps(result)
          }
      ],
      stream = True
  )

  for chunk in second_response:
    if chunk.choices[0].delta.content:
      word = chunk.choices[0].delta.content
      print(word, end="", flush = True)
      full_response += word

    else:
      stream_response = client.chat.completions.create(
          model = "gpt-4o",
          messages = messages,
          stream = True
      )

      for chunk in stream_response:
        if chunk.choices[0].delta.content:
          word = chunk.choices[0].delta.content
          print(word, end="", flush=True)
          full_response += word

  memory_log.append({"prompt": prompt, "response": full_response})

stream_with_memory("I want a peaceful flower garden with low watering needs in Sacramento.")

with open("requirements.txt", "w") as f:
    f.write("openai>=1.0.0\n")

!cat requirements.txt

from google.colab import files
files.download("requirements.txt")
