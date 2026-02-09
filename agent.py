from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider

# 1. Setup the connection using OllamaProvider
# The base_url now lives inside the provider instance
ollama_model = OpenAIChatModel(
    model_name='llama3.2',
    provider=OllamaProvider(base_url='http://localhost:11434/v1')
)

# 2. Define the Agent
agent = Agent(
    model=ollama_model,
    system_prompt="You are a precise assistant. Use tools for any math or data tasks.",
)

# 3. Define a Tool
@agent.tool_plain
def calculate_area(width: int, height: int) -> str:
    """Calculate the area of a rectangle. Use this for all area requests."""
    area = width * height
    return f"The area is {area} square units."

# 4. Run the Agent
if __name__ == "__main__":
    print("--- Local Ollama Agent Active ---")
    try:
        # result.data will contain the final string output from the tool or model
        result = agent.run_sync("I have a room that is 12 meters wide and 15 meters long. What is the area?")
        print("\nAI Response:")
        print(result.output)
    except Exception as e:
        print(f"Error: {e}")