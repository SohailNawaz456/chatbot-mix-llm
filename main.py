# === Import Required Libraries ===
import os  # For handling environment variables
from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI, set_tracing_disabled  # Chainlit agent-related classes
from dotenv import load_dotenv  # To load API keys from .env
import chainlit as cl  # Chainlit chat framework

# === Load Environment Variables ===
load_dotenv()  # Load variables from .env file
set_tracing_disabled(disabled=True)  # Disable Chainlit tracing logs for clean output

# === Fetch API Key ===
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')  # Get OpenRouter API key from .env

# === Initialize Chat History ===
history = []  # To store all user and assistant messages

# === Define Available AI Models ===
models = {
    "DeepSeek": 'deepseek/deepseek-r1:free',
    "Gemini_2_Flash": 'google/gemini-2.0-flash-exp:free',
    "Mistral": 'mistralai/devstral-small:free',
    "Qwen": 'qwen/qwen3-14b:free',
    "MetaLlama": 'meta-llama/llama-4-maverick:free'
}

# === On Chat Start: Greet User & Ask for Model Selection ===
@cl.on_chat_start
async def start_message():
    await cl.Message(content='How are you? 😜').send()  # Greet user on chat start

    # Show a model selection dropdown
    settings = await cl.ChatSettings(
        [
            cl.input_widget.Select(
                id='Model',
                label='Choose Any LLM Model',  # Label for the dropdown
                values=list(models.keys()),  # All model names as options
                initial_index=0  # Default selected model
            )
        ]
    ).send()

    # Call setup_chat to store user's selection
    await setup_chat(settings)

# === Handle Updated Model Selection by User ===
@cl.on_settings_update
async def setup_chat(settings):
    model_name = settings['Model']  # Get selected model name from user input
    cl.user_session.set('model', models[model_name])  # Store model identifier in session

    # Inform the user about their selected model
    await cl.Message(content=f'you have selected {model_name} Ai model.🤖').send()

# === Handle User Messages and Generate AI Response ===
@cl.on_message
async def my_message(msg: cl.Message):
    user_input = msg.content  # Get user message content
    history.append({"role": "user", "content": user_input})  # Add user message to conversation history

    selected_model = cl.user_session.get('model')  # Retrieve selected model from session

    # Initialize OpenRouter Async client with API key
    client = AsyncOpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url='https://openrouter.ai/api/v1',
    )

    # Set up an AI agent with instructions and selected model
    agent = Agent(
        name='my_agent',
        instructions='you are a helpful assistant',
        model=OpenAIChatCompletionsModel(model=selected_model, openai_client=client),
    )

    # Run the agent synchronously using the chat history
    result = Runner.run_sync(agent, history)

    # Send the final output of the AI agent back to the user
    await cl.Message(content=result.final_output).send()
