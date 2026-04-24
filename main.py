from dotenv import load_dotenv
import os

load_dotenv()

print("OPENAI key loaded:", bool(os.getenv("OPENAI_API_KEY")))
print("ANTHROPIC key loaded:", bool(os.getenv("ANTHROPIC_API_KEY")))
print("Setup complete.")