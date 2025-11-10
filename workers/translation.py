import os
from openai import OpenAI

# Create a single client (re-use across calls)
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not set in .env")

client = OpenAI(api_key=OPENAI_API_KEY)

def translate(text, target_region):

    if text.strip() == "":
        return ""

    response = client.responses.create(
        model="gpt-5",
        reasoning={"effort": "low"},
        instructions=(
            "You are a professional translator. "
            "Your translations account for cultural differences and local idioms. "
            "You will only respond with the translated text, without any additional commentary. "
        ),
        input=f"Translate the following marketing message for the {target_region} region: \"{text}\""
    )

    return response.output_text.strip() 
