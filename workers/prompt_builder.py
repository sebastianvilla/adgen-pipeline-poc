import os
from openai import OpenAI

# Create a single client (re-use across calls)
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not set in .env")

client = OpenAI(api_key=OPENAI_API_KEY)

def generate(theme, product, region, audience, message = "") -> str:

    response = client.responses.create(
        model="gpt-5",
        reasoning={"effort": "low"},
        instructions=(
            "You are a professional prompt builder for image backgrounds for social media ad campaigns. "
            "Your prompts account for cultural differences and local idioms. "
            "Prompts must refer only to the background. The prompt is used to fill in the background/environment around a product image. "
            "No text references should be included in the prompt, all information given to you is only for the purpose of generating a contextual prompt for the image. "
            "Your prompts must be similar in length and structure as these examples: \n"
            " - 'A vibrant city park during autumn with people jogging and walking dogs, skyscrapers in the background, clear blue sky.'\n"
            " - 'Photorealistic studio scene: product on a red background, christmas theme. High-end commercial photography style, sophisticated and premium feeling. \n"
            "You will only respond with the prompt, without any additional commentary. "
        ),
        input=f"Campaign theme: {theme}\n Product: {product}\n Region: {region}\n Audience: {audience}\n Message: {message}\n\n"
    )

    return response.output_text.strip() 
