from fastapi import FastAPI, UploadFile, File, Form
import google.generativeai as genai
from PIL import Image
import os
import re
import json

app = FastAPI()

# To store the API key once the user provides it
api_key_store = {"api_key": None}

# Prompt
prompt = (
    "Identify if you could find a vehicle in the provided image. "
    "Classify if the vehicle is 'Car', 'Bike', 'Auto rickshaw', or 'Heavy Vehicle' and provide number plate details. "
    "Provide output in 'vehicle_type', 'vehicle_number', 'vehicle_approch[in-if the vechicle front side or out-if vechicle back side]'."
)

# Endpoint to set the API key (called only once)
@app.post("/set_api_key/")
async def set_api_key(api_key: str = Form(...)):
    api_key_store["api_key"] = api_key
    genai.configure(api_key=api_key)
    return {"message": "API key set successfully"}

# Endpoint to get vehicle prediction from an image
@app.post("/get_vehicle_prediction/")
async def get_vehicle_prediction(image: UploadFile = File(...)):
    # Ensure API key is set
    if not api_key_store["api_key"]:
        return {"error": "API key not set. Please set the API key first."}

    # Save the uploaded image file to a temporary path
    image_path = f"temp_{image.filename}"
    with open(image_path, "wb") as f:
        f.write(await image.read())

    try:
        # Upload the image using Google Generative AI
        myfile = genai.upload_file(path=image_path)
        inputs = [prompt, myfile]

        # Generate content using the AI model
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(inputs)
        myfile.delete()

        # Return the response from the AI model
        # Extract JSON content using regex
        match = re.search(
            r"```json\n(.*?)```", 
            response.text, # Response from GPT
            re.DOTALL | re.IGNORECASE)
        if match:
            json_str = match.group(1).strip()

            # Convert to json object
            try:
                genai_response_json = json.loads(json_str)
                print('Decoding to JSON complete!')
            except json.JSONDecodeError:
                print(f"Error decoding JSON: {json.JSONDecodeError}")
        else:
            print("No JSON content found in the response.")

        return genai_response_json

    finally:
        # Clean up the saved image
        if os.path.exists(image_path):
            os.remove(image_path)

# Start with FastAPI docs for easy user interaction
@app.get("/")
def read_root():
    return {"message": "Welcome to the Vehicle Prediction API. Go to /docs to interact with the API."}
