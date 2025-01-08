import os
import time
import requests
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Store the timestamp of the last request for each prompt
last_requests = {}

# Function to send prompt to ComfyUI
def send_prompt(prompt_text, api_url="http://127.0.0.1:8188"):
    workflow = {
        "1": {
            "inputs": {
                "config_name": "v1-inference.yaml",
                "ckpt_name": "flux1-schnell-fp8.safetensors"
            },
            "class_type": "CheckpointLoaderSimple"
        },
        "2": {
            "inputs": {
                "text": prompt_text,
                "clip": ["1", 1]
            },
            "class_type": "CLIPTextEncode"
        },
        "3": {
            "inputs": {
                "text": "",
                "clip": ["1", 1]
            },
            "class_type": "CLIPTextEncode"
        },
        "4": {
            "inputs": {
                "width": 1024,
                "height": 1024,
                "batch_size": 1
            },
            "class_type": "EmptyLatentImage"
        },
        "5": {
            "inputs": {
                "model": ["1", 0],
                "positive": ["2", 0],
                "negative": ["3", 0],
                "latent_image": ["4", 0],
                "seed": 605612529927851,
                "steps": 4,
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0
            },
            "class_type": "KSampler"
        },
        "6": {
            "inputs": {
                "samples": ["5", 0],
                "vae": ["1", 2]
            },
            "class_type": "VAEDecode"
        },
        "7": {
            "inputs": {
                "images": ["6", 0],
                "filename_prefix": "ComfyUI"
            },
            "class_type": "SaveImage"
        }
    }

    payload = {
        "prompt": workflow
    }

    try:
        response = requests.post(f"{api_url}/prompt", json=payload)
        if response.status_code == 200:
            return response.json().get('prompt_id')
        else:
            print(f"Error sending prompt: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error: {e}")
        return None

# Function to get the latest generated image
def get_latest_image(directory, last_checked_time):
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.png')]
    files.sort(key=lambda x: os.path.getmtime(x), reverse=True)

    # Check for a file newer than the last checked time
    for file in files:
        if os.path.getmtime(file) > last_checked_time:
            return file
    return None

# Function to wait for the image with a timeout
def wait_for_image(directory, timeout=90):
    start_time = time.time()
    last_checked_time = start_time  # Starting check time
    
    while time.time() - start_time < timeout:
        latest_image = get_latest_image(directory, last_checked_time)
        if latest_image:
            return latest_image
        time.sleep(5)  # Poll every 5 seconds
    return None

@app.route('/api/generate', methods=['POST'])
def generate():
    data = request.get_json()
    if not data or 'prompt' not in data:
        return jsonify({'error': 'Prompt is required'}), 400

    prompt = data['prompt']
    current_time = time.time()
    last_time = last_requests.get(prompt, 0)

    if current_time - last_time < 5:
        return jsonify({'error': 'Duplicate request detected. Please wait before trying again.'}), 400

    last_requests[prompt] = current_time
    prompt_id = send_prompt(prompt)

    if prompt_id:
        output_directory = r'<YOUR OUTPUT FOLDER FOR COMFYUI'
        image_path = wait_for_image(output_directory)
        if image_path:
            return send_file(image_path, mimetype='image/png')
        else:
            return jsonify({'error': 'Image generation timed out.'}), 500
    else:
        return jsonify({'error': 'Failed to send prompt.'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
