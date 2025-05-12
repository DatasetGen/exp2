import websocket
import random
import uuid
import json
import urllib.request
import urllib.parse
from PIL import Image
import io


class ComfyGenerator:
    def __init__(self, server_ip="comfyui.autoescuelaseco.cloud", port=80, workflow=None):
        self.server_address = server_ip
        self.client_id = str(uuid.uuid4())
        self.prompt_template = json.loads(workflow) if workflow else {}

    def queue_prompt(self, prompt):
        data = json.dumps({"prompt": prompt, "client_id": self.client_id}).encode('utf-8')
        req = urllib.request.Request(f"https://{self.server_address}/prompt", data=data)
        return json.loads(urllib.request.urlopen(req).read())

    def get_images(self, prompt):
        prompt_id = self.queue_prompt(prompt)['prompt_id']
        output_images = {}
        current_node = ""

        ws = websocket.WebSocket()
        ws.connect(f"wss://{self.server_address}/ws?clientId={self.client_id}")

        try:
            while True:
                out = ws.recv()
                if isinstance(out, str):
                    message = json.loads(out)
                    if message['type'] == 'executing':
                        data = message['data']
                        if data['node'] is None:
                            break  # Execution is done
                        else:
                            current_node = data['node']
                else:
                    if current_node == '51':
                        images_output = output_images.get(current_node, [])
                        images_output.append(out[8:])  # Remove header
                        output_images[current_node] = images_output
        finally:
            ws.close()

        return output_images

    def generate(self):
        # Clone prompt to avoid side effects
        prompt = json.loads(json.dumps(self.prompt_template))

        images_data = self.get_images(prompt)
        pillow_images = []

        for node_id in images_data:
            for image_data in images_data[node_id]:
                image = Image.open(io.BytesIO(image_data))
                pillow_images.append(image)

        return pillow_images

import requests
import base64
from PIL import Image
from io import BytesIO
from typing import List

def image_to_base64_str(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def create_prompt_for_inpainting(image: Image.Image, mask: Image.Image, prompt: str = "") -> str:
    image_b64 = image_to_base64_str(image)
    mask_b64 = image_to_base64_str(mask)

    data = {
        "3": {
            "inputs": {
                "seed": random.randint(1,4294967294),
                "steps": 40,
                "cfg": 1,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1,
                "model": ["31", 0],
                "positive": ["38", 0],
                "negative": ["38", 1],
                "latent_image": ["38", 2]
            },
            "class_type": "KSampler",
            "_meta": {"title": "KSampler"}
        },
        "7": {
            "inputs": {
                "text": "",
                "clip": ["34", 0]
            },
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "CLIP Text Encode (Negative Prompt)"}
        },
        "8": {
            "inputs": {
                "samples": ["3", 0],
                "vae": ["32", 0]
            },
            "class_type": "VAEDecode",
            "_meta": {"title": "VAE Decode"}
        },
        "9": {
            "inputs": {
                "filename_prefix": "ComfyUI",
                "images": ["8", 0]
            },
            "class_type": "SaveImage",
            "_meta": {"title": "Save Image"}
        },
        "23": {
            "inputs": {
                "text": prompt,
                "clip": ["34", 0]
            },
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "CLIP Text Encode (Positive Prompt)"}
        },
        "26": {
            "inputs": {
                "guidance": 40,
                "conditioning": ["23", 0]
            },
            "class_type": "FluxGuidance",
            "_meta": {"title": "FluxGuidance"}
        },
        "31": {
            "inputs": {
                "unet_name": "flux1-fill-dev.safetensors",
                "weight_dtype": "default"
            },
            "class_type": "UNETLoader",
            "_meta": {"title": "Load Diffusion Model"}
        },
        "32": {
            "inputs": {
                "vae_name": "ae.safetensors"
            },
            "class_type": "VAELoader",
            "_meta": {"title": "Load VAE"}
        },
        "34": {
            "inputs": {
                "clip_name1": "flux/clip_l.safetensors",
                "clip_name2": "flux/t5xxl_fp16.safetensors",
                "type": "flux",
                "device": "default"
            },
            "class_type": "DualCLIPLoader",
            "_meta": {"title": "DualCLIPLoader"}
        },
        "38": {
            "inputs": {
                "noise_mask": False,
                "positive": ["26", 0],
                "negative": ["7", 0],
                "vae": ["32", 0],
                "pixels": ["48", 0],
                "mask": ["50", 0]
            },
            "class_type": "InpaintModelConditioning",
            "_meta": {"title": "InpaintModelConditioning"}
        },
        "48": {
            "inputs": {
                "image_base64": image_b64
            },
            "class_type": "LoadImageBase64",
            "_meta": {"title": "load image from base64 string"}
        },
        "49": {
            "inputs": {
                "image_base64": mask_b64
            },
            "class_type": "LoadImageBase64",
            "_meta": {"title": "load image from base64 string"}
        },
        "50": {
            "inputs": {
                "channel": "red",
                "image": ["49", 0]
            },
            "class_type": "ImageToMask",
            "_meta": {"title": "Convert Image to Mask"}
        },
        "51": {
            "inputs": {
                "images": ["8", 0]
            },
            "class_type": "SaveImageWebsocket",
            "_meta": {"title": "SaveImageWebsocket"}
        }
    }

    return json.dumps(data, indent=2)