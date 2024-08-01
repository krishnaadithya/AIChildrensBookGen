# flake8: noqa
import hydra
from omegaconf import OmegaConf
import torch
import os
import re
import pyrootutils
from PIL import Image, ImageDraw, ImageFont
import json
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, EulerDiscreteScheduler

from PIL import Image, ImageDraw, ImageFont
import textwrap
import os

import gradio as gr
from PIL import Image
import gradio as gr
import edge_tts
import asyncio
import tempfile
from TTS.api import TTS
import numpy as np 

import sys

from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips, TextClip, CompositeVideoClip
os.environ["COQUI_TOS_AGREED"] = "1"

BOI_TOKEN = '<img>'
EOI_TOKEN = '</img>'
IMG_TOKEN = '<img_{:05d}>'

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

device = 'cuda:0'
dtype = torch.float16
dtype_str = 'fp16'
num_img_in_tokens = 64
num_img_out_tokens = 64
instruction_prompt = '{instruction}'

tokenizer_cfg_path = 'configs/tokenizer/clm_llama_tokenizer.yaml'
image_transform_cfg_path = 'configs/processer/qwen_448_transform.yaml'
visual_encoder_cfg_path = 'configs/visual_tokenizer/qwen_vitg_448.yaml'

llm_cfg_path = 'configs/clm_models/llama2chat7b_lora.yaml'
agent_cfg_path = 'configs/clm_models/agent_7b_sft.yaml'

adapter_cfg_path = 'configs/detokenizer/detokenizer_sdxl_qwen_vit_adapted.yaml'
discrete_model_cfg_path = 'configs/discrete_model/discrete_identity.yaml'

diffusion_model_path = 'pretrained/stable-diffusion-xl-base-1.0'

save_dir = "output"

tokenizer_cfg = OmegaConf.load(tokenizer_cfg_path)
tokenizer = hydra.utils.instantiate(tokenizer_cfg)

image_transform_cfg = OmegaConf.load(image_transform_cfg_path)
image_transform = hydra.utils.instantiate(image_transform_cfg)

visual_encoder_cfg = OmegaConf.load(visual_encoder_cfg_path)
visual_encoder = hydra.utils.instantiate(visual_encoder_cfg)
visual_encoder.eval().to(device, dtype=dtype)
print('Init visual encoder done')

llm_cfg = OmegaConf.load(llm_cfg_path)
llm = hydra.utils.instantiate(llm_cfg, torch_dtype=dtype_str)
print('Init llm done.')

agent_model_cfg = OmegaConf.load(agent_cfg_path)
agent_model = hydra.utils.instantiate(agent_model_cfg, llm=llm)

agent_model.eval().to(device, dtype=dtype)
print('Init agent model Done')

noise_scheduler = EulerDiscreteScheduler.from_pretrained(diffusion_model_path, subfolder="scheduler")
print('init vae')
vae = AutoencoderKL.from_pretrained(diffusion_model_path, subfolder="vae").to(device, dtype=dtype)
print('init unet')
unet = UNet2DConditionModel.from_pretrained(diffusion_model_path, subfolder="unet").to(device, dtype=dtype)

adapter_cfg = OmegaConf.load(adapter_cfg_path)
adapter = hydra.utils.instantiate(adapter_cfg, unet=unet).to(device, dtype=dtype).eval()
print('Init adapter done')

discrete_model_cfg = OmegaConf.load(discrete_model_cfg_path)
discrete_model = hydra.utils.instantiate(discrete_model_cfg).to(device).eval()
print('Init discrete model done')

adapter.init_pipe(vae=vae,
                  scheduler=noise_scheduler,
                  visual_encoder=visual_encoder,
                  image_transform=image_transform,
                  discrete_model=discrete_model,
                  dtype=dtype,
                  device=device)

print('Init adapter pipe done')
boi_token_id = tokenizer.encode(BOI_TOKEN, add_special_tokens=False)[0]
eoi_token_id = tokenizer.encode(EOI_TOKEN, add_special_tokens=False)[0]

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

os.makedirs("tmp/", exist_ok = True)

def read_jsonl_to_dict(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            # Each line is a valid JSON object
            json_object = json.loads(line)
            data.append(json_object)
    return data

"""
def draw_text_on_image(image, text, font_size=20, max_width=30):
    draw = ImageDraw.Draw(image)
    width, height = image.size

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    wrapped_text = textwrap.fill(text, width=max_width)
    lines = wrapped_text.split('\n')

    line_heights = [font.getbbox(line)[3] - font.getbbox(line)[1] for line in lines]
    total_text_height = sum(line_heights)

    padding = 10
    y = height - total_text_height - padding

    # Draw a semi-transparent rectangle for the entire text area
    rectangle_position = (0, y - padding, width, height)
    draw.rectangle(rectangle_position, fill=(0, 0, 0, 128))  # Semi-transparent black

    for line in lines:
        # Get line width
        bbox = font.getbbox(line)
        line_width = bbox[2] - bbox[0]

        # Calculate x position to center this line
        x = (width - line_width) // 2

        # Draw the text
        draw.text((x, y), line, font=font, fill="white")

        # Move to the next line
        y += line_heights[lines.index(line)]

def create_comic_page(images, texts, images_per_row=4, font_path=None, font_size=100, spacing=50):
    
    width, height = 512, 512  # Fixed image size
    text_height = 60  # Height reserved for text
    total_images = len(images)
    rows = (total_images // images_per_row) + (total_images % images_per_row > 0)
    grid_image = Image.new('RGB', (images_per_row * (width + spacing) - spacing, rows * (height + text_height + spacing) - spacing), 'black')

    draw = ImageDraw.Draw(grid_image)
    font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default(font_size)

    # Paste images and add text in a grid with spacing
    for index, (img, text) in enumerate(zip(images, texts)):
        img = img.resize((width, height))
        x = (index % images_per_row) * (width + spacing)
        y = (index // images_per_row) * (height + text_height + spacing)
        grid_image.paste(img, (x, y))

        # Calculate text wrapping and positioning
        wrapped_text = textwrap.fill(text, width=50)  # Adjust width based on your font size and image width
        text_y = y + height + 5  # Small margin between image and text
        draw.multiline_text((x, text_y), wrapped_text, font=font, fill="white", align='center', spacing=4)

    return grid_image

def create_comic(images, texts, images_per_page=6, images_per_row=3, font_path=None, font_size=20, spacing=20):
    pages = []
    for i in range(0, len(images), images_per_page):
        page_images = images[i:i + images_per_page]
        page_texts = texts[i:i + images_per_page]
        comic_page = create_comic_page(page_images, page_texts, images_per_row, font_path, font_size, spacing)
        pages.append(comic_page)
    return pages

"""
def draw_text_on_image(picture, words, bubble_size=40, max_width=25):
    """
    Adds a fun speech bubble with words to a picture!
    """
    magic_pen = ImageDraw.Draw(picture)
    picture_width, picture_height = picture.size

    try:
        # Let's use a fun font!
        bubble_font = ImageFont.truetype("Comic_Sans_MS.ttf", bubble_size)
    except IOError:
        # Oops! If we can't find the fun font, we'll use the default one
        bubble_font = ImageFont.load_default()

    # Wrap the words so they fit nicely in the bubble
    wrapped_words = textwrap.fill(words, width=max_width)
    word_lines = wrapped_words.split('\n')

    # Figure out how tall our bubble needs to be
    line_heights = [bubble_font.getbbox(line)[3] - bubble_font.getbbox(line)[1] for line in word_lines]
    bubble_height = sum(line_heights)

    # Let's put the bubble at the bottom of the picture
    bubble_padding = 20
    y_position = picture_height - bubble_height - bubble_padding

    # Draw a see-through rectangle for our speech bubble
    bubble_shape = (0, y_position - bubble_padding, picture_width, picture_height)
    magic_pen.rectangle(bubble_shape, fill=(255, 255, 255, 200))  # Light, see-through white

    for line in word_lines:
        # Center each line of words in the bubble
        line_width = bubble_font.getbbox(line)[2] - bubble_font.getbbox(line)[0]
        x_position = (picture_width - line_width) // 2

        # Write the words in a fun color!
        magic_pen.text((x_position, y_position), line, font=bubble_font, fill="purple")

        # Move down for the next line
        y_position += line_heights[word_lines.index(line)]

def create_comic_page(pictures, captions, pictures_per_row=2, rows_per_page=3, font_path=None, font_size=50, space_between=40):
    """
    Creates a colorful comic page with pictures and captions!
    """
    picture_width, picture_height = 512, 512  # Each picture is this big
    caption_height = 120  # Space for captions
    border_size = 10  # Border around image and text
    total_pictures = len(pictures)
    
    # Calculate the height and width of the comic page
    page_width = pictures_per_row * (picture_width + space_between + 2 * border_size) - space_between
    page_height = rows_per_page * (picture_height + caption_height + space_between + 2 * border_size) - space_between

    # Create a big colorful canvas for our comic
    comic_canvas = Image.new('RGB', (page_width, page_height), 'lightblue')
    magic_pen = ImageDraw.Draw(comic_canvas)
    
    # Let's use a fun font for our captions!
    caption_font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default(size=font_size)

    # Let's add our pictures and captions to the comic!
    for i, (pic, caption) in enumerate(zip(pictures, captions)):
        pic = pic.resize((picture_width, picture_height))
        x = (i % pictures_per_row) * (picture_width + space_between + 2 * border_size) + border_size
        y = (i // pictures_per_row) * (picture_height + caption_height + space_between + 2 * border_size) + border_size

        # Draw a black border around the picture and caption
        magic_pen.rectangle([x - border_size, y - border_size, x + picture_width + border_size, y + picture_height + caption_height + border_size], outline='black', width=2)
        comic_canvas.paste(pic, (x, y))

        # Wrap the caption text so it fits nicely
        wrapped_caption = textwrap.fill(caption, width=50)  # Adjust this number to make the text fit just right
        caption_y = y + picture_height + 10  # A little space between the picture and caption
        
        # Let's write our caption in a fun color!
        line_width = magic_pen.textbbox((0, 0), wrapped_caption, font=caption_font)[2]
        centered_x = x + (picture_width - line_width) // 2
        magic_pen.multiline_text((centered_x, caption_y), wrapped_caption, font=caption_font, fill="blue", align='center', spacing=4)

    return comic_canvas

def create_comic(pictures, captions, images_per_page=6, images_per_row=2, rows_per_page=3, font_path=None, font_size=50, spacing=40):
    """
    Creates a whole comic book with multiple pages!
    """
    comic_pages = []
    for i in range(0, len(pictures), images_per_page):
        page_pictures = pictures[i:i + images_per_page]
        page_captions = captions[i:i + images_per_page]

        # Fill the remaining spots on the last page with blank images and empty captions
        while len(page_pictures) < images_per_page:
            blank_image = Image.new('RGB', (512, 512), 'white')
            page_pictures.append(blank_image)
            page_captions.append("")

        fun_comic_page = create_comic_page(page_pictures, page_captions, images_per_row, rows_per_page, font_path, font_size, spacing)
        comic_pages.append(fun_comic_page)
    
    return comic_pages


def gen_image(output):
    image_embeds_gen = output['img_gen_feat']
    images_gen = adapter.generate(image_embeds=output['img_gen_feat'], num_inference_steps=50)
    return images_gen, image_embeds_gen

def main(image_path, question, story_len = 8, window_size = 8, text_id = 1):

    output_texts, generated_images = [], []

    image = Image.open(image_path).convert('RGB')
                
    agent_model.llm.base_model.model.use_kv_cache_head = False
    image_tensor = image_transform(image).unsqueeze(0).to(device, dtype=dtype)
    
    image_tokens = BOI_TOKEN + ''.join([IMG_TOKEN.format(int(item)) for item in range(num_img_in_tokens)]) + EOI_TOKEN
    
    prompt = instruction_prompt.format_map({'instruction': question + image_tokens})
    
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = [tokenizer.bos_token_id] + input_ids
    
    boi_idx = input_ids.index(boi_token_id)
    eoi_idx = input_ids.index(eoi_token_id)
    
    input_ids = torch.tensor(input_ids).to(device, dtype=torch.long).unsqueeze(0)
    
    ids_cmp_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    
    ids_cmp_mask[0, boi_idx + 1:eoi_idx] = True
    embeds_cmp_mask = torch.tensor([True]).to(device, dtype=torch.bool)
    
    with torch.no_grad():
        image_embeds = visual_encoder(image_tensor)
    output = agent_model.generate(tokenizer=tokenizer,
                                  input_ids=input_ids,
                                  image_embeds=image_embeds,
                                  embeds_cmp_mask=embeds_cmp_mask,
                                  ids_cmp_mask=ids_cmp_mask,
                                  max_new_tokens=500,
                                  num_img_gen_tokens=num_img_out_tokens)
    text = re.sub(r'\s*<[^>]*>\s*', ' ', output['text']).strip()
    if '[INST]' in text:
        text = text.split('[INST]')[1]
    output_texts.append(text)

    while output['has_img_output'] and image_embeds.shape[0] < story_len:
        
        images_gen, image_embeds_gen = gen_image(output)
        
        original_image = images_gen[0]    
        generated_images.append(original_image)

    
        image_embeds = torch.cat((image_embeds, image_embeds_gen), dim=0)
        
        if text_id >= story_len - 1:
            break
    
        prompt = prompt + text + image_tokens
        text_id += 1
    
        input_ids = tokenizer.encode(prompt, add_special_tokens=False)
        while image_embeds.shape[0] > window_size:
            eoi_prompt_idx = prompt.index(EOI_TOKEN)
            prompt = prompt[eoi_prompt_idx + len(EOI_TOKEN) + len('[INST]'):]
            image_embeds = image_embeds[1:]
            input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    
        input_ids = [tokenizer.bos_token_id] + input_ids
    
        boi_idx = torch.where(torch.tensor(input_ids) == boi_token_id)[0].tolist()
        eoi_idx = torch.where(torch.tensor(input_ids) == eoi_token_id)[0].tolist()
    
        input_ids = torch.tensor(input_ids).to(device, dtype=torch.long).unsqueeze(0)
    
        ids_cmp_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    
        for i in range(image_embeds.shape[0]):
            ids_cmp_mask[0, boi_idx[i] + 1:eoi_idx[i]] = True
        embeds_cmp_mask = torch.tensor([True] * image_embeds.shape[0]).to(device, dtype=torch.bool)
    
        output = agent_model.generate(tokenizer=tokenizer,
                                      input_ids=input_ids,
                                      image_embeds=image_embeds,
                                      embeds_cmp_mask=embeds_cmp_mask,
                                      ids_cmp_mask=ids_cmp_mask,
                                      max_new_tokens=500,
                                      num_img_gen_tokens=num_img_out_tokens)
        text = re.sub(r'\s*<[^>]*>\s*', ' ', output['text']).strip()

        if '[INST]' in text:
            text = text.split('[INST]')[1]
        output_texts.append(text)

    return generated_images, output_texts


def images_to_pdf(comic_pages, output_path = "tmp/comic.pdf"):
    """
    Saves the comic pages as a PDF file.
    """
    if comic_pages:
        comic_pages[0].save(output_path, save_all=True, append_images=comic_pages[1:], format='PDF')

    return output_path



async def get_voices():
    voices = await edge_tts.list_voices()
    return {f"{v['ShortName']} - {v['Locale']} ({v['Gender']})": v['ShortName'] for v in voices}

async def text_to_speech_edge(text, voice, rate=-10, pitch=0):
    if not text.strip():
        return None, gr.Warning("Please enter text to convert.")
    if not voice:
        return None, gr.Warning("Please select a voice.")
    
    voice_short_name = voice.split(" - ")[0]
    rate_str = f"{rate:+d}%"
    pitch_str = f"{pitch:+d}Hz"
    communicate = edge_tts.Communicate(text, voice_short_name, rate=rate_str, pitch=pitch_str)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        tmp_path = tmp_file.name
        await communicate.save(tmp_path)
    return tmp_path, None

def clone(text, audio):
    tts.tts_to_file(text=text, speaker_wav=audio, language="en", file_path="tmp/output.wav")
    return "tmp/output.wav"

def process_input(text, audio_input, voice):
    if audio_input:
        return clone(text, audio_input)
    else:
        audio_path, _ = asyncio.run(text_to_speech_edge(text, voice))
        return audio_path

def create_video(images, audio_paths, fps=24):
    clips = []
    temp_files = []

    for i, (image, audio_path) in enumerate(zip(images, audio_paths)):
        # Resize image
        image = image.resize((512, 512))
        
        # Save image
        image_path = f"tmp/temp_image_{i}.jpg"
        image.save(image_path)
        temp_files.append(image_path)
        
        audio_clip = AudioFileClip(audio_path)
        duration = audio_clip.duration
        
        # Create video clip
        img_clip = ImageClip(image_path).set_duration(duration)
        video_clip = img_clip.set_audio(audio_clip)
        clips.append(video_clip)
    
    # Concatenate clips
    final_clip = concatenate_videoclips(clips, method="compose")
    
    # Write video file
    video_path = "tmp/comic_video.mp4"
    try:
        final_clip.write_videofile(video_path, codec="libx264", audio_codec="aac", fps=fps)
    except Exception as e:
        print(f"Error writing video file: {e}")
        # Clean up temporary files
        for file in temp_files:
            if os.path.exists(file):
                os.remove(file)
        raise  # Re-raise the exception after cleanup
    
    # Clean up temporary files
    for file in temp_files:
        if os.path.exists(file):
            os.remove(file)
    
    return video_path

def app(image_path, question, story_len):
    font_path = None
    
    generated_images, output_texts = main(image_path, question, story_len=story_len, window_size=8, text_id=1)
    
    comic_pages = create_comic(generated_images, output_texts, images_per_row=2, font_path=font_path, font_size=20, spacing=30)
    pdf_path = images_to_pdf(comic_pages)
    
    return [comic.convert("RGB") for comic in comic_pages], pdf_path, generated_images, output_texts

def show_image(images, index):
    if index < 0 or index >= len(images):
        return None, index
    return images[index], index

def next_image(images, index):
    new_index = (index + 1) % len(images)
    return images[new_index], new_index

def prev_image(images, index):
    new_index = (index - 1) % len(images)
    return images[new_index], new_index

def generate_audio_for_texts(texts, audio_input, voice):
    audio_paths = []
    for text in texts:
        audio_path = process_input(text, audio_input, voice)
        audio_paths.append(audio_path)
    return audio_paths
        
def generate_video_wrapper(images, texts, audio_input, voice):
    audio_paths = generate_audio_for_texts(texts, audio_input, voice)
    video_path = create_video(images, audio_paths)
    return video_path
    
async def create_demo():
    voices = await get_voices()
    
    with gr.Blocks() as demo:

        markdown_content = markdown_content = """
# Curious George Adventure Creator: AI Comic Book Generator

Welcome to the future of storytelling with the Curious George Adventure Creator! Dive into the playful and adventurous world of Curious George like never before. Our innovative AI model, designed for children and fans of the lovable monkey, brings George's stories to life with just a starting image and a chosen theme.

The Curious George Adventure Creator allows you to craft unique episodes featuring the sweet and curious African monkey, Curious George, and his ever-patient friend, "The Man in the Yellow Hat." George's adventures, often filled with playful curiosity and unforeseen trouble, are brought to life through our cutting-edge AI technology.

### Key Features

1. **Easy Episode Creation**: Simply upload a starting image and select a theme, and our AI model will generate a personalized Curious George episode. Watch George explore, learn, and get into his usual delightful mishaps, all tailored to your input.
   
2. **Learning and Fun**: Each episode emphasizes themes of learning, forgiveness, and curiosity. It's not just entertainment; it's an educational experience wrapped in fun and adventure.

3. **Voice Options**:
   - **Loan Your Voice**: Bring a personal touch to your episode by lending your own voice to the characters.
   - **Audio Library**: Choose from a variety of pre-recorded voices, including cloned voices that perfectly match the characters.
"""

        logo_path = "asset/logo.png"
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Image(logo_path)
            with gr.Column(scale=2):
                gr.Markdown(markdown_content)
        
        

        # Input Section
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Upload your starting image")
                gr.Markdown("This could be anything like farmland, forest, or sea.")
                image_input = gr.Image(type="filepath", label="Upload Image", elem_id="image_input")
                
                gr.Markdown("### Write a scenario")
                gr.Markdown('Example: "George the monkey goes to a farm and explores."')
                scenario_input = gr.Textbox(label="Enter your scenario:", elem_id="scenario_input")

                story_len = gr.Slider(2, 10, value=8, step = 1, label="Count", info="Choose between 2 and 10")

                
                generate_button = gr.Button("Generate Comic", elem_id="generate_button")
            with gr.Column(scale=2):
                # Output Section
                image_output = gr.Image(label="Generated Image", elem_id="image_output", height=1200)
                with gr.Row():
                    prev_button = gr.Button("Previous", elem_id="prev_button")
                    next_button = gr.Button("Next", elem_id="next_button")


        with gr.Row():
            gr.Markdown(
                        """
                        **If you liked the comic, you can download it using the button below:**
                        """
                    )
            pdf_output = gr.File(label="Download the Generated Story as a pdf")


        # Voice-over Section
        gr.Markdown(
            """
            ## Voice-over Options
            You can add a personalized touch to your comic by generating a voice-over. You have two options:
            - **Voice Reference Audio File:** Upload an audio file of your voice to clone it.
            - **Select Voice:** Choose from a variety of pre-recorded voices available in our system.
            """
        )
        
        with gr.Row():
            audio_input = gr.Audio(type="filepath", label="Clone your voice by clicking the record buttong/upload a reference voice")
            voice = gr.Dropdown(choices=[""] + list(voices.keys()), label="Select Voice", value="")
        
        generate_video_button = gr.Button("Generate Video")
        video_output = gr.Video(label="Generated Video")
        
        index = gr.State(0)
        generated_images = gr.State([])
        output_texts = gr.State([])
        comic_pages = gr.State([])
        
        generate_button.click(
            app,
            inputs=[image_input, scenario_input, story_len],
            outputs=[comic_pages, pdf_output, generated_images, output_texts]
        ).then(
            show_image,
            inputs=[comic_pages, index],
            outputs=[image_output, index]
        )
        
        prev_button.click(
            prev_image,
            inputs=[comic_pages, index],
            outputs=[image_output, index]
        )
        
        next_button.click(
            next_image,
            inputs=[comic_pages, index],
            outputs=[image_output, index]
        )
        
        generate_video_button.click(
            generate_video_wrapper,
            inputs=[generated_images, output_texts, audio_input, voice],
            outputs=[video_output]
        )
    
    return demo

if __name__ == "__main__":
    demo = asyncio.run(create_demo())
    demo.launch(share=True)