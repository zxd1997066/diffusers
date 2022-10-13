
import math
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline
import intel_extension_for_pytorch as ipex

prompt = ["a photo of an astronaut riding a horse on mars"]
batch_size = 1
prompt = prompt * batch_size

device = 'cpu'
seed = 666
generator = torch.Generator(device).manual_seed(seed)

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

model = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4', use_auth_token=True)
model = model.to(device)

# to channels last
model.unet = model.unet.to(memory_format=torch.channels_last)
model.vae = model.vae.to(memory_format=torch.channels_last)
model.text_encoder = model.text_encoder.to(memory_format=torch.channels_last)
model.safety_checker = model.safety_checker.to(memory_format=torch.channels_last)
# optimize with ipex
model.unet = ipex.optimize(model.unet.eval(), dtype=torch.bfloat16, inplace=True)
model.vae = ipex.optimize(model.vae.eval(), dtype=torch.bfloat16, inplace=True)
model.text_encoder = ipex.optimize(model.text_encoder.eval(), dtype=torch.bfloat16, inplace=True)
model.safety_checker = ipex.optimize(model.safety_checker.eval(), dtype=torch.bfloat16, inplace=True)

# compute
with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
    images = model(prompt, guidance_scale=7.5, num_inference_steps=50, generator=generator)["images"]

# save image
r = math.ceil(math.sqrt(batch_size))
c = math.ceil(batch_size / r)
grid = image_grid(images, rows=r, cols=c)
grid.save("astronaut_rides_horse-{}x{}.png".format(r, c))
