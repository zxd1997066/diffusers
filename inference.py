
import os
import torch
import time
import argparse
import sys
from PIL import Image
from diffusers import StableDiffusionPipeline, DDPMPipeline

MODEL_CLASSES = {
    "stable-diffusion": (StableDiffusionPipeline),
    "ddpm": (DDPMPipeline),
}


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

def test(args, model, prompt):
    total_sample = 0
    total_time = 0.0
    if args.profile:
        if torch.cuda.is_available():
            prof_act = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        else:
            prof_act = [torch.profiler.ProfilerActivity.CPU]
        with torch.profiler.profile(
            activities=prof_act,
            record_shapes=True,
            schedule=torch.profiler.schedule(
                wait=int(args.num_iter/2),
                warmup=2,
                active=1,
            ),
            on_trace_ready=trace_handler,
        ) as p:
            for i in range(args.num_iter):
                elapsed = time.time()
                if args.model_type == 'stable-diffusion':
                    images = model(prompt, num_inference_steps=args.num_inference_steps, args=args)["images"]
                else:
                    images = model(batch_size=args.per_device_eval_batch_size, args=args).images[0]
                if torch.cuda.is_available(): torch.cuda.synchronize()
                p.step()
                elapsed = time.time() - elapsed
                print("Iteration: {}, inference time: {} sec.".format(i, elapsed), flush=True)
                if i >= args.num_warmup:
                    total_time += elapsed
                    total_sample += 1
    else:
        for i in range(args.num_iter):
            elapsed = time.time()
            if args.model_type == 'stable-diffusion':
                images = model(prompt, num_inference_steps=args.num_inference_steps, args=args)["images"]
            else:
                images = model(batch_size=args.per_device_eval_batch_size, args=args).images[0]
            if torch.cuda.is_available(): torch.cuda.synchronize()
            elapsed = time.time() - elapsed
            print("Iteration: {}, inference time: {} sec.".format(i, elapsed), flush=True)
            if i >= args.num_warmup:
                total_time += elapsed
                total_sample += 1

    print("\n", "-"*20, "Summary", "-"*20)
    latency = total_time / total_sample * 1000
    throughput = total_sample / total_time
    print("inference Latency: {} ms".format(latency))
    print("inference Throughput: {} samples/s".format(throughput))

    # save image
    if args.save_image:
        if args.model_name_or_path != 'stable-diffusion':
            images.save(f"generated_image.png")
        else:
            from math import ceil
            grid = image_grid(images, rows=args.image_rows, cols=ceil(args.batch_size / args.image_rows))
            grid.save("astronaut_rides_horse-" + args.precision + ".png")

def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cpu_time_total")
    print(output)
    import pathlib
    timeline_dir = str(pathlib.Path.cwd()) + '/timeline/'
    if not os.path.exists(timeline_dir):
        try:
            os.makedirs(timeline_dir)
        except:
            pass
    timeline_file = timeline_dir + 'timeline-' + str(torch.backends.quantized.engine) + '-' + \
                'sd-' + str(p.step_num) + '-' + str(os.getpid()) + '.json'
    p.export_chrome_trace(timeline_file)


if __name__ == '__main__':
    #
    parser = argparse.ArgumentParser(description='PyTorch Inference')
    parser.add_argument("--model_name_or_path", type=str, default='CompVis/stable-diffusion-v1-4', help="model name")
    parser.add_argument("--model_type", type=str, default='', help="model_type")
    parser.add_argument('--prompt', default=["a photo of an astronaut riding a horse on mars"], type=list, help='prompt')
    parser.add_argument('--device', default="cpu", type=str, help='cpu, cuda or xpu')
    parser.add_argument('--precision', default="float32", type=str, help='precision')
    parser.add_argument('--channels_last', default=1, type=int, help='Use NHWC or not')
    parser.add_argument('--profile', action='store_true', default=False, help='collect timeline')
    parser.add_argument('--num_iter', default=1, type=int, help='test iterations')
    parser.add_argument('--num_warmup', default=0, type=int, help='test warmup')
    parser.add_argument('--per_device_eval_batch_size', default=1, type=int, help='per_device_eval_batch_size')
    parser.add_argument('--num_inference_steps', default=50, type=int, help='num_inference_steps')
    parser.add_argument('--save_image', action='store_true', default=False, help='save image')
    parser.add_argument('--image_rows', default=1, type=int, help='saved image array')
    #
    parser.add_argument('--ipex', action='store_true', default=False, help='enable ipex')
    parser.add_argument('--jit', action='store_true', default=False, help='enable JIT')
    #
    parser.add_argument('--do_eval', action='store_true', default=False, help='useless')
    parser.add_argument('--overwrite_output_dir', action='store_true', default=False, help='useless')
    parser.add_argument('--output_dir', default='', type=str, help='useless')
    args = parser.parse_args()
    print(args)

    # Intialize
    model_class = MODEL_CLASSES[args.model_type]
    model = model_class.from_pretrained(args.model_name_or_path)
    prompt = args.prompt * args.per_device_eval_batch_size
    if args.device == 'cuda':
        model = model.to("cuda")
    if args.channels_last:
        # model = model.to(memory_format=torch.channels_last)
        model.unet = model.unet.to(memory_format=torch.channels_last)
        if args.model_type == 'stable-diffusion':
            model.vae = model.vae.to(memory_format=torch.channels_last)
            model.text_encoder = model.text_encoder.to(memory_format=torch.channels_last)
            model.safety_checker = model.safety_checker.to(memory_format=torch.channels_last)
        print("---- Use NHWC model.")
    if args.ipex:
        import intel_extension_for_pytorch as ipex
        print("---- Use IPEX")
        if args.precision == "bfloat16":
            model.unet = ipex.optimize(model.unet.eval(), dtype=torch.bfloat16, inplace=True)
            if args.model_type == 'stable-diffusion':
                model.vae = ipex.optimize(model.vae.eval(), dtype=torch.bfloat16, inplace=True)
                model.text_encoder = ipex.optimize(model.text_encoder.eval(), dtype=torch.bfloat16, inplace=True)
                model.safety_checker = ipex.optimize(model.safety_checker.eval(), dtype=torch.bfloat16, inplace=True)
        else:
            model.unet = ipex.optimize(model.unet.eval(), dtype=torch.float32, inplace=True)
            if args.model_type == 'stable-diffusion':
                model.vae = ipex.optimize(model.vae.eval(), dtype=torch.float32, inplace=True)
                model.text_encoder = ipex.optimize(model.text_encoder.eval(), dtype=torch.float32, inplace=True)
                model.safety_checker = ipex.optimize(model.safety_checker.eval(), dtype=torch.float32, inplace=True)

    # start test
    if args.precision == "bfloat16":
        print("---- Use AMP bfloat16")
        with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
            test(args, model, prompt)
    elif args.precision == "float16":
        print("---- Use AMP float16")
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
            test(args, model, prompt)
    else:
        test(args, model, prompt)

