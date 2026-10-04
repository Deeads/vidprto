# Photo Sequence Generation with Motion Vectors and Stable Diffusion

An entry-level photo sequence generation method that uses motion vectors and Stable Diffusion img2img to create smooth animated sequences.

## Overview

This prototype implements a photo sequence generation pipeline that:
- Loads a background image
- Defines motion vectors (x, y) using linear interpolation
- Applies motion vectors to transform each frame
- Feeds each frame through Stable Diffusion img2img with low denoising strength (0.2-0.3)
- Renders frame sequence
- Exports MP4 video

The low denoising strength ensures each frame retains details while being slightly refreshed by the diffusion model, creating smooth transitions.

## Requirements

- Python 3.8+
- Pillow (PIL)
- NumPy
- MoviePy
- PyTorch (for Stable Diffusion)
- diffusers (Hugging Face)
- transformers
- accelerate

**Note:** Stable Diffusion models are large (~4GB). First run will download the model from Hugging Face.

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. **GPU Support (Optional but Recommended):**
   - For faster processing, install PyTorch with CUDA support
   - The script will automatically use GPU if available, otherwise CPU

## Usage

1. **Prepare your assets:**
   - Place `background.png` in the project root (any resolution, RGB format)

2. **Run the script:**
```bash
python main.py
```

3. **Output:**
   - Frames will be saved to `frames/` directory
   - Final video will be saved as `output.mp4`

**Note:** Processing can take a while since each frame goes through Stable Diffusion. On CPU, expect ~10-30 seconds per frame. On GPU, expect ~1-3 seconds per frame.

## Animation Parameters

Currently hardcoded in `main.py`:
- **Duration:** 2 seconds
- **FPS:** 24 frames per second
- **Start Motion Vector:** (0, 0) - no movement
- **End Motion Vector:** (100, -30) - shift 100px right, 30px up
- **Denoising Strength:** 0.25 (low = retains details, subtle refresh)
- **Stable Diffusion Prompt:** "high quality, detailed, cinematic"

## How It Works

1. **Motion Vectors:** Define how much to shift each frame (dx, dy)
2. **Linear Interpolation:** Smoothly transition motion vectors across frames
3. **Image Translation:** Apply motion vector to shift the background
4. **Stable Diffusion img2img:** Process each frame with low denoising strength
   - Low strength (0.2-0.3) = retains structure, adds subtle variation
   - Creates smooth transitions between frames
5. **Video Compilation:** Assemble frames into MP4

## Project Structure

```
project-root/
    main.py              # Main photo sequence generation script
    requirements.txt     # Python dependencies
    background.png       # Background image (you provide)
    frames/              # Generated frame sequence
    output.mp4           # Final video output
```

## Function Architecture

The script is organized into modular functions:

- `load_background()` - Loads background image
- `tween_motion_vectors()` - Computes interpolated motion vectors
- `apply_motion_vector()` - Translates image by motion vector
- `setup_diffusion_model()` - Initializes Stable Diffusion img2img pipeline
- `process_frame_with_diffusion()` - Processes frame through diffusion model
- `generate_frame_sequence()` - Generates all frames (motion + diffusion)
- `compile_video()` - Assembles frames into MP4 video
- `run()` - Orchestrates the entire pipeline

## Notes

- This is an entry-level photo sequence generation method
- Low denoising strength (0.2-0.3) is key for retaining details
- Processing time depends on hardware (GPU recommended)
- First run downloads the Stable Diffusion model (~4GB)
- All code is modular and well-commented for easy expansion

