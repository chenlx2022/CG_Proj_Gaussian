


import torch
import os
import time
import numpy as np
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser

# Project imports
from scene import Scene
from gaussian_renderer import render, GaussianModel
from arguments import ModelParams, PipelineParams, get_combined_args
from utils.general_utils import safe_state
import torchvision


def render_view_set(output_dir, view_set_name, iteration, views, gaussians, pipeline, background):
    # setup output dirs
    render_dir = Path(output_dir) / view_set_name / f"ours_{iteration}" / "renders"
    gt_dir = Path(output_dir) / view_set_name / f"ours_{iteration}" / "gt"
    
    render_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Rendering {view_set_name} set ({len(views)} views)")
    print(f"Output: {render_dir}")
    print(f"{'='*60}\n")
    
    # warmup to avoid init overhead
    print("Warming up CUDA kernels...")
    warmup_views = min(3, len(views))
    for view in views[:warmup_views]:
        render(view, gaussians, pipeline, background)
    torch.cuda.synchronize()
    print(f"✓ Warmed up with {warmup_views} frames\n")
    
    frame_times = []
    total_start = time.perf_counter()
    
    for idx, view in enumerate(tqdm(views, desc=f"Rendering {view_set_name}")):
        torch.cuda.synchronize()
        frame_start = time.perf_counter()
        
        render_output = render(view, gaussians, pipeline, background)
        
        torch.cuda.synchronize()
        frame_end = time.perf_counter()
        frame_times.append((frame_end - frame_start) * 1000)
        
        rendered_image = render_output["render"]
        depth_map = render_output["depth"]
        
        gt_image = view.original_image[0:3, :, :]
        
        output_path = render_dir / f"{idx:05d}.png"
        torchvision.utils.save_image(rendered_image, str(output_path))
        
        gt_path = gt_dir / f"{idx:05d}.png"
        torchvision.utils.save_image(gt_image, str(gt_path))
        
        # save first depth map
        if idx == 0:
            depth_path = render_dir.parent / "depth" / f"{idx:05d}.png"
            depth_path.parent.mkdir(exist_ok=True)
            depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-6)
            torchvision.utils.save_image(depth_normalized.unsqueeze(0), str(depth_path))
    
    total_end = time.perf_counter()
    total_time = (total_end - total_start) * 1000
    
    print(f"\n{'='*60}")
    print(f"✓ Saved {len(views)} rendered images to {render_dir}")
    print(f"{'='*60}")
    
    if len(frame_times) > 0:
        frame_times = np.array(frame_times)
        avg_time = np.mean(frame_times)
        median_time = np.median(frame_times)
        std_time = np.std(frame_times)
        min_time = np.min(frame_times)
        max_time = np.max(frame_times)
        avg_fps = 1000.0 / avg_time
        
        print(f"\n📊 Performance Statistics ({view_set_name} set):")
        print(f"{'─'*60}")
        print(f"  Total time:        {total_time:.2f} ms  ({total_time/1000:.2f} s)")
        print(f"  Number of frames:  {len(views)}")
        print(f"  {'─'*58}")
        print(f"  Average frame time: {avg_time:.2f} ms  (FPS: {avg_fps:.2f})")
        print(f"  Median frame time:  {median_time:.2f} ms  (FPS: {1000/median_time:.2f})")
        print(f"  Std deviation:      {std_time:.2f} ms")
        print(f"  Fastest frame:      {min_time:.2f} ms  (FPS: {1000/min_time:.2f})")
        print(f"  Slowest frame:      {max_time:.2f} ms  (FPS: {1000/max_time:.2f})")
        print(f"{'='*60}\n")
    else:
        print(f"\n⚠️  No frames to render in {view_set_name} set")
        print(f"{'='*60}\n")


def main(model_params, pipeline_params, args):
    print("\n" + "="*60)
    print("3D Gaussian Splatting - Inference Rendering")
    print("="*60)
    print(f"Model path: {args.model_path}")
    print(f"Iteration: {args.iteration}")
    print("="*60 + "\n")
    
    safe_state(args.quiet)
    overall_start = time.perf_counter()
    
    with torch.no_grad():
        dataset = model_params.extract(args)
        pipeline = pipeline_params.extract(args)
        
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
        
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        print(f"✓ Loaded model at iteration {scene.loaded_iter}")
        print(f"✓ Number of Gaussians: {len(gaussians.get_xyz)}")
        print(f"✓ SH degree: {gaussians.active_sh_degree}")
        print(f"✓ Background: {'white' if dataset.white_background else 'black'}\n")
        
        if not args.skip_train:
            train_cameras = scene.getTrainCameras()
            render_view_set(dataset.model_path, "train", scene.loaded_iter,
                          train_cameras, gaussians, pipeline, background)
        
        if not args.skip_test:
            test_cameras = scene.getTestCameras()
            render_view_set(dataset.model_path, "test", scene.loaded_iter,
                          test_cameras, gaussians, pipeline, background)
    
    overall_end = time.perf_counter()
    overall_time = overall_end - overall_start
    
    print("\n" + "="*60)
    print("✓ Rendering complete!")
    print("="*60)
    print(f"⏱️  Total execution time: {overall_time:.2f} seconds")
    print("="*60 + "\n")


if __name__ == "__main__":
    parser = ArgumentParser(description="Render 3D Gaussian Splatting scenes")
    
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    
    parser.add_argument("--iteration", type=int, default=-1,
                        help="Iteration to load (-1 for latest)")
    parser.add_argument("--skip_train", action="store_true",
                        help="Skip rendering training views")
    parser.add_argument("--skip_test", action="store_true",
                        help="Skip rendering test views")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress non-essential output")
    
    args = get_combined_args(parser)
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model path does not exist: {args.model_path}")
        exit(1)
    
    main(model, pipeline, args)
