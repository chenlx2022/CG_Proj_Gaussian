#!/usr/bin/env python3
#
# Simplified 3D Gaussian Splatting Rendering Script
# For CG Course Final Project - Inference Only
#
# This script demonstrates the core rendering pipeline without
# training-specific features. It's designed to be readable and
# to clearly show the data flow from trained model to rendered images.
#

import torch
import os
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
    """
    Render a set of camera views and save results.
    
    Args:
        output_dir: Base output directory
        view_set_name: Name of view set ("train" or "test")
        iteration: Model iteration number
        views: List of camera viewpoints
        gaussians: GaussianModel instance
        pipeline: Pipeline configuration
        background: Background color tensor [3] on GPU
    """
    # Setup output directories
    render_dir = Path(output_dir) / view_set_name / f"ours_{iteration}" / "renders"
    gt_dir = Path(output_dir) / view_set_name / f"ours_{iteration}" / "gt"
    
    render_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Rendering {view_set_name} set ({len(views)} views)")
    print(f"Output: {render_dir}")
    print(f"{'='*60}\n")
    
    # Render each view
    for idx, view in enumerate(tqdm(views, desc=f"Rendering {view_set_name}")):
        # Core rendering call - simplified interface
        render_output = render(view, gaussians, pipeline, background)
        
        rendered_image = render_output["render"]  # [3, H, W]
        depth_map = render_output["depth"]        # [H, W]
        
        # Get ground truth image
        gt_image = view.original_image[0:3, :, :]  # [3, H, W]
        
        # Save rendered image
        output_path = render_dir / f"{idx:05d}.png"
        torchvision.utils.save_image(rendered_image, str(output_path))
        
        # Save ground truth for comparison
        gt_path = gt_dir / f"{idx:05d}.png"
        torchvision.utils.save_image(gt_image, str(gt_path))
        
        # Optional: Save depth map
        if idx == 0:  # Save first depth map as example
            depth_path = render_dir.parent / "depth" / f"{idx:05d}.png"
            depth_path.parent.mkdir(exist_ok=True)
            # Normalize depth for visualization
            depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-6)
            torchvision.utils.save_image(depth_normalized.unsqueeze(0), str(depth_path))
    
    print(f"✓ Saved {len(views)} rendered images to {render_dir}")


def main(model_params, pipeline_params, args):
    """
    Main rendering pipeline.
    
    Workflow:
        1. Load trained Gaussian model
        2. Load camera viewpoints
        3. Render each view
        4. Save results
    """
    print("\n" + "="*60)
    print("3D Gaussian Splatting - Inference Rendering")
    print("="*60)
    print(f"Model path: {args.model_path}")
    print(f"Iteration: {args.iteration}")
    print("="*60 + "\n")
    
    # Initialize RNG for reproducibility
    safe_state(args.quiet)
    
    # Disable gradient computation (inference only)
    with torch.no_grad():
        # Extract model and pipeline configurations from parsed args
        dataset = model_params.extract(args)
        pipeline = pipeline_params.extract(args)
        
        # Initialize Gaussian model
        gaussians = GaussianModel(dataset.sh_degree)
        
        # Load scene (cameras + trained Gaussians)
        scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
        
        # Setup background color
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        print(f"✓ Loaded model at iteration {scene.loaded_iter}")
        print(f"✓ Number of Gaussians: {len(gaussians.get_xyz)}")
        print(f"✓ SH degree: {gaussians.active_sh_degree}")
        print(f"✓ Background: {'white' if dataset.white_background else 'black'}\n")
        
        # Render training views
        if not args.skip_train:
            train_cameras = scene.getTrainCameras()
            render_view_set(
                dataset.model_path,
                "train",
                scene.loaded_iter,
                train_cameras,
                gaussians,
                pipeline,
                background
            )
        
        # Render test views
        if not args.skip_test:
            test_cameras = scene.getTestCameras()
            render_view_set(
                dataset.model_path,
                "test",
                scene.loaded_iter,
                test_cameras,
                gaussians,
                pipeline,
                background
            )
    
    print("\n" + "="*60)
    print("✓ Rendering complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Command line argument parser
    parser = ArgumentParser(description="Render 3D Gaussian Splatting scenes")
    
    # Model and pipeline parameter groups
    # These objects will parse their specific arguments from the command line
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    
    # Rendering options
    parser.add_argument("--iteration", type=int, default=-1,
                        help="Iteration to load (-1 for latest)")
    parser.add_argument("--skip_train", action="store_true",
                        help="Skip rendering training views")
    parser.add_argument("--skip_test", action="store_true",
                        help="Skip rendering test views")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress non-essential output")
    
    # Parse all arguments
    args = get_combined_args(parser)
    
    # Validate arguments
    if not os.path.exists(args.model_path):
        print(f"Error: Model path does not exist: {args.model_path}")
        exit(1)
    
    # Run rendering
    # Pass the parameter objects and parsed args separately
    main(model, pipeline, args)
