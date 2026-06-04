# Controllable Drone Scene Simulation - Baseline Tracking

This repository contains the foundational computer vision and data extraction pipeline implemented as preparation for the **Editable Scene Simulation for Autonomous Drones** research project (Summer of Open AI Research 2026 - Oreon Labs).

## Pipeline Demo
https://www.pexels.com/pt-br/video/ao-redor-31855544/

## Objective
The goal of this baseline is to demonstrate the extraction of structured temporal data from raw drone footage, which is a prerequisite for controllable scene-editing and synthetic data generation. Instead of just rendering bounding boxes, this pipeline manually handles frame-by-frame memory management to prevent OOM errors on constrained hardware and exports the tracking coordinates into a standardized format.

## Architecture & Data Extraction
- `processar_drone.py`: The main execution script. It utilizes OpenCV for manual stream management and YOLOv8 (nano) with ByteTrack for instance segmentation and object tracking.
- `drone_trajectories.json`: The extracted database. It contains frame-by-frame tracking IDs and `xyxy` bounding box coordinates, acting as the structured API between the vision model and any subsequent 3D simulation or scene-editing engine.

## Quick Start
```bash
# Isolate the environment
python3 -m venv venv
source venv/bin/activate

# Install strictly required dependencies
pip install ultralytics opencv-python

# Run the manual tracking and extraction pipeline
python processar_drone.py
