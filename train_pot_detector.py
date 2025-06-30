# ~/sim_ws/train_pot_detector.py
import os
from ultralytics import YOLO

if __name__ == '__main__':
    # Load a pre-trained YOLOv8n model (small, fast, good starting point for fine-tuning)
    model = YOLO('yolov8n.pt')

    # Path to your data.yaml file
    data_yaml_path = os.path.expanduser('~/yolo_pot_dataset/data.yaml')

    # Train the model
    # epochs: Number of training iterations. Start with 50-100.
    # imgsz: Image size for training. 640 is standard.
    # batch: Number of images per training batch. Adjust based on your computer's RAM/VRAM.
    # name: A name for the training run, results saved under runs/detect/{name}
    # device: 'cpu' to use CPU, '0' or 'cuda:0' for GPU if you have one.
    results = model.train(
        data=data_yaml_path,
        epochs=50, # You can increase this if accuracy isn't good enough
        imgsz=640,
        batch=4,    # Reduce if you get out-of-memory errors
        name='flower_pot_detector',
        save=True,
        verbose=True,
        device='cpu' # Change to '0' or 'cuda:0' if you have an NVIDIA GPU and CUDA set up
    )

    print("\nTraining completed! Check results in: ", results.save_dir)
    print("Your best model weights are usually saved at:", os.path.join(results.save_dir, 'weights', 'best.pt'))