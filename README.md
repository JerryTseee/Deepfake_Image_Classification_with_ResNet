# Deepfake Image Classification

This project trains a ResNet50-based image classifier on the dataset in `dataset/FINAL_DATASET.csv`. The code is split into separate modules for dataset preparation, training, and inference so each stage can be run independently.

## Project Layout

- `prepare_dataset.py` downloads and organizes images by split.
- `train.py` trains the model (last few layers of cnn) and saves the best checkpoint.
- `inference.py` loads the saved checkpoint and evaluates it on the test split.
- `run.py` runs the full pipeline end to end.
- `src/` contains shared code for config, data handling, model definition, training, inference, and orchestration.

## Requirements

Install the Python dependencies used by the project:

```bash
pip install pandas numpy torch torchvision pillow requests tqdm scikit-learn matplotlib
```

If you are using a CUDA-capable machine, install the PyTorch build that matches your GPU and driver setup from the official PyTorch instructions.

## Data

The repository expects this file to exist:

- `dataset/FINAL_DATASET.csv`

Running the preparation step will create these folders if needed:

- `dataset/images/train`
- `dataset/images/val`
- `dataset/images/test`

Downloaded image files are stored under the matching split directory.

## How To Run

Run each stage separately:

```bash
python prepare_dataset.py
python train.py
python inference.py
```

Or run the full pipeline with one command:

```bash
python run.py
```

## Outputs

- `best_resnet_deepfake.pth` stores the best model checkpoint.
- `training_curves.png` stores the training loss and validation accuracy plot.

## Notes

- Validation uses the explicit `val` split when it exists in the CSV.
- If the CSV does not contain a validation split, one is created from the training rows.
- The model labels are derived from `label_numeric`, 0 means FAKE and 1 means REAL