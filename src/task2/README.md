# Task 2: Multimodal Pipeline (NER + Image Classification)

This task implements a multimodal pipeline that combines Named Entity Recognition (NER) and image classification to verify if a text description matches an image content.

## Components

1. **Named Entity Recognition** - Extracts animal names from text using transformer models
2. **Image Classification** - Classifies animal images using ResNet18
3. **Pipeline** - Combines both models to verify if entities in text match the image content

## Dataset

The pipeline uses the Animals-10 dataset with 10 classes:
- bear, cat, cow, dog, elephant, giraffe, horse, monkey, sheep, zebra

## Models

1. **NER Model**: Uses HuggingFace transformer `dslim/bert-base-NER`
2. **Image Classifier**: ResNet18 trained on the animal dataset

## Usage

### Training the Image Classifier

```bash
python train_image.py --data_dir <path> --output_dir <path>
```

### Running the Pipeline

```bash
python pipeline.py --text "There is a cow in the picture" --image <path_to_image>
```

### Example

```bash
python pipeline.py \
  --text "There is a zebra in the picture" \
  --image data/animals10_demo/zebra/zebra_1.jpg \
  --image_model models/image/resnet18.pth \
  --classes bear cat cow dog elephant giraffe horse monkey sheep zebra
```

## Output

The pipeline returns a boolean value indicating whether any entity extracted from the text matches the predicted image class.