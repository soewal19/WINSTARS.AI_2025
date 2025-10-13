Usage summary:

1. Create virtual env and install:
   python3 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt

2. Task1 (MNIST):
   python src/task1/train_mnist.py --algo all --epochs 2 --limit 2000 --out_dir models/mnist
   python src/task1/inference_mnist.py --algo cnn --model models/mnist/cnn_model_tf

3. Task2 (Animals + NER):
   - Download Animals-10 via Kaggle and place into data/animals10/
   - Train image model:
       python src/task2/train_image_tf.py --data_dir data/animals10/train --val_dir data/animals10/test --out_dir models/image --epochs 2
   - Train NER (toy example):
       python src/task2/ner_train.py --out models/task2/ner_model
   - Run pipeline:
       python src/task2/pipeline_tf.py --text "There is a tiger" --image_path data/animals10/test/tiger/xxx.jpg --model_dir models/image/resnet50_animals10_tf --classes_json models/image/classes.json
