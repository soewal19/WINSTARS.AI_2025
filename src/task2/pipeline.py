import argparse
from src.task2.infer_ner import extract_entities
from src.task2.infer_image import predict_image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--image_model', default=None)
    parser.add_argument('--classes', nargs='*', default=None)
    args = parser.parse_args()
    ents = extract_entities(args.text)
    cls, prob = predict_image(args.image, model_path=args.image_model, classes=args.classes)
    print('Entities:', ents)
    print('Image prediction:', cls, prob)
    match = False
    for e in ents:
        if e and cls and e.lower() in cls.lower():
            match = True
    print('Match:', match)
    return match

if __name__=='__main__':
    main()
