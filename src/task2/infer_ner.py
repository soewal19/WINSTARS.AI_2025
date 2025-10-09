from transformers import pipeline

def extract_entities(text, model='dslim/bert-base-NER'):
    try:
        nlp = pipeline('ner', model=model, tokenizer=model, aggregation_strategy='simple')
    except TypeError:
        nlp = pipeline('ner', model=model, tokenizer=model)
    ents = nlp(text)
    return [e.get('word') for e in ents]
