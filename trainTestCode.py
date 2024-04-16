import spacy
import random
from spacy.util import minibatch, compounding

# Load SpaCy with a blank Armenian model
nlp = spacy.blank("hy")

# Example training data (replace this with your own labeled data)
TRAIN_DATA = [
    ("Արմեն անձնակազմի պետ գործակալ է։", {"entities": [(0, 5, "PERSON")]}),
    ("Ու՞րախական անուն եք։", {"entities": [(0, 7, "ORG")]}),
    ("Երևանի գավառակագություններից մեկը Սասունցին է։", {"entities": [(0, 12, "LOCATION")]}),
    # Add more data...
]

# Add named entity recognizer to the pipeline
ner = nlp.create_pipe("ner")
nlp.add_pipe(ner, last=True)

# Add labels
for _, annotations in TRAIN_DATA:
    for ent in annotations.get("entities"):
        ner.add_label(ent[2])

# Disable other pipelines during training
pipe_exceptions = ["ner"]
other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

# Train the model
with nlp.disable_pipes(*other_pipes), open("log.txt", "w") as f:
    optimizer = nlp.begin_training()
    for itn in range(10):  # Adjust number of iterations
        random.shuffle(TRAIN_DATA)
        losses = {}
        batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
        for batch in batches:
            texts, annotations = zip(*batch)
            nlp.update(texts, annotations, drop=0.5, losses=losses)
        print(losses, file=f)

# Save the trained model
nlp.to_disk("armenian_ner_model")

# Load the trained model
nlp_loaded = spacy.load("armenian_ner_model")

# Test the model
test_text = "Արմեն անձնակազմի պետ գործակալ է։"
doc = nlp_loaded(test_text)
for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)
