from transformers import pipeline

MODEL_ID = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

clf = pipeline("sentiment-analysis", model=MODEL_ID)

texts = [
    "I love this course!",
    "This was a terrible experience.",
]

for t in texts:
    result = clf(t)
    print(result)