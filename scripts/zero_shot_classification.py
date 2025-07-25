from transformers import pipeline

sentences = [
    "Our new product uses recycled materials and renewable energy in production.",
    "We claim to be eco-friendly without providing any supporting data."
]

classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')

labels = ['clean', 'greenwashing']
results = classifier(sentences, candidate_labels=labels)

for sent, res in zip(sentences, results):
    label = res['labels'][0]
    score = res['scores'][0]
    print(f"{sent} => {label} ({score:.2f})")
