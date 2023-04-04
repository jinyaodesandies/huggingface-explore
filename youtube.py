from transformers import pipeline
classifier = pipeline("sentiment-analysis")
res= classifier("ive been waiting for huggingface my whole life")
print(res)
