from transformers import pipeline

classifier =  pipeline("zero-shot-classification")
#pipeline is the thing you use to define what kind of model you want
# classifier is like a function call where you say what you want.
sequence = "this is a course about python list comprehension"
#sequence is the text you want it to read
canidate_labels=["education", "politics", "business"]
#canidata labels are what it might have to do with
res = classifier(sequence,canidate_labels)
#calls the sequence.
print(res)
