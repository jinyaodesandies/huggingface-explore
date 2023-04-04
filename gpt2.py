from transformers import pipeline

generator= pipeline("text-generation", model= "distilgpt2")
# telling the app what kind of thing and what specific model to use
res=generator("In this course, we will learn how to",
max_length=30,
num_return_sequences=2,
)
#what text, what length of text, how many returns respectively seperated by commas.
print(res)
