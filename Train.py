from transformers import BertTokenizer, TextDataset, DataCollatorForLanguageModeling, TrainingArguments, Trainer, GPT2LMHeadModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

dataset = TextDataset(tokenizer= AutoTokenizer,file_path="token.txt", block_size=1000000, )



# Initialize the tokenizer

# Test encoding and decoding
text = dataset
encoded = tokenizer.encode(text)
decoded = tokenizer.decode(encoded)

# Check the output
print("Encoded:", encoded)
print("Decoded:", decoded)


# Create the DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(tokenizer=key, mlm=False)

# Initialize trainer
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,
    per_device_train_batch_size=16,
    save_steps=1000,
    save_total_limit=2
)

# Instantiate the GPT2LMHeadModel
Gpt = GPT2LMHeadModel.from_pretrained('gpt2')

trainer = Trainer(
    model=Gpt,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator
)

# Start training
trainer.train()
