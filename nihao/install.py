from transformers import GPT2Tokenizer, GPT2Model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')
model = GPT2Model.from_pretrained('gpt2-xl')
text = "hello"
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
print(output)