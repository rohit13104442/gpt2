from flask import Flask, request, jsonify, render_template
from flasgger import Swagger
from transformers import GPT2LMHeadModel, GPT2Tokenizer,TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments,GPT2Config
import pdfplumber
import os
import tensorflow as tf
import tensorflow_datasets as tfds
import torch

from torch.utils.data.dataset import Dataset

app = Flask(__name__)
Swagger(app)

#  Endpoint for uploading PDF file
@app.route('/api/upload_pdf', methods=['POST'])
def upload_pdf():

    """
    Upload PDF File
    ---
    consumes:
      - multipart/form-data
    parameters:
      - in: formData
        name: pdf_file
        type: file
        required: true
        description: The PDF file to upload
    responses:
      200:
        description: Successfully uploaded PDF file
        content:
          application/json:
            schema:
              type: object
              properties:
                message:
                  type: string
    """
        

    pdf_file = request.files.get('pdf_file')

  # Define a function to convert PDF to text
    def convert_pdf_to_text(pdf_path, text_path):
      # Open the PDF file
        with pdfplumber.open(pdf_path) as pdf:
            text = ''     # Initialize an empty string to store extracted text
            # Iterate through each page of the PDF
            for page in pdf.pages:
                text += page.extract_text()
        # Open a text file in write mode with UTF-8 encoding
        with open(text_path, 'w', encoding='utf-8') as text_file:
            text_file.write(text)      # Write the extracted text to the text file

    # Define the paths to the PDF and text files
    pdf_path = pdf_file
    text_path = 'untitled.txt'

    # Convert the PDF to text
    convert_pdf_to_text(pdf_path, text_path)
    file_path = 'untitled.txt'

# Load and display the first few lines of the file
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()  # Read the first 1000 characters to get a sense of the content

    print(text)
    # !pip install transformers tensorflow
    # !pip install accelerate>=0.20.1
    # !pip install transformers -U
    model_name = 'gpt2'
    # config = GPT2Config.from_pretrained("gpt2", weight_decay=0.01, hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1)
    # model = GPT2LMHeadModel(config=config)
    # Initialize the GPT-2 model and tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
#     max_sequence_length = 10  # Example value, adjust as needed

# # Initialize the tokenizer with the desired maximum sequence length
#     tokenizer = GPT2Tokenizer.from_pretrained("gpt2", max_model_input_sizes={"gpt2": max_sequence_length})

    # Ensure the tokenizer uses the same EOS token as the model's original tokenizer
    tokenizer.pad_token = tokenizer.eos_token

    # Assuming the text has been preprocessed if necessary
    train_path = 'untitled.txt'
   
    # Create a TextDataset using the tokenizer and the preprocessed text data
    train_dataset = TextDataset(
      tokenizer=tokenizer,
      file_path=train_path,
      block_size=128)
    model.save_pretrained("./gpt2-finetuned-book")
    tokenizer.save_pretrained("./gpt2-finetuned-book")
    

    # # Initialize the DataCollatorForLanguageModeling
    data_collator = DataCollatorForLanguageModeling(
      tokenizer=tokenizer, mlm=False)
    # Define the training arguments
    training_args = TrainingArguments(
      output_dir="./gpt2-finetuned-book",
      overwrite_output_dir=False,
      num_train_epochs=3,             # an "epoch" refers to one complete pass through the entire training dataset during model training.
      per_device_train_batch_size=8,
      save_steps=5000,
      save_total_limit=2
    )
    # Initialize the Trainer with the defined model, training arguments, data collator, and dataset
    trainer = Trainer(

      model=model,
      args=training_args,
      data_collator=data_collator,
      train_dataset=train_dataset
    )
    # Start model training

    trainer.train()
    trainer.save_model()

    model.eval() 
    return jsonify({'message': 'PDF file uploaded successfully'})

@app.route('/api/query_output', methods=['POST'])
def query_output():
    """
    Query Output Based on Trained GPT-2 Model
    ---
    consumes:
      - application/json
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          properties:
            prompt_text:
              type: string
    responses:
      200:
        description: Generated text based on the prompt
        content:
          application/json:
            schema:
              type: object
              properties:
                generated_text:
                  type: string
    """
    data = request.get_json()
    prompt_text = data.get('prompt_text', '')
    def generate_text(model_path, sequence, max_length):
      model = GPT2LMHeadModel.from_pretrained(model_path)
      tokenizer = GPT2Tokenizer.from_pretrained(model_path)
      ids = tokenizer.encode(f'{sequence}', return_tensors='pt')
      final_outputs = model.generate(

        ids,
        do_sample=True,
        max_length=max_length,
        pad_token_id=model.config.eos_token_id,
        top_k=50,
        top_p=0.95
      )
    
      return(tokenizer.decode(final_outputs[0], skip_special_tokens=True))
    # def generate_text(prompt_text, model, tokenizer1, max_length=1000, num_return_sequences=1):

    # # Encode the prompt text
    #   input_ids = tokenizer1.encode(prompt_text, return_tensors='pt')

    # # Generate sequences
    #   output_sequences = model.generate(

    #     input_ids=input_ids,
    #     max_length=max_length,
    #     repetition_penalty=1.2,
    #     do_sample=True,
    #     num_return_sequences=num_return_sequences,
    #     pad_token_id=tokenizer1.eos_token_id,
    #     temperature=1.0
    #     )

    # # Decode and store each sequence
    #   generated_texts = []
    #   for i, generated_sequence in enumerate(output_sequences):

    #     generated_text = tokenizer1.decode(generated_sequence, skip_special_tokens=True)
    #     generated_texts.append(generated_text)

    #   return generated_texts

    model_name = './gpt2-finetuned-book'
   

    generated_texts = generate_text(model_name, prompt_text, max_length=1000)
    return jsonify({'generated_texts': generated_texts})


# @app.route('/apidocs')

# def render_swagger_ui():
#     return render_template('swaggerui.html')

if __name__ == '__main__':
    app.run(debug=True, port=5001)
