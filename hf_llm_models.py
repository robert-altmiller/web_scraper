# Databricks notebook source
# DBTITLE 1,Library Imports
import torch
import numpy as np
from pyspark.sql.functions import *
from pyspark.sql.types import *
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoModelForCausalLM, AutoModelForSeq2SeqLM, pipeline

# COMMAND ----------

# DBTITLE 1,Hugging Face Model to Get Sentiment as Positive or Negative from Text
# intialize tokenizer and model
print("initializing hf model: textattack/distilbert-base-uncased-CoLA.....\n")
tokenizer_sentiment = AutoTokenizer.from_pretrained("textattack/distilbert-base-uncased-CoLA")
model_sentiment = AutoModelForSequenceClassification.from_pretrained("textattack/distilbert-base-uncased-CoLA")

def get_hf_sentiment(text: str, model = model_sentiment, tokenizer = tokenizer_sentiment) -> str:
  """get sentence sentiment using hf model: distilbert-base-uncased-CoLA"""

  inputs = tokenizer.encode_plus(
      text,
      add_special_tokens = True,
      return_tensors = "pt",
      truncation = True,
      max_length = 512,
  )

  outputs = model(**inputs)
  logits = outputs.logits
  predicted_label = torch.argmax(logits, dim=1).item()
  predicted_sentiment = "Positive" if predicted_label == 1 else "Negative"
  return predicted_sentiment

# spark user defined function (UDF)
sentimentUDF = udf(lambda x: get_hf_sentiment(x), StringType()).asNondeterministic()

# COMMAND ----------

# DBTITLE 1,Hugging Face Model to Get Emotion from Text
# intialize classifer pipeline
print("initializing hf pipeline: michellejieli/emotion_text_classifier.....\n")
classifier = pipeline("sentiment-analysis", model = "michellejieli/emotion_text_classifier")

def get_hf_emotion(text: str) -> np.array:
  """get hugging face text emotion using hf model: emotion_text_classifier"""
  return classifier(text)

# spark user defined function (UDF)
emotionUDF = udf(lambda x: get_hf_emotion(x), StringType()).asNondeterministic()

# COMMAND ----------

# DBTITLE 1,Hugging Face Model to Get Keywords From Text
# initial tokenizer and model
print("initializing hf model: yanekyuk/bert-uncased-keyword-extractor.....\n")
tokenizer_keyword = AutoTokenizer.from_pretrained("yanekyuk/bert-uncased-keyword-extractor")
model_keyword = AutoModelForTokenClassification.from_pretrained("yanekyuk/bert-uncased-keyword-extractor")
ner_model = pipeline('ner', model = model_keyword, tokenizer = tokenizer_keyword)

def get_hf_keywords(text: str, model = ner_model) -> list:
    """get hugging face text keywords using hf model: bert-uncased-keyword-extractor"""    
    

    keywords = model(text)
    
    # add one additional fake entry at the end so we can pick up the last key phrase in the for loop below
    keywords.append({'entity': 'B-KEY', 'score': 0, 'index': 0, 'word': 'None', 'start': 0, 'end': 0})

    # Iterate keywords and create keyphrases, get average scores for appended words
    keyphrase = ''
    keyphrase_score = []
    keywords_dict = {}
    for i in range(len(keywords)):
        entity = keywords[i]["entity"]
        score = keywords[i]["score"]
        word = keywords[i]["word"]

        if entity.startswith("B"):
            # Add the previous keyphrase to final_keywords
            if keyphrase_score:
                avg_score = sum(keyphrase_score) / len(keyphrase_score)
                if "#" in keyphrase:
                    keyphrase = modify_python_str(keyphrase.strip(), removepunctuation=True, removespaces=True)
                
                # Consolidate duplicate keywords and calculate average scores
                if keyphrase in keywords_dict:
                    keywords_dict[keyphrase]["total_score"] += avg_score
                    keywords_dict[keyphrase]["count"] += 1
                else:
                    keywords_dict[keyphrase] = {"total_score": avg_score, "count": 1}

            # Reset keyphrase and keyphrase_score
            keyphrase = ''
            keyphrase_score = []

        # Append word to keyphrase and score to keyphrase_score
        keyphrase += word + " "
        keyphrase_score.append(score)

    # Convert the consolidated dictionary to a list of dictionaries with unique keywords and average scores
    final_keywords = [{"keyword": keyword, "score": keywords_dict[keyword]["total_score"] / keywords_dict[keyword]["count"]} for keyword in keywords_dict]

    return final_keywords

# spark user defined function (UDF)
keywordsUDF = udf(lambda x: get_hf_keywords(x), StringType()).asNondeterministic()

# COMMAND ----------

# DBTITLE 1,Hugging Face Model to Get Summary of Text
# initialize tokenizer and model
print("initializing hf model: facebook/bart-large-cnn.....\n")
tokenizer_bart = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
model_bart = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

def get_hf_summary(input_text: str, tokenizer = tokenizer_bart, model = model_bart) -> str:
    """get hugging face text summary using bart-large-cnn"""    

    # Encode input sequence
    input_ids = tokenizer.encode(input_text, max_length = 1024, truncation = True, return_tensors = "pt")
    
    # Generate output sequence
    output_ids = model.generate(input_ids, early_stopping=False)
    
    # Decode output sequence
    output_str = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output_str

# COMMAND ----------

# DBTITLE 1,Hugging Face Model to Get Text Completion From Incomplete Text
def get_hf_text_completion(text: str, max_length: int) -> str:
  """get hugging face text completion using hf model: databricks/dolly-v2-3b"""
  
  # initial tokenizer and model
  print("initializing hf model: databricks/dolly-v2-12b.....\n")
  tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v2-12b", padding_side = "left")
  model = AutoModelForCausalLM.from_pretrained("databricks/dolly-v2-12b", device_map = "auto", torch_dtype = torch.bfloat16)
  
  # text generation function
  def generate(text: str, max_length: int) -> str:
      input_ids = tokenizer.encode(text, return_tensors = "pt")
      input_ids = input_ids.to(model.device)
      generated_output = model.generate(input_ids, max_length = max_length, pad_token_id = tokenizer.pad_token_id)
      dd = tokenizer.decode(generated_output[0])
      return dd

  # return generated text    
  return generate(text, max_length)

# unit test
# results = get_hf_text_completion("you are a data engineer write apache spark scala code for the problem Read a csv file from adls with schema emp_id, first_name, last_name filter on first_name = 'John' and write the output to another adls path \nanswer: here is the apache spark scala code", max_length = 1000)
# print(results)

# COMMAND ----------

# DBTITLE 1,Hugging Face Model for Grammar Correction in Text
print("initializing hf pipeline: pszemraj/flan-t5-large-grammar-synthesis.....\n")
# pipeline definition
corrector = pipeline('text2text-generation', 'pszemraj/flan-t5-large-grammar-synthesis')

def get_hf_grammar_correction(text: str) -> str:
  """get hugging face text grammar correction using hf model: pszemraj/flan-t5-large-grammar-synthesis"""
  return corrector(text)[0]["generated_text"]


def get_levenstein_dist_percent(text1: str, text2: str) -> float:
  """get levenstein distance between two text strings"""
  from Levenshtein import distance as lev
  val = lev(text1, text2)
  return val, round(float(val/len(text1)),2)


def get_keep_text(text: str) -> bool:
  """
  determine if text is good or bad based on a % of the number changes 
  to go from the original to the gramatically correct text
  """
  text1 = text
  text2 = get_hf_grammar_correction(text)
  l_dist, l_dist_percent = get_levenstein_dist_percent(text1, text2)
  print(f"original_text: {text1}, text_length: {len(text1)}")
  print(f"fixed_text: {text2}, text_length: {len(text2)}")
  print(f"levenstein_val: {l_dist}, levenstein_percent: {l_dist_percent}")

  keep_text = False
  if l_dist_percent < .3: keep_text = True
  print(f"keep_text: {keep_text}") 
  return keep_text

# unit test
# text = """SEARCH HOMEPAGE CHEAT SHEET TOP 10 RIGHT NOW 1 Trump Furious That Jury Wasn\u2019t Told Name of Carroll\u2019s Cat BUT HER CAT!"""
# answer = get_keep_text(text)

# COMMAND ----------

# from transformers import pipeline
# summarizer = pipeline("summarization", model="philschmid/bart-large-cnn-samsum")

# conversation = '''Jeff: Can I train a 🤗 Transformers model on Amazon SageMaker? 
# Philipp: Sure you can use the new Hugging Face Deep Learning Container. 
# Jeff: ok.
# Jeff: and how can I get started? 
# Jeff: where can I find documentation? 
# Philipp: ok, ok you can find everything here. https://huggingface.co/blog/the-partnership-amazon-sagemaker-and-hugging-face                                           
# '''
# summarizer(conversation)
