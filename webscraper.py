# Databricks notebook source
# DBTITLE 1,PIP Install Requirements
# MAGIC %pip install -r "/Workspace/Repos/robert.altmiller@databricks.com/web_scraper/requirements.txt"

# COMMAND ----------

# DBTITLE 1,Library Imports
import requests, os, json, string, hashlib, re, random, time
from pyspark.sql.functions import *
from nltk import FreqDist
import pandas as pd
from bs4 import BeautifulSoup

# COMMAND ----------

# DBTITLE 1,Generic Helper Functions
# MAGIC %run "./generic_helpers"

# COMMAND ----------

# DBTITLE 1,Hugging Face (HF) Large Language Models for Inference
# MAGIC %run "./hf_llm_models"

# COMMAND ----------

# DBTITLE 1,Webscraper Helper Functions
# MAGIC %run "./webscraper_helpers"

# COMMAND ----------

def get_html_data(url_base = None, html_link_filter = None):
    # file name as md5 hash
    filename = str_to_md5_hash(url_base)

    # get html data recursively (links, cleaned / raw data)
    html_data, html_cleaned_data = get_html_page_data(url_base)

    html_links = get_html_page_links(url_base, html_data, html_link_filter)

    # enrich html cleaned data with metrics for analyzing and cleaning data
    summary_json, sentences_summary = enrich_html_page_data_with_metrics(filename, html_data, html_links, html_cleaned_data)
    
    return summary_json, sentences_summary


    # html_links = get_html_page_links(url_base, html_data, html_link_filter)

    # write_local_json_file(url_base, html_cleaned_data, html_links)

    # if len(html_links) > 0: # then process more pages recursively using html_link_filter
    #     html_links_random = random.sample(html_links, len(html_links))
    #     for link in html_links_random:
    #         #time.sleep(float(random.randrange(0, 2))/100)
    #         # check if website has already been processed
    #         foldername = get_folder_name(link)
    #         filename = str_to_md5_hash(link)
    #         filepath = f"scraped_data/{foldername}/{filename}.json"
    #         if os.path.exists(filepath) == False: # prevents recursive forever loops
    #             print(f"path does not exists -- {filepath} - {link}")
    #             get_html_data(url_base = link, html_link_filter = html_link_filter)
    #         else:
    #             print(f"path already exists -- {filepath} - {link}")
    #             continue
    # return html_links


"""web scraper"""
html_link_filter = "cnn.com"

url_base = "https://edition.cnn.com/2023/05/06/politics/nationalism-conservatism-what-matters/index.html"
summary_json, sentences_json = get_html_data(url_base, html_link_filter)

df_summary = spark.createDataFrame(pd.read_json(summary_json)) \
    .select("filename_md5", "sentences_total", "sentences_summary", "sentences_html_summary", "sentences_html_summary_links") \
    .cache()


df_sentences = flatten_df(spark.createDataFrame(pd.read_json(sentences_json)) \
    .select("filename_md5", "sentences_total", explode(col("sentences_detail")).alias("sentences_detail")), prefix = "") \
    .cache()
df_sentences = df_sentences \
  .withColumn("hf_sentence_sentiment", sentimentUDF(col("sentence"))) \
  .withColumn("hf_sentence_emotion", emotionUDF(col("sentence")))

# COMMAND ----------

# DBTITLE 1,Display Summary Data
display(df_summary)

# COMMAND ----------

# DBTITLE 1,Display Sentences Data
display(df_sentences)

# COMMAND ----------


