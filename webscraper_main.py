# Databricks notebook source
# DBTITLE 1,PIP Install Requirements
# MAGIC %pip install -r "/Workspace/Repos/robert.altmiller@databricks.com/web_scraper/requirements.txt"

# COMMAND ----------

# DBTITLE 1,Library Imports
import requests, os, json, string, hashlib, re, random, time
from fake_useragent import UserAgent
from pyspark.sql.functions import *
from nltk import FreqDist
import pandas as pd
from bs4 import BeautifulSoup

# COMMAND ----------

# DBTITLE 1,Generic Helper Functions
# MAGIC %run "./generic_helpers"

# COMMAND ----------

# DBTITLE 1,Delta Lake Helper Functions
# MAGIC %run "./delta_helpers"

# COMMAND ----------

# DBTITLE 1,Hugging Face (HF) Large Language Models for Inference Functions
# MAGIC %run "./hf_llm_models"

# COMMAND ----------

# DBTITLE 1,Webscraper Helper Functions
# MAGIC %run "./webscraper_helpers"

# COMMAND ----------

# DBTITLE 1,Generate Web Scraped Data Recursively
def get_html_data(url_base = None, html_link_filter = None):
    """get html data process"""
    # file name as md5 hash
    filename = str_to_md5_hash(url_base)

    # get html data recursively (links, cleaned / raw data)
    html_data, html_cleaned_data = get_html_page_data(url_base)
    html_links = get_html_page_links(url_base, html_data, html_link_filter)

    # enrich html cleaned data with metrics for analyzing and cleaning data and append results to delta table
    summary_json, sentences_json = enrich_html_page_data_with_metrics(url_base, filename, html_data, html_cleaned_data, html_links)
    write_local_json_file(url_base, "summary", summary_json)
    write_local_json_file(url_base, "sentences", sentences_json)
    write_results_to_delta_lake(summary_json, sentences_json)
    
    if len(html_links) > 0: # then process more pages recursively using html_link_filter
        html_links_random = random.sample(html_links, len(html_links))
        for link in html_links_random:
            time.sleep(float(random.randrange(0, 5)) / 100)
            # check if website has already been processed
            if link != "ERROR - NO HTML Links Exist":
              print(f"next link being processed: {link}\n\n")
              foldername = get_folder_name(link)
              filename = str_to_md5_hash(link)
              filepath = f"scraped_data/{foldername}/{filename}_summary.json"
              if os.path.exists(f"./{filepath}") == False: # prevents recursive forever loops
                  print(f"path does not exists -- {filepath} - {link}\n\n")
                  get_html_data(url_base = link, html_link_filter = html_link_filter)
              else:
                  print(f"path already exists -- {filepath} - {link}\n\n")
                  continue

    return summary_json, sentences_json


"""main web scraper program"""
html_link_filter = "snowflake.com/blog"
url_base = "https://www.snowflake.com/blog/"
summary_json, sentences_json = get_html_data(url_base, html_link_filter)
print("finished processing available website urls...\n")
