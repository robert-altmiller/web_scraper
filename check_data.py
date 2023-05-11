# Databricks notebook source
# DBTITLE 1,Library Imports
import json
from pyspark.sql.functions import *

# COMMAND ----------

# DBTITLE 1,Delta Helper Functions
# MAGIC %run "./delta_helpers"

# COMMAND ----------

dbutils.fs.ls(f"/Workspace{get_local_notebook_path().rsplit('/', 1)[0]}/df_summary")

# COMMAND ----------

# DBTITLE 1,Display Web Scraped Summary Data
dfsummary = spark.read.load(f"/Workspace{get_local_notebook_path().rsplit('/', 1)[0]}/df_summary", format = "delta") \
  .dropDuplicates()
display(dfsummary)

# COMMAND ----------

# DBTITLE 1,Display Web Scraped Sentences Data
dfsentences = spark.read.load(f"/Workspace{get_local_notebook_path().rsplit('/', 1)[0]}/df_sentences", format = "delta")
dfsentences = dfsentences \
  .select("sentence_key", "url_base", "filename_md5", "sentences_total", "sentence", "sentence_avg_chars_per_word", "sentence_length", "length_to_sentence", "sentence_summary_ratio", "sentence_total_words", "sentence_words_that_repeat_count", "sentence_words_that_repeat_total_words_ratio", "hf_sentence_sentiment", "hf_sentence_emotion", "hf_sentences_keywords") \
  .dropDuplicates(["filename_md5","sentence_key"]) \
  .sort(asc("filename_md5"), asc("sentence_key"))
display(dfsentences)

# COMMAND ----------

df = dfsentences \
  .groupby("url_base") \
  .agg(
    count(col("url_base"))
  )
display(df)

# COMMAND ----------


