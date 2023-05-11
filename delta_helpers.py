# Databricks notebook source
# DBTITLE 1,Get Local Notebook Path
def get_local_notebook_path():
    """get local notebook path"""
    jsondata = json.loads(dbutils.notebook.entry_point.getDbutils().notebook().getContext().toJson())
    nb_path = jsondata['extraContext']['notebook_path']
    return nb_path

# COMMAND ----------

# DBTITLE 1,Write Web Scraped Results to Delta Lake
def write_results_to_delta_lake(summary_json = None, sentences_json = None, mode = "append"):
  """
  write results to delta lake tables in databricks
  mode can be 'append' or 'overwrite'
  """
  # df_summary spark data for all website page content
  df_summary = spark.createDataFrame(pd.read_json(summary_json)) \
      .select("url_base", "filename_md5", "sentences_total", "sentences_summary", "sentences_html_summary", "sentences_html_summary_links") \
  # write to delta lake
  df_summary.write.format("delta").option("mergeSchema", "true").mode(mode).save(f"/Workspace{get_local_notebook_path().rsplit('/', 1)[0]}/df_summary")

  # df_sentences spark dataframe for all sentences website page
  df_sentences = flatten_df(spark.createDataFrame(pd.read_json(sentences_json)) \
      .select("url_base", "filename_md5", "sentences_total", explode(col("sentences_detail")).alias("sentences_detail")), prefix = "") \
      .cache()
  # enrich df_sentences with sentiment and emotion from hugging face LLMs
  df_sentences = df_sentences \
    .withColumn("hf_sentence_sentiment", sentimentUDF(col("sentence"))) \
    .withColumn("hf_sentence_emotion", emotionUDF(col("sentence")))
  # write to delta lake
  df_sentences.write.format("delta").option("mergeSchema", "true").mode(mode).save(f"/Workspace{get_local_notebook_path().rsplit('/', 1)[0]}/df_sentences")

  # unpersist cached dataframes 
  df_sentences.unpersist()
