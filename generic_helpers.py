# Databricks notebook source
# DBTITLE 1,Library Imports
#library imports
import os, json, ast, string, re
import numpy as np
from nltk import FreqDist

# COMMAND ----------

# DBTITLE 1,Python String to MD5 Hash
def str_to_md5_hash(inputstr = None):
  """encode string using md5 hash"""
  return hashlib.md5(inputstr.encode('utf-8')).hexdigest()


# COMMAND ----------

# DBTITLE 1,Exclude File Extensions Check
def exclude_file_ext_check(inputstr = None):
    """check for excluded filename extensions"""
    exclude_file_name_ext = [".pdf", ".xls" ".xlsx", ".doc", ".docx", ".ppt", ".pptx"]
    for fileext in exclude_file_name_ext:
        if fileext in inputstr: return True
    return False

# COMMAND ----------

# DBTITLE 1,Modify a Python String with Edits and Changes
def modify_python_str(
        inputstr: str, 
        lowercase: bool = False,
        uppercase: bool = False,
        removenumbers: bool = False,
        removespaces: bool = False,
        removepunctuation: bool = False,
        singledashes: bool = False,
        periodforunderscore: bool = False
    )-> string:
    """modified a python string with edits and changes"""
    if lowercase: inputstr = inputstr.lower()
    if uppercase: inputstr = inputstr.upper()
    if periodforunderscore: inputstr = inputstr.replace(".", "_")
    if removenumbers: inputstr = re.sub(r'[0-9]', '', inputstr)
    if removespaces: inputstr = inputstr.replace(' ', '')
    if removepunctuation:
        punctuation = [punct for punct in str(string.punctuation)]
        punctuation.remove("-")
        for punct in punctuation:
            inputstr = inputstr.replace(punct, '')
    if singledashes: inputstr = re.sub(r'(-)+', r'-', inputstr)
    return inputstr

# COMMAND ----------

# DBTITLE 1,Remove Null and Blank Attributes from Complex Nested Json Dictionary
def remove_blank_attributes(json_dict):
  """
  recursive function to remove all null, none, [], and other blank
  attributes from a complex nested json dictionary
  """
  cleaned_dict = {}
  for key, value in json_dict.items():
    if value:
      if isinstance(value, dict):
        cleaned_value = remove_blank_attributes(value)
        if cleaned_value: cleaned_dict[key] = cleaned_value
      elif isinstance(value, list):
        cleaned_list = []
        for item in value:
          if isinstance(item, dict):
            cleaned_item = remove_blank_attributes(item)
            if cleaned_item: cleaned_list.append(cleaned_item)
          elif item: cleaned_list.append(item)
        if cleaned_list: cleaned_dict[key] = cleaned_list
      else: cleaned_dict[key] = value
  return cleaned_dict

# COMMAND ----------

# DBTITLE 1,Flatten Complex Nested Json
def format_json_col_as_struct(df = None, payload_col = None):
    """
    format spark json string column as a struct
    return type is a spark dataframe with json string columns formatted as a struct
    """
    json_schema = spark.read.json(df.rdd.map(lambda row: row[payload_col])).schema
    df = df.withColumn(payload_col, from_json(col(payload_col), json_schema))
    return df


def flatten_df(nested_df, prefix):
    """
    flatten nested json struct columns
    return type is spark dataframe with json struct column exploded into multiple spark columns
    """
    flat_cols = [c[0] for c in nested_df.dtypes if c[1][:6] != 'struct']
    nested_cols = [c[0] for c in nested_df.dtypes if c[1][:6] == 'struct']
    flat_df = nested_df.select(flat_cols +
                                [col(nc + '.' + c).alias(prefix + c)
                                for nc in nested_cols
                                for c in nested_df.select(nc + '.*').columns])
    return flat_df

# COMMAND ----------

# DBTITLE 1,Write Json File Content To Local Folder
def write_local_json_file(url_base = None, fname_prefix = None, data = None):
    """write scraped data to local"""
    url_base = modify_python_str(url_base, lowercase = True)
    # folder name (base website address used for scraping)
    foldername = get_folder_name(url_base)
    print(f"foldername: {foldername}")
    # file name (md5 hash website address used for scraping)
    filename = f"{str_to_md5_hash(url_base)}_{fname_prefix}.json"
    print(f"filename: {filename}")
    # local folder path
    local_folder_path = f"scraped_data/{foldername}\n\n"
    if os.path.isdir(local_folder_path) == False: os.makedirs(local_folder_path)
    with open(f"./{local_folder_path}/{filename}", "w") as f:
       f.write(data)
