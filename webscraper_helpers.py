# Databricks notebook source
# DBTITLE 1,Get HTML Page Data - Raw and Cleaned
def get_html_page_data(url = None):
    """get html page data - raw and cleaned"""
    html_data = requests.get(url = url).text
    clean_data = ' '.join(BeautifulSoup(html_data, "html.parser").stripped_strings)
    return html_data, clean_data.strip()

# COMMAND ----------

# DBTITLE 1,Get Local Folder Name For Writing Scraped Website Data
def get_folder_name(url_base = None):
    """
    get local folder name for writing scraped website data
    this can also be used for creating a delta table partition
    """
    # folder name (base website address used for scraping)
    foldername = f"{url_base.split('//')[1].split('/')[0]}"
    # remove puncuation from folder name
    foldername = modify_python_str(foldername, periodforunderscore = True)
    return foldername

# COMMAND ----------

# DBTITLE 1,Get Word Frequency Distribution From an Input String (Freq. Count > 1)
def get_word_frequency(inputstr: str) -> list:
  """get word frequency using nltk library"""
  from nltk import FreqDist

  def remove_single_occurrence_words(lst: list) -> list:
    """remove word frequencies with count == 1"""
    # create a dictionary to store the word frequency
    freq_dict = {}
    for word, freq in lst:
      freq_dict[word] = freq_dict.get(word, 0) + freq
    # remove the words with frequency 1
    new_lst = [(word, freq) for word, freq in lst if freq_dict[word] > 1]
    return new_lst

  # start execution here
  words = modify_python_str(inputstr, removepunctuation = True).split(" ")
  fdist1 = FreqDist(words)
  return remove_single_occurrence_words(fdist1.most_common())


# unit test
# text = "I am doing good and are you doign good?"
# print(get_word_frequency(text))

# COMMAND ----------

# DBTITLE 1,Get Content Metrics (Word Count, Word Frequency, Avg. Chars per Word, etc)
def enrich_html_page_data_with_metrics(filename: str, htmlcontent: str, htmllinks: str, cleanedcontent: str, sentence_summ_keep_ratio: float = 100) -> list:
  """
  get html clean page data split into sentences and enriched with metrics
  and hugging face Large Language Model data.
  """
  summ_length = len(cleanedcontent.strip())
  sentences = split_into_sentences(cleanedcontent)
  sentences_total = len(sentences)
  sentences_final = []
  sentences_summary = ''
  length_to_sent = 0
  counter = 1
  for sentence in sentences:
    sentence = sentence.strip()
    sentence = modify_python_str(sentence, removepunctuation = True)
    sent_word_count = len(sentence.split(" "))
    sent_letter_count = (len(sentence) - sent_word_count) + 1 # we subtract word_count in order to not count the spaces between the words.
    sent_avg_chars_per_word = sent_letter_count / sent_word_count # average number of characters per word in the sentence.
    length_to_sent += len(sentence) # this is the length from the start to get to the sentence (includes the length of the sentence too).
    sent_word_freq_count = len(get_word_frequency(sentence)) # count of words where word frequency > 1
    sent_length = len(sentence)
    sent_summ_ratio = (sent_length / summ_length) # sentence summary ratio is the ([sentence length] / [summary length])
        
    # filter out content which is bad (word count > 40 fiter out)
    # if filter_content_by_word_count(sent_word_count) == True: 
    #   continue

    if sent_summ_ratio > 0 and sent_summ_ratio < sentence_summ_keep_ratio: # get summary data that matters based on sentence length.
      sentences_summary = sentences_summary + sentence + ". "
      sentences_final.append({
        "sentence_key": counter,
        "sentence": sentence,
        "hf_sentence": get_hf_summary(sentence),
        "sentence_summary_ratio": sent_summ_ratio, 
        "sentence_length": sent_length, 
        "length_to_sentence": length_to_sent, 
        "sentence_total_words": sent_word_count, 
        "sentence_avg_chars_per_word": sent_avg_chars_per_word, 
        "sentence_words_that_repeat_count": sent_word_freq_count, 
        "sentence_words_that_repeat_total_words_ratio": sent_word_freq_count / sent_word_count, 
        "sentences_keywords": get_hf_keywords(sentence)
        #"sentence_word_frequency": get_word_frequency(f'{sentence}')
      })
      counter += 1
        
  # sentences summary word frequencies
  sentences_summary_word_freq = get_word_frequency(f'''{sentences_summary}''')
  sentences_summary_word_freq_count = len(sentences_summary_word_freq)

  summary_json = json.dumps([
    remove_blank_attributes({
      "filename_md5": filename,
      "sentences_total": sentences_total,
      "sentences_summary": sentences_summary,
      "sentences_html_summary": htmlcontent,
      "sentences_html_summary_links": htmllinks
    })
  ])

  sentences_json = json.dumps([
    remove_blank_attributes({
      "filename_md5": filename,
      "sentences_total": sentences_total,
      #"sentences_summary_word_frequency_count": sentences_summary_word_freq_count,
      #"sentence_summary_word_frequency": sentences_summary_word_freq, # count of words where word frequency > 1
      "sentences_detail": sentences_final
    })
  ])

  return summary_json, sentences_json

# COMMAND ----------

# DBTITLE 1,Filter Scraped Web Content Using Content Metrics
def filter_content_by_word_count(wordcount: int) -> bool:
    """filter our irrelevant content"""
    # in the english language most sentences should have 40 words on less with average being 15 - 20 words
    if wordcount > 40: return True # filter out sentences with word length > 40
    else: return False    

# COMMAND ----------

# DBTITLE 1,Get HTML Page Links / Urls
def get_html_page_links(url_base = None, html_data = None, html_link_filter = None):
    """get all links and remove duplicate links"""
    soup = BeautifulSoup(html_data, 'html.parser')
    embedded_urls = []
    for link in soup.find_all('a'):
        href_link = link.get('href')
        if href_link != None:
            emebedded_url = create_valid_page_links(url_base, href_link)
            if emebedded_url != None and html_link_filter in emebedded_url and exclude_file_ext_check(emebedded_url) == False: # html link filter
                embedded_urls.append(emebedded_url)
    return list(set(embedded_urls)) # remove duplicates (if any)


def create_valid_page_links(url_base = None, url_sub = None):
    """create a valid url for all html page links"""
    url_sub = url_sub.replace("\\","/")
    if len(url_sub) > 1: # we might have a valid url
        if url_sub.startswith("http") == True:
            return url_sub # valid url
        else: 
            return None

# COMMAND ----------

# DBTITLE 1,Split Web Scraped Content into Sentences
def split_into_sentences(text: str) -> list[str]:
    """
    Split the text into sentences.

    If the text contains substrings "<prd>" or "<stop>", they would lead 
    to incorrect splitting because they are used as markers for splitting.

    :param text: text to be split into sentences
    :type text: str

    :return: list of sentences
    :rtype: list[str]
    """

    # regex for alphabets, prefixes, suffixes, starters, acronyms, websites, digits, and multiple dots
    alphabets= "([A-Za-z])"
    prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
    suffixes = "(Inc|Ltd|Jr|Sr|Co)"
    starters = "(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
    acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
    websites = "[.](com|net|org|io|gov|edu|me)"
    digits = "([0-9])"
    multiple_dots = r'\.{2,}'

    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text)
    text = re.sub(multiple_dots, lambda match: "<prd>" * len(match.group(0)) + "<stop>", text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = [s.strip() for s in sentences]
    if sentences and not sentences[-1]: sentences = sentences[:-1]
    return sentences
