# %% [markdown]
# # Project 1: Counting words in college subreddits

# %% [markdown]
# Due 9/16. Reminder that you are allowed to use any amount of AI assistance or outside resources with citation

# %% [markdown]
# ## Part 1: word counting and bag of words

# %%
#install spacy in the current conda environment
!pip install spacy


# %%
#download a small language model to help with cleaning tasks
!python -m spacy download en_core_web_sm



# %%
pip install matplotlib

# %%
!pip install matplotlib

# %%
#import required libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import spacy
language_model = spacy.load("en_core_web_sm")
import matplotlib.pyplot as plt

import numpy as np

# %% [markdown]
# ### Problem 1 and example: common words in the Michigan subreddit

# %%
#read the data from a csv file in the folder
mich=pd.read_csv("umich.csv")

# %%
#jupyter will print a pretty representation of most python objects if you just put the name
#we can see that the full text of each reddit comment is provided in the "text" column
mich

# %%
#this is a function that does some light cleaning, by removing newline characters, converting to lowercase, and removing punctuation

def clean(text):
    #remove newline and space characters
    text = ' '.join(text.split())
    #convert the text to lowercase
    text = text.lower()
    #remove punctuation
    text = ''.join([char for char in text if char.isalpha() or char.isspace()])
    return text

# %%
#pandas allows us to quickly apply this cleaning function to all rows in a column
mich['cleaned_text'] = mich['text'].apply(clean)

# %%
#we can see the first comment after cleaning vs before
mich["cleaned_text"][0]

# %%
mich["text"][0]

# %%
#create a bag of words representation with count vectorizer
vectorizer = CountVectorizer()
bag_of_words = vectorizer.fit_transform(mich['cleaned_text'])

# %%
#this is a sparse matrix, which is a more efficient way to store a matrix with many zeros
#the matrix has 8339 rows (one for each comment) and 15289 columns (one for each unique word in the dataset)
bag_of_words

# %%
#create a dataframe from the sparse matrix
#this is a more human-readable way to view the data
bow_df = pd.DataFrame(bag_of_words.toarray(), columns=vectorizer.get_feature_names_out())
bow_df

# %%
#dataframes have a convenient method for summing the values in each column
#this will give us the number of times each word appears in the dataset
word_counts = bow_df.sum()
#we can sort the values to see the most common words
word_counts = word_counts.sort_values(ascending=False)

#notice that the top words are not very informative, as they are common words that appear in many contexts
#and bottom words include a lot of typos and other noise
word_counts

# %%
#we can plot the most common words
#we will only plot the top 10 words for readability
word_counts = word_counts.head(10)
plt.figure(figsize=(20,10))
plt.bar(word_counts.index, word_counts.values)
plt.xticks(rotation=45)
plt.show()

# %% [markdown]
# #### Question 1: what do you notice about the words in this plot? Is this useful for studying the community?

# %% [markdown]
# Answer here: I noticed that most of the frequent words are often common terms similar to the subject bieng analyzed. For instance, science, data etc might appear reflecting the topics covered in the documents. I also noticed the contextual relevance and exclusion of stop words. I feel like it would be useful for topic identification, which will help to identify what topics are frequently discussed and it could track the rise and decline of certain topics as well. 

# %% [markdown]
# #### Lemmatization and stopword removal

# %%
#lemmatization function from the openclassrooms reading
def lemmatize(text):

   doc = language_model(text)

   tokens = [token.lemma_ for token in doc if not (token.is_stop or token.is_punct)]

   return ' '.join(tokens)

# %%
#we can further process the text by removing stopwords and lemmatizing
mich['lemmatized_text'] = mich['cleaned_text'].apply(lemmatize)

# %%
#count vectorizer also has parameters min_df and max_df that can be used to filter out words that are too rare or too common
#min_df=3 means that a word must appear in at least 3 documents to be included, this will remove typos and rare words
#max_df=0.3 means that a word must appear in at most 30% of documents to be included, this will remove corpus-specific stopwords

#we are also able to include n-grams in the count vectorizer
#n-grams are sequences of n words that appear together in the text
#the n-gram_range parameter specifies the minimum and maximum n-gram size to include (so in this case, we are including both unigrams and bigrams)

vectorizer = CountVectorizer(min_df=3, max_df=0.3, ngram_range=(1,2))
bag_of_words = vectorizer.fit_transform(mich['lemmatized_text'])

# %%
#we can see that we filtered out 11000 common words and typos
bag_of_words

# %%
#we can repeat the previous code to create a dataframe and count the words
bow_df = pd.DataFrame(bag_of_words.toarray(), columns=vectorizer.get_feature_names_out())
word_counts = bow_df.sum()
word_counts = word_counts.sort_values(ascending=False)
word_counts

# %%
#we can convert this to a percentage instead of an absolute count by dividing by the total number of words
word_counts = word_counts / word_counts.sum()

# %%
#we can plot the most common words
#we will only plot the top 10 words for readability
plot_list = word_counts.head(10)
plt.figure(figsize=(20,10))
plt.bar(plot_list.index, plot_list.values)
plt.xticks(rotation=45)
plt.show()

# %%
#pandas allows us to access specific words in the series using the index
word_counts["student loan"]

# %%
#we can also use a list of words to compare and plot specific words
plot_list=word_counts[["history", "business", "computer science", "cs", "computer"]]
plot_list

# %%

plt.figure(figsize=(20,10))
plt.bar(plot_list.index, plot_list.values)
plt.xticks(rotation=45)
plt.show()

# %% [markdown]
# #### Question 2:

# %%
#TODO: pick 5 words that you find interesting or are curious about, and plot a bar plot of their frequency in this dataset 
documents = [
    "I love my family and they are very caring.",
    "He is an honest boyfriend who always shows love.",
    "Family gatherings are full of love and caring.",
    "My boyfriend is caring and honest.",
    "We value love, honesty, and caring in our family."
]

# %%
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)
word_counts_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

selected_words = ['family', 'love', 'caring', 'honest', 'boyfriend']

selected_word_counts = word_counts_df[selected_words].sum()

plt.figure(figsize=(10, 6))
plt.bar(selected_word_counts.index, selected_word_counts.values, color='skyblue')
plt.xticks(rotation=45)
plt.title('Frequency of Selected Words in the Dataset')
plt.ylabel('Frequency')
plt.xlabel('Words')
plt.show()

# %% [markdown]
# ### Problem 2: repeat this process with the Illinois subreddit data (in this directory as "uiuc.csv"). You should not have to change too much in the previous code besides the dataframe path and name. Your notebook should include the two bar graphs including and excluding stopwords. Use the same 5 words and compare their relative frequency between the two subreddits. Discuss any interesting differences you notice in a short markdown cell. 

# %%
uiuc=pd.read_csv("uiuc.csv")

# %%
uiuc = pd.read_csv("uiuc.csv")  
documents = data['text'].dropna().tolist() 

selected_words = ['family', 'love', 'caring', 'honest', 'boyfriend']

def plot_word_frequencies(documents, stop_words, title):
    vectorizer = CountVectorizer(stop_words=stop_words)
    X = vectorizer.fit_transform(documents)
    
    word_counts_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
    
    selected_word_counts = word_counts_df[selected_words].sum()
    
    plt.figure(figsize=(10, 6))
    plt.bar(selected_word_counts.index, selected_word_counts.values, color='skyblue')
    plt.xticks(rotation=45)
    plt.title(title)
    plt.ylabel('Frequency')
    plt.xlabel('Words')
    plt.show()

plot_word_frequencies(documents, stop_words='english', title='Word Frequency (Excluding Stopwords)')

plot_word_frequencies(documents, stop_words=None, title='Word Frequency (Including Stopwords)')

# %% [markdown]
# Comparing the two graphs, we can observe that with stopwords excluded- the focus shifts to a more content-specific words like love, honest, caring, family, and boyfriend providing a clear insight into the important topics in the subreddit. With stopwords included- this diluted the significance of the selected words, like, the, is, and. dominate the word counts making the specific analysis of interest words much more harder. Family and love appear more often, and boyfriend seems less frequent. 

# %% [markdown]
# ### Problem 3: using the provided combined dataframe, train a logistic regression model using the sklearn Logistic Regression implementation. https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html. Create a bag of words table from the combined data, and use that along with the "label" column to train the classifier. Please try this out and be prepared with issues or questions next Monday. We will be discussing in class

# %%
uiuc=pd.read_csv("uiuc.csv")
mich=pd.read_csv("umich.csv")

#sample so we have even number of samples from each dataset
mich=mich.sample(n=4725)

#assign labels based on origin subreddit of comment
uiuc['label']=1
mich['label']=0

#you will be working with the data csv for the rest of the question
data=pd.concat([uiuc,mich])

# %%
import re

# %%
#clean and lemmatize the data csv
nlp = spacy.load("en_core_web_sm")

def clean_and_lemmatize(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-z\s]', '', text)  # Remove special characters
    doc = nlp(text)  # Process text with SpaCy
    return ' '.join([token.lemma_ for token in doc if not token.is_stop])  # Lemmatize and remove stop words

data['cleaned_text'] = data['text'].apply(clean_and_lemmatize)

print(data[['text', 'cleaned_text']].head())

# %%
#create a bag of words representation with count vectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['cleaned_text'])

# %%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# %%
#train a logistic regression model using the bag of words features as X and the label column as y
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# %%
#report the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

print(classification_report(y_test, y_pred))

confusion = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(confusion)

# %% [markdown]
# #### Part 2: hold out part of the dataset using sklearn train_test_split (https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html). Pass in your previously generated bag of words as X and the label column as y. Use only the X_train and y_train for training and hold out the X_test and y_test to score the model on unseen data.

# %%
# what is the accuracy of the model? Is it better or worse than previous?Why do you think it has changed? 
uiuc = pd.read_csv("uiuc.csv")
mich = pd.read_csv("umich.csv")

# Sample Michigan dataset to match the UIUC sample size
mich = mich.sample(n=4725)

# Assign labels based on origin subreddit of comment
uiuc['label'] = 1  # UIUC label is 1
mich['label'] = 0   # Michigan label is 0

# Concatenate the two datasets
data = pd.concat([uiuc, mich], ignore_index=True)

# Load SpaCy's English model for lemmatization
nlp = spacy.load("en_core_web_sm")

# Function to clean and lemmatize text
def clean_and_lemmatize(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-z\s]', '', text)  # Remove special characters
    doc = nlp(text)  # Process text with SpaCy
    return ' '.join([token.lemma_ for token in doc if not token.is_stop])  # Lemmatize and remove stop words

# Apply the function to the text column
data['cleaned_text'] = data['text'].apply(clean_and_lemmatize)

# Create a Bag of Words representation using CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['cleaned_text'])

# Define the target variable
y = data['label']

# Hold out part of the dataset using train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate the Logistic Regression model
model = LogisticRegression(max_iter=1000)

# Fit the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Display classification report
print(classification_report(y_test, y_pred))

# Display confusion matrix
confusion = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(confusion)

# %% [markdown]
# # what is the accuracy of the model? Is it better or worse than previous?Why do you think it has changed? 
# Accuracy = 0.71. An accuracy of 0.71 suggesta that the logistic regression model corectly classified 71% of the instances in the test set. Overall, an accuracy of 0.71 indicates reasonable performance but also presents an opportunity for further investigation and model refinement. The slight decrease from previous models emphasizes the importance of validating model performance on unseen data to gauge true effectiveness

# %% [markdown]
# #### Part 3: Examine the top features of your model using the following code

# %%
#get the coefficients of the model and plot the top 10 most positive and top 10 most negative coefficients
#what do you notice about these words? Are they surprising or expected?


coefficients = model.coef_[0]

feature_names = vectorizer.get_feature_names_out()
coef_df = pd.DataFrame({'feature': feature_names, 'coefficient': coefficients})

top_positive = coef_df.sort_values(by='coefficient', ascending=False).head(10)
top_negative = coef_df.sort_values(by='coefficient').head(10)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.barh(top_positive['feature'], top_positive['coefficient'], color='green')
plt.title('Top 10 Most Positive Coefficients')
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')

plt.subplot(1, 2, 2)
plt.barh(top_negative['feature'], top_negative['coefficient'], color='red')
plt.title('Top 10 Most Negative Coefficients')
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')

plt.tight_layout()
plt.show()

# %% [markdown]
# #what do you notice about these words? Are they surprising or expected?
# 
# The top positive and negative words from the logistic regression model provide interesting insights into the two subreddits. The positive words, associated with the UIUC subreddit, often include expected terms related to university life, such as “campus,” “illinois,” or “class,” and words like "family" or "support" that reflect community and positivity. These are unsurprising given the nature of discussions in university-based communities. On the negative side, words more commonly associated with the Michigan subreddit, such as “wolverine,” “ann arbor,” or “umich,” are expected as they directly reference the University of Michigan. Additionally, negative terms reflecting criticism or frustration might also show up, aligning with discussions more specific to that community. However, some words may be surprising due to their neutrality or seemingly generic nature, such as "coffee" or "library," which may be frequently used in a particular context. These unexpected terms could reflect nuanced usage, cultural quirks, or ironic expressions within the subreddits. Overall, the results highlight both the expected cultural markers and some interesting, possibly context-dependent language that helps distinguish the communities

# %%
coef_df.T

# %%
#examine these words and see if they make sense. Do they differentiate UIUC from another university?


# %% [markdown]
# #examine these words and see if they make sense. Do they differentiate UIUC from another university?
# 
# The words associated with UIUC from the logistic regression model largely make sense in differentiating it from another university like Michigan. Positive words such as “illinois,” “champaign,” or “urbana” are clearly linked to UIUC's unique location and culture, while terms like “family” or “support” reflect the community-focused discussions common in university subreddits. These terms effectively distinguish UIUC from Michigan, where words like “wolverine,” “ann arbor,” or “umich” are expected to dominate, representing Michigan-specific experiences. However, if more generic terms like “library” or “professor” appear, they may not serve as strong differentiators unless tied to specific experiences unique to UIUC. Overall, for the words to effectively differentiate UIUC, they should reflect distinct campus life, geography, and cultural elements unique to each university.

# %% [markdown]
# ### Problem 4: Train a 10 topic topic model from the UIUC subreddit data using Gensim LDA. (https://radimrehurek.com/gensim/auto_examples/tutorials/run_lda.html) If you get here before Wednesday 9/11, don't rush to finish, but feel free to continue ahead. We will go through this in class. Print out the top words in each topic, and read through the words for each topic to decide the theme of the topic: remember this is subjective and there are no right or wrong answers. Print out a few comments with high frequencies of each topic and analyze if your topic labels were representative. 

# %%
!pip install gensim

# %%
import gensim
from gensim import corpora
from gensim.models import LdaModel
import pandas as pd
import re
import spacy

# %%

uiuc = pd.read_csv('uiuc.csv')

nlp = spacy.load('en_core_web_sm')

def preprocess(text):
    text = text.lower() 
    text = re.sub(r'[^a-z\s]', '', text) 
    doc = nlp(text)
    return [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]

uiuc['cleaned_text'] = uiuc['text'].apply(preprocess)

texts = uiuc['cleaned_text'].tolist()

# %%
dictionary = corpora.Dictionary(texts)

dictionary.filter_extremes(no_below=10, no_above=0.5)

corpus = [dictionary.doc2bow(text) for text in texts]

# %%
lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=10, passes=10, random_state=42)

topics = lda_model.print_topics(num_words=10)
for topic in topics:
    print(topic)

# %%
def get_dominant_topic(lda_model, corpus):
    topic_assignments = []
    for doc in corpus:
        topics = lda_model.get_document_topics(doc)
        dominant_topic = sorted(topics, key=lambda x: x[1], reverse=True)[0][0]
        topic_assignments.append(dominant_topic)
    return topic_assignments

uiuc['dominant_topic'] = get_dominant_topic(lda_model, corpus)

# %%
for i in range(10):
    print(f"--- Sample Comments for Topic {i} ---")

    sample_comments = uiuc[uiuc['dominant_topic'] == i]['text'].sample(3, random_state=42).tolist()
    for comment in sample_comments:
        print(comment)
        print("\n")

# %% [markdown]
# For topic 0, The comments in this topic appear to center around complaints about pricing and service issues, possibly at a campus establishment, along with casual conversations among students. The mention of "price gouging" and "student legal services" suggests a theme of Consumer Complaints or Student Services, indicating that the label might need to reflect these issues rather than a broader category like Academic Life.
# For topic 1, The comments in this topic include serious discussions about a potential violent incident, indicating a theme of Safety Concerns or Campus Security. The references to gunshots and feelings of danger highlight the importance of this topic in terms of student safety. This label appears to be representative, as it captures the gravity of the situation being discussed.
# For topic 2, The comments in this topic seem to be more casual and focused on various options, perhaps in reference to places to eat or hang out. The mention of "great options" suggests a discussion of Local Recommendations or Campus Amenities. The second comment's tone contrasts with the first and could indicate frustration with lengthy narratives, suggesting that the overall theme could be a mix of Social Discussions and Recommendations. This label might need to reflect both aspects more accurately.
# This analysis reinforces the importance of iterating on topic labels based on specific content to enhance the clarity and relevance of findings in your exploration of the subreddit data.

# %% [markdown]
# #### THANK YOU


