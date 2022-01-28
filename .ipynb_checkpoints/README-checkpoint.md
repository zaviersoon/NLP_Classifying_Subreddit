# Project 3 Web APIs & NLP

### Problem Statement

Men In Black (MIB) is a government entity that researches and monitors the sentiment of people on aliens, whether conspiracies that are made are a little too close to things that people should not know about. As a data science team working for Reddit, MIB request us to explore and develop a model to be able to accurately classify reddit posts and identify the key words that differentiates posts related from "Aliens" to "Space" in reddit. This would facilitate MIB's intelligence collection efforts about sentiments from ordinary citizens. This information can also be useful for other associations to promote its marketing effort on social media, events and podcasts.

Three classification models, Logistic Regression, Naive Bayes and Random Forest will be developed to assist with the problem statement. The performance of the model will be assessed based on its Accuracy and F1-score on unseen test data.

### Background

Are there aliens out there? The real starting point for UFO speculation, and possible government involvement, dates to Roswell, New Mexico, in July 1947. Project Blue Book was a program to investigate UFO sightings by the United States Air Force from March 1952 to its termination on December 17, 1969. A U.S. government report([*source*](https://www.dni.gov/files/ODNI/documents/assessments/Prelimary-Assessment-UAP-20210625.pdf)) on UFOs says it found no evidence of aliens but acknowledged 143 reports of "unidentified aerial phenomena" since 2004 that could not be explained. The report was released on 25th June 2021 by the Office of the Director of National Intelligence with substantial input from the military. The study is part of the most significant public effort so far to deal with decades of speculation, rumor and unhinged conspiracy theories about UFOs. 

Some of the most intriguing cases come from Navy pilots who reported seeing UFOs — and filming some of them — off the East Coast of the U.S. over a period of months in 2014 and 2015. The pilots, including some who have spoken publicly, say the mystery objects moved with exceptional speed, agility and acceleration that they had never seen before. And in some incidents, the pilots said the objects went underwater.

### Contents

- Part 1 Data Collection
- Part 2 Exploratory Data Analysis(EDA) & Cleaning
- Part 3 Preprocessing and Modeling

### Data Dictionary

|Feature|Type|Dataset|Description|
|---|---|---|---|
|subreddit|object|df_combine.csv|Subreddit to which the post/submission belongs (1 = Aliens, 0 = Space)|
|title|object|df_combine.csv|Title of post/submission| 
|selftext|object|df_combine.csv|Body of post/submission|
|content|object|df_combine.csv|Feature engineering of title and selftext| 
|content_length|int64|df_combine.csv|The length of each content|
|content_word_count|int64|df_combine.csv|The number of words in each content|
|content_token_lemme|object|df_combine.csv|Content which tokenized and lemmatized| 
|content_token_stem|object|df_combine.csv|Content which tokenized and stemmed| 

### Conclusions

|Model|Vectorizer|Train Score|Test Score|F1_score|AUC|Remarks|
|---|---|---|---|---|---|---|
|LogisticRegression(random_state=42)|CountVectorizer(stop_words='english')|0.9774|0.8991|0.8996|0.96|With Hyperparameter Tuning| 
|LogisticRegression(random_state=42)|TfidfVectorizer(stop_words='english')|0.9718|0.9101|0.914|0.96|With Hyperparameter Tuning|
|MultinomialNB()|CountVectorizer(stop_words='english')|0.9718|0.8969|0.9011|0.96|With Hyperparameter Tuning|
|MultinomialNB()|TfidfVectorizer(stop_words='english')|0.9793|0.8969|0.9006|0.96|With Hyperparameter Tuning|
|RandomForestClassifier(random_state=42)|CountVectorizer(stop_words='english')|1.0|0.8728|0.8835|0.96|Without Hyperparameter Tuning|
|RandomForestClassifier(random_state=42)|TfidfVectorizer(stop_words='english')|1.0|0.8728|0.8835|0.96|Without Hyperparameter Tuning|

As we can see, all the models R2 accuracy (87.2% ~ 91%) and F1-score (88.3% ~ 9.4%) are fairly similar but still predicts much better than our baseline score(50.6%). All the models are very good at distinguishing between the <mark style="background-color: lightgrey">r/aliens</mark> (positive class) and <mark style="background-color: lightgrey">r/space</mark> (negative class) with AUC score equal to 0.96. This shows that <mark style="background-color: lightgrey">r/aliens</mark> and <mark style="background-color: lightgrey">r/space</mark> really do just talk about very different, distinct things. Overall, Logistic Regression with TF-IDF Vectorizer produces the highest accuracy results and least amount of misclassification among other models. After careful consideration and judgement, we decided to choose Logistic Regression with TF-IDF Vectorizer model as our production model because:

- It is easier to implement and interpret. 
- The model coefficients can be easily interpreted as indicators of feature importance. The model is using the coefficients of each word in the post and calculates the odds of it belonging to <mark style="background-color: lightgrey">r/aliens</mark> or belong to <mark style="background-color: lightgrey">r/space</mark>.
    
### Recommendations

Based on our model, the key words that best differentiates posts related from <mark style="background-color: lightgrey">r/aliens</mark> to <mark style="background-color: lightgrey">r/space</mark> are:

The top 5 words with the highest odd for <mark style="background-color: lightgrey">r/aliens</mark>:
- alien
- ufo
- think 
- like
- human

The top 5 words with the highest odd for <mark style="background-color: lightgrey">r/space</mark>:
- jwst
- space
- webb
- telescope
- launch

The more the above listed words appear in a posts, the better the model to differentiates posts related to <mark style="background-color: lightgrey">r/aliens</mark> from <mark style="background-color: lightgrey">r/space</mark>.

### Limitations & Improvement

As demonstrated by the false predictions above, our model does have some limitations, especially when it comes to predicting <mark style="background-color: lightgrey">r/aliens</mark> posts. Mentioning James Webb Space Telescope or NASA throws off our model, even though it could potentially be an issue pertinent to <mark style="background-color: lightgrey">r/aliens</mark>. 

Areas for improvement and future exploration:
1. Use word similarities (e.g. word2vec) to classify posts instead of frequency.
2. Try Support Vector Machine algorithm for classification.
3. Explore relationship between post content, number of comments, and upvote ratio.
4. Implement sentiment analysis such as positive and negative Comment.
5. Develop set of StopWords for the model.

### References

1. https://www.upgrad.com/blog/multinomial-naive-bayes-explained/
2. https://www.geeksforgeeks.org/advantages-and-disadvantages-of-logistic-regression/
3. https://www.upgrad.com/blog/multinomial-naive-bayes-explained/
4. https://www.mygreatlearning.com/blog/random-forest-algorithm/
5. https://www.analyticsvidhya.com/blog/2020/06/auc-roc-curve-machine-learning/
6. https://towardsdatascience.com/what-metrics-should-we-use-on-imbalanced-data-set-precision-recall-roc-e2e79252aeba
7. https://towardsdatascience.com/interpreting-coefficients-in-linear-and-logistic-regression-6ddf1295f6f1?gi=171712960afd
