import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import HDBSCAN, DBSCAN
from sklearn.manifold import TSNE
import plotly.graph_objects as go
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv('/Users/sgipson/Documents/Climate_Change_Sentiment/archive/The Climate Change Twitter Dataset.csv')
print(df.head())
print(df.shape)
print(df.columns)

print(df.shape)
df = df.dropna()
print(df.shape)

print(df['aggressiveness'].value_counts())
print(df['sentiment'].value_counts())
print(df['stance'].value_counts())
print(df['topic'].value_counts())
print(df['gender'].value_counts())

# positive and negative mean sentiment overtime
df['year'] = pd.to_datetime(df['created_at']).dt.year
df['date'] = pd.to_datetime(df['created_at']).dt.date
print(df.groupby('year')['date'].count())
df = df[df['year'] > 2006] # very little data for this one

"""
# sentiment distribution
plt.hist(df['sentiment'])
plt.title("Sentiment Distribution")
plt.savefig('sentiment-distribution.png')
plt.show()


df_pos = df[df['sentiment'] > 0]
df_neutral = df[df['sentiment'] == 0]
df_neg = df[df['sentiment'] < 0]


mean = df.groupby('year')['sentiment'].mean()
print(mean)
plt.plot(mean.index, mean.values)
plt.title('Mean Sentiment Per Year')
plt.xlabel('Year')
plt.ylabel('Average Sentiment')
plt.axhline(y=0, color='black', linestyle='--', label='Neutral')
plt.savefig('mean-sentiment-per-year.png')
plt.show()

# plot pos and negative together
mean_pos = df_pos.groupby('year')['sentiment'].mean()
print(mean_pos)
plt.plot(mean_pos.index, mean_pos.values, label='Positive Sentiment')
mean_neg = df_neg.groupby('year')['sentiment'].mean()
print(mean_neg)
plt.plot(mean_neg.index, mean_neg.values, label='Negative Sentiment')
plt.title('Mean Positive and Negative Sentiment Per Year')
plt.xlabel('Year')
plt.axhline(y=0, color='black', linestyle='--', label='Neutral')
plt.ylabel('Average Sentiment')
plt.savefig('pos-neg-mean-sentiment-per-year.png')
plt.show()


# plot the ratio of positive and negative sentiment tweets
counts = df.groupby('year')['sentiment'].count()
counts_pos = df_pos.groupby('year')['sentiment'].count()
counts_neg = df_neg.groupby('year')['sentiment'].count()

plt.plot(counts_pos.index, counts_pos.values/counts.values, label='Positive Sentiment')
plt.plot(counts_neg.index, counts_neg.values/counts.values, label='Negative Sentiment')
plt.title('Change of Sentiment Ratio Overtime')
plt.xlabel('Year')
plt.ylabel('Ratio of Tweets')
plt.legend()
plt.axhline(y=0.5, color='black', linestyle='--', label='50%')
plt.savefig('pos-neg-sentiment-ratio-per-year.png')
plt.show()


# plot the stance overtime
df_yes = df[df['stance'] == 'believer']
df_no = df[df['stance'] == 'denier']
print(df_no)
df_neu_stance = df[df['stance'] == 'neutral']

counts = df.groupby('year')['stance'].count()
print(counts)
counts_yes = df_yes.groupby('year')['stance'].count()
print(counts_yes.values/counts.values)
counts_no = df_no.groupby('year')['stance'].count()
print(counts_no.values/counts.values)
counts_neutral = df_neu_stance.groupby('year')['stance'].count()
print(counts_neutral)

plt.plot(counts.index, counts_yes.values/counts.values, label='Believer')
plt.plot(counts.index, counts_no.values/counts.values, label='Denier')
plt.plot(counts.index, counts_neutral.values/counts.values, label='Neutral')
plt.title('Change in Ratio of Climate Change Tweet Stance Overtime')
plt.ylim(0, 1)
plt.xlabel('Year')
plt.ylabel('Ratio of Tweets')
plt.legend()
plt.axhline(y=0.5, color='black', linestyle='--', label='50%')
plt.savefig('stance-overtime-ratio-per-year.png')
plt.show()




df = df[df['topic'] != 'Undefined / One Word Hashtags'] # not useful topic
topics = df['topic'].unique()

# look at how the mean sentiment per topic changes overtime
# sort by means for easier comparison
topic_data = []
topic_mean_sentiments = pd.DataFrame()

for topic in topics:
    df_topic = df[df['topic'] == topic]
    topic_mean = df_topic['sentiment'].mean()
    topic_data.append((topic_mean, topic))


sorted_means = sorted(topic_data, key=lambda x: x[0], reverse=True)
print(sorted_means)

for tup in sorted_means:
    topic = tup[1]
    df_topic = df[df['topic'] == topic]
    topic_year_means = df_topic.groupby('year')['sentiment'].mean()
    topic_mean_sentiments[topic] = topic_year_means.values

topic_mean_sentiments.set_index(df['year'].unique())
print(topic_mean_sentiments)
print(topic_mean_sentiments.index)
plt.figure(figsize=(10, 6))
plt.imshow(topic_mean_sentiments.T)
plt.colorbar(label='Mean Sentiment')
plt.xlabel('Year')
plt.ylabel('Topic')
plt.xticks(np.arange(0, len(df['year'].unique()), 1), df['year'].unique(), rotation=45)
plt.yticks(np.arange(0, len(topics), 1), topic_mean_sentiments.columns)
plt.title('Mean Sentiment of Climate Change Topic Overtime')
plt.tight_layout()
plt.savefig('topic-sentiments-overtime.png')
plt.show()

tweets_count_per_year = df.groupby('year')['id'].count() # number of tweets per year
# show aggressiveness overtime
df_agg = df[df['aggressiveness'] == 'aggressive']
aggressive_tweets_per_year = df_agg.groupby('year')['id'].count()
mean_agg_ratio = (aggressive_tweets_per_year.values/tweets_count_per_year.values).mean()
plt.plot(aggressive_tweets_per_year.index, aggressive_tweets_per_year.values/tweets_count_per_year.values)
plt.axhline(y=mean_agg_ratio, color='black', linestyle='--', label='Mean Ratio')
plt.title('Ratio of Aggressive Tweets Overtime')
plt.savefig('aggressive-tweets-overtime.png')
plt.xlabel('Year')
plt.ylabel('Ratio of Aggressive Tweets')
plt.legend()
plt.show()


# show the ratio of times each topic was mentioned per year
# show its sentiment too
df = df[df['topic'] != 'Undefined / One Word Hashtags'] # not useful topic
topics = df['topic'].unique()

for topic in topics:
    df_topic = df[df['topic'] == topic]
    topic_tweets_per_year = df_topic.groupby('year')['id'].count()
    print(topic_tweets_per_year)
    plt.plot(tweets_count_per_year.index, topic_tweets_per_year.values/tweets_count_per_year.values, label=topic)
plt.legend()
plt.title('Ratio of Climate Change Tweet Topics per Year')
plt.xlabel('Year')
plt.ylabel('Ratio of Topic Tweets')
plt.savefig('tweets-topics-per-year.png')
plt.show()
"""

df = df[df['topic'] != 'Undefined / One Word Hashtags'] # not useful topic
topics = df['topic'].unique()

# look at how the mean sentiment per topic changes overtime
# sort by means for easier comparison
topic_data = []
topic_stance_ratio = pd.DataFrame()

for topic in topics:
    df_topic = df[df['topic'] == topic]
    all_stances = len(df_topic['stance'])
    pos_stances = len(df_topic[df_topic['stance'] == 'believer'])
    pos_ratio = pos_stances / all_stances
    topic_data.append((pos_ratio, topic))


sorted_means = sorted(topic_data, key=lambda x: x[0], reverse=True)
print(sorted_means)

for tup in sorted_means:
    topic = tup[1]
    df_topic = df[df['topic'] == topic]
    pos_stances = df_topic[df_topic['stance'] == 'believer']
    topic_year_ratio = pos_stances.groupby('year')['stance'].count()/df_topic.groupby('year')['stance'].count()
    topic_stance_ratio[topic] = topic_year_ratio.values

topic_stance_ratio.set_index(df['year'].unique())
print(topic_stance_ratio)
print(topic_stance_ratio .index)
plt.figure(figsize=(10, 6))
plt.imshow(topic_stance_ratio .T)
plt.colorbar(label='Ratio of Climate Change Believer Tweets')
plt.xlabel('Year')
plt.ylabel('Topic')
plt.xticks(np.arange(0, len(df['year'].unique()), 1), df['year'].unique(), rotation=45)
plt.yticks(np.arange(0, len(topics), 1), topic_stance_ratio .columns)
plt.title('Ratio of Climate Change Believers Overtime')
plt.tight_layout()
plt.savefig('topic-stances-overtime.png')
plt.show()

# clustering
df = df[df['topic'] != 'Undefined / One Word Hashtags'] # not useful topic
df = df[df['gender'] != 'undefined']
sample_df = df.sample(n=100000, random_state=42)
onehot = pd.get_dummies(sample_df[['gender', 'aggressiveness', 'topic', 'stance', 'year']]) # do gender, aggressiveness, topic, stance
# onehot = pd.get_dummies(sample_df[['topic', 'stance', 'year']]) # maybe just do topic and stance and year
# then add sentiment
print(onehot)
X = onehot
# X = np.hstack((onehot, df['sentiment'].values.reshape(-1, 1)))

print(X)
tsne = TSNE(n_components=2, random_state=42, perplexity= 30, learning_rate=200) # try perplexity= 50 learning rate=200 # try adding sentiment to this
X_tsne = tsne.fit_transform(X)
plt.figure(figsize=(8, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.5)
plt.title('t-SNE Visualization')
plt.savefig('t-SNE.png')
plt.show()

# cluster tsne with hdbscan
# do the plotly graph with labels on date and topic
hdb = HDBSCAN(min_cluster_size=2)
hdb.fit(X_tsne)
cluster = hdb.labels_
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=cluster, cmap='Paired', s=5)
plt.title("HDBSCAN")
plt.savefig('tsne-hdbscan.png')
plt.show()

gender = df['gender'].values
aggressive = df['aggressiveness'].values
topic = df['topic'].values
stance = df['stance'].values
year = df['year'].values
sentiment = df['sentiment'].values
# labels = dict({'topic': topics, 'year': years, 'stance': df['stance'].values, 'sentiment': df['sentiment'].values, 'aggressiveness': df['aggressiveness'].values})
labels = zip(gender, aggressive, topic, stance, year, sentiment)
hover_text = [f"gender: {label[0]}, aggressiveness: {label[1]}, year: {label[4]}, topic: {label[2]}, stance: {label[3]}, sentiment: {label[5]}" for label in labels]
fig = go.Figure(data=go.Scatter(x=X_tsne[:, 0],
                                y=X_tsne[:, 1],
                                marker=dict(color=cluster, size=10, colorscale='RDYlBu'),
                                mode='markers',
                                hovertext=hover_text))
fig.update_layout(title="t-SNE with HDBSCAN")
fig.show()

# clustering with sentiment as a value
df = df[df['topic'] != 'Undefined / One Word Hashtags'] # not useful topic
df = df[df['gender'] != 'undefined']
sample_df = df.sample(n=100000, random_state=42)
onehot = pd.get_dummies(sample_df[['gender', 'aggressiveness', 'topic', 'stance', 'year']]) # do gender, aggressiveness, topic, stance
# onehot = pd.get_dummies(sample_df[['topic', 'stance', 'year']]) # maybe just do topic and stance and year
# then add sentiment
print(onehot)
X = np.hstack((onehot, sample_df['sentiment'].values.reshape(-1, 1)))

print(X)
tsne = TSNE(n_components=2, random_state=42, perplexity= 30, learning_rate=200) # try perplexity= 50 learning rate=200 # try adding sentiment to this
X_tsne = tsne.fit_transform(X)
plt.figure(figsize=(8, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.5)
plt.title('t-SNE Visualization with Sentiment Values Included')
plt.savefig('t-SNE-with-sentiment.png')
plt.show()

# cluster tsne with hdbscan
# do the plotly graph with labels on date and topic
hdb = HDBSCAN(min_cluster_size=2)
hdb.fit(X_tsne)
cluster = hdb.labels_
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=cluster, cmap='Paired', s=5)
plt.title("HDBSCAN")
plt.savefig('tsne-hdbscan-with-sentiment.png')
plt.show()

gender = df['gender'].values
aggressive = df['aggressiveness'].values
topic = df['topic'].values
stance = df['stance'].values
year = df['year'].values
sentiment = df['sentiment'].values
# labels = dict({'topic': topics, 'year': years, 'stance': df['stance'].values, 'sentiment': df['sentiment'].values, 'aggressiveness': df['aggressiveness'].values})
labels = zip(gender, aggressive, topic, stance, year, sentiment)
hover_text = [f"gender: {label[0]}, aggressiveness: {label[1]}, year: {label[4]}, topic: {label[2]}, stance: {label[3]}, sentiment: {label[5]}" for label in labels]
fig = go.Figure(data=go.Scatter(x=X_tsne[:, 0],
                                y=X_tsne[:, 1],
                                marker=dict(color=cluster, size=10, colorscale='RDYlBu'),
                                mode='markers',
                                hovertext=hover_text))
fig.update_layout(title="t-SNE with HDBSCAN and Sentiment Values Included")
fig.show()
"""
color_mapping = {'believer': 'blue', 'denier': 'red', 'neutral': 'gray'}
fig = go.Figure(data=go.Scatter(x=df['date'],
                                y=df['sentiment'],
                                marker=dict(color=df['stance'].map(color_mapping), size=10, colorscale='RDYlBu'),
                                mode='markers',
                                hovertext=hover_text))
fig.update_layout(title="scattering")
fig.show()
"""