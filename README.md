# Climate_Change_Twitter_Data

  This dataset includes data for over 15 million tweets over 13 years, each belonging to one of ten climate change related topics. Each tweet has a sentiment score (-1 being the most negative, 1 being the most positive), a label declaring whether or not the tweet used aggressive language, a date of creation, and a category label for either supporting the belief in man-made climate change, denying it, or remaining neutral. 

  To evaluate how the climate-change discussion on Twitter has changed, I've graphed various groupings of these variables overtime. Of these, the heatmap of the ratio of climate change believer tweets per topic overtime may be the most informative. I created this after graphing each stance as a ratio overtime--since the "believer" stance has greatly increased in the past few years, I wanted to know more about which topics have experienced higher ratios of "believers" overtime. I did the same with sentiment overtime (including positive and negative on the same scale), however the stance ratio shows a more useful variation across topics and overtime.

  Additionally, I used t-SNE with a variety of parameters to attempt to show how tweets were similar or dissimilar in the euclidean space based on the tweet metadata available. I used one hot encoding to represent each tweet as the gender of the tweet author, whether the tweet contained aggressive language, the year of creation, climate change topic, and climate change stance conveyed by the tweet. I considered including sentiment, however the cluster resulting without including this metric is significantly more interpretable than when including sentiment. I believe this is becuase sentiment is not on a one-hot encoded scale and contains negative values, while all other values are wither 1 or 0--this may affect the dimension reduction poorly as, t-SNE represents each point in relation to other points spatially. The cluster graph includes a smaller sample size of the full dataset. To view the metadata for each point, run the code to generate the plotly graph.

  The next step of the clustering would be to match each data point not with its metadata, but instead with the text of the tweet. However, retrieving this text would require hydrating, which at this moment in time I am unfamiliar with (I'm planning to test it out over the summer).

Dataset retrieved from https://www.kaggle.com/datasets/deffro/the-climate-change-twitter-dataset?select=The+Climate+Change+Twitter+Dataset.csv

References:

Effrosynidis, Dimitrios, Georgios Sylaios, and Avi Arampatzis. "Exploring climate change on Twitter using seven aspects: Stance, sentiment, aggressiveness, temperature, gender, topics, and disasters." Plos one 17.9 (2022): e0274213.

Effrosynidis, Dimitrios, et al. "The climate change Twitter dataset." Expert Systems with Applications 204 (2022): 117541.
