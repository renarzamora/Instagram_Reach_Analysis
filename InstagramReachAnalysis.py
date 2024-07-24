import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveRegressor
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import os

class InstagramReachAnalyzer:
    def __init__(self):
        self.data = None
    
    def load_data(self, filepath):
        self.data = pd.read_csv(filepath, encoding = 'latin1')
        print('Data loaded successfully')
    
    def eda_process(self):
        print(self.data.head())

        # checking whether the dataset contains any null values or not
        self.data.isnull().sum()

        # we checked that each column has one null value so we'll drop all null values
        self.data.dropna(inplace=True)

        # Let's have a look at the insights of the columns to understand the data type of all the columns
        self.data.info()

        # Now let’s jump analyzing the reach of the Instagram posts. We will first have a look at the distribution of impressions 
        # it have received from home
        plt.figure(figsize=(10,8))
        plt.style.use('fivethirtyeight')
        plt.title('Distributions of Impressions From Home')
        sns.histplot(self.data['From Home'])
        plt.show()

        # The impressions received from the Home section on Instagram show how much the posts reach the followers. 
        # Looking at the impressions from Home, we can say that it is difficult to reach all followers on a daily basis. 
        # Now let's take a look at the distribution of impressions you received from hashtags.

        plt.figure(figsize=(10,8))
        plt.title('Distributions of Impressions from Hashtags')
        sns.histplot(self.data['From Hashtags'])
        plt.show()

        # Hashtags are tools we use to categorize our posts on Instagram so that we can reach more people based 
        # on the kind of content we are creating. 
        # Looking at hashtag impressions shows that not all posts can be reached using hashtags, but many new users can be reached 
        # from hashtags. Now let’s have a look at the distribution of impressions received from the explore section of Instagram

        plt.figure(figsize=(10,8))
        plt.title('Distribution of Impressions from Explore')
        sns.histplot(self.data['From Explore'])
        plt.show()

        # The explore section of Instagram is the recommendation system of Instagram. It recommends posts to the users based on their 
        # preferences and interests. By looking at the impressions received from the explore section, We can see that Instagram 
        # does not recommend these posts much to the users. Some posts have received a good reach from the explore section, 
        # but it’s still very low compared to the reach  received from hashtags.

        # Now let's take a look at the percentage of impressions obtained from various sources on Instagram.
        home = self.data['From Home'].sum()
        hashtags = self.data['From Hashtags'].sum()
        explore = self.data['From Explore'].sum()
        others = self.data['From Other'].sum()

        labels = ['From Home','From Hashtags','From Explore','From Other']
        values = [home, hashtags, explore, others]

        fig = px.pie(self.data, values = values, names = labels,
                     title='Impressions On Instragram From Various Sources', hole=0.5)
        fig.show()

        # So we can see the above donut plot shows that almost 50 per cent of the reach is from the followers, 
        # 38.1 per cent is from hashtags, 9.14 per cent is from the explore section, and 3.01 per cent is from other sources.

        # Analyzing Content
        # Now let’s analyze the content of the Instagram posts. The dataset has two columns, namely caption and hashtags, 
        # which will help us understand the kind of content posting on Instagram.
        # Let’s create a wordcloud of the caption column to look at the most used words in the caption of the Instagram posts

        text = ' '.join(i for i in self.data.Caption)
        stopwords = set(STOPWORDS)
        wordcloud = WordCloud(stopwords=stopwords, background_color='white').generate(text)
        plt.style.use('classic')
        plt.figure(figsize=(12,10))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.show()

        # Now let’s create a wordcloud of the hashtags column to look at the most used hashtags in the Instagram posts:
        text = ' '.join(i for i in self.data.Hashtags)
        stopwords = set(STOPWORDS)
        wordcloud = WordCloud(stopwords=stopwords, background_color='white').generate(text)
        plt.style.use('classic')
        plt.figure(figsize=(12,10))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.show()

        # Analyzing Relationships
        # Now let’s analyze relationships to find the most important factors of our Instagram reach. 
        # It will also help us in understanding how the Instagram algorithm works.
        # Let’s have a look at the relationship between the number of likes and the number of impressions on the Instagram posts.

        figure = px.scatter(data_frame=self.data, x='Impressions',
                            y='Likes', size='Likes', trendline='ols',
                            title='Relationships Between Like and Impressions')
        figure.show()

        # There is a linear relationship between the number of likes and the reach got on Instagram. 
        # Now let’s see the relationship between the number of comments and the number of impressions on the Instagram posts.
        figure = px.scatter(data_frame=self.data, x='Impressions',
                            y='Comments', size='Comments', trendline='ols',
                            title='Relationships Between Impressions and Comments')
        figure.show()

        # It looks like the number of comments we get on a post doesn’t affect its reach. Now let’s have a look at the relationship 
        # between the number of shares and the number of impressions.
        figure=px.scatter(data_frame=self.data, x='Impressions',
                          y='Shares', size='Shares', trendline='ols',
                          title='Relationships Between Impressions and Total shares')
        figure.show()
        
        # A more number of shares will result in a higher reach, but shares don’t affect the reach of a post as much as likes do. 
        # Now let’s have a look at the relationship between the number of saves and the number of impressions.
        figure = px.scatter(data_frame=self.data, x='Impressions',
                            y='Saves', size='Saves', trendline='ols',
                            title='Relationships Between Posts Saves and Total Impressions')
        figure.show()

        # There is a linear relationship between the number of times the post is saved and the reach of the Instagram post. 
        # Now let’s have a look at the correlation of all the columns with the Impressions column
        df = self.data[['Impressions', 'Likes', 'From Hashtags', 'Follows', 'Profile Visits', 'Saves','From Home', 'From Explore', 'Shares', 'From Other', 'Comments']]
        #correlation = self.data.corr()
        correlation = df.corr()
        print(correlation['Impressions'].sort_values(ascending=False))

        # So we see say that more likes and saves will help you get more reach on Instagram. The higher number of shares 
        # will also help you get more reach, but a low number of shares will not affect your reach either

        # Analyzing Conversion Rate
        # In Instagram, conversation rate means how many followers you are getting from the number of profile visits from a post. 
        # The formula that you can use to calculate conversion rate is (Follows/Profile Visits) * 100. 
        # Now let’s have a look at the conversation rate of my Instagram account
        conversion_rate = (self.data['Follows'].sum() / self.data['Profile Visits'].sum() * 100)
        print('Conversion Rate=',conversion_rate)

        # So the conversation rate of the Instagram account is 31% which sounds like a very good conversation rate. 
        # Let’s have a look at the relationship between the total profile visits and the number of followers gained 
        # from all profile visits
        figure=px.scatter(data_frame=self.data, x='Profile Visits',
                          y='Follows', size='Follows', trendline='ols',
                          title='Relationships Between Profile Visits and Followers gained')
        figure.show()
        print('EDA process finished')
        # The relationship between profile visits and followers gained is also linear.

    def Instagram_Reach_Prediction_Model(self):
        # We will train a machine learning model to predict the reach of an Instagram post. 
        # Let’s split the data into training and test sets before training the model.

        x = np.array(self.data[['Likes','Saves','Comments', 'Shares','Profile Visits', 'Follows']])
        y = np.array(self.data['Impressions'])
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        # Now we can train a machine learning model to predict the reach of an Instagram post using Python
        model = PassiveAggressiveRegressor()
        model.fit(x_train, y_train)
        score = model.score(x_test, y_test)
        print('Score=',score)
        
        # Now We'll predict the reach of an Instagram post by giving inputs to the machine learning model.
        # Features = [['Likes','Saves','Comments', 'Shares','Profile Visits','Follows']]
        features = np.array([[282.0, 233.0, 4.0, 9.0, 165.0, 54.0]])
        prediction = model.predict(features)
        print('Predictions=',prediction)
        print('Prediction process finished')

# usage example
if __name__ == '__main__':
    analyzer = InstagramReachAnalyzer()
    filepath = os.getcwd()+'\Instagram data.csv'
    analyzer.load_data(filepath)
    analyzer.eda_process()
    analyzer.Instagram_Reach_Prediction_Model()

# So this is how we can analyze and predict the reach of Instagram posts with machine learning using Python. 
# If a content creator wants to do well on Instagram in a long run, they have to look at the data of their Instagram reach. 
# That is where the use of Data Science in social media comes in.

