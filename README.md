# Ames, IA - House Sale Price Estimation

[Problem Statement](#Problem-Statement)  
[Summary](#Summary)  
[Data Dictionary](#Data-Dictionary)  
[Data Pulling](#Data-Pulling)  
[Data Discovery](#Data-Discovery)    
[Pipeline Modeling](#Pipeline-Modeling)  
[Conclusions and Recommendations](#Conclusions-and-Recommendations)  
[What Next](#What-Next)


### Problem Statement

Home Flippers, LLC, a company focused on flipping homes in the Ames, IA region, contracted me to build a model to predict home values in the area. They were looking to use this model so they could calculate predicted prices for houses on the market in relation to historical sale prices of comparable houses. This information is of concern and value to them to enable them to identify opportunities with the greatest upside after making renovations or remodeling the house for sale. 

Obviously the majority of the upside in sale price from a home flip comes from the improvements made during the renovations, but by identifying undervalued homes in need of repair would allow them to calculate the potential upside compared to the amount of investment they put into the renovation.

___
### Summary

Things to look into in this report:
- Which house features have the greatest influence on house sale price?
- How accurate can I get a model to accurately predict house prices based on an existing dataset?
- Which houses present the biggest upside opportunity for someone intending to renovate and flip a house?
- Generate a list of houses of various sizes which sold for the least and meet the previous condition.

___
### Data Dictionary

|Feature|Type|Source|Description|
|---|---|---|---|
|**title**|*object*|Pushshift API|Title of the Reddit post|
|**selftext**|*object*|Pushshift API|Body text of the Reddit post|
|**author**|*object*|Pushshift API|Username of the person who wrote the post|
|**subreddit**|*object*|Pushshift API|Name of the subreddit that the post was pulled from|
|**sub_target**|*int*||Subreddit converted into binary for modeling|
|**combined**|*object*||Concatenated value from title and selftext|
|**char_count**|*int*||Total number of characters in the combined tex|
|**word_count**|*int*||Total number of words in the combined tex|
|**tokens**|*object*||Tokenized text from combined text|
|**stemmed**|*object*||Stemmed text from the tokenized text|
|**stock_sentiment**|*float*||Sentiment score based on stock/investment specific terminology|
|**compound_sent**|*float*||Compound sentiment score from VADER analysis|
|**stock_sentiment_sign**|*object*||stock_sentiment converted into "positive", 'neutral', and "negative"|
|**compound_sent_sign**|*object*||compound_sent converted into "positive", 'neutral', and "negative"|

___
### [Data Pulling]()


___
### [Data Discovery]()

- My initial dive into the data began by reading through the data documentation and considering which features I thought might have the greatest influence, positive or negative, on sale price.
- Next, I created a list of the categorical variables I was investigating and counted their values to determine what changes I would have to make to the data during the cleaning and mapping process.
- I then produced a heatmap to show those features and their correlations to the sale price.

<div align="center"><img src="./assets/images/saleprice_heatmap.png" width="35%" height="35%"></div>

- After reviewing the heatmap, I produced a pairplot of the top three correlated features to investigate their relation to sale price.
- One thing I took notice of was the outliers present in Living Area Above Grade and how they didn't follow the upward trend of price correlation.

<div align="center"><img src="./assets/images/pairplot.png" width="90%" height="90%"></div>
 
- One thing to mention about this project was that it was highly iterative. The steps I took above, especially evaluating value counts for categorical (and sometimes numeric) columns were repeated on new features throughout the refinement process.

___
### [Pipeline Modeling]()

After importing my cleaned and engineered data, I first created and plotted my baseline model. In addition to plotting my null hypothesis (the mean sale price) on the vivsualization, I plotted the OLS regression line to get an idea of what that looked like and for consideration.

<div align="center"><img src="./assets/images/null_hypothesis.png" width="75%" height="75%"></div>

I then proceeded to select my features out of all the categories I cleaned/mapped and feature engineered. By use of another function, feat_sel(), I was able to streamline the feature selection process so I could easily comment features in/out and just rerun my entire notebook to test the new model.

I trained/test/split my data, instantiated and fit the model, calculated the coefficients, and calculated my predictions and residuals. I then calculated the MSE and RMSE for both the training and validation data as well as calculating the R-squared scores.

|Metric|Split|Score|
|---|---|---|
|MSE|Training|0.010568052522169484|
|RMSE|Training|0.10280103366294273|
|R2|Training|0.9375076375371625|
|MSE|Validation|0.016373309956852496|
|RMSE|Validation|0.1279582352052907|
|R2|Validation|0.903518251292407|

In order to better visualize the data and check for and major issues, I plotted my residuals looking for an even distribution of positive and negative residuals and to ensure there wasn't spread. My residuals had several values that were further on the negative side, but I found removing such data points from my original training data skewed my model's results and hurt the model performance. 

<div align="center"><img src="./assets/images/residual_spread.png" width="60%" height="60%"></div>

I also drew a histogram to visualize the residuals distribution and to ensure it was normally distributed. Again, the residuals showed as skewing a bit to the negative side, but for the most part were normally distributed.

<div align="center"><img src="./assets/images/residual_distribution.png" width="60%" height="60%"></div>

I then wanted to check how my model performed compared to my null hypothesis. To do so, I had to exponentiate my values before comparing them to the null RMSE.

|Metric|Model|Score|
|---|---|---|
|RMSE|Baseline/Null|79276.56098763691|
|RMSE|Linear Regression|19263.461294487737|

This was a very noticeable improvement over my baseline model. At one point I had concerns of overfitting my LR model because I kept adding features and the RMSE and R2 scores kept improving ever so slightly, although my final RMSE on the test data got worse. I ultimately tried to balance my features and model to find the sweet spot in fit.

One last thing to consider for the validity of my Linear Regression model is that it meets the LINEM requirements. I would say my model passes Linearity as can be seen in my baseline model. The data collected was independent of each other, as best I can tell and is to be assumed, as no house sale affected another house sale directly. The distribution of errors was relatively normal with some minor skewing to the left. There was also equal variance of errors except for a few outliers spread in the negative direction. And lastly, I didn't find any multicollinear features resulting in correlation higher than 0.95.

___
### [Conclusions and Recommendations]

As part of my conclusion and post-model visualizations, I needed to pickle my LR model from my modeling notebook. After loading my pickle and any other necessary dataframes I visualized the coefficient weights for the top 10 most influential features.

<div align="center"><img src="./assets/images/top_features.png" width="70%" height="70%"></div>

This chart was very helpful in visualizing the coefficients for my various features. It was no surprise that the most influential feature was the living area above grade. This is what typically represents the square footage of a house since unless a basement has a [legal ingress or egress](https://www.homelight.com/blog/buyer-are-basements-included-in-square-footage/#:~:text=Legal%20ingress%20or%20egress), its square footage is usually not calculated in the total SF of the house. Other factors which had a significant negative influence on the sale price of a house were being zoned in either a Commercial, Agricultural, or Industrial zone (all mapped to ms_zoning_other), and a functional rating of Salvage, Severely Damaged, and Major Deductions 2 (all mapped to functional_Maj2). Other functional ratings in the top 10 features all had a positive influence on sale price.

The next thing I wanted to take a look at for my conclusion was the distribution of two groups of homes plotted by their Above Grade Living Area and their Sale Price. The two groups I plotted were 'Home built before 1960 that were remodeled between 2000 and 2010' and 'Homes built or last remodeled before 1960'.

<div align="center"><img src="./assets/images/remodeled_or_not.png" width="75%" height="75%"></div>

This was an important visualization for my problem statement as this plot demonstrated the sale price gap between recently remodeled houses and non-recently remodeled houses. It is easy to see that the older homes without remodel tended to skew lower in their sale price at all square footages. While there are still some homes that are in the range of the remodeled houses, but the average sale price at each square footage bin is lower for the not remodeled homes. I considered these lowest priced homes as having the most opportunity for remodeling, renovation, and flipping. Obviously each home would need to be evaluated individually to consider the upside opportunity in renovating and flipping it, but using the model I generated to get as accurate a sale price would enable flippers to find homes that were selling for less than their comp set.

In order to convey my findings to Home Flippers, LLC, I separated the [up to] ten lowest priced houses in each 500 SF bin, starting with 500 - 1000 SF. In the first image below of the master table, I highlighted areas that I saw as opportunities for improvement (or renovation) in green and challenging areas in red. The areas in red were primarily contraints of the houses such as worse foundations and poor conditions having to do with location like being locataed adjacent to a railroad.

<div align="center"><img src="./assets/images/flip_opportunities.png" width="100%" height="100%"></div>  

Any green areas, on the other hand, were shortcomings of the current house but things easily improved upon. Considering Home Flippers, LLC has different flippers that have their specialities, this would allow them to pick and choose houses to flip based on the renovation strengths of the company.

<div align="center"><img src="./assets/images/flip_opportunities_cropped.png" width="100%" height="100%"></div>

Ultimately, the accuracy of the model is what drives the value in this project for HF, LLC. By having an accurate model to predict house prices, they are more easily able to identify houses that are undervalued based on the sale price.

___

### What Next

Things I would like to look into as the project continues:
- The first improvement to this project I would make is introducing a K-Nearest Neighbors logistical regression model which could be used in conjunction with my Linear Regression model. Currently my LR model is able to accurately predict housing prices, but it would be optimal to be able to compare a specific house and evaluate its *potential* sale price based on the comp set for houses in the market. This would allow Home Flippers, LLC to identify an undervalued house and then estimate how much potential profit there is in the flip by factoring in their estimated renovation costs and then comparing it to the nearest neighbor(s) of a comparable house with the improved quality/condition/functionality ratings in its comp set.
- The second item I would like to work on is extrapolating this model and testing it on other housing markets. While I suspect that this model may still be relatively accurate for neighboring towns to Ames, I know it would have to be tweaked or completely reworked if going to a neighboring county or state. Many of the features used in my LR model may have more or less influence (stronger or weaker coefficients) depending on the part of the country the market is located in.
- One last thing I would also like to investigate is if this model is still accurate in present day. Considering the housing sale data is from 2006 - 2010, it would be extremely beneficial to evaluate how the model performs in present day for any real world application. One important factor which may or may not affect the present day effectiveness of the model is that during 2008 and 2009 the country was experiencing a recession, especially in the housing market. Even though I didn't find any significant influence based on the year sold in the data, it is still something I would need to investigate and consider.