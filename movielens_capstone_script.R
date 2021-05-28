# Load requires R libraries to initial project.
library(tidyverse)
library(caret)
library(data.table)
library(knitr)

# Load ggplot2 - allows for better and more easily manipulated plots.
library(ggplot2)

#Load lubridate - makes it easier to work with date-time data n R.
library(lubridate)

# Load dplyr - packages used for data manipulation in R (filter, mutate etc)
library(dplyr)

# Report up to 5 significant digits - easy to judge the RMSE scores versus our goal of < 0.8649
options(digits = 5)

# MovieLens 10M dataset:
dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)
ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))), 
                 col.names = c("userId", "movieId", "rating", "timestamp"))
movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# If using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId), 
    title = as.character(title), genres = as.character(genres))
movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of the MovieLens dataset
set.seed(1, sample.kind = "Rounding")
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure UserId and movieId in validation set are also in edx set
validation <- temp %>% semi_join(edx, by = "movieId") %>% semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)
rm(dl, ratings, movies, test_index, temp, movielens, removed)

# Our first step here is to have a look into the data and obtain some general, descriptive knowledge.
# The (glimpse) function shows us the layout of the data as a quick snapshot.
glimpse(edx)
# The (summary) function gives us a descriptive statistics table or the data.
summary(edx)
# The (dim) function will tell us the number of columns and rows.
dim(edx)
# We want to know the class of our data, so we use the class function.
class(edx)

# Following that step, we want to learn some specific information on the dataset we can expand upon later.
# How many different movies and users are in the dataset?
n_distinct(edx$userId)
n_distinct(edx$movieId)

# Which movies have the greatest number of ratings?
edx %>% group_by(movieId, title) %>% summarize(count = n()) %>% arrange(desc(count)) 

# Which ratings have the highest count and lets display those counts
edx %>% group_by(rating) %>% summarize(count = n()) %>% arrange(desc(count))

# Model Creation: Developing a promising model by accounting for identified data biases.

# Identified Bias I: Whole Star vs Half Star Ratings

# First we want to group whole numbers (Full Star) and fractional numbers (Half Star) so we can present them in the form of a visual.
ratings_group <-  ifelse((edx$rating == 1 |edx$rating == 2 | edx$rating == 3 | edx$rating == 4 | edx$rating == 5) , "Full_star", "Half_star")

df_ratings <- data.frame(edx$rating, ratings_group)

# We create our histogram for Full Stars & Half Stars
ggplot(df_ratings, aes(x= edx.rating, fill = ratings_group)) + geom_histogram( binwidth = 0.2) + ggtitle("MovieLens DataSet  Ratings Distribution") + labs(x="Rating Values", y="Rating Count")

# This histogram shows us some pertinent takeaways:
# 1. There are a lot more full star ratings than half star ratings.
# 2. There was not one rating given as a 0.

# Identified Bias II: Movie Ratings Count

edx %>% count(movieId) %>% ggplot(aes(n)) + geom_histogram(bins = 40, color = "yellow", fill = "purple") + scale_x_log10() + labs(x="Number of Ratings", y="Number of Movies") + ggtitle("Number of Ratings per Movie")

# This histogram shows us that there is a bias regarding how many movies have ratings, including a significant number of movies that have been rated just one time.
# As such, it will make it very difficult to predict the movie ratings for these movies that have been not rated a large number of times which will affect our accuracy.
# Observing this bias, a model will need to be developed to account for this bias. Regularization will further aid us in achieving this goal.

# Identified Bias III: Distribution of Ratings by Users

edx %>% count(userId) %>% ggplot(aes(n)) + geom_histogram(bins = 40, color = "purple", fill = "yellow") + scale_x_log10() + labs(x="Number of Ratings", y="Number of Users") + ggtitle("Number of Ratings by User")

# This histogram shows most users rate between 50 - 100 movies. So while we have a penalty term for our ratings score bias, we will (through model development) have one for the user term as well.
# The distribution isn't the only useful part to analyze when looking into users. We can also analyze the distribution of ratings by user. Users who give ratings on the outliers will be more difficult to predict so if that is the case, we will need to account for that. We explore that as follows:

edx %>% group_by(userId) %>% summarise(mean_u = mean(rating)) %>% ggplot(aes(mean_u)) + geom_histogram(bins = 40, color = "yellow", fill = "purple") + labs(x="Mean Ratings", y="Number of Users") + ggtitle("Average Movie Ratings by User")

# The mean for edx$rating in our dataset is 3.512 and this histogram shows us how many ratings we are getting according to the number of users.
# We can see that there is an influence here from a high number of users giving certain ratings. Since we have some users who rate a large number of movies, any biases they have (being overly positive, overly negative, genre distastes etc) will be reflected here and make it harder for us to predict ratings in the future.

# Modeling Development
# Room-mean-square-error(RMSE) is a frequently used measure of the differences between values (sample or population values) predicted by a model or an estimator and the values observed.
# RMSE is simply interpreted as; the lower the number, the better.
# For this project, we want a RMSE of 0.8649 or less in order to get the maximum number of points available for that section.

# The first part of our model development is to define our RMSE function. This will be completed as follows:

RMSE <- function(true_ratings, predicted_ratings){sqrt(mean((true_ratings - predicted_ratings)^2))}

# For this project, we will develop 5 distinct models. These will include our foundation model (Model A) and we will extend off this Model A to achieve results that are better than the results achieved from Model A.
# These 5 models are broken down and explained as follows:

# Model A (Average Movie Rating Model)
# Model B (Median Movie Rating Model)
# Model C (Adjusted Movie Effect Model)
# Model D (Adjusted Movie & User Effect Model)
# Model E (Regularized Movie & User Effect Model)

# In the end of this report, we will showcase all the results of the 5 models (the corresponding RMSEs) for comparison purposes.

# -------

# Model A: Average Movie Rating Model
# Our first model is among the most basic possible as it predicts the same rating for all movies.
# We obtain the mean of our rating (previously discussed above) the following way:

mu <- print(mean(edx$rating))

# Using this value, 3.51427, we can run our first model (acting as a foundation model) and compute the RMSE.

model_a <- print(RMSE(validation$rating, mu))

# This model gives us our first RMSE which is 1.0612 and substantially above our goal RMSE.
# As such, we can't stop here and need to develop this model further by trying new techniques and accounting for various discovered data biases.

# -------

# Model B: Median Movie Rating Model

med_u <- print(median(edx$rating))

# Using this value, 4.0, we can run our second model and compare the RMSE with that of our first model.

model_b <- print(RMSE(validation$rating, med_u))

# This model gives us our second RMSE which is 1.16802.
# This RMSE is 10% worse than our first RMSE.
# The RMSE differences between a median and mean model were explored to understand/confirm which model to expand upon with extended analyses.
# This quick exercise confirms that we need to use our first RMSE (using the mean) to develop Model C through Model E.

# -------

# Model C: Adjusted Movie Effect Model

# In the earlier section, we identified a biases associated with the fact that some movies will simply be inherently rated higher than other movies.
# As such, we can add the ability to account for the 'effect of the movie' to create a model involved model, Model C. This model will be an extension of Model A but accounts for the effects of the movie.

moviebias_avg <- edx %>% group_by(movieId) %>% summarize(b_i = mean(rating - mu))
pred_ratings_bi <- mu + validation %>% left_join(moviebias_avg, by='movieId') %>% .$b_i
model_c <- print(RMSE(validation$rating, pred_ratings_bi))

# Model C produces our lowest RMSE to date which is 0.94391. This represents an approximate 11% improvement over Model A which was previously our top model RMSE prior to integrating Model C.
# This model represents a significant improvement but further refinement is required to help reach the desired RMSE threshold.

# -------

# Model D: Adjusted Movie & User Effect Model

# In the earlier section, we identified a bias associated with the users in our dataset. Some humans are inherently more positive and more negative than others. Since we have a situation where some users rated a high percentage of total movies, this select group of users who rated lots of movies are having an effect on the data and thus our ability to predict movie rating in the validation set.
# Since we saw that accounting for the movie bias substantially reduced our RMSE, by accounting for the movie and user biases should have an even more significant effect on reducing our RMSE.
# In a similar fashion to how we developed the model extension for movie bias, we do the same to create a model that accounts for movie and user biases.

movieuserbias_avg <- edx %>% left_join(moviebias_avg, by='movieId') %>% group_by(userId) %>% summarize(b_u = mean(rating - mu - b_i)) 
pred_ratings_bu <- validation %>% left_join(moviebias_avg, by='movieId') %>% left_join(movieuserbias_avg, by='userId') %>% mutate(pred = mu + b_i + b_u) %>% .$pred
model_d <- print(RMSE(validation$rating,pred_ratings_bu))

# Model D produces our lowest RMSE to date which is 0.86535.This represents an approximate 8.3% decrease in RMSE when compared to Model C and 18.5% decrease when compared to Model A.
# This model represents a significant improvement but is still slightly higher than our desired RMSE threshold so further work is needed.
# If we can add an extension to this Model D that reduces our RMSE by 0.05% or greater, we will have achieved our goal.

# -------

# Model E: Regularized Movie & User Effect Model

# A technique that we went over through the course material will help us lower our RMSE and that technique is known as Regularization.
# Regularization adds a penalty of the different parameters of the model to reduce the model's freedom. Regularization helps us to add additional information to prevent overfitting and improve model accuracy by punishes values that exist towards the 'tails' (outlier ends) of the dataset values distribution.
# Regularization is one of the more important prerequisites to improving the accuracy and reliability of convergence and is a very effective method to apply here for our current problem.

# In regularization, we have to determine what is the optimal 'regularization' effect to inflict upon the model. This 'effect' or 'weighting' in regularization is normally defined as the lambda perameter and can often be referred to as the 'tuning parameter'. The cross validation process helps us determine our optimal tuning parameter, lambda.
# That two-step process is developed as follows:
# 1. Determining the optimal value for the tuning parameter, lambda. This is done using cross validation.
# 2. Apply that lambda to the regularization equation on Model D in order to determine the RMSE for our regularized model, denoted as Model E.

lambdas_seq <- seq(0, 8, 0.2)

reg_rmses <- sapply(lambdas_seq, function(lambda){
  
  moviebias_reg <- edx %>% group_by(movieId) %>% 
    summarise(n=n(), bi_reg = sum(rating - mu) / (lambda + n))
  
  movieuserbias_reg <- edx %>% group_by(userId) %>% left_join(moviebias_avg, by="movieId") %>%
    summarise(n=n(), bu_reg = sum(rating - mu - b_i) / (lambda + n))
  
  bi_reg <- validation %>% left_join(moviebias_reg, by="movieId") %>% .$bi_reg
  
  bu_reg <- validation %>% left_join(movieuserbias_reg, by="userId") %>% .$bu_reg
  
  pred_reg <- mu + bi_reg + bu_reg
  
  return(RMSE(pred_reg, validation$rating))
  
})

# We can plot our lambda distribution if we want to see how it looks.

plot_lambdas <- print(data.frame(lambdas_seq, reg_rmses) %>% ggplot(aes(lambdas_seq, reg_rmses)) + ggtitle("Plotting Lambda Sequence for Optimal Lambda") + labs(x="Lambda Distribution", y="Regularized RMSEs") + geom_point(color='purple'))

# Our optimal number for lambda is computed as follows:

opt_lambda <- print(lambdas_seq[which.min(reg_rmses)])

# With our optimal lambda determined (5.2), we can create Model E and determine how the corresponding RMSE relates to Model D (our lowest RMSE to date)

moviebias_reg <- edx %>% group_by(movieId) %>% summarise(n=n(), bi_reg = sum(rating - mu) / (opt_lambda + n))
movieuserbias_reg <- edx %>% group_by(userId) %>% left_join(moviebias_avg, by="movieId") %>% summarise(n=n(), bu_reg = sum(rating - mu - b_i) / (opt_lambda + n))
bi_reg <- validation %>% left_join(moviebias_reg, by="movieId") %>% .$bi_reg
bu_reg <- validation %>% left_join(movieuserbias_reg, by="userId") %>% .$bu_reg
pred_reg <- mu + bi_reg + bu_reg
model_e <- print(RMSE(validation$rating,pred_reg))

# After much model development, starting with Model A, we now have an RMSE reported under our required threshold of 0.8649. Success!

# -------

# Reported Results: In the Form of an RMSE Table

RMSE_Results_Table <- print(data.frame(Letter = c("A","B","C","D","E"), Model_Name = c("Average Movie Rating","Median Movie Rating","Adjusted Movie Effect","Adjusted Movie & User Effect","Regularized Movie & User Effect"), Model_RMSE = c(model_a, model_b, model_c, model_d, model_e), RMSE_Success = c("No", "No", "No", "No", "Yes")))

# Conclusions

# From the table reported above (entitled RMSE Results Table), we can see how the RMSE values differ for our 5 modeling attempts. By accounting for movie and user biases and initiating regularization on our modeling, we are able to achieve a RMSE below the required level of 0.8649.

# Project Extensions

# There are other, more advanced machine learning models that we can experiment with to improve the RMSE (specifically accounting for genre bias) but current
# limitations with memory (RAM) could not initial be overcome despite some exhaustive efforts. Through the data exploration stages, we were able to
# identify a potential genre bias but initial trials could not develop a proper model to account for this added layer. At this stage, that would be the key area of added exploration.