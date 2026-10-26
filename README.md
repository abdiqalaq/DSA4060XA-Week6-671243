# DSA4060XA — Week 6 Guided Practical Lab
## Evaluating Recommender Systems: Rating Prediction, Ranking Quality, and Beyond-Accuracy Metrics

**Student Name:** Abdiqalaq Issack  
**Student ID:** 671243  
**Course:** DSA4060XA — Recommender Systems  
**Institution:** United States International University–Africa (USIU-Africa)  
**Practical:** Week 6 — Evaluating Recommender Systems  
**Programming Language:** Python  
**Development Environment:** Visual Studio Code and Jupyter Notebook

---

## 1. Project Overview

Recommender systems are widely used in digital platforms to help users discover products, movies, music, and other content that aligns with their interests. Streaming services, in particular, depend on recommendation algorithms to organize large content catalogues and provide personalized viewing experiences.

However, evaluating a recommender system involves more than measuring how accurately it predicts user ratings. A model may produce accurate numerical predictions while failing to recommend relevant movies within its highest-ranked results. Similarly, a model may recommend a broad and diverse selection of movies without necessarily identifying content that users are most likely to enjoy.

This practical investigates these differences by implementing and evaluating three recommendation approaches:

1. Popularity-Based Recommendation
2. Content-Based Filtering
3. Item-Based Collaborative Filtering

The models are evaluated using three categories of metrics:

- **Rating prediction accuracy:** Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
- **Top-N ranking quality:** Precision@3, Recall@3, and Normalized Discounted Cumulative Gain (NDCG@3).
- **Beyond-accuracy performance:** Catalogue Coverage@3, Novelty@3, and Diversity@3.

The project also includes an extension experiment comparing Top-3 and Top-5 recommendation performance.

The final objective is to identify a suitable recommendation approach for a hypothetical streaming platform seeking to help users discover relevant movies without allowing a small number of popular titles to dominate recommendation lists.

---

## 2. Problem Statement

Different recommendation algorithms optimize different aspects of recommendation quality.

A popularity-based recommender may identify generally well-rated movies but repeatedly recommend the same titles to many users.

A content-based recommender may produce recommendations aligned with individual genre preferences, but these recommendations can become repetitive when they focus on similar content.

Collaborative filtering may discover relationships between movies that are not apparent from their genres, but it can struggle when historical interactions are limited.

These differences create an important evaluation problem: **How can recommendation models be compared fairly when prediction accuracy, relevance, coverage, novelty, and diversity may produce conflicting results?**

This project addresses the problem by evaluating all three models on a common dataset and using consistent training and testing procedures.

Rather than selecting a model based on a single metric, the study examines the trade-offs among several performance dimensions.

---

## 3. Project Objectives

### 3.1 General Objective

To implement and evaluate three movie recommendation models using rating prediction, ranking quality, and beyond-accuracy metrics, and determine their suitability for a streaming-platform recommendation scenario.

### 3.2 Specific Objectives

1. Load, inspect, and explore the provided movie and ratings datasets.
2. Analyze the distribution of user ratings.
3. Apply a chronological leave-one-out training and testing split.
4. Implement a popularity-based recommendation baseline.
5. Construct a content-based recommendation model using movie genres and user preference profiles.
6. Implement item-based collaborative filtering using cosine similarity.
7. Evaluate rating prediction performance using MAE and RMSE.
8. Generate personalized Top-3 recommendation lists.
9. Evaluate ranking performance using Precision@3, Recall@3, and NDCG@3.
10. Measure catalogue coverage, recommendation novelty, and genre diversity.
11. Compare the strengths and weaknesses of the three models.
12. Recommend a suitable approach for a streaming-platform business scenario.
13. Investigate the effect of increasing the recommendation list length from K = 3 to K = 5.

---

## 4. Dataset Description

### 4.1 Data Sources

The practical uses two provided CSV datasets.

**Movie dataset: `DSA4060_Week6_Movies.csv`**

This dataset contains movie identifiers, titles, and genre information.

| Column | Description |
|---|---|
| `movie_id` | Unique identifier assigned to each movie |
| `title` | Movie title |
| `genres` | One or more movie genres separated by `\|` |

Example movies include Galaxy Mission, Love in Nairobi, Final Battle, Family Weekend, and Future World.

**Ratings dataset: `DSA4060_Week6_Ratings.csv`**

This dataset records user interactions with movies.

| Column | Description |
|---|---|
| `user_id` | Unique user identifier |
| `movie_id` | Identifier of the rated movie |
| `rating` | Numerical rating assigned by the user |
| `timestamp` | Ordering information used to identify the latest interaction |

### 4.2 Dataset Characteristics

The initial data inspection produced the following results:

| Characteristic | Value |
|---|---:|
| Unique users | 12 |
| Movies | 20 |
| Observed ratings | 120 |
| Possible user–movie interactions | 240 |
| Observed interaction density | 50% |
| Missing interaction proportion | 50% |
| Training interactions | 108 |
| Test interactions | 12 |

The dataset contains 12 users and 20 movies, giving 240 possible user–movie combinations. Since only 120 ratings were observed, the interaction matrix has a density of 50%.

This means that half of the possible user–movie interactions are unobserved.

An unobserved interaction should not automatically be interpreted as a negative preference. It may simply indicate that a user has not watched or rated the movie.

### 4.3 Rating Distribution

The observed rating distribution was:

| Rating | Frequency | Percentage |
|---|---:|---:|
| 1 | 0 | 0.0% |
| 2 | 29 | 24.2% |
| 3 | 49 | 40.8% |
| 4 | 28 | 23.3% |
| 5 | 14 | 11.7% |
| **Total** | **120** | **100%** |

Rating 3 was the most frequent, accounting for approximately 40.8% of all observations.

Ratings of 4 and 5 together accounted for 35% of the dataset.

This distribution indicates that moderate ratings are more common than very high ratings. Consequently, models predicting values near the middle of the rating scale may achieve reasonable average prediction errors.

However, the rating distribution alone does not determine how effectively a model will rank relevant movies.

---

## 5. Development Environment and Technologies

The practical was implemented in Python using a Jupyter Notebook in Visual Studio Code.

### 5.1 Libraries

| Library | Purpose |
|---|---|
| pandas | Data loading, manipulation, grouping, and evaluation tables |
| NumPy | Numerical calculations, arrays, logarithms, and averaging |
| Matplotlib | Visualization of rating distributions |
| scikit-learn | Error metrics and cosine similarity |
| Jupyter Notebook | Interactive execution, explanations, and reproducible outputs |

### 5.2 Main Techniques

The implementation uses:

- Pandas DataFrames and pivot tables
- Chronological leave-one-out evaluation
- Grouped mean rating calculations
- Multi-hot encoding of movie genres
- Rating-weighted user preference vectors
- Cosine similarity
- Similarity-weighted rating prediction
- Top-K candidate ranking
- Offline ranking metrics
- Catalogue-level and list-level recommendation analysis

---

## 6. Methodology

### 6.1 Overall Workflow

The project follows the sequence below:

1. Load the movie and ratings datasets.
2. Inspect dataset structure and rating distributions.
3. Split the ratings into training and testing interactions.
4. Build three recommendation models.
5. Generate predicted ratings for the held-out test interactions.
6. Calculate MAE and RMSE.
7. Define relevant test movies.
8. Generate Top-3 recommendations for each user.
9. Calculate ranking quality metrics.
10. Evaluate coverage, novelty, and diversity.
11. Consolidate results into a unified evaluation table.
12. Interpret model trade-offs.
13. Develop a streaming-platform recommendation.
14. Conduct a Top-3 versus Top-5 extension.

### 6.2 Chronological Leave-One-Out Split

The ratings were sorted by user identifier and timestamp.

For each user, the latest interaction was selected as the test observation, while all earlier interactions were assigned to the training set.

This produced:

- **108 training interactions (90%)**
- **12 test interactions (10%)**

Each of the 12 users contributed one held-out interaction.

The split preserves the temporal ordering of interactions and provides a more realistic evaluation than randomly selecting test ratings without considering interaction order.

All user preference profiles, popularity statistics, and item similarity calculations were constructed from the training data.

The held-out test ratings were used for evaluation rather than model construction.

### 6.3 Evaluation Considerations

The same held-out test interactions were used for all three models, supporting a consistent comparison.

Nevertheless, the small test set introduces substantial uncertainty. Each individual interaction can noticeably affect the average evaluation metrics.

Therefore, the findings should be interpreted as results from a guided experimental dataset rather than definitive evidence of performance on a large streaming platform.

---

## 7. Model 1: Popularity-Based Recommendation

### 7.1 Model Description

The popularity baseline predicts a movie's rating using its average rating in the training dataset.

For a movie with observed training ratings, the predicted rating is:

**Predicted rating = Mean training rating of the movie**

If a movie has no observed training rating, the global training mean is used.

The global training mean was approximately **3.222**.

### 7.2 Implementation

The model groups training interactions by `movie_id` and calculates:

- Mean rating
- Number of observed ratings

The resulting movie-level statistics are used for prediction and recommendation ranking.

### 7.3 Examples of Highly Rated Movies

| Movie | Mean Training Rating | Rating Count |
|---|---:|---:|
| Broken Hearts | 4.167 | 6 |
| Galaxy Mission | 4.000 | 3 |
| The Promise | 4.000 | 2 |
| Family Weekend | 3.800 | 5 |
| Hidden Truth | 3.667 | 6 |

These results demonstrate that some highly ranked movies have relatively few observed ratings.

### 7.4 Strengths

The popularity baseline is straightforward to implement, computationally inexpensive, and interpretable.

It also provides a useful reference for determining whether more personalized models improve performance.

### 7.5 Limitations

The model does not directly account for individual user preferences.

It can repeatedly recommend similar highly rated movies across users.

Additionally, a movie with very few ratings may achieve a high mean rating, even when that estimate is uncertain.

The implementation does not impose a minimum rating-count threshold or apply Bayesian smoothing to the popularity scores.

---

## 8. Model 2: Content-Based Filtering

### 8.1 Model Description

Content-Based Filtering recommends movies based on similarities between movie characteristics and a user's historical preferences.

The practical uses movie genres as content features.

### 8.2 Genre Encoding

The `genres` column contains multiple genres separated by the `|` character.

Multi-hot encoding converts these genre combinations into binary feature vectors.

For example:

- Action|Sci-Fi becomes active indicators for Action and Sci-Fi.
- Romance|Drama becomes active indicators for Romance and Drama.

The resulting feature matrix contains **20 movies and 9 genre features**.

The genres are:

Action, Adventure, Comedy, Crime, Drama, Family, Mystery, Romance, and Sci-Fi.

### 8.3 User Preference Profiles

Each user's preference profile is constructed from the genres of previously rated movies.

Genre indicators are weighted by the user's historical ratings and averaged across their training interactions.

This approach gives greater contributions to genres associated with higher-rated movies while also reflecting how frequently each genre occurs in the user's history.

### 8.4 Example: User 1's Preference Profile

| Genre | Preference Score |
|---|---:|
| Drama | 1.667 |
| Sci-Fi | 1.000 |
| Comedy | 0.889 |
| Romance | 0.667 |
| Action | 0.556 |
| Family | 0.556 |
| Crime | 0.333 |
| Mystery | 0.333 |
| Adventure | 0.222 |

Drama and Sci-Fi were User 1's strongest genre preferences.

These results are consistent with the user's highly rated movies:

- Galaxy Mission — Action|Sci-Fi — Rating 5
- Digital Dreams — Sci-Fi|Drama — Rating 4

Drama achieved the highest preference score partly because it appeared frequently in User 1's rating history.

Sci-Fi achieved the second-highest score despite appearing in fewer movies because both Sci-Fi movies received relatively high ratings.

### 8.5 Rating Prediction

Cosine similarity measures alignment between the user's preference vector and a candidate movie's genre vector.

The practical transforms similarity into a predicted rating using:

**Predicted Rating = 1 + 4 × Cosine Similarity**

The result is clipped to the rating scale of 1–5.

This is a simplified prediction rule for the practical, not a statistically calibrated rating model.

### 8.6 Strengths

Content-Based Filtering incorporates individual genre preferences and can recommend movies using their characteristics.

It does not require other users to have rated a candidate movie, provided the movie's genre features are available.

### 8.7 Limitations

Genre information alone may not capture important factors such as storytelling quality, actors, pacing, or user mood.

The approach may also produce relatively similar recommendations when users have concentrated genre preferences.

In this implementation, the direct transformation of cosine similarity into rating values can contribute to prediction errors.

---

## 9. Model 3: Item-Based Collaborative Filtering

### 9.1 Model Description

Item-Based Collaborative Filtering identifies relationships between movies based on patterns in historical user ratings.

Unlike content-based filtering, it does not rely on genre descriptions to calculate item similarity.

Instead, it compares movies according to how users have rated them.

### 9.2 User–Item Rating Matrix

The training data was converted into a matrix where:

- Rows represent users.
- Columns represent movies.
- Values represent observed ratings.
- Missing values indicate unobserved interactions.

The matrix was then transposed so that movies became rows and users became columns.

Missing ratings were replaced with zeros for cosine similarity calculation.

### 9.3 Item Similarity Matrix

Cosine similarity was calculated between movie rating vectors.

The resulting similarity matrix had dimensions:

**20 × 20**

Selected similarities included:

| Movie Pair | Cosine Similarity |
|---|---:|
| Movie 2 and Movie 3 | 0.646 |
| Movie 2 and Movie 4 | 0.569 |
| Movie 2 and Movie 5 | 0.544 |
| Movie 1 and Movie 4 | 0.441 |
| Movie 1 and Movie 5 | 0.133 |

Movie 2 and Movie 3 exhibited relatively strong alignment in their rating patterns.

Movie 1 and Movie 5 exhibited substantially weaker alignment.

### 9.4 Rating Prediction

The model predicts a target movie's rating using a similarity-weighted average of the user's historical ratings.

For each previously rated movie, its rating is multiplied by its similarity to the target movie.

The weighted ratings are summed and divided by the sum of positive similarities.

If no positive similarities are available, the global training mean is used.

### 9.5 Strengths

Item-Based Collaborative Filtering can identify relationships between movies even when their genre descriptions differ.

It uses collective interaction patterns and can provide recommendations that are not restricted to a user's most frequent genres.

### 9.6 Limitations

The model depends on historical rating information.

Movies with few interactions may have unreliable similarity estimates.

Replacing missing ratings with zeros may influence similarity scores.

The method may also require additional computational resources as the number of users and movies increases.

---

## 10. Rating Prediction Evaluation

### 10.1 Mean Absolute Error

Mean Absolute Error measures the average absolute difference between actual and predicted ratings.

**MAE = Average of |Actual Rating − Predicted Rating|**

Lower MAE indicates more accurate predictions.

### 10.2 Root Mean Squared Error

Root Mean Squared Error measures the square root of the average squared prediction error.

**RMSE = Square Root of Mean Squared Error**

RMSE penalizes larger prediction errors more strongly than MAE.

### 10.3 Results

| Model | MAE | RMSE |
|---|---:|---:|
| Popularity Baseline | 1.086 | 1.201 |
| Content-Based Filtering | 1.094 | 1.381 |
| **Item-Based Collaborative Filtering** | **0.924** | **1.070** |

### 10.4 Interpretation

Item-Based Collaborative Filtering achieved the lowest MAE and RMSE.

Relative to the Popularity Baseline, its MAE was approximately 14.9% lower and its RMSE approximately 10.9% lower.

Content-Based Filtering performed slightly worse than the baseline in MAE and more noticeably worse in RMSE.

The results suggest that item-based similarity patterns were more useful for predicting the held-out ratings in this dataset than the simplified genre-based prediction approach.

However, rating prediction accuracy does not necessarily determine how well a model ranks candidate movies.

---

## 11. Top-N Recommendation Generation

Each model generated Top-3 recommendations for every user.

Movies already rated by the user in the training data were excluded from the candidate list.

The held-out test movies remained eligible so that ranking evaluation could determine whether the models successfully recommended them.

### 11.1 Example: User 1

| Rank | Popularity Baseline | Content-Based | Item-Based CF |
|---|---|---|---|
| 1 | Broken Hearts | Campus Life | The Last Case |
| 2 | The Promise | Broken Hearts | Laugh Out Loud |
| 3 | Campus Life | The Promise | The Promise |

The popularity and content-based models recommended the same three movies in different orders.

Item-Based Collaborative Filtering generated a more distinct list.

Only The Promise appeared in all three recommendation lists.

### 11.2 User 1's Held-Out Interaction

User 1's held-out movie was Movie 14, which received an actual rating of 4.

Because the relevance threshold was 4, this movie was classified as relevant.

None of the three models included Movie 14 in User 1's Top-3 recommendations.

This illustrates why generating plausible recommendations is not equivalent to retrieving known relevant test items.

---

## 12. Ranking Evaluation

### 12.1 Relevance Definition

A held-out movie was classified as relevant when its actual rating was at least 4.

The test dataset contained:

| Classification | Count |
|---|---:|
| Relevant | 5 |
| Non-relevant | 7 |
| Total | 12 |

Thus, 41.7% of held-out interactions were relevant.

### 12.2 Precision@3

Precision@3 measures the proportion of the three recommended movies that are known to be relevant.

**Precision@3 = Relevant recommendations in Top-3 / 3**

### 12.3 Recall@3

Recall@3 measures the proportion of known relevant held-out movies retrieved in the Top-3 recommendations.

**Recall@3 = Relevant test movies retrieved / Total known relevant test movies**

In this practical, users with no relevant held-out movie were assigned a recall value of zero.

### 12.4 NDCG@3

Normalized Discounted Cumulative Gain measures how highly relevant movies appear in recommendation rankings.

A relevant movie appearing first receives a greater contribution than one appearing second or third.

NDCG normalizes the ranking score relative to the ideal ranking.

Users without a relevant held-out movie were assigned an NDCG value of zero.

### 12.5 Ranking Results

| Model | Precision@3 | Recall@3 | NDCG@3 |
|---|---:|---:|---:|
| Popularity Baseline | 0.028 | 0.083 | 0.042 |
| **Content-Based Filtering** | **0.028** | **0.083** | **0.083** |
| Item-Based CF | 0.000 | 0.000 | 0.000 |

### 12.6 Interpretation

Content-Based Filtering achieved the highest NDCG@3.

Both Content-Based Filtering and the Popularity Baseline retrieved one known relevant held-out movie across the 12 users.

However, the content-based model ranked its retrieved relevant movie higher.

Item-Based Collaborative Filtering did not retrieve any of the five known relevant test movies within its Top-3 recommendations.

This is particularly important because Item-Based CF achieved the best rating prediction accuracy.

The result demonstrates that rating prediction and recommendation ranking are different evaluation tasks.

### 12.7 Evaluation Limitation

Each user had only one held-out test movie.

Consequently, the evaluation does not establish whether other recommended movies would have been relevant if the users had rated them.

The ranking results should therefore be interpreted as retrieval performance against the available held-out labels, not as a complete assessment of recommendation relevance.

---

## 13. Catalogue Coverage Evaluation

### 13.1 Definition

Catalogue coverage measures the proportion of available movies that appear in at least one recommendation list.

**Coverage = Unique Recommended Movies / Total Catalogue Movies**

### 13.2 Results

| Model | Unique Movies Recommended | Coverage@3 |
|---|---:|---:|
| Popularity Baseline | 7 | 35% |
| Content-Based Filtering | 14 | 70% |
| **Item-Based CF** | **17** | **85%** |

### 13.3 Interpretation

Item-Based Collaborative Filtering achieved the highest coverage, recommending 17 of the 20 available movies across users.

Content-Based Filtering achieved 70% coverage, twice that of the Popularity Baseline.

The Popularity Baseline recommended only seven distinct movies, indicating that its recommendations were concentrated on a relatively small subset of the catalogue.

For a streaming platform, broader coverage may help expose users to a larger selection of titles.

However, coverage does not indicate whether recommended movies are relevant.

---

## 14. Recommendation Novelty Evaluation

### 14.1 Definition

Novelty measures how uncommon recommended movies are according to their interaction frequency in the training dataset.

Movies with fewer observed ratings receive higher novelty scores.

The implementation uses a smoothed interaction-frequency probability:

**Probability(movie) = (Movie Training Rating Count + 1) / (Total Training Ratings + Catalogue Size)**

The novelty score is then calculated as:

**Novelty(movie) = −log₂(Probability(movie))**

The model-level novelty score is the average across all Top-3 recommendation positions for all users.

### 14.2 Results

| Model | Novelty@3 |
|---|---:|
| **Popularity Baseline** | **4.733** |
| Content-Based Filtering | 4.453 |
| Item-Based CF | 4.720 |

### 14.3 Interpretation

The Popularity Baseline achieved the highest novelty score, although its advantage over Item-Based Collaborative Filtering was only 0.013.

This result is possible because the baseline ranks movies using average ratings rather than the number of ratings received.

A movie may have a high average rating while remaining relatively infrequently rated.

Content-Based Filtering achieved the lowest novelty score, suggesting that its recommendations tended to include relatively more frequently rated movies.

The findings demonstrate that novelty and catalogue coverage measure different aspects of recommendation behaviour.

A model may repeatedly recommend a small selection of uncommon movies and therefore achieve high novelty but low coverage.

---

## 15. Recommendation Diversity Evaluation

### 15.1 Definition

Diversity measures differences among movies within an individual recommendation list.

The implementation calculates genre-based cosine similarity between every pair of movies in each Top-3 list.

Pairwise dissimilarity is calculated as:

**Dissimilarity = 1 − Cosine Similarity**

The diversity score is the average pairwise dissimilarity within each recommendation list, averaged across all users.

### 15.2 Results

| Model | Diversity@3 |
|---|---:|
| Popularity Baseline | 0.736 |
| Content-Based Filtering | 0.542 |
| **Item-Based CF** | **0.780** |

### 15.3 Interpretation

Item-Based Collaborative Filtering achieved the highest genre diversity.

The Popularity Baseline also produced relatively diverse recommendations.

Content-Based Filtering achieved the lowest diversity score.

This is consistent with the content-based approach, which prioritizes movies that align with a user's existing genre preferences.

Although this can improve personalization, it may also result in recommendations containing similar genre combinations.

The results demonstrate that strong personalization does not necessarily produce the most diverse recommendation lists.

---

## 16. Unified Model Evaluation

The complete comparison is presented below.

| Model | MAE ↓ | RMSE ↓ | Precision@3 ↑ | Recall@3 ↑ | NDCG@3 ↑ | Coverage@3 ↑ | Novelty@3 ↑ | Diversity@3 ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Popularity Baseline | 1.086 | 1.201 | 0.028 | 0.083 | 0.042 | 0.350 | **4.733** | 0.736 |
| Content-Based | 1.094 | 1.381 | **0.028** | **0.083** | **0.083** | 0.700 | 4.453 | 0.542 |
| Item-Based CF | **0.924** | **1.070** | 0.000 | 0.000 | 0.000 | **0.850** | 4.720 | **0.780** |

**Interpretation of arrows:**

- ↓ Lower is better.
- ↑ Higher is better.

Precision@3 and Recall@3 are tied between Content-Based Filtering and the Popularity Baseline.

### 16.1 Best Rating Prediction Model

**Item-Based Collaborative Filtering**

It achieved the lowest MAE and RMSE.

### 16.2 Best Ranking Model

**Content-Based Filtering**

It achieved the highest NDCG@3 and tied for the highest Precision@3 and Recall@3.

### 16.3 Best Catalogue Coverage

**Item-Based Collaborative Filtering**

It recommended 85% of the available catalogue.

### 16.4 Best Novelty

**Popularity Baseline**

It achieved a novelty score of 4.733, marginally above Item-Based CF.

### 16.5 Best Diversity

**Item-Based Collaborative Filtering**

It achieved a diversity score of 0.780.

---

## 17. Discussion of Model Trade-offs

### 17.1 Rating Accuracy versus Ranking Performance

One of the most significant findings is the difference between rating prediction accuracy and ranking performance.

Item-Based Collaborative Filtering achieved the lowest prediction errors but failed to retrieve known relevant movies within its Top-3 lists.

Content-Based Filtering performed worse at predicting numerical ratings but achieved the strongest ranking results.

This suggests that minimizing rating prediction error is not sufficient for selecting an effective Top-N recommender.

### 17.2 Personalization versus Catalogue Exposure

The popularity model recommended only 35% of the catalogue.

Both personalized approaches achieved broader coverage.

Content-Based Filtering recommended 70% of the catalogue, while Item-Based Collaborative Filtering recommended 85%.

This suggests that personalized recommendation approaches can distribute exposure across more titles.

However, greater coverage does not guarantee higher relevance.

### 17.3 Novelty versus Coverage

The Popularity Baseline achieved the highest novelty but the lowest catalogue coverage.

This is not contradictory because the two metrics measure different properties.

Novelty concerns the frequency with which recommended movies appear in historical interactions.

Coverage concerns how many distinct catalogue items are recommended across users.

A recommender can achieve high novelty by repeatedly recommending a small number of relatively uncommon movies.

### 17.4 Personalization versus Diversity

Content-Based Filtering produced the least diverse lists.

This may occur because the model recommends movies that closely match existing genre preferences.

Item-Based Collaborative Filtering achieved the greatest genre diversity, potentially because collective rating patterns connect movies from different genres.

### 17.5 Why No Single Metric Is Sufficient

A model with low MAE may not retrieve relevant items in its Top-N recommendations.

A model with high coverage may recommend many irrelevant items.

A model with high novelty may repeatedly recommend a small selection of uncommon movies.

A model with high diversity may include movies that do not match a user's interests.

Therefore, model selection should reflect the objectives of the intended application and balance multiple evaluation criteria.

---

## 18. Streaming Platform Business Recommendation

### 18.1 Business Scenario

Consider a streaming platform that wants to:

- Help users discover movies relevant to their interests.
- Reduce excessive repetition of the same popular titles.
- Increase exposure to a broader selection of available content.
- Maintain a personalized viewing experience.

### 18.2 Recommended Initial Model

**Content-Based Filtering**

The content-based approach is recommended as the initial model for this scenario.

### 18.3 Justification

**Relevance:** Content-Based Filtering achieved the highest observed NDCG@3 of 0.083.

**Catalogue exposure:** Its coverage of 70% was twice the popularity baseline's 35%.

**Personalization:** It incorporates individual users' historical genre preferences.

**Balance:** It offers a more defensible balance between observed ranking relevance and catalogue exposure than the other two approaches.

### 18.4 Why Not Item-Based CF Alone?

Item-Based Collaborative Filtering achieved the highest coverage and diversity.

However, its observed Top-3 ranking metrics were zero.

Therefore, the evaluation does not provide sufficient evidence to select it as the sole recommendation model when relevance is a central business objective.

### 18.5 Why Not the Popularity Baseline Alone?

The Popularity Baseline achieved only 35% coverage.

This concentration conflicts with the platform's objective of avoiding excessive repetition of a small set of titles.

### 18.6 Potential Hybrid Improvement

A future hybrid system could combine:

1. Content-based relevance scoring.
2. Collaborative-filtering candidate generation.
3. Diversity-aware reranking.

The objective would be to preserve personalized relevance while increasing catalogue exposure and reducing repetitive genre recommendations.

This hybrid approach is a proposed future improvement, not an implemented or evaluated model in the current practical.

### 18.7 Business Recommendation Summary

Content-Based Filtering is the preferred initial approach because it provides the strongest observed ranking performance and broader catalogue exposure than the popularity baseline.

However, further evaluation with larger datasets and more complete relevance information would be required before real-world deployment.

---

## 19. Extension Experiment: Top-3 versus Top-5

### 19.1 Purpose

The extension examines how increasing the number of recommendations affects ranking performance.

The original practical evaluated Top-3 recommendation lists.

The extension increases the list length to Top-5 while maintaining the same dataset, models, and relevance threshold.

### 19.2 Experimental Setup

Two recommendation list lengths were compared:

- K = 3
- K = 5

For each model, Precision, Recall, and NDCG were recalculated and averaged across all 12 users.

### 19.3 Results

| K | Model | Precision | Recall | NDCG |
|---|---|---:|---:|---:|
| 3 | Content-Based | 0.028 | 0.083 | 0.083 |
| 3 | Item-Based CF | 0.000 | 0.000 | 0.000 |
| 3 | Popularity Baseline | 0.028 | 0.083 | 0.042 |
| 5 | Content-Based | 0.033 | 0.167 | 0.116 |
| 5 | Item-Based CF | 0.017 | 0.083 | 0.032 |
| 5 | Popularity Baseline | 0.017 | 0.083 | 0.042 |

### 19.4 Content-Based Filtering

Content-Based Filtering improved across all three ranking metrics.

Recall increased from 0.083 to 0.167.

NDCG increased from 0.083 to 0.116.

The model retrieved two known relevant test movies at K = 5, compared with one at K = 3.

### 19.5 Item-Based Collaborative Filtering

Item-Based CF improved from zero ranking scores at K = 3.

At K = 5, it achieved:

- Precision: 0.017
- Recall: 0.083
- NDCG: 0.032

This indicates that one known relevant movie appeared within the Top-5 recommendations but not within the Top-3.

### 19.6 Popularity Baseline

The Popularity Baseline did not retrieve any additional known relevant test movies.

Recall remained 0.083 and NDCG remained 0.042.

Precision decreased because the recommendation list became longer without producing additional relevant hits.

### 19.7 Extension Conclusion

Increasing K improved the observed ranking performance of Content-Based Filtering and Item-Based Collaborative Filtering.

However, the improvement was not uniform across models.

Content-Based Filtering remained the strongest ranking approach at K = 5.

The extension demonstrates that recommendation list length is an important evaluation parameter and that models should be compared at consistent values of K.

---

## 20. Study Limitations

### 20.1 Small Dataset

The experiment uses only 12 users, 20 movies, and 120 observed ratings.

This limits the statistical reliability and generalizability of the findings.

### 20.2 One Held-Out Interaction per User

Each user contributes only one test interaction.

This provides limited information about the full range of movies each user might consider relevant.

### 20.3 Limited Positive Relevance Labels

Only five held-out interactions were classified as relevant.

As a result, ranking metrics are sensitive to a small number of successful or unsuccessful recommendations.

### 20.4 Unobserved Items Are Not Necessarily Irrelevant

The evaluation only knows whether held-out movies were relevant.

Other recommended movies may also be relevant, but their relevance is not established by the available test labels.

Consequently, observed Precision and Recall values may not reflect complete real-world recommendation quality.

### 20.5 Simplified Content-Based Rating Prediction

The content-based model transforms cosine similarity directly into a 1–5 rating estimate.

This transformation is convenient for teaching but is not calibrated against observed rating behaviour.

### 20.6 Missing Ratings in Collaborative Filtering

Missing interactions are replaced with zeros when computing item similarity.

This assumption may affect similarity estimates, particularly for movies with limited ratings.

### 20.7 Limited Content Features

The content-based model uses genres only.

Other movie characteristics, including descriptions, actors, directors, release years, and user reviews, are not incorporated.

### 20.8 Offline Evaluation

The practical does not include online experiments, live user feedback, click-through rates, watch completion rates, or satisfaction measurements.

Therefore, offline metric improvements may not translate directly into improved user engagement.

### 20.9 No Statistical Significance Testing

The evaluation compares point estimates from a small held-out test set.

It does not include confidence intervals, repeated splits, or significance tests.

Small numerical differences should therefore be interpreted cautiously.

---

## 21. Recommendations for Future Work

Several improvements could strengthen the evaluation.

### 21.1 Expand the Dataset

Use a larger dataset with more users, movies, and interactions to improve the reliability of model comparisons.

### 21.2 Evaluate Multiple Test Interactions

Use a chronological split that retains several test interactions per user, providing more complete information for ranking evaluation.

### 21.3 Improve Content-Based Features

Include movie descriptions, actors, directors, release dates, and textual embeddings to represent content more comprehensively.

### 21.4 Improve Collaborative Filtering

Investigate alternative similarity measures, minimum co-rating requirements, rating normalization, and neighbourhood selection.

### 21.5 Calibrate Rating Predictions

Replace the simple cosine-similarity-to-rating transformation with a prediction method trained or calibrated on observed ratings.

### 21.6 Develop a Hybrid Recommender

Combine content-based relevance and collaborative-filtering signals, potentially followed by diversity-aware reranking.

### 21.7 Explore Additional Ranking Metrics

Consider Hit Rate@K, Mean Reciprocal Rank, Mean Average Precision, and ranking metrics evaluated over multiple held-out relevant items.

### 21.8 Evaluate Larger Recommendation Lists

Compare several K values to understand how recommendation length influences precision, recall, diversity, and coverage.

### 21.9 Conduct Online Evaluation

Where possible, evaluate models using real user engagement and satisfaction outcomes.

---

## 22. Installation and Reproducibility

### 22.1 Requirements

- Python 3
- Visual Studio Code or Jupyter Notebook
- pandas
- NumPy
- Matplotlib
- scikit-learn
- Jupyter

### 22.2 Clone the Repository

Clone the GitHub repository using:

`git clone <YOUR-REPOSITORY-URL>`

Then navigate to the project directory:

`cd DSA4060XA-Week6-671243`

Replace the placeholder URL with the actual GitHub repository URL.

### 22.3 Install Dependencies

Install the required Python packages:

`python -m pip install pandas numpy matplotlib scikit-learn jupyter`

Depending on the operating system and Python configuration, the command may use `python3` instead of `python`.

### 22.4 Open the Notebook

Open:

`DSA4060XA_Week6_Guided_Evaluation_Lab.ipynb`

Select the appropriate Python environment or Jupyter kernel.

### 22.5 Dataset Placement

Ensure both datasets are accessible using the file paths specified in the notebook:

- `DSA4060_Week6_Movies.csv`
- `DSA4060_Week6_Ratings.csv`

If the datasets are stored in another directory, update the loading paths accordingly.

### 22.6 Execute the Practical

Run the notebook cells sequentially, starting with library imports and data loading.

For final verification, restart the kernel and run all cells from the beginning.

The notebook should generate:

- Dataset inspection outputs
- Rating distribution visualization
- Training and testing datasets
- Popularity model results
- Content-based user preference profiles
- Item similarity matrix
- Rating prediction evaluation tables
- Top-3 recommendation lists
- Ranking evaluation results
- Coverage, novelty, and diversity results
- Unified evaluation table
- Top-3 versus Top-5 extension results

### 22.7 Reproducibility Notes

The chronological leave-one-out split is deterministic for the provided dataset and timestamp ordering.

No random model initialization is required by the three implemented recommendation methods.

To reproduce the reported results, use the same dataset, model definitions, relevance threshold, and evaluation conventions.

---

## 23. Repository Structure

The intended repository structure is:

    DSA4060XA-Week6-671243/
    |
    |-- DSA4060XA_Week6_Guided_Evaluation_Lab.ipynb
    |
    |-- DSA4060_Week6_Movies.csv
    |
    |-- DSA4060_Week6_Ratings.csv
    |
    |-- README.md

### File Descriptions

**DSA4060XA_Week6_Guided_Evaluation_Lab.ipynb**

Contains the complete practical, including data exploration, model implementation, evaluation calculations, results, interpretations, and extension experiment.

**DSA4060_Week6_Movies.csv**

Contains movie identifiers, titles, and genre information.

**DSA4060_Week6_Ratings.csv**

Contains user identifiers, movie identifiers, ratings, and timestamps.

**README.md**

Provides the project overview, methodology, evaluation results, interpretation, conclusions, and execution instructions.

---

## 24. Key Findings

The practical produced the following main findings:

1. **Item-Based Collaborative Filtering was the strongest rating prediction model**, achieving MAE of 0.924 and RMSE of 1.070.

2. **Content-Based Filtering was the strongest observed ranking model**, achieving NDCG@3 of 0.083 and NDCG@5 of 0.116.

3. **Item-Based Collaborative Filtering achieved the highest catalogue coverage**, recommending 85% of available movies across users.

4. **The Popularity Baseline achieved the highest novelty**, with a score of 4.733, marginally exceeding Item-Based CF.

5. **Item-Based Collaborative Filtering achieved the highest genre diversity**, with Diversity@3 of 0.780.

6. **The Popularity Baseline had the lowest catalogue coverage**, exposing users to only 35% of available movies.

7. **Increasing K from 3 to 5 improved the ranking results of Content-Based Filtering and Item-Based CF**, demonstrating that recommendation list length influences evaluation outcomes.

8. **No single model performed best across all metrics**, emphasizing the importance of multi-dimensional evaluation.

---

## 25. Conclusion

This practical implemented and evaluated three movie recommendation approaches using a common dataset and consistent evaluation procedures.

The results demonstrate that recommendation quality cannot be adequately measured using rating prediction accuracy alone.

Item-Based Collaborative Filtering achieved the strongest numerical rating prediction performance, the highest catalogue coverage, and the greatest genre diversity. However, it failed to retrieve known relevant test movies within the Top-3 recommendations.

Content-Based Filtering achieved the strongest observed ranking performance and substantially greater catalogue coverage than the Popularity Baseline.

The Popularity Baseline remained useful as a simple benchmark and achieved the highest interaction-based novelty score, but its recommendations were concentrated on a relatively small portion of the catalogue.

The extension experiment showed that increasing the recommendation list length from three to five improved ranking performance for Content-Based Filtering and Item-Based Collaborative Filtering.

For the streaming-platform business scenario, Content-Based Filtering was selected as the most appropriate initial approach because it offered the strongest observed relevance ranking and broader catalogue exposure than the popularity baseline.

Nevertheless, the results are exploratory because of the small dataset and limited held-out relevance information.

The central lesson from this practical is that **an effective recommender system must balance prediction accuracy, recommendation relevance, catalogue exposure, novelty, and diversity according to its intended business objectives.**

---

## 26. Academic Information

**Student:** Abdiqalaq Issack  
**Student ID:** 671243  
**Course:** DSA4060XA — Recommender Systems  
**Institution:** United States International University–Africa (USIU-Africa)  
**Assignment:** Week 6 Guided Practical Lab — Evaluating Recommender Systems

This repository documents the implementation, evaluation, interpretation, and extension of three recommendation approaches using the datasets provided for the practical.