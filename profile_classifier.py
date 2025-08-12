# General imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack

# NLP imports
from bs4 import BeautifulSoup
import re
# NLTK for natural language processing
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('brown')

# Performance metrics imports
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve


# -------------------------  Extract and Load --------------------------------------
PATH = 'https://raw.githubusercontent.com/masudf6/bigdata_analytics/main/A2_2024_Released/twitter_user_data.csv'
df = pd.read_csv(PATH, encoding='ISO-8859-1')

# Drop the repeating info cols
cols_to_drop = ['_unit_id', '_unit_state', '_trusted_judgments', '_last_judgment_at',
               'gender_gold', 'profile_yn_gold', 'profileimage', 'tweet_coord','tweet_id', 'tweet_created', 'name', 'tweet_location', 'user_timezone']

df = df.drop(columns=cols_to_drop)
df.info()
df.head()


# --------------------------- Drop Entries with unusable Label -------------------------
# Function to plot gender distribution
def plot_gender_distribution(df):
  # Fill missing values in gender with 'blank'
  df.loc[:, 'gender'] = df['gender'].fillna('blank')

  plt.figure(figsize=(8, 6))
  ax = sns.countplot(data=df, x='gender')

  for p in ax.patches:
      ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                  ha='center', va='baseline', fontsize=12, color='black', xytext=(0, 5),
                  textcoords='offset points')

  plt.title('Gender Distribution')
  plt.show()

plot_gender_distribution(df)

# Filter out the unknowns and blanks
df = df.query("gender != 'unknown' and gender != 'blank'")
plot_gender_distribution(df)


# ------------------------------------- Data Prep ----------------------------------------

# Handle NULLS
def handle_nulls(df):

  # Check for missing values across the entire dataset
  missing_values = df.isna().sum()
  print(f"Missing values per column before: {missing_values[missing_values > 0]}")

  df['gender:confidence'] = df['gender:confidence'].fillna(0.5)
  df.fillna("not available", inplace=True)
  print(f'Missing value after handling blanks: {df.isna().sum().sum()}')

  return df

df = handle_nulls(df)


""" Text transformation """
def denoise_text(text):

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", '', text)

    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()

    # Remove user mentions, hashtags, special characters, digits, and brackets
    text = re.sub(r'(@\w+|#\w+|[^a-zA-Z\s]|\[[^]]*\])', '', text)

    # Remove extra whitespaces and strip leading/trailing spaces
    text = ' '.join(text.split())
    text = text.strip()

    return text

def standardize_text(text):
    return text.lower()

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    return ' '.join([word for word in text.split() if word not in stop_words])

def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

def stem_text(text):
    stemmer = PorterStemmer()
    return ' '.join([stemmer.stem(word) for word in text.split()])

def clean_text(text):
  return stem_text(lemmatize_text(remove_stopwords(standardize_text(denoise_text(text)))))


# TF-IDF matrix
def tfidf_sparse(df):

  # Initialize and fit the vectorizer
  tfidf = TfidfVectorizer(tokenizer=lambda x: x.split(), lowercase=False)
  tfidf_matrix = tfidf.fit_transform(df)

  # Get the token names
  tokens = tfidf.get_feature_names_out()

  # Convert to DataFrame for manipulation
  tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tokens)

  # Sum the TF-IDF scores across all documents for each token
  token_importance = tfidf_df.sum(axis=0)

  # Sort tokens by their importance
  sorted_importance = token_importance.sort_values(ascending=False)

  # Calculate cumulative sum of token importance
  cumulative_importance = np.cumsum(sorted_importance)

  # Select tokens contributing to the top 90% importance
  threshold = cumulative_importance[-1] * 0.90
  top_90_tokens = cumulative_importance[cumulative_importance <= threshold].index.tolist()

  # Initialize a new TF-IDF Vectorizer with the top 90% tokens
  tfidf_top_90 = TfidfVectorizer(tokenizer=lambda x: x.split(), lowercase=False, vocabulary=top_90_tokens)

  # Fit and transform the text column with the reduced vocabulary
  tfidf_matrix_top_90 = tfidf_top_90.fit_transform(df)

  # Optionally, convert the result to a DataFrame
  # tfidf_df_top_90 = pd.DataFrame(tfidf_matrix_top_90.toarray(), columns=tfidf_top_90.get_feature_names_out())
  # tfidf_df_top_90.head()

  """ Plots """
  fig, ax = plt.subplots(1, 2, figsize=(15, 6))

  # Barplot for top 20 tokens by importance using Seaborn
  top_20_tokens = sorted_importance.head(20).reset_index()
  top_20_tokens.columns = ['Token', 'TF-IDF Importance']
  sns.barplot(x='TF-IDF Importance', y='Token', data=top_20_tokens, ax=ax[0])
  ax[0].set_title('Top Tokens by TF-IDF')

  # Elbow curve for cumulative importance using Seaborn
  sns.lineplot(x=range(len(cumulative_importance)), y=cumulative_importance, ax=ax[1])
  ax[1].set_xlabel('Number of Tokens')
  ax[1].set_ylabel('Cumulative TF-IDF Importance')
  ax[1].set_title('Elbow Method for Token Selection')
  ax[1].axhline(y=threshold, color='r', linestyle='--', label='90% Threshold')
  ax[1].legend()

  plt.tight_layout()
  plt.show()

  return tfidf_matrix_top_90


def transform_text_features(df):

  df['text'] = df['text'].apply(clean_text)
  df['description'] = df['description'].apply(clean_text)
  print(df[['text', 'description']].head())

  tweet = tfidf_sparse(df['text'])
  description = tfidf_sparse(df['description'])

  df = df.drop(columns=['text', 'description'])

  return df, tweet, description

df, tweet, description = transform_text_features(df)

df.head()

""" Categorical (Hex value) transformation"""
# Function to clean and format the hex color
def clean_hex_color(hex_color):
    # Step 1: Strip whitespaces
    hex_color = hex_color.strip()

    # Step 2: Remove leading '#' if it exists
    if hex_color.startswith('#'):
        hex_color = hex_color[1:]

    # Step 3: Remove any invalid characters (non-hex digits)
    hex_color = ''.join(c for c in hex_color if c in '0123456789ABCDEFabcdef')

    # Step 4: Adjust the hex value length
    if len(hex_color) == 6:
        return hex_color  # Return as is if it's exactly 6 characters
    elif len(hex_color) > 6:
        return hex_color[:6]  # Strip the last characters if longer than 6
    elif hex_color == 0:
        return '000000'
    else:
        # If shorter than 6, add zeros at the end
        return hex_color.ljust(6, '0')

# Function to convert hex to RGB
def hex_to_rgb(hex_color):
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def transform_categorical_features(df):

  # Apply the cleaning function to the 'color_hex' column
  df['link_color'] = df['link_color'].apply(clean_hex_color)
  df['sidebar_color'] = df['sidebar_color'].apply(clean_hex_color)

  # Apply the function to create new columns for R, G, and B
  df[['li_R', 'li_G', 'li_B']] = pd.DataFrame(df['link_color'].apply(hex_to_rgb).tolist(), index=df.index)
  df[['sb_R', 'sb_G', 'sb_B']] = pd.DataFrame(df['sidebar_color'].apply(hex_to_rgb).tolist(), index=df.index)

  # Drop the original hex color columns
  df = df.drop(columns=['link_color', 'sidebar_color'])

  return df

df = transform_categorical_features(df)
df.head()


""" Date-time transformation"""
def transform_date_time_feature(df):

  # Convert 'created' column to datetime
  df['created'] = pd.to_datetime(df['created'])

  # Calculate the age in years
  current_date = pd.Timestamp.now()
  df['created'] = ((current_date - df['created']).dt.days / 365).astype(int)
  df.rename(columns={'created': 'account_age'}, inplace=True)

  return df

df = transform_date_time_feature(df)
df.head()

""" Boolean transformation"""
def transform_boolean_features(df):

  # Convert 'golden' and 'profile_yn' into numeric (0 and 1) values
  df['_golden'] = df['_golden'].astype(int)  # 'golden' is already boolean (True/False)

  df['profile_yn'] = df['profile_yn'].map({'yes': 1, 'no': -1})  # Assuming 'profile_yn' is 'yes' or 'no'
  df['profile_yn'] = df['profile_yn'] * df['profile_yn:confidence']
  df = df.drop(columns=['profile_yn:confidence'])

  return df

df = transform_boolean_features(df)
df.head()


""" Label Transformation"""
def label_transformation(df):

  df['isHuman'] = df['gender'].replace({'male': 'human', 'female': 'human', 'brand': 'nonhuman'})
  df['isHuman'] = df['isHuman'].map({'human': 1, 'nonhuman': 0})

  df = df.drop(columns=['gender'])

  return df

df = label_transformation(df)
df.head()

# Show whether the data is imbalanced and by how much
# Calculate the counts of 1s and 0s
counts = df['isHuman'].value_counts()
total_entries = len(df)
percentages = (counts / total_entries) * 100

# Plot percentages with annotations
plt.figure(figsize=(8, 6))
ax = sns.barplot(x=percentages.index, y=percentages.values, palette='viridis')
plt.title('Percentages of Humans(1) and NonHumans(0)')
plt.xlabel('Value')
plt.ylabel('Percentage')

# Annotate the bars
for p in ax.patches:
    ax.annotate(f'{p.get_height():.1f}%', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 5), textcoords='offset points')

plt.show()


# --------------------------- Data Splits -----------------------------------
y = df['isHuman']
df = df.drop(columns=['isHuman'])

df_color = df[['li_R', 'li_G', 'li_B', 'sb_R', 'sb_G', 'sb_B']]
df_count = df[['fav_number', 'retweet_count', 'tweet_count']]

def scaler(df):
  scaler = StandardScaler()
  scaled_features = scaler.fit_transform(df)
  return scaled_features

# Full data
scaled_features = scaler(df)
# Combine the scaled features with the TF-IDF matrices
X = hstack([tweet, description, scaled_features])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Text data only
X_text = hstack([tweet, description])
X_train_text, X_test_text, y_train_text, y_test_text = train_test_split(X_text, y, test_size=0.2, random_state=42, stratify=y)

# Dense data only (W/O text)
scaled_dense = scaler(df)
X_train_dense, X_test_dense, y_train_dense, y_test_dense = train_test_split(df, y, test_size=0.2, random_state=42, stratify=y)

# Color data only
scaled_color_data = scaler(df_color)
X_train_color, X_test_color, y_train_color, y_test_color = train_test_split(scaled_color_data, y, test_size=0.2, random_state=42, stratify=y)

# Count data only
scaled_count_data = scaler(df_count)
X_train_count, X_test_count, y_train_count, y_test_count = train_test_split(scaled_count_data, y, test_size=0.2, random_state=42, stratify=y)


# -------------------------- Model imports ----------------------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

# Initialize the Logistic Regression model
lr = LogisticRegression(max_iter=1000, random_state=42)
knn = KNeighborsClassifier()
dt = DecisionTreeClassifier()
rf = RandomForestClassifier()
mnb = MultinomialNB()
svm = SVC(probability=True)


# Train the model on the training data
# models = [lr, knn, dt, rf, svm]

# for model in models:
#   model.fit(X_train, y_train)

""" Because the models take up a lot of computation and time resources, except for Multinomial on Text and LR on full data, everything
    has been commented out."""

mnb.fit(X_train_text, y_train_text)
lr.fit(X_train, y_train)
svm.fit(X_train_color, y_train_color)


# lr.fit(X_train_text, y_train_text)
# lr = LogisticRegression(max_iter=1000, random_state=42)
# lr.fit(X_train_dense, y_train_dense)
# lr = LogisticRegression()
# lr.fit(X_train_color, y_train_color)
# lr = LogisticRegression()
# lr.fit(X_train_count, y_train_count)

# knn = KNeighborsClassifier()
# knn.fit(X_train_count, y_train_count)



# Param Grids
# param_grid_lr = {
#     'C': [0.01, 0.1, 1, 10, 100],  # Regularization strength
#     'penalty': ['l1', 'l2', 'elasticnet', 'none'],  # Regularization type
#     'solver': ['saga'],  # Only saga optimization algorithm chosen because we have a fairly large dataset
#     'max_iter': [100, 200, 300],  # Number of iterations
#     'class_weight': ['balanced'],  # Handling imbalanced data
#     'l1_ratio': [0.1, 0.5, 0.7]  # Only applicable for elasticnet
# }


# param_grid_knn = {
#     'n_neighbors': [3, 5, 7, 9],  # Number of neighbors to use
#     'weights': ['uniform', 'distance'],  # Weight function for prediction
#     'metric': ['euclidean', 'manhattan'],  # Distance metric to use
# }

# param_grid_dt = {
#     'criterion': ['gini', 'entropy'],  # Function to measure quality of split
#     'max_depth': [None, 10, 20, 30, 40, 50],  # Maximum depth of the tree
#     'min_samples_split': [2, 10, 20],  # Minimum number of samples to split a node
#     'min_samples_leaf': [1, 5, 10],  # Minimum number of samples at leaf node
#     'max_features': [None, 'sqrt', 'log2'],  # Number of features to consider when looking for best split
# }

# param_grid_rf = {
#     'n_estimators': [100, 200, 300],  # Number of trees in the forest
#     'max_depth': [None, 10, 20, 30],  # Maximum depth of the tree
#     'min_samples_split': [2, 10],  # Minimum number of samples required to split an internal node
#     'min_samples_leaf': [1, 4],  # Minimum number of samples required at each leaf node
#     'bootstrap': [True, False],  # Whether to use bootstrap samples when building trees
#     'max_features': ['sqrt', 'log2'],  # Number of features to consider for best split
# }

# param_grid_svm = {
#     'C': [0.1, 1, 10, 100],  # Regularization parameter
#     'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],  # Kernel type
#     'degree': [2, 3, 4],  # Degree of the polynomial kernel (only for 'poly')
#     'gamma': ['scale', 'auto'],  # Kernel coefficient (used with 'rbf', 'poly', 'sigmoid')
#     'probability': [True]  # Whether to enable probability estimates
# }

# param_grid_nb = {
    # 'var_smoothing': [1e-9, 1e-8, 1e-7]  # Smoothing parameter to prevent zero probabilities (GaussianNB)
# }

# RandomizedSearch cross folding
# from sklearn.model_selection import RandomizedSearchCV

# random_search = RandomizedSearchCV(
#     estimator=lr,
#     param_distributions=param_grid_lr,
#     n_iter=5,
#     scoring='f1',
#     cv=5,
#     n_jobs=-1,
#     random_state=42
# )

# # Fit RandomizedSearchCV
# random_search.fit(X_train, y_train)

# best_estimator = random_search.best_estimator_
# best_score = random_search.best_score_
# print(f"Best F1 score: {best_score:.3f}")
# print(f"Best parameters: {random_search.best_params_}")

# ---------------------------- Reasult Evaluation and performance metrics ----------------------------------------
def performance_metrics(model, X_test, y_test):

  # Predict probabilities and classes
  y_pred = model.predict(X_test)
  try:
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # For ROC and PR curves
  except AttributeError:
    # If the model does not support predict_proba, skip ROC and Precision-Recall curve
    y_pred_proba = None

  # Generate confusion matrix
  cm = confusion_matrix(y_test, y_pred)

  # Calculate ROC curve and AUC
  fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
  roc_auc = auc(fpr, tpr)

  # Calculate Precision-Recall curve
  precision_, recall_, _ = precision_recall_curve(y_test, y_pred_proba)

  # Calculate metrics
  accuracy = accuracy_score(y_test, y_pred)
  precision = precision_score(y_test, y_pred)
  recall = recall_score(y_test, y_pred)
  f1 = f1_score(y_test, y_pred)

  # Print the metrics
  print(f"Accuracy: {accuracy:.4f}")
  print(f"Precision: {precision:.4f}")
  print(f"Recall: {recall:.4f}")
  print(f"F1 Score: {f1:.4f}")


  """ Plots """

  # Create subplots for CM, ROC, and Precision-Recall Curve
  fig, ax = plt.subplots(1, 3, figsize=(18, 6))

  # Plot Confusion Matrix with Seaborn
  group_names = ['TN', 'FP', 'FN', 'TP']
  group_counts = [f"{value}" for value in cm.flatten()]
  labels = [f"{name}\n{count}" for name, count in zip(group_names, group_counts)]
  labels = np.asarray(labels).reshape(2, 2)

  sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', cbar=False, ax=ax[0], annot_kws={"size": 16})
  ax[0].set_title('Confusion Matrix')
  ax[0].set_xlabel('Predicted')
  ax[0].set_ylabel('Actual')

  # Plot ROC Curve
  if y_pred_proba is not None:
    ax[1].plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.2f})')
    ax[1].plot([0, 1], [0, 1], color='red', linestyle='--')
    ax[1].set_title('ROC Curve')
    ax[1].set_xlabel('False Positive Rate')
    ax[1].set_ylabel('True Positive Rate')
    ax[1].legend(loc="lower right")
  else:
    ax[1].set_title('ROC Curve (Not available)')

  # Plot Precision-Recall Curve
  if y_pred_proba is not None:
    ax[2].plot(recall_, precision_, color='green')
    ax[2].set_title('Precision-Recall Curve')
    ax[2].set_xlabel('Recall')
    ax[2].set_ylabel('Precision')
  else:
    ax[2].set_title('Precision-Recall Curve (Not available)')

  # Adjust layout
  plt.tight_layout()
  plt.show()


""" Commenting this out because time and computaion demanding"""
# for model in models:
#   print(model)
#   performance_metrics(model, X_test, y_test)

performance_metrics(lr, X_test, y_test)
performance_metrics(mnb, X_test_text, y_test_text)
performance_metrics(svm, X_test_color, y_test_color)