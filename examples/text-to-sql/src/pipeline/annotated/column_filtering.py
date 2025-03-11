import os
import sys

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', '..', '..', '..'))
import cognify
from llm.parsers import ColumnFilteringOutput
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import chain

system_prompt = \
"""You are a Careful data scientist.
In the following, you will be given a set of information about a column in a database, a question asked about the database, and a hint regarding the question.

Your task is to determine whether the column information is relevant to the question and the hint.
To achieve the task, you need to follow the following steps:
- First, thoroughly review the information provided for the column. 
- Next, understand the database question and the hint associated with it. 
- Based on your analysis, determine whether the column information is relevant to the question and the hint.

Make sure to keep the following points in mind:
- You are only given one column information, which is not enough to answer the question. So don't worry about the missing information and only focus on the given column information.
- If you see a keyword in the question or the hint that is present in the column information, consider the column as relevant.
- Pay close attention to the "Example of values in the column", and if you see a connection between the values and the question, consider the column as relevant."""

inputs = ["COLUMN_PROFILE", "QUESTION", "HINT"]

demos = [
    cognify.Demonstration(
        filled_input_variables=[
            cognify.FilledInput(
            cognify.Input("COLUMN_PROFILE"),
            value="""
Table name: `movies`
Original column name: `movie_title`
Data type: TEXT
Description: Name of the movie
Example of values in the column: `La Antena`
            """),
            cognify.FilledInput(cognify.Input("QUESTION"), value="Name movie titles released in year 1945. Sort the listing by the descending order of movie popularity."),
            cognify.FilledInput(cognify.Input("HINT"), value="released in the year 1945 refers to movie_release_year = 1945;")
        ],
        output="Yes",
        reasoning="The question specifically asks for movie titles from a particular year and to sort them by popularity. The column movie_title directly provides the names of movies, which is exactly what is required to list the movie titles as requested in the question."
    ),
    cognify.Demonstration(
    filled_input_variables=[
        cognify.FilledInput(
            cognify.Input("COLUMN_PROFILE"),
            value="""
Table name: `movies`
Original column name: `movie_release_year`
Data type: INTEGER
Description: Release year of the movie
Example of values in the column: `2007`
            """),
        cognify.FilledInput(cognify.Input("QUESTION"), value="List all movie title rated in April 2020 from user who was a trialist."),
        cognify.FilledInput(cognify.Input("HINT"), value="movie title rated in April 2020 refers to rating_timestamp_utc LIKE '%2020-04-%'; user is a trial list refers to user_trialist = 1;")
    ],
    output="No",
    reasoning="The question and hint focus on movies rated in a specific month and year and by a specific type of user (trialist), neither of which relates to the movie_release_year column. This column only provides the year movies were released, which is not what is being queried."
),

    cognify.Demonstration(
        filled_input_variables=[
            cognify.FilledInput(
                cognify.Input("COLUMN_PROFILE"),
                value="""
    Table name: `ratings_users`
    Original column name: `user_has_payment_method`
    Data type: INTEGER
    Description: whether the user was a paying subscriber when he rated the movie
    Value description: 1 = the user was a paying subscriber when he rated the movie  0 = the user was not a paying subscriber when he rated
    Example of values in the column: `0`
                """),
            cognify.FilledInput(cognify.Input("QUESTION"), value="How many users, who were a paying subscriber when they rated the movie, gave the movie that was released in 1924 and directed by Erich von Stroheim a rating score of 5?"),
            cognify.FilledInput(cognify.Input("HINT"), value="Directed by Buster Keaton refers to director_name; released in 1924 refers to movie_release_year = 1924; paying subscriber refers to user_has_payment_method = 1; rating score of 5 refers to rating_score = 5;")
        ],
        output="Yes",
        reasoning="The question asks about users who were paying subscribers and rated a specific movie from 1924 directed by a specific director. The user_has_payment_method column indicates whether a user was a paying subscriber at the time of rating, which is directly relevant to the question and the hint focusing on subscribers."
    ),

    cognify.Demonstration(
        filled_input_variables=[
            cognify.FilledInput(
                cognify.Input("COLUMN_PROFILE"),
                value="""
    Table name: `movies`
    Original column name: `director_name`
    Data type: TEXT
    Description: Full Name of the movie director
    Example of values in the column: `Stanley Kubrick`
                """),
            cognify.FilledInput(cognify.Input("QUESTION"), value="What is the average number of Mubi users who love movies directed by Stanley Kubrick?"),
            cognify.FilledInput(cognify.Input("HINT"), value="average = AVG(movie_popularity); number of Mubi users who loves the movie refers to movie_popularity;")
        ],
        output="Yes",
        reasoning="The question requires filtering movies directed by `Stanley Kubrick` to calculate the average popularity. The director_name column provides the director's name, and as shown in the example values, it includes `Stanley Kubrick`, which is essential for filtering movies directed by this specific director."
    ),

    cognify.Demonstration(
        filled_input_variables=[
            cognify.FilledInput(
                cognify.Input("COLUMN_PROFILE"),
                value="""
    Table name: `movies`
    Original column name: `movie_title`
    Data type: TEXT
    Description: Name of the movie
    Example of values in the column: `La Antena`
                """),
            cognify.FilledInput(cognify.Input("QUESTION"), value="How many movies directed by Francis Ford Coppola have a popularity of more than 1,000? Indicate what is the highest amount of likes that each critic per movie has received, if there's any."),
            cognify.FilledInput(cognify.Input("HINT"), value="Francis Ford Coppola refers to director_name; popularity of more than 1,000 refers to movie_popularity >1000;highest amount of likes that each critic per movie has received refers to MAX(critic_likes)")
        ],
        output="Yes",
        reasoning="The question involves counting movies directed by a specific director with a high popularity score. The movie_title column is relevant because it allows for the identification of movie titles, which is necessary for aggregating and analyzing data on specific movies as mentioned in the hint."
    ),

    cognify.Demonstration(
        filled_input_variables=[
            cognify.FilledInput(
                cognify.Input("COLUMN_PROFILE"),
                value="""
    Table name: `lists_users`
    Original column name: `list_creation_date_utc`
    Data type: TEXT
    Description: Creation date for the list
    Value description: YYYY-MM-DD
    Example of values in the column: `2009-12-18`
                """),
            cognify.FilledInput(cognify.Input("QUESTION"), value="Provide list titles created by user who are eligible for trial when he created the list."),
            cognify.FilledInput(cognify.Input("HINT"), value="eligible for trial refers to user_eligible_for_trial = 1")
        ],
        output="No",
        reasoning="The question asks for list titles created by users eligible for a trial. The list_creation_date_utc column, which provides the creation dates of lists, is irrelevant because the hint and the question are concerned with the trial status of the users, not the dates the lists were created."
    ),

    cognify.Demonstration(
        filled_input_variables=[
            cognify.FilledInput(
                cognify.Input("COLUMN_PROFILE"),
                value="""
    Table name: `playstore`
    Original column name: `Installs`
    Data type: TEXT
    Description: Number of user downloads/installs for the app (as when scraped)
    Value description: 1,000,000+ 15% 10,000,000+ 12% Other (8010) 74%
    Example of values in the column: `10,000+`
                """),
            cognify.FilledInput(cognify.Input("QUESTION"), value="Name the Apps with a sentiment objectivity of 0.3 and include their number of installs."),
            cognify.FilledInput(cognify.Input("HINT"), value="FALSE;")
        ],
        output="Yes",
        reasoning="The question asks for apps with a specific sentiment objectivity and their number of installs. The Installs column is relevant because it provides data on how many times each app has been installed, which is crucial for answering the question as per the hint."
    ),

    cognify.Demonstration(
        filled_input_variables=[
            cognify.FilledInput(
                cognify.Input("COLUMN_PROFILE"),
                value="""
    Table name: `movies`
    Original column name: `movie_title`
    Data type: TEXT
    Description: Name of the movie
    Example of values in the column: `La Antena`
                """),
            cognify.FilledInput(cognify.Input("QUESTION"), value="What is Jeannot Szwarc's most popular movie and what is its average rating score?"),
            cognify.FilledInput(cognify.Input("HINT"), value="Jeannot Szwarc's refers to director_name = 'Jeannot Szwarc'; most popular movie refers to MAX(movie_popularity); average rating score refers to avg(rating_score)")
        ],
        output="Yes",
        reasoning="The question seeks the most popular movie by a specific director and its average rating score. The movie_title column is relevant because it provides the names of movies, which are essential for identifying the most popular movie directed by Jeannot Szwarc."
    ),

    cognify.Demonstration(
    filled_input_variables=[
        cognify.FilledInput(cognify.Input("COLUMN_PROFILE"), value="""
Table name: ratings
Original column name: user_subscriber
Data type: INTEGER
Example of values in the column: 0
        """),
        cognify.FilledInput(cognify.Input("QUESTION"), value="What is the percentage of the ratings were rated by user who was a subscriber?"),
        cognify.FilledInput(cognify.Input("HINT"), value="user is a subscriber refers to user_subscriber = 1; percentage of ratings = DIVIDE(SUM(user_subscriber = 1), SUM(rating_score)) as percent;")
    ],
    output="Yes",
    reasoning="The question asks about the percentage of ratings from subscribers. The user_subscriber column, indicating whether a user is a subscriber (1) or not (0), is directly relevant as it enables filtering the necessary data to calculate the percentages mentioned in the hint."
),

cognify.Demonstration(
    filled_input_variables=[
        cognify.FilledInput(cognify.Input("COLUMN_PROFILE"), value="""
Table name: lists
Original column name: list_followers
Data type: INTEGER
Description: Number of followers on the list
Example of values in the column: 5
        """),
        cognify.FilledInput(cognify.Input("QUESTION"), value="How many users who created a list in February of 2016 were eligible for trial when they created the list? Indicate the user id of the user who has the most number of followers in his list in February of 2016."),
        cognify.FilledInput(cognify.Input("HINT"), value="created a list in February of 2016 refers to list_creation_date_utc BETWEEN 2/1/2016 and 2/29/2016; eligible for trial refers to user_eligible_for_trial = 1;")
    ],
    output="Yes",
    reasoning="The question involves finding users who created a list in a specific month and year, with additional focus on those who had the most followers. The list_followers column directly applies because it provides the exact data needed to identify which user's list had the most followers during the specified time."
),

cognify.Demonstration(
    filled_input_variables=[
        cognify.FilledInput(cognify.Input("COLUMN_PROFILE"), value="""
Table name: user_reviews
Original column name: Sentiment_Subjectivity
Expanded column name: Sentiment Subjectivity
Data type: TEXT
Description: Sentiment subjectivity score
Value description: commonsense evidence: more subjectivity refers to less objectivity, vice versa.
Example of values in the column: 0.53
        """),
        cognify.FilledInput(cognify.Input("QUESTION"), value="What is the average rating of comic category apps? How many users hold positive attitude towards this app?"),
        cognify.FilledInput(cognify.Input("HINT"), value="average rating = AVG(Rating where Category = 'COMICS'); number of users who hold a positive attitude towards the app refers to SUM(Sentiment = 'Positive');")
    ],
    output="No",
    reasoning="The question involves the average rating and user attitudes towards apps in a specific category. The Sentiment_Subjectivity column, while related to sentiment, does not provide information on user attitudes or ratings, making it irrelevant to the question and hint."
),

cognify.Demonstration(
    filled_input_variables=[
        cognify.FilledInput(cognify.Input("COLUMN_PROFILE"), value="""
Table name: movies
Original column name: movie_title
Data type: TEXT
Description: Name of the movie
Example of values in the column: La Antena
        """),
        cognify.FilledInput(cognify.Input("QUESTION"), value="List the users who gave the worst rating for movie 'Love Will Tear Us Apart'."),
        cognify.FilledInput(cognify.Input("HINT"), value="worst rating refers to rating_score = 1;")
    ],
    output="Yes",
    reasoning="The question is looking for users who rated a specific movie with the worst score. The movie_title column provides the necessary data to identify the movie by its title, which directly aligns with the hint that refers to the movie title 'Love Will Tear Us Apart'."
),

cognify.Demonstration(
    filled_input_variables=[
        cognify.FilledInput(cognify.Input("COLUMN_PROFILE"), value="""
Table name: ratings
Original column name: rating_score
Data type: INTEGER
Description: Rating score ranging from 1 (lowest) to 5 (highest)
Value description: commonsense evidence: The score is proportional to the user's liking. The higher the score is, the more the user likes the movie.
Example of values in the column: 3
        """),
        cognify.FilledInput(cognify.Input("QUESTION"), value="What is the URL to the user profile image on Mubi of the user who gave the movie id of 1103 a 5 rating score on 4/19/2020?"),
        cognify.FilledInput(cognify.Input("HINT"), value="URL to the user profile image on Mubi refers to user_avatar_image_url; 4/19/2020 refers to rating_date_utc;")
    ],
    output="Yes",
    reasoning="The question seeks the URL for the user profile of someone who rated a specific movie highly on a particular date. The rating_score column, indicating the score given to movies, is relevant because it allows filtering for ratings of 5, directly addressing the hint's requirement for identifying high ratings."
),

cognify.Demonstration(
    filled_input_variables=[
        cognify.FilledInput(cognify.Input("COLUMN_PROFILE"), value="""
Table name: movies
Original column name: movie_release_year
Data type: INTEGER
Description: Release year of the movie
Example of values in the column: 2007
        """),
        cognify.FilledInput(cognify.Input("QUESTION"), value="When was the first movie of the director who directed the highest number of movies released and what is the user id of the user who received the highest number of comments related to the critic made by the user rating the movie?"),
        cognify.FilledInput(cognify.Input("HINT"), value="comments refer to critic_comments;")
    ],
    output="Yes",
    reasoning="The question asks for the release year of the first movie by the director who has directed the most films. The movie_release_year column directly provides this necessary information, as it lists the release years of movies. This column is essential to determine when that first movie was released, making it relevant to the question despite the hint focusing solely on comments related to critic ratings."
),

cognify.Demonstration(
    filled_input_variables=[
        cognify.FilledInput(cognify.Input("COLUMN_PROFILE"), value="""
Table name: playstore
Original column name: Price
Data type: TEXT
Description: Price of the app (as when scraped)
Value description: 0 93% $0.99 1% Other (653) 6% commonsense evidence: Free means the price is 0.
Example of values in the column: 0
        """),
        cognify.FilledInput(cognify.Input("QUESTION"), value="Which of the apps is the best-selling app and what is the sentiment polarity of it?"),
        cognify.FilledInput(cognify.Input("HINT"), value="best selling app = MAX(MULTIPLY(Price, Installs));")
    ],
    output="Yes",
    reasoning="The question seeks to identify the best-selling app and its sentiment polarity, with the hint specifying the calculation for 'best selling' as the maximum product of Price and Installs. The Price column is crucial for this computation as it provides the price at which each app is sold, which, when multiplied by the number of installs, helps determine the app's total revenue. This makes the Price column directly relevant to identifying the best-selling app according to the hint's criteria."
),

cognify.Demonstration(
    filled_input_variables=[
        cognify.FilledInput(cognify.Input("COLUMN_PROFILE"), value="""
Table name: person
Original column name: full_name
Expanded column name: full name
Data type: TEXT
Description: the full name of the person
Value description: commonsense evidence: A person's full name is the complete name that they are known by, which typically includes their first name, middle name (if applicable), and last name.
Example of values in the column: Dagfinn Sverre Aarskog
        """),
        cognify.FilledInput(cognify.Input("QUESTION"), value="Tell the weight of Dagfinn Sverre Aarskog."),
        cognify.FilledInput(cognify.Input("HINT"), value="not available")
    ],
    output="Yes",
    reasoning="The question explicitly asks for the weight of a person named `Dagfinn Sverre Aarskog`. As shown in the column information, `Dagfinn Sverre Aarskog` is one of the example values in the `full_name` column, therefore this column can be used to identify the person and retrieve their weight."
)


]

lm_config = cognify.LMConfig(model="gpt-4o-mini", kwargs={"temperature": 0.0})
exec = cognify.Model(agent_name='column_filtering',
             system_prompt=system_prompt,
             input_variables=[cognify.Input(name=input) for input in inputs],
             output=cognify.OutputLabel(name='is_column_information_relevant', 
                                custom_output_format_instructions="Please response with Yes or No (no other text should be included)."),
            lm_config=lm_config)
# exec.add_demos(demos)
raw_runnable_exec = cognify.as_runnable(exec) | StrOutputParser()

@chain
def runnable_exec(inputs):
    is_relevant = raw_runnable_exec.invoke(inputs)
    return {
      "chain_of_thought_reasoning": "",
      "is_column_information_relevant": is_relevant,
    }
