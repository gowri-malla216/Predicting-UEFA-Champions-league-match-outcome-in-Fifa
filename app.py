from flask import Flask, render_template, request
from pyspark.sql import SparkSession
import fickling

app = Flask(__name__)

# Initialize Spark session
spark = SparkSession.builder.appName("FIFA_Column_Selection").getOrCreate()

# Replace 'your_data_file.csv' with the actual path to your dataset
data_path = './data/female_players_legacy.csv'

# Read the data into a Spark DataFrame
data = spark.read.csv(data_path, header=True, inferSchema=True)

# Select relevant columns
selected_columns = ["fifa_version", "player_positions", "overall", "potential", "age", "height_cm", "weight_kg",
                    "weak_foot", "skill_moves", "club_name", "long_name",
                    "attacking_crossing", "attacking_finishing", "attacking_heading_accuracy",
                    "attacking_short_passing", "attacking_volleys", "skill_dribbling", "skill_curve",
                    "skill_fk_accuracy", "skill_long_passing", "skill_ball_control", "movement_acceleration",
                    "movement_sprint_speed", "movement_agility", "movement_reactions", "movement_balance",
                    "power_shot_power", "power_jumping", "power_stamina", "power_strength", "power_long_shots",
                    "mentality_aggression", "mentality_interceptions", "mentality_positioning", "mentality_vision",
                    "mentality_penalties", "defending_marking_awareness", "defending_standing_tackle",
                    "defending_sliding_tackle"]

# Get unique FIFA versions in the dataset
fifa_versions = data.select("fifa_version").distinct().rdd.flatMap(lambda x: x).collect()

# Create separate DataFrames for each FIFA version with selected columns
version_dataframes = {version: data.filter(data.fifa_version == version).select(selected_columns) for version in fifa_versions}

# Accessing data for a specific FIFA version, for example, FIFA 20
fifa_20_data = version_dataframes[20]

def load_model(fifa_version):
    with open(f'model_{fifa_version}.pkl', 'rb') as file:
        model = fickling.load(file)
    return model

def calculate_win_probability(overall_attack, overall_mid, overall_defense):
    # Replace this function with your actual win probability calculation
    return 0.75  # Placeholder value

def get_teams_rdd(fifa_version):
    return version_dataframes[fifa_version].rdd.map(lambda row: row['club_name']).distinct()

def get_players_rdd(fifa_version, club_name):
    return version_dataframes[fifa_version].rdd.filter(lambda row: row['club_name'] == club_name)

@app.route('/')
def index():
    return render_template('index.html', fifa_versions=fifa_versions)

@app.route('/teams', methods=['POST'])
def get_teams():
    fifa_version = request.form.get('fifa_version')
    teams = get_teams_rdd(fifa_version).collect()
    return render_template('teams.html', teams=teams)

@app.route('/players', methods=['POST'])
def get_players():
    # Get selected FIFA version and club name from the form
    fifa_version = request.form.get('fifa_version')
    club_name = request.form.get('club_name')

    # Filter player data based on selected FIFA version and club name
    filtered_players_rdd = get_players_rdd(fifa_version, club_name)

    # Extract unique player names and positions for each player
    players_info = []
    for player in filtered_players_rdd.collect():
        long_names = player['long_name']  # Replace 'long_name' with your actual column name
        positions = player['player_positions']  # Replace 'player_positions' with your actual column name

        players_info.append({
            'long_names': long_names,
            'positions': positions
        })

    # Render the players.html template with the filtered player information
    return render_template('players.html', fifa_version=fifa_version, players=players_info)

@app.route('/predict', methods=['POST'])
def predict():
    fifa_version = request.form.get('fifa_version')

    # # Load the corresponding Spark DataFrame for the selected FIFA version
    # # Replace 'your_spark_data.csv' with the actual path to your Spark data
    # spark_data_path = 'your_spark_data.csv'
    # spark_data = spark.read.csv(spark_data_path, header=True, inferSchema=True)

    # # Perform feature engineering (replace with your actual feature engineering steps)
    # assembler = VectorAssembler(inputCols=["overall", "potential", "age", "height_cm", "weight_kg",
    #                                        "weak_foot", "skill_moves"], outputCol="features")
    # transformed_data = assembler.transform(spark_data)

    # Load the pre-trained model
    model = load_model(fifa_version)

    # Collect player ratings from the form
    player_ratings = {}
    for player in request.form:
        if player.endswith('_search'):
            player_ratings[player[:-7]] = float(request.form.get(player))

    # Create a Spark DataFrame for the input data
    input_data = spark.createDataFrame([(fifa_version, *list(player_ratings.values()))],
                                       ["fifa_version", *list(player_ratings.keys())])

    # Transform the input data for prediction
    input_transformed = assembler.transform(input_data)

    # Make predictions using the pre-trained model
    predictions = model.transform(input_transformed)

    # Calculate team overall attack, overall mid, overall defense
    attack_players = ["F", "W", "S"]
    mid_players = ["M"]
    defense_players = ["B"]

    attack_rating = predictions.filter(predictions.player_positions.isin(attack_players)).agg({"prediction": "avg"}).collect()[0][0]
    mid_rating = predictions.filter(predictions.player_positions.isin(mid_players)).agg({"prediction": "avg"}).collect()[0][0]
    defense_rating = predictions.filter(predictions.player_positions.isin(defense_players)).agg({"prediction": "avg"}).collect()[0][0]

    # Use the overall ratings to calculate win probability
    overall_attack = (attack_rating + mid_rating) / 2
    overall_mid = mid_rating
    overall_defense = (mid_rating + defense_rating) / 2

    win_probability = calculate_win_probability(overall_attack, overall_mid, overall_defense)
    spark.stop()

    return render_template('result.html', win_probability=win_probability)

if __name__ == '__main__':
    app.run(debug=True)
