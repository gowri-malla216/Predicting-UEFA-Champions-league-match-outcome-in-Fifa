from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import col, udf
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import explode
from pyspark.sql.functions import split
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.feature import PCA
from pyspark.ml.linalg import Vectors
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import PipelineModel

# Create a Spark session
spark = SparkSession.builder.appName("FIFA_Column_Selection").getOrCreate()

# Replace 'your_data_file.csv' with the actual path to your dataset
data_path = 'male_players.csv'

# Read the data into a Spark DataFrame
data = spark.read.csv(data_path, header=True, inferSchema=True)

# Select relevant columns
selected_columns = ["fifa_version", "player_positions", "overall", "potential", "age", "height_cm", "weight_kg",
                    "weak_foot", "skill_moves", "club_name","long_name",
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
fifa_20_data = fifa_20_data.filter(~fifa_20_data['player_positions'].contains("GK"))
fifa_20_data = fifa_20_data.sample(False, 0.1, seed=42).limit(100000)
fifa_data_exploded = fifa_20_data.withColumn('position', explode(split(fifa_20_data['player_positions'], ', ')))

# Index the 'position' column
indexer = StringIndexer(inputCol='position', outputCol='indexed_position')
indexer_model = indexer.fit(fifa_data_exploded)
fifa_data_indexed = indexer_model.transform(fifa_data_exploded)
columns_to_remove = ['fifa_version', 'player_positions', 'position','club_name', 'long_name']
fifa_data_indexed = fifa_data_indexed.drop(*columns_to_remove)
fifa_data_indexed.show(10)
positions = indexer_model.labels
# Assembling features
feature_columns =["overall", "potential", "age", "height_cm", "weight_kg",
                    "weak_foot", "skill_moves", 
                    "attacking_crossing", "attacking_finishing", "attacking_heading_accuracy",
                    "attacking_short_passing", "attacking_volleys", "skill_dribbling", "skill_curve",
                    "skill_fk_accuracy", "skill_long_passing", "skill_ball_control", "movement_acceleration",
                    "movement_sprint_speed", "movement_agility", "movement_reactions", "movement_balance",
                    "power_shot_power", "power_jumping", "power_stamina", "power_strength", "power_long_shots",
                    "mentality_aggression", "mentality_interceptions", "mentality_positioning", "mentality_vision",
                    "mentality_penalties", "defending_marking_awareness", "defending_standing_tackle",
                    "defending_sliding_tackle", 'indexed_position' ]
assembler = VectorAssembler(inputCols=feature_columns, outputCol='features')

fifa_data_indexed = assembler.transform(fifa_data_indexed)

# Standard Scaling (excluding indexed_player_positions)
scaler = StandardScaler(inputCol='features', outputCol='scaled_features', withStd=True, withMean=True)
scaler_model = scaler.fit(fifa_data_indexed)
fifa_data_scaled = scaler_model.transform(fifa_data_indexed)

# Perform PCA directly on DataFrame
num_principal_components = 8  # Choose accordingly
pca = PCA(k=num_principal_components, inputCol="scaled_features", outputCol="pca_features")
pca_model = pca.fit(fifa_data_scaled)
fifa_data_pca = pca_model.transform(fifa_data_scaled)
# Split data into train and test
train_data, test_data = fifa_data_pca.randomSplit([0.8, 0.2], seed=42)

# Define Random Forest model
rf = RandomForestRegressor(numTrees=5000, featuresCol='pca_features', labelCol='overall')

# Create a pipeline
pipeline = Pipeline(stages=[rf])

# Train the model
model = pipeline.fit(train_data)
model.save(./"fifa_20_rf_model")
# Make predictions on the test set
predictions = model.transform(test_data)

# Evaluate the model
evaluator = RegressionEvaluator(labelCol='overall', predictionCol='prediction', metricName='rmse')
rmse = evaluator.evaluate(predictions)
print(f"Root Mean Squared Error (RMSE): {rmse}")