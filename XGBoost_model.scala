import org.apache.spark.ml.{Pipeline, PipelineStage, PipelineModel}
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer, VectorAssembler, CountVectorizer, StringIndexer, CountVectorizerModel, OneHotEncoder }
import ml.dmlc.xgboost4j.scala.spark.{XGBoostRegressor, XGBoostRegressionModel}


def vectorize(col: String, binary: Boolean, numFeats: Int, minDocFreq: Int) = {
  Array[PipelineStage](
    new CountVectorizer()
      .setBinary(binary).setVocabSize(numFeats).setMinDF(minDocFreq)
      .setInputCol(col).setOutputCol(col + "_tf")
  )
}

def train_test_XGBoost(trainingSet: DataFrame, testSet: DataFrame) = {
  val vectorAssembler = new VectorAssembler().setInputCols(
    Array(List_of_Features))
          .setOutputCol(featuresArray)

  val booster = new XGBoostRegressor()
                                    .setObjective("rank:ndcg")
                                    .setTrainTestRatio(0.8)
                                    .setEvalMetric("ndcg")
                                    .setNumRound(200)
                                    .setNumWorkers(20)
                                    .setFeaturesCol("all_features")
                                    .setLabelCol("label")
                                    .setGroupCol("userIndex")
                                    .setNumEarlyStoppingRounds(5)
                                    .setMaximizeEvaluationMetrics(true)
                                    

  val stages = (
    vectorize("id", binary=true, numFeats=1000, minDocFreq=100) 
    ++ vectorize("feature_1", binary=true, numFeats=1000, minDocFreq=100)
    ++ vectorize("feature_2", binary=true, numFeats=1000, minDocFreq=100)
    ++ vectorize("Feature_3", binary=true, numFeats=1000, minDocFreq=100)
    ++ Array[PipelineStage](vectorAssembler, booster)
  )


  val pipeline = new Pipeline().setStages(stages)
  val XGBoost_model = pipeline.fit(trainingSet)
  val predictions = XGBoost_model.transform(testSet)
  
  //return 
  (model, booster, predictions)
  
}


val featuresArray = Array("feature_name_1", "feature_name_2", "feature_name_3", ...)
val (model, booster, predictions) = train_test_XGBoost(trainingSet, testSet, featuresArray)


val predictions_TrainSet = model.transform(trainingSet).cache
val predictions_TestSet = model.transform(testSet).cache
val predictions_EvaluationSet = model.transform(evaluationSet).cache

















