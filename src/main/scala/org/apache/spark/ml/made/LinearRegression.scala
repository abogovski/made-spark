package org.apache.spark.ml.made

import breeze.linalg.{DenseVector => BDV}
import org.apache.spark.ml.linalg.{Vector, VectorUDT, Vectors}
import org.apache.spark.ml.param.shared.{HasFeaturesCol, HasLabelCol, HasMaxIter, HasPredictionCol, HasStepSize}
import org.apache.spark.ml.{Estimator}
import org.apache.spark.ml.regression.RegressionModel
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.MetadataUtils.getNumFeatures
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsReader, DefaultParamsWritable, DefaultParamsWriter, Identifiable, MLReadable, MLReader, MLWritable, MLWriter, SchemaUtils}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset, Encoder}
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.types.StructType

private[made] trait LinearRegressionParams
    extends HasFeaturesCol with HasLabelCol with HasPredictionCol with HasStepSize with HasMaxIter {

  def setLabelCol(value: String): this.type = set(labelCol, value)
  def setStepSize(value: Double): this.type = set(stepSize, value)
  def setMaxIter(value: Int): this.type = set(maxIter, value)

  setDefault(maxIter -> 100)
  setDefault(stepSize -> 0.85)

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, getFeaturesCol, new VectorUDT())
    SchemaUtils.checkColumnType(schema, getLabelCol, new VectorUDT())

    if (schema.fieldNames.contains($(predictionCol))) {
      SchemaUtils.checkColumnType(schema, getPredictionCol, new VectorUDT())
      schema
    } else {
      SchemaUtils.appendColumn(schema, schema(getLabelCol).copy(name = getPredictionCol))
    }
  }
}

class LinearRegression (override val uid: String)
  extends Estimator[LinearRegressionModel]
    with LinearRegressionParams with DefaultParamsWritable
{
  def this() = this(Identifiable.randomUID("MADE_LinearReggression"))

  override def copy(extra: ParamMap): LinearRegression = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

  override def fit(dataset: Dataset[_]): LinearRegressionModel = {
    val Xy = prepareXyRDD(dataset); Xy.persist()

    val (numRows, numFeatures) = (dataset.count(), getNumFeatures(dataset, $(featuresCol)))
    val coeffs = BDV.fill(numFeatures){0.0}
    var intercept = 0.0

    for (_ <- 0 until $(maxIter)) {
      val (coeffsStep, interceptStep) = calcStep(Xy, coeffs, intercept)
      coeffs -= $(stepSize) / numRows * coeffsStep
      intercept -= $(stepSize) / numRows * interceptStep
    }
    Xy.unpersist()
    copyValues(new LinearRegressionModel(uid, Vectors.fromBreeze(coeffs), intercept)).setParent((this))
  }

  private def prepareXyRDD(dataset: Dataset[_]): RDD[(BDV[Double], Double)] = {
    implicit val vecEncoder: Encoder[Vector] = ExpressionEncoder()
    implicit val fpEncoder: Encoder[Double] = ExpressionEncoder()

    dataset.select(
      dataset($(featuresCol)).as[Vector], dataset($(labelCol)).as[Double]
    ).rdd.map {
      case (x, y) => (x.asBreeze.toDenseVector, y)
    }
  }

  private def calcStep(
      Xy: RDD[(BDV[Double], Double)],
      coeffs: BDV[Double],
      intercept: Double): (BDV[Double], Double) = {

    val coeffsSpark = Vectors.fromBreeze(coeffs)

    Xy.mapPartitions({
      iterator: Iterator[(BDV[Double], Double)] =>
        val w: BDV[Double] = coeffsSpark.asBreeze.toDenseVector
        iterator.map {
          case (x, y) =>
            val eps = x.dot(w) + intercept - y
            (eps * x, eps)
        }
    }).reduce({
      case ((wLeft, bLeft), (wRight, bRight)) => (wLeft + wRight, bLeft + bRight)
    })
  } : (BDV[Double], Double)
}

object LinearRegression extends DefaultParamsReadable[LinearRegression]

class LinearRegressionModel private[made] (
    override val uid: String,
    val coefficients: Vector,
    val intercept: Double
  )
  extends RegressionModel[Vector, LinearRegressionModel]
    with LinearRegressionParams
    with MLWritable {

  private[made] def this(coeffs: Vector, intercept: Double) =
    this(Identifiable.randomUID("MADE_LinearRegressionModel"), coeffs, intercept)

  override val numFeatures = coefficients.size

  override def predict(features: Vector): Double = {
    features.dot(coefficients) + intercept
  }

  override def write: MLWriter = new DefaultParamsWriter(this) {
    override protected def saveImpl(path: String): Unit = {
      super.saveImpl(path)
      val data: (Vector, Double) = (coefficients.asInstanceOf[Vector] -> intercept.asInstanceOf[Double])
      sqlContext.createDataFrame(Seq(data)).write.parquet(path + "/data")
    }
  }
  override def copy(extra: ParamMap): LinearRegressionModel = {
    copyValues(new LinearRegressionModel(uid, coefficients, intercept), extra)
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    val transformUdf = dataset.sqlContext.udf.register(uid + "_transform",
        (x: Vector) => this.predict(x)
    )
    dataset.withColumn($(predictionCol), transformUdf(dataset($(featuresCol))))
  }
}

object LinearRegressionModel extends MLReadable[LinearRegressionModel] {
  override def read: MLReader[LinearRegressionModel] = new MLReader[LinearRegressionModel] {
    override def load(path: String): LinearRegressionModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc)

      val vectors = sqlContext.read.parquet(path + "/data")

      // Used to convert untyped dataframes to datasets with vectors
      implicit val vecEncoder : Encoder[Vector] = ExpressionEncoder()
      implicit val fpEncoder: Encoder[Double] = ExpressionEncoder()

      val (coeffs, intercept) = vectors.select(vectors("_1").as[Vector], vectors("_2").as[Double]).first()

      val uid = Identifiable.randomUID("MADE_LinearRegressionModel")
      val model = new LinearRegressionModel(uid, coeffs, intercept)
      metadata.getAndSetParams(model)
      model
    }
  }
}
