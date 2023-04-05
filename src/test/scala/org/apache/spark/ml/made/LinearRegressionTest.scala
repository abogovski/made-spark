package org.apache.spark.ml.made

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}
import com.google.common.io.Files
import org.scalatest._
import flatspec._
import matchers._
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.DataFrame

class LinearRegressionTest extends AnyFlatSpec with should.Matchers with WithSpark {
  val paramsTol = 0.01

  lazy val coeffs: Vector = Vectors.fromBreeze(LinearRegressionTest._coeffs)
  lazy val intercept: Double = LinearRegressionTest._intercept
  lazy val data: DataFrame = LinearRegressionTest._data

  "Regressor" should "fit to synthesized linear model" in {
    val model = new LinearRegression().setMaxIter(100).setStepSize(0.85).fit(data)
    model.coefficients.size should be(3)
    (model.coefficients(0) - coeffs(0)).abs should be < paramsTol
    (model.coefficients(1) - coeffs(1)).abs should be < paramsTol
    (model.coefficients(2) - coeffs(2)).abs should be < paramsTol
    (model.intercept - intercept).abs should be < paramsTol
  }

  "Model" should "correctly do predictions" in {
    val model = new LinearRegressionModel(coeffs.copy, intercept)
    for (row <- data.collect()) {
      (model.predict(row(0).asInstanceOf[Vector]) - row(1).asInstanceOf[Double]).abs should be < paramsTol
    }

    val transformedData = model.transform(data.select("features", "label")); // select ensures dataset is copied
    for (row <- transformedData.select( "prediction", "label").collect()) {
      (row(0).asInstanceOf[Double] - row(1).asInstanceOf[Double]).abs should be < paramsTol
    }
  }

  "Serialization" should "should preserve model" in {
    val uid = Identifiable.randomUID("MADE_LinearRegressionModel")
    val originalModel = new LinearRegressionModel(uid, coeffs.copy, intercept)

    val tmpFolder = Files.createTempDir()
    originalModel.write.overwrite().save("file://" + tmpFolder.getAbsolutePath)

    val preservedModel: LinearRegressionModel = LinearRegressionModel.load(tmpFolder.getAbsolutePath)
    preservedModel.coefficients.size should be(3)
    preservedModel.coefficients should equal(coeffs)
    preservedModel.intercept should be(intercept)
  }
};

object LinearRegressionTest extends WithSpark {
  import spark.implicits._

  lazy val _features: BDM[Double] = BDM.rand(100000, 3)
  lazy val _coeffs: BDV[Double] = BDV(1.5, 0.3, -0.7)
  lazy val _intercept = 1.23 /* was not specified in problem statement */
  lazy val _labels = (_features * _coeffs.asDenseMatrix.t) + _intercept

  lazy val _data = (0 until _features.rows).map(
    i => Vectors.fromBreeze(_features(i,::).t) -> _labels(i, 0)
  ).toDF("features" , "label")
}