/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.ml.param

/* NOTE TO DEVELOPERS:
 * If you add these parameter traits into your algorithm, you need to add a setter method as well.
 */

private[ml] trait HasRegParam extends Params {
  /** param for regularization parameter */
  val regParam: DoubleParam = new DoubleParam(this, "regParam", "regularization parameter")
  def getRegParam: Double = get(regParam)
}

private[ml] trait HasMaxIter extends Params {
  /** param for max number of iterations */
  val maxIter: IntParam = new IntParam(this, "maxIter", "max number of iterations")
  def getMaxIter: Int = get(maxIter)
}

private[ml] trait HasFeaturesCol extends Params {
  /** param for features column name */
  val featuresCol: Param[String] =
    new Param(this, "featuresCol", "features column name", Some("features"))
  def getFeaturesCol: String = get(featuresCol)
}

private[ml] trait HasLabelCol extends Params {
  /** param for label column name */
  val labelCol: Param[String] = new Param(this, "labelCol", "label column name", Some("label"))
  def getLabelCol: String = get(labelCol)
}

private[ml] trait HasPredictionCol extends Params {
  /** param for prediction column name */
  val predictionCol: Param[String] =
    new Param(this, "predictionCol", "prediction column name", Some("prediction"))
  def getPredictionCol: String = get(predictionCol)
}

private[ml] trait HasRawPredictionCol extends Params {
  /** param for raw prediction column name */
  val rawPredictionCol: Param[String] =
    new Param(this, "rawPredictionCol", "raw prediction (a.k.a. confidence) column name",
      Some("rawPrediction"))
  def getRawPredictionCol: String = get(rawPredictionCol)
}

private[ml] trait HasProbabilityCol extends Params {
  /** param for predicted class conditional probabilities column name */
  val probabilityCol: Param[String] =
    new Param(this, "probabilityCol", "column name for predicted class conditional probabilities",
      Some("probability"))
  def getProbabilityCol: String = get(probabilityCol)
}

private[ml] trait HasThreshold extends Params {
  /** param for threshold in (binary) prediction */
  val threshold: DoubleParam = new DoubleParam(this, "threshold", "threshold in prediction")
  def getThreshold: Double = get(threshold)
}

private[ml] trait HasInputCol extends Params {
  /** param for input column name */
  val inputCol: Param[String] = new Param(this, "inputCol", "input column name")
  def getInputCol: String = get(inputCol)
}

private[ml] trait HasOutputCol extends Params {
  /** param for output column name */
  val outputCol: Param[String] = new Param(this, "outputCol", "output column name")
  def getOutputCol: String = get(outputCol)
}
