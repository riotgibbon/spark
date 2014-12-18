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

package org.apache.spark.examples.mllib

import org.apache.spark.mllib.feature.{IDF, HashingTF}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.clustering.GaussianMixtureModelEM
import org.apache.spark.mllib.random.UniformGenerator

object DenseGmmEMTiming {
  def main(args: Array[String]): Unit = {
    if (args.length != 1 && args.length != 2) {
      println("usage: DenseGmmEMTiming <input file> (<instanceLimit>)")
    } else {
      val instanceLimit = if (args.length == 1) {
        None
      } else {
        Some(args(1).toInt)
      }
      run(args(0), instanceLimit)
    }
  }

  private def runTest(data: RDD[Seq[String]], k: Int, numFeatures: Int): Unit = {
    val convergenceTol = 0.01

    val tfFeaturesRDD = new HashingTF(numFeatures).transform(data)
    val idfModel = new IDF(minDocFreq = 0).fit(tfFeaturesRDD)
    val featuresRDD = idfModel.transform(tfFeaturesRDD)
      .mapPartitionsWithIndex { case (partitionIndex, iterator) =>
      // Add random noise since GMM is fragile.
      val rng = new UniformGenerator()
      rng.setSeed(partitionIndex + 1)
      iterator.map { features => Vectors.dense(features.toArray.map(_ + rng.nextValue())) }
    }.cache()
    val numInstances = featuresRDD.count()

    val startTime = System.nanoTime()
    val clusters = new GaussianMixtureModelEM()
      .setK(k)
      .setConvergenceTol(convergenceTol)
      .run(featuresRDD)
    val elapsedTime = (System.nanoTime() - startTime) / 1e9
    val kmeansCost = clusters.computeCost(featuresRDD) / numInstances.toDouble
    println(s"$numInstances\t$k\t$numFeatures\t$elapsedTime\t$kmeansCost")
  }

  def run(inputFile: String, instanceLimit: Option[Int]) {
    val conf = new SparkConf().setAppName("Spark EM Sample")
    val ctx  = new SparkContext(conf)
    
    val origWordsRDD = ctx.textFile(inputFile).map { line =>
      line.trim.split(' ').toSeq
    }
    val origWordsRDDcount = origWordsRDD.count()
    val wordsRDD = if (instanceLimit.isEmpty || origWordsRDDcount <= instanceLimit.get) {
      origWordsRDD
    } else {
      origWordsRDD.sample(withReplacement = true,
        fraction = instanceLimit.get / origWordsRDDcount.toDouble)
    }
    wordsRDD.cache()
    val ks = Array(2, 4, 16, 64, 256)
    val numFeaturess = Array(10, 100)
    println("\nnumInstances\tk\tnumFeatures\ttime(sec)\tavg kmeansCost")
    for (k <- ks) {
      for (numFeatures <- numFeaturess) {
        runTest(wordsRDD, k, numFeatures)
      }
    }
    println()
  }
}
