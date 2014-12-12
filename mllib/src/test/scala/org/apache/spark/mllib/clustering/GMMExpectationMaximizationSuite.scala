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

package org.apache.spark.mllib.clustering

import org.scalatest.FunSuite

import org.apache.spark.mllib.linalg.{Vectors, Matrices}
import org.apache.spark.mllib.util.MLlibTestSparkContext
import org.apache.spark.mllib.util.TestingUtils._

class GMMExpectationMaximizationSuite extends FunSuite with MLlibTestSparkContext {
  test("single cluster") {
    val data = sc.parallelize(Array(
        Vectors.dense(6.0, 9.0),
        Vectors.dense(5.0, 10.0),
        Vectors.dense(4.0, 11.0)
      ))
    
    // expectations
    val Ew = 1.0
    val Emu = Vectors.dense(5.0, 10.0)
    val Esigma = Matrices.dense(2, 2, Array(2.0 / 3.0, -2.0 / 3.0, -2.0 / 3.0, 2.0 / 3.0))
    
    val gmm = new GaussianMixtureModelEM().setK(1).run(data)
                
    assert(gmm.weight(0) ~== Ew absTol 1E-5)
    assert(gmm.mu(0) ~== Emu absTol 1E-5)
    assert(gmm.sigma(0) ~== Esigma absTol 1E-5)
  }
}