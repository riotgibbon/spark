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

import breeze.linalg.{DenseVector => BreezeVector, DenseMatrix => BreezeMatrix}
import breeze.linalg.{Transpose, det, inv}
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.{Matrices, Vector, Vectors}
import org.apache.spark.{Accumulator, AccumulatorParam, SparkContext}
import org.apache.spark.SparkContext.DoubleAccumulatorParam

/**
 * Expectation-Maximization for multivariate Gaussian Mixture Models.
 * 
 */
object GMMExpectationMaximization {
  /**
   * Trains a GMM using the given parameters
   * 
   * @param data training points stored as RDD[Vector]
   * @param k the number of Gaussians in the mixture
   * @param maxIterations the maximum number of iterations to perform
   * @param delta change in log-likelihood at which convergence is considered achieved
   */
  def train(data: RDD[Vector], k: Int, maxIterations: Int, delta: Double): GaussianMixtureModel = {
    new GMMExpectationMaximization().setK(k)
      .setMaxIterations(maxIterations)
      .setDelta(delta)
      .run(data)
  }
  
  /**
   * Trains a GMM using the given parameters
   * 
   * @param data training points stored as RDD[Vector]
   * @param k the number of Gaussians in the mixture
   * @param maxIterations the maximum number of iterations to perform
   */
  def train(data: RDD[Vector], k: Int, maxIterations: Int): GaussianMixtureModel = {
    new GMMExpectationMaximization().setK(k).setMaxIterations(maxIterations).run(data)
  }
  
  /**
   * Trains a GMM using the given parameters
   * 
   * @param data training points stored as RDD[Vector]
   * @param k the number of Gaussians in the mixture
   * @param delta change in log-likelihood at which convergence is considered achieved
   */
  def train(data: RDD[Vector], k: Int, delta: Double): GaussianMixtureModel = {
    new GMMExpectationMaximization().setK(k).setDelta(delta).run(data)
  }
  
  /**
   * Trains a GMM using the given parameters
   * 
   * @param data training points stored as RDD[Vector]
   * @param k the number of Gaussians in the mixture
   */
  def train(data: RDD[Vector], k: Int): GaussianMixtureModel = {
    new GMMExpectationMaximization().setK(k).run(data)
  }
}

/**
 * This class performs multivariate Gaussian expectation maximization.  It will 
 * maximize the log-likelihood for a mixture of k Gaussians, iterating until
 * the log-likelihood changes by less than delta, or until it has reached
 * the max number of iterations.  
 */
class GMMExpectationMaximization private (
    private var k: Int, 
    private var delta: Double, 
    private var maxIterations: Int) extends Serializable {
      
  // Type aliases for convenience
  private type DenseDoubleVector = BreezeVector[Double]
  private type DenseDoubleMatrix = BreezeMatrix[Double]
  
  // number of samples per cluster to use when initializing Gaussians
  private val nSamples = 5;
  
  // A default instance, 2 Gaussians, 100 iterations, 0.01 log-likelihood threshold
  def this() = this(2, 0.01, 100)
  
  /** Set the number of Gaussians in the mixture model.  Default: 2 */
  def setK(k: Int): this.type = {
    this.k = k
    this
  }
  
  /** Set the maximum number of iterations to run. Default: 100 */
  def setMaxIterations(maxIterations: Int): this.type = {
    this.maxIterations = maxIterations
    this
  }
  
  /**
   * Set the largest change in log-likelihood at which convergence is 
   * considered to have occurred.
   */
  def setDelta(delta: Double): this.type = {
    this.delta = delta
    this
  }
  
  /** Machine precision value used to ensure matrix conditioning */
  private val eps = math.pow(2.0, -52)
  
  /** Perform expectation maximization */
  def run(data: RDD[Vector]): GaussianMixtureModel = {
    val ctx = data.sparkContext
    
    // we will operate on the data as breeze data
    val breezeData = data.map{ u => u.toBreeze.toDenseVector }.cache()
    
    // Get length of the input vectors
    val d = breezeData.first.length 
    
    // For each Gaussian, we will initialize the mean as the average
    // of some random samples from the data
    val samples = breezeData.takeSample(true, k * nSamples, scala.util.Random.nextInt)
    
    // C will be array of (weight, mean, covariance) tuples
    // we start with uniform weights, a random mean from the data, and
    // diagonal covariance matrices using component variances
    // derived from the samples 
    var C = (0 until k).map(i => (1.0/k, 
                                  vec_mean(samples.slice(i * nSamples, (i + 1) * nSamples)), 
                                  init_cov(samples.slice(i * nSamples, (i + 1) * nSamples)))
                           ).toArray
    
    val acc_w     = new Array[Accumulator[Double]](k)
    val acc_mu    = new Array[Accumulator[DenseDoubleVector]](k)
    val acc_sigma = new Array[Accumulator[DenseDoubleMatrix]](k)
    
    var llh = Double.MinValue // current log-likelihood 
    var llhp = 0.0            // previous log-likelihood
    
    var i, iter = 0
    do {
      // reset accumulators
      for(i <- 0 until k){
        acc_w(i)     = ctx.accumulator(0.0)
        acc_mu(i)    = ctx.accumulator(
                      BreezeVector.zeros[Double](d))(DenseDoubleVectorAccumulatorParam)
        acc_sigma(i) = ctx.accumulator(
                      BreezeMatrix.zeros[Double](d,d))(DenseDoubleMatrixAccumulatorParam)
      }
      
      val log_likelihood = ctx.accumulator(0.0)
            
      // broadcast the current weights and distributions to all nodes
      val dists = ctx.broadcast((0 until k).map(i => 
                                  new MultivariateGaussian(C(i)._2, C(i)._3)).toArray)
      val weights = ctx.broadcast((0 until k).map(i => C(i)._1).toArray)
      
      // calculate partial assignments for each sample in the data
      // (often referred to as the "E" step in literature)
      breezeData.foreach(x => {  
        val p = (0 until k).map(i => 
          eps + weights.value(i) * dists.value(i).pdf(x)).toArray
        val norm = sum(p)
        
        log_likelihood += math.log(norm)  
          
        // accumulate weighted sums  
        val xxt = x * new Transpose(x)
        for(i <- 0 until k){
          p(i) /= norm
          acc_w(i) += p(i)
          acc_mu(i) += x * p(i)
          acc_sigma(i) += xxt * p(i)
        }  
      })
      
      // Collect the computed sums
      val W = (0 until k).map(i => acc_w(i).value).toArray
      val MU = (0 until k).map(i => acc_mu(i).value).toArray
      val SIGMA = (0 until k).map(i => acc_sigma(i).value).toArray
      
      // Create new distributions based on the partial assignments
      // (often referred to as the "M" step in literature)
      C = (0 until k).map(i => {
            val weight = W(i) / sum(W)
            val mu = MU(i) / W(i)
            val sigma = SIGMA(i) / W(i) - mu * new Transpose(mu)
            (weight, mu, sigma)
          }).toArray
      
      llhp = llh; // current becomes previous
      llh = log_likelihood.value // this is the freshly computed log-likelihood
      iter += 1
    } while(iter < maxIterations && Math.abs(llh-llhp) > delta)
    
    // Need to convert the breeze matrices to MLlib matrices
    val weights = (0 until k).map(i => C(i)._1).toArray
    val means   = (0 until k).map(i => Vectors.fromBreeze(C(i)._2)).toArray
    val sigmas  = (0 until k).map(i => Matrices.fromBreeze(C(i)._3)).toArray
    new GaussianMixtureModel(weights, means, sigmas)
  }
  
  /** Sum the values in array of doubles */
  private def sum(x : Array[Double]) : Double = {
    var s : Double = 0.0
    (0 until x.length).foreach(j => s += x(j))
    s
  }
  
  /** Average of dense breeze vectors */
  private def vec_mean(x : Array[DenseDoubleVector]) : DenseDoubleVector = {
    val v = BreezeVector.zeros[Double](x(0).length)
    (0 until x.length).foreach(j => v += x(j))
    v / x.length.asInstanceOf[Double] 
  }
  
  /**
   * Construct matrix where diagonal entries are element-wise
   * variance of input vectors (computes biased variance)
   */
  private def init_cov(x : Array[DenseDoubleVector]) : DenseDoubleMatrix = {
    val mu = vec_mean(x)
    val ss = BreezeVector.zeros[Double](x(0).length)
    val result = BreezeMatrix.eye[Double](ss.length)
    (0 until x.length).map(i => (x(i) - mu) :^ 2.0).foreach(u => ss += u)
    (0 until ss.length).foreach(i => result(i,i) = ss(i) / x.length)
    result
  }
  
  /** AccumulatorParam for Dense Breeze Vectors */
  private object DenseDoubleVectorAccumulatorParam extends AccumulatorParam[DenseDoubleVector] {
    def zero(initialVector : DenseDoubleVector) : DenseDoubleVector = {
      BreezeVector.zeros[Double](initialVector.length)
    }
    
    def addInPlace(a : DenseDoubleVector, b : DenseDoubleVector) : DenseDoubleVector = {
      a += b
    }
  }
  
  /** AccumulatorParam for Dense Breeze Matrices */
  private object DenseDoubleMatrixAccumulatorParam extends AccumulatorParam[DenseDoubleMatrix] {
    def zero(initialVector : DenseDoubleMatrix) : DenseDoubleMatrix = {
      BreezeMatrix.zeros[Double](initialVector.rows, initialVector.cols)
    }
    
    def addInPlace(a : DenseDoubleMatrix, b : DenseDoubleMatrix) : DenseDoubleMatrix = {
      a += b
    }
  }  
  
  /** 
   * Utility class to implement the density function for multivariate Gaussian distribution.
   * Breeze provides this functionality, but it requires the Apache Commons Math library,
   * so this class is here so-as to not introduce a new dependency in Spark.
   */
  private class MultivariateGaussian(val mu : DenseDoubleVector, val sigma : DenseDoubleMatrix) 
      extends Serializable {
    private val sigma_inv_2 = inv(sigma) * -0.5
    private val U = math.pow(2.0*math.Pi, -mu.length/2.0) * math.pow(det(sigma), -0.5)
    
    def pdf(x : DenseDoubleVector) : Double = {
      val delta = x - mu
      val delta_t = new Transpose(delta)
      U * math.exp(delta_t * sigma_inv_2 * delta)
    }
  }
}
