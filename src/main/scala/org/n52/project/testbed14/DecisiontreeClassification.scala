package org.n52.project.testbed14

import astraea.spark.rasterframes._
import astraea.spark.rasterframes.ml.{ NoDataFilter, TileExploder }
import geotrellis.raster._
import geotrellis.raster.render._
import geotrellis.raster.io.geotiff.SinglebandGeoTiff
import org.apache.spark.sql._
import java.io.File
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.ml.tuning.CrossValidator
import org.slf4j.LoggerFactory

class DecisiontreeClassification {

  implicit val spark = SparkSession.builder().
    master("local[*]").appName(getClass.getName).getOrCreate().withRasterFrames
  spark.sparkContext.setLogLevel("ERROR")

  val logger = LoggerFactory.getLogger(classOf[DecisiontreeClassification])
  
  import spark.implicits._

  // Utility for reading imagery from our test data set
  def readTiff(name: String): SinglebandGeoTiff = SinglebandGeoTiff(s"d:/tmp/resources/$name")

  def run(args: Array[String]) {

    val filenamePattern = "L8-%s-Elkton-VA.tiff"
    val bandNumbers = 2 to 7
    val bandColNames = bandNumbers.map(b ⇒ s"band_$b").toArray
    val tileSize = 10

    logger.info("Array length: " + args.length)
    
    // For each identified band, load the associated image file
    val joinedRF = bandNumbers.
      map { b ⇒ (b, filenamePattern.format("B" + b)) }.
      map { case (b, f) ⇒ (b, readTiff(f)) }.
      map { case (b, t) ⇒ t.projectedRaster.toRF(tileSize, tileSize, s"band_$b") }.
      reduce(_ spatialJoin _)

    joinedRF.printSchema()

    val targetCol = "target"

    val target = readTiff(filenamePattern.format("Labels")).
      mapTile(_.convert(DoubleConstantNoDataCellType)).
      projectedRaster.
      toRF(tileSize, tileSize, targetCol)

    target.select(aggStats(target(targetCol))).show

    val abt = joinedRF.spatialJoin(target)

    val exploder = new TileExploder()

    val noDataFilter = new NoDataFilter().
      setInputCols(bandColNames :+ targetCol)

    val assembler = new VectorAssembler().
      setInputCols(bandColNames).
      setOutputCol("features")

    val classifier = new DecisionTreeClassifier().
      setLabelCol(targetCol).
      setFeaturesCol(assembler.getOutputCol)

    val pipeline = new Pipeline().
      setStages(Array(exploder, noDataFilter, assembler, classifier))

    val evaluator = new MulticlassClassificationEvaluator().
      setLabelCol(targetCol).
      setPredictionCol("prediction").
      setMetricName("accuracy")

    val paramGrid = new ParamGridBuilder().
      addGrid(classifier.maxDepth, Array(2, 3, 4)).
      build()

    val trainer = new CrossValidator().
      setEstimator(pipeline).
      setEvaluator(evaluator).
      setEstimatorParamMaps(paramGrid).
      setNumFolds(4)

    val model = trainer.fit(abt)

    val metrics = model.getEstimatorParamMaps.
      map(_.toSeq.map(p ⇒ s"${p.param.name} = ${p.value}")).
      map(_.mkString(", ")).
      zip(model.avgMetrics)

    metrics.toSeq.toDF("params", "metric").show(false)

    val scored = model.bestModel.transform(joinedRF)

    scored.groupBy($"prediction" as "class").count().show

    val tlm = joinedRF.tileLayerMetadata.left.get

    val retiled = scored.groupBy($"spatial_key").agg(
      assembleTile(
        $"column_index", $"row_index", $"prediction",
        tlm.tileCols, tlm.tileRows, ByteConstantNoDataCellType))

    val rf = retiled.asRF($"spatial_key", tlm)

    val raster = rf.toRaster($"prediction", 186, 169)

    val clusterColors = IndexedColorMap.fromColorMap(
      ColorRamps.Viridis.toColorMap((0 until 3).toArray))

    raster.tile.renderPng(clusterColors).write("d:/tmp/classified.png")
  }
}