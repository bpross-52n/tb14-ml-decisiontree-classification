package org.n52.project.testbed14

import astraea.spark.rasterframes._
import astraea.spark.rasterframes.ml.{ NoDataFilter, TileExploder }
import geotrellis.raster._
import geotrellis.raster.render._
import geotrellis.raster.io.geotiff.SinglebandGeoTiff
import astraea.spark.rasterframes.datasource.geotiff._
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
  spark.sparkContext.setLogLevel("INFO")

  val logger = LoggerFactory.getLogger(classOf[DecisiontreeClassification])

  import spark.implicits._

  // Utility for reading imagery from our test data set
  def readTiff(name: String): SinglebandGeoTiff = SinglebandGeoTiff(s"D:/52n/Projekte/Testbed-14/ML/$name")

  def run(args: Array[String]) {

//    val filenamePattern = "L8-%s-Elkton-VA.tiff"
    val bandNumbers = 2 to 7
    val bandColNames = bandNumbers.map(b ⇒ s"band_$b").toArray
    val tileSize = 10
    
    logger.debug("Array length: " + args.length)
    
    var tiffName = "IMG_PHR1B_P_201509271105571_ORT_1974032101-001_R1C1_subset2_downsized3.tif"

    var labelName = "labels4.tif"
    
    if(args.length > 0){
      tiffName = args(0)
    }
    
    if(args.length > 1){
    	labelName = args(1)
    }

    // For each identified band, load the associated image file
//    val joinedRF = bandNumbers.
//      map { b ⇒ (b, filenamePattern.format("B" + b)) }.
//      map { case (b, f) ⇒ (b, readTiff(f)) }.
//      map { case (b, t) ⇒ t.projectedRaster.toRF(tileSize, tileSize, s"band_$b") }.
//      reduce(_ spatialJoin _)
//
//    joinedRF.printSchema()

//    val tiffFilePath = new File("D:/52n/Projekte/Testbed-14/ML/IMG_PHR1B_P_201509271105571_ORT_1974032101-001_R1C1.tif")
//    
//    val tiffRF = spark.read.
//      geotiff.
//      loadRF(tiffFilePath.toURI)

    val tiffRF = readTiff(tiffName).
      projectedRaster.
      toRF(tileSize, tileSize, "band_2")
      
    tiffRF.printSchema()
      
//    val labelFilePath = new File("D:/52n/Projekte/Testbed-14/ML/labels.tif")
//          
//    val labelRF = spark.read.
//      geotiff.
//      loadRF(labelFilePath.toURI)

    val targetCol = "target"
    
    val target = readTiff(labelName).
      mapTile(_.convert(DoubleConstantNoDataCellType)).
      projectedRaster.
      toRF(tileSize, tileSize, targetCol)

    target.select(aggStats(target(targetCol))).show

    val abt = tiffRF.spatialJoin(target)

    val exploder = new TileExploder()

    val noDataFilter = new NoDataFilter().
//      setInputCols(1)
      setInputCols(bandColNames :+ targetCol)

    val assembler = new VectorAssembler().
      setInputCols(Array("band_2")).
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

    val scored = model.bestModel.transform(tiffRF)

    scored.groupBy($"prediction" as "class").count().show

    val tlm = tiffRF.tileLayerMetadata.left.get

    val retiled = scored.groupBy($"spatial_key").agg(
      assembleTile(
        $"column_index", $"row_index", $"prediction",
        tlm.tileCols, tlm.tileRows, ByteConstantNoDataCellType))

    val rf = retiled.asRF($"spatial_key", tlm)

    val raster = rf.toRaster($"prediction", 250, 331)

    val clusterColors = IndexedColorMap.fromColorMap(
      ColorRamps.Viridis.toColorMap((0 until 3).toArray))

    raster.tile.renderPng(clusterColors).write("d:/tmp/classified23536848355.png")
  }
}