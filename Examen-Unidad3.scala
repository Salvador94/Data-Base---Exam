// 1- Import un sesion spark
import org.apache.spark.sql.SparkSession

// 7- Import VectorAssembler and Vector
import org.apache.spark.ml.feature.{VectorAssembler,StringIndexer,VectorIndexer,OneHotEncoder}
import org.apache.spark.ml.linalg.Vectors

// 4- Importar la libreria de Kmeans para el algoritmo de agrupacion
import org.apache.spark.ml.clustering.KMeans

// 2- Utilice el siguiente codigo para reportar errores reducidos
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

// 3- Cree un instancia de la sesion Spark
val spark = SparkSession.builder().getOrCreate()

// 5- Cargar el dataset de Wholesale Customers Data
val dataset = spark.read.option("header","true").option("inferSchema","true").csv("Wholesale customers data.csv")

// 6- Seleccionar la siguiente columna para conjuntos de entrenamiento
val feature_data = dataset.select($"Fresh", $"Milk", $"Grocery", $"Frozen", $"Detergents_Paper", $"Delicassen")

// 8- Crea un nuevo objetivo VectorAssembler para las columnas de caracteristicas como un conjunto de entradas, recordando que no hay etiquetas
val assembler = new VectorAssembler().setInputCols(Array("Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen")).setOutputCol("features")

// 9- Utilice el objetivo assembler para transformar feature_data
val training_data = assembler.transform(feature_data).select()

// 10- Crea un modelo Kmeans con K=3
val model = kmeans.fit(training_data)
val kmeans = new KMeans().setK(3).setSeed(1L)

// 11- Evaluar los grupos utilizado WSSSE (Within set sum of squared errors)
val WSSE = model.computeCost(dataset)
println(s"Within set sum of Squared Errors = $WSSE")

// 12- Mostrar los resultados
println("Cluster Centers: ")
model.clusterCenters.foreach(println)
