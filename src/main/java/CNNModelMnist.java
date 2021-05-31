import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.Random;

public class CNNModelMnist {
    public static void main(String[] args) throws Exception {
        long seed = 1234;
        double learningRate = 0.01;
        long height = 28;
        long width = 28;
        long depth = 1;
        int outputSize = 10;
        MultiLayerConfiguration multiLayerConfiguration = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .updater(new Sgd(learningRate))
                .list()
                .setInputType(InputType.convolutionalFlat(height, width, depth ))
                .layer(0, new ConvolutionLayer.Builder()
                        .nIn(depth)
                        .nOut(20)
                        .activation(Activation.RELU)
                        .kernelSize(5,5)
                        .stride(1, 1)
                        .build())
                .layer(1 ,new SubsamplingLayer.Builder()
                        .kernelSize(2, 2)
                        .stride(1, 1)
                        .poolingType(SubsamplingLayer.PoolingType.MAX)
                        .build())
                .layer(2, new ConvolutionLayer.Builder()
                        .nOut(50)
                        .activation(Activation.RELU)
                        .kernelSize(5, 5)
                        .stride(1, 1)
                        .build())
                .layer(3 ,new SubsamplingLayer.Builder()
                        .kernelSize(2, 2)
                        .stride(1, 1)
                        .poolingType(SubsamplingLayer.PoolingType.MAX)
                        .build())
                .layer(4, new DenseLayer.Builder()
                        .nOut(500)
                        .activation(Activation.RELU)
                        .build())
                .layer(5, new OutputLayer.Builder()
                        .nIn(500)
                        .nOut(outputSize)
                        .activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .build())
                .build();
        //hSystem.out.println(multiLayerConfiguration.toJson());

        MultiLayerNetwork model = new MultiLayerNetwork(multiLayerConfiguration);
        model.init();
        System.out.println("-------------- Training --------------------");
        String path = System.getProperty("user.home")+"\\Desktop\\mnist";
        File fileTrain = new File(path+"\\training");
        FileSplit fileSpliTrain = new FileSplit(fileTrain, NativeImageLoader.ALLOWED_FORMATS, new Random(seed));
        RecordReader recordReaderTrain = new ImageRecordReader(height, width, depth, new ParentPathLabelGenerator());
        recordReaderTrain.initialize(fileSpliTrain);
        int batchSize = 54;
        DataSetIterator dataSetIteratorTrain = new RecordReaderDataSetIterator(recordReaderTrain, batchSize, 1, outputSize);
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        dataSetIteratorTrain.setPreProcessor(scaler);
        int numEpoch = 1;

        UIServer uiServer = UIServer.getInstance();
        StatsStorage inMemoryStatsStorage = new InMemoryStatsStorage();
        uiServer.attach(inMemoryStatsStorage);
        model.setListeners(new StatsListener(inMemoryStatsStorage));

        for (int i = 0; i<numEpoch; i++)
        {
            model.fit(dataSetIteratorTrain);
        }
        System.out.println("-------------- Model is trained --------------------");
        System.out.println("-------------- Evaluation --------------------");
        File fileTest = new File(path+"\\testing");
        FileSplit fileSpliTest = new FileSplit(fileTrain, NativeImageLoader.ALLOWED_FORMATS);
        RecordReader recordReaderTest = new ImageRecordReader(height, width, depth, new ParentPathLabelGenerator());
        recordReaderTest.initialize(fileSpliTest);
        DataSetIterator dataSetIteratorTest = new RecordReaderDataSetIterator(recordReaderTest, batchSize, 1, outputSize);
        dataSetIteratorTest.setPreProcessor(scaler);

        Evaluation evaluation = new Evaluation();
        while (dataSetIteratorTest.hasNext())
        {
            DataSet dataSet = dataSetIteratorTest.next();
            INDArray features = dataSet.getFeatures();
            INDArray labels = dataSet.getLabels();

            INDArray predicted = model.output(features);
            evaluation.eval(predicted, labels);
        }
        System.out.println(evaluation.stats());
       /* while (dataSetIteratorTrain.hasNext())
        {
            DataSet dataSet = dataSetIteratorTrain.next();
            INDArray features = dataSet.getFeatures();
            INDArray labels = dataSet.getLabels();
            System.out.println(features.shapeInfoToString());
            System.out.println(labels.shapeInfoToString());
            System.out.println("------------------------");

        }*/

    }
}
