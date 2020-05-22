package com.example.vrpdrapp;

import android.content.Context;
import android.util.Log;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

public class EMNISTNet {

    public static String[] CLASS_LABELS = new String[] {
            "0","1","2","3","4","5","6","7","8","9",
            "A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z",
            "a","b","d","e","f","g","h","n","q","r","t" };

    private static final String TAG = EMNISTNet.class.getSimpleName();

    private Module module = null;

    public EMNISTNet(Context context, String moduleName) {
        Log.i(TAG, "Loading EMNISTNet... ");

        try {
            this.module = Module.load(getAssetFilePath(context, moduleName));
        } catch (IOException e) {
            Log.e(TAG, "Error reading asset: Failed to load pytorch module from "+moduleName);
        }
    }

    public String predict(Mat inputImage) {
        Mat image = new Mat(28, 28, CvType.CV_8UC1);
        Imgproc.resize(inputImage, image, new Size(28, 28)); // resize image to 28x28

        image.convertTo(image, CvType.CV_32FC1, 1.0/255.0); // convert to type float32 with 1 single channel

        long[] shape = new long[4]; // expected input shape: 1x1x28x28
        shape[0] = 1;
        shape[1] = 1;
        shape[2] = image.width();
        shape[3] = image.height();

        float[] imageData = matToFloatArray(image);

        final Tensor inputTensor = Tensor.fromBlob(imageData, shape);

        final Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();

        final float[] scores = outputTensor.getDataAsFloatArray();

        int maxScoreIdx = findMaxScoreIdx(scores);

        String className = CLASS_LABELS[maxScoreIdx];

        return  className;
    }

    private float[] matToFloatArray(Mat image) {
        float[] data = new float[(int)(image.total()*image.channels())];
        image.get(0,0, data);

        return data;
    }

    private int findMaxScoreIdx(float[] scores) {
        float maxScore = -Float.MAX_VALUE;
        int maxScoreIdx = -1;
        for (int i = 0; i < scores.length; i++) {
            if (scores[i] > maxScore) {
                maxScore = scores[i];
                maxScoreIdx = i;
            }
        }

        return maxScoreIdx;
    }

    /**
     * Copies specified asset to the file in /files app directory and returns this file absolute path.
     *
     * @return absolute file path
     */
    public static String getAssetFilePath(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }
}
