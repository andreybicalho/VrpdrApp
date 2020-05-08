package com.example.vrpdrapp;

import android.content.Context;
import android.content.res.AssetManager;
import android.util.Log;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgproc.Imgproc;
import org.opencv.utils.Converters;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class Yolo {

    private static final String TAG = Yolo.class.getSimpleName();

    private Context context;

    private Net net;

    private List<String> classNames;

    private Size inputImageSize;

    private float confidenceThreshold;

    private float nonMaxSupressThreshold;


    public Yolo(Context context, int width, int height, String classesFilename, String modelArchitectureFilename, String modelWeightsFilename, float confidenceThreshold, float nonMaxSupressThreshold) {
        this.context = context;

        this.inputImageSize = new Size(width, height);

        this.classNames = loadClassesNames(classesFilename);

        loadNet(modelArchitectureFilename, modelWeightsFilename);

        this.confidenceThreshold = confidenceThreshold;

        this.nonMaxSupressThreshold = nonMaxSupressThreshold;
    }

    // Upload file to storage and return a path.
    private String getAssetPath(String file) {
        AssetManager assetManager = context.getAssets();

        BufferedInputStream inputStream = null;
        try {
            // Read data from assets.
            inputStream = new BufferedInputStream(assetManager.open(file));
            byte[] data = new byte[inputStream.available()];
            inputStream.read(data);
            inputStream.close();

            // Create copy file in storage.
            File outFile = new File(context.getFilesDir(), file);
            FileOutputStream os = new FileOutputStream(outFile);
            os.write(data);
            os.close();
            // Return a path to file which may be read in common way.
            return outFile.getAbsolutePath();
        } catch (IOException ex) {
            Log.i(TAG, "Failed to upload a file");
            return "";
        }
    }

    private List<String> getOutputNames(Net net) {
        List<String> names = new ArrayList<>();

        List<Integer> outLayers = net.getUnconnectedOutLayers().toList();
        List<String> layersNames = net.getLayerNames();

        outLayers.forEach((item) -> names.add(layersNames.get(item - 1)));
        return names;
    }

    private List<String> loadClassesNames(String classesFilename) {
        String classesFilenamePath = getAssetPath(classesFilename);

        File file = new File(classesFilenamePath);
        List<String> classes = new ArrayList<>();
        try {
            Scanner sc = new Scanner(file);
            while(sc.hasNextLine()) {
                classes.add(sc.nextLine());
            }
        } catch (FileNotFoundException e) {
            Log.e(TAG, classesFilename+" file not found!");
        }

        return classes;
    }

    private void loadNet(String modelArchitectureFilename, String modelWeightsFilename) {
        Log.i(TAG, "Loading YOLO Net...");

        String modelArchitecture = getAssetPath(modelArchitectureFilename);
        String modelWeights = getAssetPath(modelWeightsFilename);

        net = Dnn.readNetFromDarknet(modelArchitecture, modelWeights);
    }

    private List<Rect> nonMaxSupression(Mat inputImage, List<Mat> netOutputs, boolean drawBoundingBox) {
        List<Integer> clsIds = new ArrayList<>();
        List<Float> confs = new ArrayList<>();
        List<Rect> rects = new ArrayList<>();
        for (int i = 0; i < netOutputs.size(); ++i)
        {
            // each row is a candidate detection, the 1st 4 numbers are
            // [center_x, center_y, width, height], followed by (N-4) class probabilities
            Mat level = netOutputs.get(i);
            for (int j = 0; j < level.rows(); ++j)
            {
                Mat row = level.row(j);
                Mat scores = row.colRange(5, level.cols());
                Core.MinMaxLocResult mm = Core.minMaxLoc(scores);
                float confidence = (float)mm.maxVal;
                Point classIdPoint = mm.maxLoc;
                if (confidence > confidenceThreshold)
                {
                    //Log.i(TAG, "Found one object of class id "+classIdPoint.x+" with a confidence of "+confidence*100+"%.");
                    int centerX = (int)(row.get(0,0)[0] * inputImage.cols());
                    int centerY = (int)(row.get(0,1)[0] * inputImage.rows());
                    int width   = (int)(row.get(0,2)[0] * inputImage.cols());
                    int height  = (int)(row.get(0,3)[0] * inputImage.rows());
                    int left    = centerX - width  / 2;
                    int top     = centerY - height / 2;

                    clsIds.add((int)classIdPoint.x);
                    confs.add((float)confidence);
                    rects.add(new Rect(left, top, width, height));
                }
            }
        }

        if(confs.isEmpty())
            return new ArrayList<>();

        // Apply non-maximum suppression procedure.
        MatOfFloat confidences = new MatOfFloat(Converters.vector_float_to_Mat(confs));
        Rect[] boxesArray = rects.toArray(new Rect[0]);
        MatOfRect boxes = new MatOfRect(boxesArray);
        MatOfInt indices = new MatOfInt();
        Dnn.NMSBoxes(boxes, confidences, confidenceThreshold, nonMaxSupressThreshold, indices);

        int [] ind = indices.toArray();
        List<Rect> boundingBoxes = new ArrayList<>();
        for (int i = 0; i < ind.length; ++i)
        {
            int idx = ind[i];
            float conf = confs.get(idx);
            Rect box = boxesArray[idx];
            boundingBoxes.add(box);
            int classId = clsIds.get(idx);

            // Draw bounding boxes
            if(drawBoundingBox) {
                //Log.i(TAG, "Classe: "+idx);
                String label = String.format("%s [%.0f%%]", classNames.get(classId), 100 * conf);
                drawBoundingBox(inputImage, conf, label, 2.0f, box, 2);
            }
        }

        return boundingBoxes;
    }

    public void drawBoundingBox(Mat inputImage, float confidence, String label, float fontScale, Rect boundingBox, int thickness) {
        if(confidence > 0.8) {
            Scalar color = new Scalar(0, 255, 0);
            Imgproc.rectangle(inputImage, boundingBox.tl(), boundingBox.br(), color, thickness);
            Imgproc.putText(inputImage, label, new Point(boundingBox.x, boundingBox.y), Core.FONT_HERSHEY_PLAIN, fontScale, color, thickness);
        }
        else if (confidence > 0.55 && confidence < 0.8) {
            Scalar color = new Scalar(255, 255, 0);
            Imgproc.rectangle(inputImage, boundingBox.tl(), boundingBox.br(), color, thickness);
            Imgproc.putText(inputImage, label, new Point(boundingBox.x, boundingBox.y), Core.FONT_HERSHEY_PLAIN, fontScale, color, thickness);
        }
        else {
            Scalar color = new Scalar(255, 0, 0);
            Imgproc.rectangle(inputImage, boundingBox.tl(), boundingBox.br(), color, thickness);
            Imgproc.putText(inputImage, label, new Point(boundingBox.x, boundingBox.y), Core.FONT_HERSHEY_PLAIN, 5.0, color, thickness);
        }
    }

    public List<Rect> detect(Mat inputImage, boolean drawBoundingBox) {
        Mat blob = Dnn.blobFromImage(inputImage, 0.00392, inputImageSize, new Scalar(0, 0, 0), false, false);

        net.setInput(blob);

        List<Mat> result = new ArrayList<>();

        List<String> outBlobNames = getOutputNames(net);

        net.forward(result, outBlobNames);

        return nonMaxSupression(inputImage, result, drawBoundingBox);
    }
}
