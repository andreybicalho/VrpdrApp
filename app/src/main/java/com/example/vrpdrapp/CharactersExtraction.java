package com.example.vrpdrapp;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

public class CharactersExtraction {

    private Mat finalProcessedImage;

    private float minContourAreaRatio = 0.02f;
    private float maxContourAreaRatio = 0.1f;

    public CharactersExtraction(float minContourAreaRatio, float maxContourAreaRatio) {
        this.minContourAreaRatio = minContourAreaRatio;
        this.maxContourAreaRatio = maxContourAreaRatio;
    }

    public List<Mat> extract(Mat inputImage) {
        // initialize images
        Mat grayImg = new Mat(inputImage.height(), inputImage.width(), CvType.CV_8UC1);
        Mat blurImg = new Mat(inputImage.height(), inputImage.width(), CvType.CV_8UC1);
        Mat morphImg = new Mat(inputImage.height(), inputImage.width(), CvType.CV_8UC1);
        Mat thresholdImg = new Mat(inputImage.height(), inputImage.width(), CvType.CV_8UC1);

        // to gray
        Imgproc.cvtColor(inputImage, grayImg, Imgproc.COLOR_RGB2GRAY);

        Mat se = buildStructuringElement(3, Imgproc.CV_SHAPE_RECT);
        Imgproc.morphologyEx(grayImg, morphImg, Imgproc.MORPH_CLOSE, se);

        // Otsu's thresholding after Gaussian filtering
        Imgproc.GaussianBlur(morphImg, blurImg, new Size(5,5), 0);
        Imgproc.threshold(blurImg, thresholdImg, 0,255,Imgproc.THRESH_BINARY_INV+Imgproc.THRESH_OTSU);

        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(thresholdImg, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);

        List<Mat> chars = new ArrayList<>();
        float totalArea = inputImage.width() * inputImage.height();

        for (MatOfPoint contour : contours) {
            Rect contourBoundingBox = Imgproc.boundingRect(contour);
            float roiArea = (float) contourBoundingBox.area();
            float roiAreaRatio = roiArea / totalArea;

            if(roiAreaRatio >= minContourAreaRatio && roiAreaRatio <= maxContourAreaRatio) {
                Mat digit = new Mat(thresholdImg, contourBoundingBox);
                chars.add(digit);
            }
        }

        if(finalProcessedImage != null)
            finalProcessedImage.release();

        finalProcessedImage = thresholdImg.clone();

        grayImg.release();
        morphImg.release();
        blurImg.release();
        thresholdImg.release();

        return chars;
    }

    public List<Mat> extract1(Mat inputImage) {
        // initialize images
        Mat grayImg = new Mat(inputImage.height(), inputImage.width(), CvType.CV_8UC1);
        Mat thresholdImg = new Mat(inputImage.height(), inputImage.width(), CvType.CV_8UC1);

        // to gray
        Imgproc.cvtColor(inputImage, grayImg, Imgproc.COLOR_RGB2GRAY);

        // Otsu's thresholding
        Imgproc.threshold(grayImg, thresholdImg, 0,255,Imgproc.THRESH_BINARY_INV+Imgproc.THRESH_OTSU);

        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(thresholdImg, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);

        List<Mat> chars = new ArrayList<>();
        float totalArea = inputImage.width() * inputImage.height();

        for (MatOfPoint contour : contours) {
            Rect contourBoundingBox = Imgproc.boundingRect(contour);
            float roiArea = (float) contourBoundingBox.area();
            float roiAreaRatio = roiArea / totalArea;

            if(roiAreaRatio >= minContourAreaRatio && roiAreaRatio <= maxContourAreaRatio) {
                Mat digit = new Mat(thresholdImg, contourBoundingBox);
                chars.add(digit);
            }
        }

        if(finalProcessedImage != null)
            finalProcessedImage.release();

        finalProcessedImage = thresholdImg.clone();

        grayImg.release();
        thresholdImg.release();

        return chars;
    }

    private Mat buildStructuringElement(int kernelSize, int elementType) {
        Mat element = Imgproc.getStructuringElement(elementType,
                new Size(2 * kernelSize + 1, 2 * kernelSize + 1),
                new Point(kernelSize, kernelSize));

        return element;
    }

    public Mat getFinalProcessedImage() {
        return this.finalProcessedImage;
    }

    // NOTE: testing
    public List<Mat> test(Mat inputImage, int kernelSize) {
        Mat grayImg = new Mat(inputImage.height(), inputImage.width(), CvType.CV_8UC1);

        if(finalProcessedImage != null)
            finalProcessedImage.release();

        finalProcessedImage = new Mat(inputImage.height(), inputImage.width(), CvType.CV_8UC1);

        Imgproc.cvtColor(inputImage, grayImg, Imgproc.COLOR_RGB2GRAY);


        Mat se = buildStructuringElement(kernelSize, Imgproc.CV_SHAPE_RECT);
        Imgproc.morphologyEx(grayImg, finalProcessedImage, Imgproc.MORPH_CLOSE, se);

        grayImg.release();

        return new ArrayList<>();
    }
}
