package com.example.vrpdrapp;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
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
        Mat grayImg = new Mat(inputImage.height(), inputImage.width(), CvType.CV_8UC1);
        Imgproc.cvtColor(inputImage, grayImg, Imgproc.COLOR_RGB2GRAY);

        Mat thresholdImg = new Mat(inputImage.height(), inputImage.width(), CvType.CV_8UC1);
        Imgproc.threshold(grayImg, thresholdImg, 0,255,Imgproc.THRESH_BINARY_INV+Imgproc.THRESH_OTSU);

        Mat watershedImg = new Mat(inputImage.height(), inputImage.width(), CvType.CV_8UC1);
        markerBasedWatershed(inputImage, thresholdImg, watershedImg);
        watershedImg.convertTo(watershedImg, CvType.CV_8UC1);

        extractContours(watershedImg, thresholdImg);

        List<Mat> chars = extractContours(thresholdImg, null);

        if(finalProcessedImage != null)
            finalProcessedImage.release();

        finalProcessedImage = thresholdImg.clone();

        grayImg.release();
        watershedImg.release();
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

    private Mat skeletonize(Mat inputImage) {
        Mat img = inputImage.clone();
        Mat se = buildStructuringElement(3, Imgproc.CV_SHAPE_CROSS);

        Mat skel = new Mat(img.height(), img.width(), CvType.CV_8UC1, Scalar.all(0));
        Mat skeleton = new Mat(img.height(), img.width(), CvType.CV_8UC1, Scalar.all(0));
        Mat openImg = new Mat(img.height(), img.width(), CvType.CV_8UC1);
        Mat auxImg = new Mat(img.height(), img.width(), CvType.CV_8UC1);
        Mat erodedImg = new Mat(img.height(), img.width(), CvType.CV_8UC1);


        while(true) {
            Imgproc.morphologyEx(img, openImg, Imgproc.MORPH_OPEN, se);
            Core.subtract(img, openImg, auxImg);
            Imgproc.morphologyEx(img, erodedImg, Imgproc.MORPH_ERODE, se);
            Core.bitwise_or(skel, auxImg, skeleton);
            skel = skeleton.clone();
            img = erodedImg.clone();

            if(Core.countNonZero(img) == 0) {
                break;
            }
        }

        img.release();
        se.release();
        skel.release();
        openImg.release();
        auxImg.release();
        erodedImg.release();

        return skeleton;
    }

    private void markerBasedWatershed(Mat image, Mat preMarkerImg, Mat outputImg) {
        Mat skeleton = skeletonize(preMarkerImg);
        Imgproc.connectedComponents(skeleton, outputImg);

        Imgproc.watershed(image, outputImg);
    }

    private List<Mat> extractContours(Mat inputImage, Mat outputMaskedImg) {
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(inputImage, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);

        List<Mat> result = new ArrayList<>();
        float totalArea = inputImage.width() * inputImage.height();

        List<MatOfPoint> contoursUsedForMasking = new ArrayList<>();

        for (MatOfPoint contour : contours) {
            Rect contourBoundingBox = Imgproc.boundingRect(contour);
            float roiArea = (float) contourBoundingBox.area();
            float roiAreaRatio = roiArea / totalArea;

            if(roiAreaRatio >= minContourAreaRatio && roiAreaRatio <= maxContourAreaRatio) {
                contoursUsedForMasking.add(contour);
                Mat digit = new Mat(inputImage, contourBoundingBox);
                result.add(digit);
            }
        }

        if(outputMaskedImg != null) {
            Imgproc.drawContours(outputMaskedImg, contoursUsedForMasking, -1, new Scalar(255, 255, 255), 1);
        }

        return result;
    }
}
