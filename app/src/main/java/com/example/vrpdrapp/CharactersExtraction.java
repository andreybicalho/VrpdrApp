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

    private static final String TAG = CharactersExtraction.class.getSimpleName();

    private Mat finalProcessedImage;

    private Mat kernelDefault;

    private float minContourAreaRatio = 0.02f;
    private float maxContourAreaRatio = 0.1f;

    public CharactersExtraction(float minContourAreaRatio, float maxContourAreaRatio) {
        this.minContourAreaRatio = minContourAreaRatio;
        this.maxContourAreaRatio = maxContourAreaRatio;

        kernelDefault = buildStructuringElement(3, Imgproc.CV_SHAPE_CROSS);
    }

    public List<Mat> extract(Mat inputImage) {
        Mat grayImg = new Mat(inputImage.height(), inputImage.width(), CvType.CV_8UC1);
        Imgproc.cvtColor(inputImage, grayImg, Imgproc.COLOR_RGB2GRAY);

        Mat thresholdImg = new Mat(inputImage.height(), inputImage.width(), CvType.CV_8UC1);
        Imgproc.threshold(grayImg, thresholdImg, 0,255,Imgproc.THRESH_BINARY_INV+Imgproc.THRESH_OTSU);

        Mat watershedImg = new Mat(inputImage.height(), inputImage.width(), CvType.CV_8UC1);
        //skeletonMarkerBasedWatershed(inputImage, thresholdImg, watershedImg);
        intersectionLinesMarkerBasedWatershedSegmentation(inputImage, thresholdImg, watershedImg);

        Mat maskedImg = new Mat(inputImage.height(), inputImage.width(), CvType.CV_8UC1, Scalar.all(0));
        extractBoxedContours(watershedImg, maskedImg);
        Core.bitwise_and(maskedImg, thresholdImg, maskedImg);
        List<Mat> chars = extractBoxedContours(maskedImg, maskedImg);
        Core.bitwise_and(maskedImg, thresholdImg, maskedImg);

        if(finalProcessedImage != null)
            finalProcessedImage.release();

        finalProcessedImage = maskedImg.clone();

        grayImg.release();
        watershedImg.release();
        thresholdImg.release();
        maskedImg.release();

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
        Mat erodedImg = inputImage.clone();

        Mat skeleton = new Mat(inputImage.height(), inputImage.width(), CvType.CV_8UC1, Scalar.all(0));
        Mat openImg = new Mat(inputImage.height(), inputImage.width(), CvType.CV_8UC1);
        Mat auxImg = new Mat(inputImage.height(), inputImage.width(), CvType.CV_8UC1);

        while(true) {
            Imgproc.morphologyEx(erodedImg, openImg, Imgproc.MORPH_OPEN, kernelDefault);
            Core.subtract(erodedImg, openImg, auxImg);
            Imgproc.morphologyEx(erodedImg, erodedImg, Imgproc.MORPH_ERODE, kernelDefault);
            Core.bitwise_or(skeleton, auxImg, skeleton);

            if(Core.countNonZero(erodedImg) == 0) {
                break;
            }
        }

        erodedImg.release();
        openImg.release();
        auxImg.release();

        return skeleton;
    }

    private void skeletonMarkerBasedWatershed(Mat image, Mat preMarkerImg, Mat outputImg) {
        Mat skeleton = skeletonize(preMarkerImg);
        Imgproc.connectedComponents(skeleton, outputImg);

        skeleton.release();

        Imgproc.watershed(image, outputImg);

        watershedToBw(outputImg);
        outputImg.convertTo(outputImg, CvType.CV_8UC1);
    }

    private void intersectionLinesMarkerBasedWatershedSegmentation(Mat image, Mat preMarkerImg, Mat outputImg) {
        Mat intersectionImg = new Mat(preMarkerImg.height(), preMarkerImg.width(), CvType.CV_8UC1, Scalar.all(0));
        int h1 = preMarkerImg.height() / 2;
        int h2 = h1 + preMarkerImg.height() / 4;

        Imgproc.line(intersectionImg, new Point(0, h1), new Point(preMarkerImg.width(), h1), new Scalar(255), 3);
        Imgproc.line(intersectionImg, new Point(0, h2), new Point(preMarkerImg.width(), h2), new Scalar(255), 3);
        Core.bitwise_and(intersectionImg, preMarkerImg, intersectionImg);

        Imgproc.connectedComponents(intersectionImg, outputImg);

        Imgproc.watershed(image, outputImg);

        watershedToBw(outputImg);
        outputImg.convertTo(outputImg, CvType.CV_8UC1);
    }

    private void watershedToBw(Mat image) {
        int[] imgData = new int[(int) (image.total() * image.channels())];
        int pixelValue;
        image.get(0, 0, imgData);
        for (int i = 0; i < image.rows(); i++) {
            for (int j = 0; j < image.cols(); j++) {
                pixelValue = imgData[(i * image.cols() + j)];
                if(pixelValue == -1) {
                    imgData[(i * image.cols() + j)] = 255;
                } else if(pixelValue != 255) {
                    imgData[(i * image.cols() + j)] = 0;
                }
            }
        }
        image.put(0, 0, imgData);
        return;
    }

    private List<Mat> extractBoxedContours(Mat inputImage, Mat outputMask) {
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(inputImage, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);
        contours.sort((m1, m2) -> {
            Rect rect1 = Imgproc.boundingRect(m1);
            Rect rect2 = Imgproc.boundingRect(m2);

            if (rect1.x > rect2.x) {
                return 1;
            } else if (rect1.x < rect2.x) {
                return -1;
            } else {
                return 0;
            }
        });

        List<Mat> result = new ArrayList<>();
        float totalArea = inputImage.width() * inputImage.height();

        for (MatOfPoint contour : contours) {
            Rect contourBoundingBox = Imgproc.boundingRect(contour);
            float roiArea = (float) contourBoundingBox.area();
            float roiAreaRatio = roiArea / totalArea;

            if(roiAreaRatio >= minContourAreaRatio && roiAreaRatio <= maxContourAreaRatio) {
                Mat digit = new Mat(inputImage, contourBoundingBox);
                result.add(digit);

                if(outputMask != null) {
                    Imgproc.rectangle(outputMask,
                            new Point(contourBoundingBox.x, contourBoundingBox.y),
                            new Point(contourBoundingBox.x + contourBoundingBox.width, contourBoundingBox.y + contourBoundingBox.height),
                            new Scalar(255),
                            -1);
                }
            }
        }

        return result;
    }
}
