package com.example.vrpdrapp;

import android.graphics.Bitmap;
import android.graphics.Color;
import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceView;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {
    private static final String TAG = MainActivity.class.getSimpleName();

    private CameraBridgeViewBase cameraBridgeViewBase;
    private BaseLoaderCallback baseLoaderCallback;

    private Mat currentFrame = null;
    private Mat cachedFrame = null;
    private Yolo yolo;

    private CharactersExtraction charactersExtraction;
    boolean ocrProcessing = false;

    private String ocrPrediction;

    private EMNISTNet emnistNet;

    // NOTE: used for debugging
    private boolean debugPreview = false;
    private HashMap<Integer, Mat> characters = new HashMap<>();
    private HashMap<Integer, String> predChars = new HashMap<>();
    private int selectedCharacterIndex = 0;
    private Mat cachedRoi;
    private Mat cachedProcessedRoi;
    private List<Mat> cachedDigits;
    private List<String> cachedPreds;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        cameraBridgeViewBase = (JavaCameraView) findViewById(R.id.CameraView);
        cameraBridgeViewBase.setVisibility(SurfaceView.VISIBLE);
        cameraBridgeViewBase.setCvCameraViewListener(this);

        baseLoaderCallback = new BaseLoaderCallback(this) {
            @Override
            public void onManagerConnected(int status) {
                super.onManagerConnected(status);
                switch (status) {
                    case BaseLoaderCallback.SUCCESS:
                        cameraBridgeViewBase.enableView();
                        break;
                    default:
                        super.onManagerConnected(status);
                        break;
                }
            }
        };

        Button predictButton = (Button) findViewById(R.id.recog_button);
        predictButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                ocrProcessing = true;
            }
        });

        Button clearButton = (Button) findViewById(R.id.clear_button);
        clearButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                ocrProcessing = false;
                ocrPrediction = "";
                characters.clear();

                ((ImageView) findViewById(R.id.crop_lp_preview)).setImageResource(0);
                ((ImageView) findViewById(R.id.lp_preprocessing_preview)).setImageResource(0);
                ((ImageView) findViewById(R.id.digit_preview)).setImageResource(0);

                ((TextView) findViewById(R.id.ocr_prediction)).setText("");
                ((TextView) findViewById(R.id.digit_prediction)).setText("");

                if(cachedFrame != null) {
                    cachedFrame.release();
                }
            }
        });


        // NOTE: used for debugging
        Button toggleDebugPreviewButton = (Button) findViewById(R.id.toggle_debug_preview);
        toggleDebugPreviewButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                debugPreview = !debugPreview;

                toggleDebugPreviewButton.setBackgroundColor(debugPreview ? Color.GREEN : Color.RED);

                enableDebugPreview(debugPreview);

                if(debugPreview) {
                    ocrProcessing = false;
                    showMatOnImageView(cachedRoi, findViewById(R.id.crop_lp_preview));
                    showMatOnImageView(cachedProcessedRoi, findViewById(R.id.lp_preprocessing_preview));
                    showCharactersOnDebugPreview(cachedDigits, cachedPreds);
                }
            }
        });

        Button digitBack = (Button) findViewById(R.id.digit_back);
        digitBack.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if(!characters.isEmpty()) {
                    --selectedCharacterIndex;
                    if (selectedCharacterIndex < 0) {
                        selectedCharacterIndex = characters.size() - 1;
                    }

                    showMatOnImageView(characters.get(selectedCharacterIndex), findViewById(R.id.digit_preview));
                    TextView digitPredictionTextView = findViewById(R.id.digit_prediction);
                    digitPredictionTextView.setText(predChars.get(selectedCharacterIndex));
                }
            }
        });
        Button digitForward = (Button) findViewById(R.id.digit_forward);
        digitForward.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if(!characters.isEmpty()) {
                    ++selectedCharacterIndex;
                    if (selectedCharacterIndex >= characters.size()) {
                        selectedCharacterIndex = 0;
                    }
                    showMatOnImageView(characters.get(selectedCharacterIndex), findViewById(R.id.digit_preview));
                    TextView digitPredictionTextView = findViewById(R.id.digit_prediction);
                    digitPredictionTextView.setText(predChars.get(selectedCharacterIndex));
                }
            }
        });

        enableDebugPreview(debugPreview);
    }

    private void predict() {
        if(currentFrame == null) return;

        Imgproc.cvtColor(currentFrame, currentFrame, Imgproc.COLOR_RGBA2RGB);
        cachedFrame = currentFrame.clone();

        List<Rect> boundingBoxes = yolo.detect(currentFrame, true);

        if(boundingBoxes != null && !boundingBoxes.isEmpty()) {
            for (Rect boundingBox : boundingBoxes) {
                //Log.i(TAG, "ROI(x,y,w,h) ---> ROI("+boundingBox.x+"," +boundingBox.y+","+boundingBox.width+","+boundingBox.height+")");
                //drawLabeledBoundingBox(processingFrame, "Test", 2.0f, new Scalar(0, 0, 255), new Scalar(0, 0, 255), boundingBox, 2);
                if(boundingBox.x < 0 || boundingBox.y < 0
                        || boundingBox.width < 0 || boundingBox.height < 0
                        || boundingBox.x > currentFrame.width() || boundingBox.y > currentFrame.height()
                        || boundingBox.width > currentFrame.width() || boundingBox.height > currentFrame.height()) {
                    Log.w(TAG, "BAD ROI(x,y,w,h) ---> ROI("+boundingBox.x+"," +boundingBox.y+","+boundingBox.width+","+boundingBox.height+")");
                    ocrProcessing = false;
                    return;
                }

                Log.i(TAG, "ROI(x,y,w,h) ---> ROI("+boundingBox.x+"," +boundingBox.y+","+boundingBox.width+","+boundingBox.height+")");
                Mat roi = new Mat(cachedFrame, boundingBox);
                List<Mat> chars = charactersExtraction.extract1(roi);

                List<String> preds = predictCharacters(chars);
                ocrPrediction = String.join("", preds);
                drawPredictionBoundingBox(currentFrame, boundingBox, ocrPrediction);

                if(debugPreview) {
                    cachedRoi = roi.clone();
                    cachedProcessedRoi = charactersExtraction.getFinalProcessedImage().clone();
                    cachedDigits = chars;
                    cachedPreds = preds;
                }

                roi.release();
            }
        }
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        currentFrame = inputFrame.rgba();

        if(ocrProcessing) {
            predict();
        }

        return currentFrame;
    }

    private List<String> predictCharacters(List<Mat> characters) {
        List<String> predictChars = new ArrayList<>();
        for (Mat ch : characters) {
            predictChars.add(emnistNet.predict(ch));
        }

        return predictChars;
    }

    private void drawLabeledBoundingBox(Mat inputImage, String label, float fontScale, Scalar recColor, Scalar textColor, Rect boundingBox, int thickness) {
        Imgproc.rectangle(inputImage, boundingBox.tl(), boundingBox.br(), recColor, thickness, Core.FILLED);
        Imgproc.putText(inputImage, label, new Point(boundingBox.x, boundingBox.y), Core.FONT_HERSHEY_PLAIN, fontScale, textColor, thickness);
    }

    private void drawPredictionBoundingBox(Mat inOutImage, Rect boundingBox, String prediction) {
        int x = boundingBox.x;
        int y = boundingBox.y + boundingBox.height;
        int w = boundingBox.x + (int) (1.3 * Imgproc.getTextSize(prediction, Core.FONT_HERSHEY_PLAIN, 0.7f, 2, null).width);
        int h = (boundingBox.y + boundingBox.height) + 25;
        Imgproc.rectangle(inOutImage, new Point(x, y), new Point(w, h), new Scalar(0, 255, 255), 2, Core.FILLED);
        Imgproc.putText(inOutImage, prediction, new Point(x+5, y+20), Core.FONT_HERSHEY_PLAIN, 0.7f, new Scalar(0, 255, 255), 2);
    }

    private void showCharactersOnDebugPreview(List<Mat> chars, List<String> preds) {
        if(chars != null && !chars.isEmpty()) {
            showMatOnImageView(chars.get(0), this.findViewById(R.id.digit_preview));

            characters.clear();
            int index = 0;
            for(Mat ch : chars) {
                characters.put(index, ch);
                predChars.put(index, preds.get(index) != null ? preds.get(index) : "?");
                ++index;
            }
        }

        ((TextView) findViewById(R.id.ocr_prediction)).setText(ocrPrediction);
    }

    private void showMatOnImageView(Mat image, ImageView imageView) {
        if (image == null) return;

        Bitmap bitmap = null;

        bitmap = Bitmap.createBitmap(image.width(), image.height(),Bitmap.Config.ARGB_8888);

        Utils.matToBitmap(image, bitmap);

        imageView.setAdjustViewBounds(true);
        imageView.setMaxWidth(image.width());
        imageView.setMaxHeight(image.height());
        imageView.setImageBitmap(bitmap);
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        Log.i(TAG, "Camera View Started  - Resolution: "+width+"x"+height);

        yolo = new Yolo(this,
                768, 416,
                "classes.names",
                "yolov3_license_plates_tiny.cfg",
                "yolov3_license_plates_tiny_best.weights",
                0.6f,
                0.5f);

        charactersExtraction = new CharactersExtraction(0.03f, 0.23f);

        emnistNet = new EMNISTNet(this, "emnist_net_custom_mobile.pth");

        Log.i(TAG, "Modules loaded!");
    }

    @Override
    public void onCameraViewStopped() {
        if(currentFrame != null) {
            currentFrame.release();
        }
        if(cachedFrame != null) {
            cachedFrame.release();
        }
    }

    @Override
    protected void onResume() {
        super.onResume();

        if(!OpenCVLoader.initDebug()) {
            Toast.makeText(getApplicationContext(), "OpenCV debug couldn't load properly!", Toast.LENGTH_SHORT).show();
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_4_0, this, baseLoaderCallback);
            return;
        }

        baseLoaderCallback.onManagerConnected(baseLoaderCallback.SUCCESS);
    }

    @Override
    protected void onPause() {
        super.onPause();

        if(cameraBridgeViewBase != null){
            cameraBridgeViewBase.disableView();
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();

        if (cameraBridgeViewBase != null){
            cameraBridgeViewBase.disableView();
        }
    }

    private void enableDebugPreview(boolean enable) {
        int visibility = enable ? View.VISIBLE : View.INVISIBLE;

        ImageView cropLpPreview = findViewById(R.id.crop_lp_preview);
        cropLpPreview.setEnabled(enable);
        cropLpPreview.setVisibility(visibility);

        ImageView lpPreprocessingPreview = findViewById(R.id.lp_preprocessing_preview);
        lpPreprocessingPreview.setEnabled(enable);
        lpPreprocessingPreview.setVisibility(visibility);

        ImageView digitPreview = findViewById(R.id.digit_preview);
        digitPreview.setEnabled(enable);
        digitPreview.setVisibility(visibility);

        Button digitBack = findViewById(R.id.digit_back);
        digitBack.setEnabled(enable);
        digitBack.setVisibility(visibility);

        Button digitForward = findViewById(R.id.digit_forward);
        digitForward.setEnabled(enable);
        digitForward.setVisibility(visibility);

        TextView ocrPrediction = findViewById(R.id.ocr_prediction);
        ocrPrediction.setEnabled(enable);
        ocrPrediction.setVisibility(visibility);
        ocrPrediction.setBackgroundColor(Color.BLACK);
        ocrPrediction.setTextColor(Color.CYAN);

        TextView digitPredictionTextView = findViewById(R.id.digit_prediction);
        digitPredictionTextView.setEnabled(enable);
        digitPredictionTextView.setVisibility(visibility);
        digitPredictionTextView.setBackgroundColor(Color.BLACK);
        digitPredictionTextView.setTextColor(Color.CYAN);
    }
}

