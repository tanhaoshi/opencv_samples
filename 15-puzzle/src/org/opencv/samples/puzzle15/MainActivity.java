package org.opencv.samples.puzzle15;

import android.Manifest;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.hardware.Camera;
import android.os.Bundle;
import android.support.annotation.NonNull;
import android.support.v7.app.AppCompatActivity;
import android.util.DisplayMetrics;
import android.util.Log;
import android.view.TextureView;
import android.view.View;
import android.view.ViewTreeObserver;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.DMatch;
import org.opencv.core.KeyPoint;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.Features2d;
import org.opencv.features2d.SIFT;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.samples.puzzle15.camera.CameraHelper;
import org.opencv.samples.puzzle15.camera.CameraListener;
import org.opencv.samples.puzzle15.permissions.EasyPermission;
import org.opencv.samples.puzzle15.utils.ImageRotateUtil;

import java.io.ByteArrayOutputStream;
import java.util.LinkedList;
import java.util.List;

public class MainActivity extends AppCompatActivity implements EasyPermission.PermissionCallback,ViewTreeObserver.OnGlobalLayoutListener{

    /**
     * opencv 根据图片特征 寻找相似度相当的图片
     * https://blog.csdn.net/qq_42670220/article/details/108623752
     */

    private static final String TAG = "MainActivity";

    private int matchesPointCount = 0;

    private boolean tempOpenCVFlag = false;

//    ImageView startIdCard,resizeIdCard,grayIdCard,thresholdIdCard;

    private ImageView   startIdCard,matchImage,subImage;
    private TextureView textureView;

    private CameraHelper cameraHelper;
    private Integer rgbCameraId = Camera.CameraInfo.CAMERA_FACING_BACK;

    MatOfKeyPoint templateKeyPoints;
    Mat templateImage;

    private static final int REGISTER_STATUS_READY = 0;
    private static final int REGISTER_STATUS_PROCESSING = 1;
    private static final int REGISTER_STATUS_DONE = 2;
    private int registerStatus = REGISTER_STATUS_DONE;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_camera);

        startIdCard  = findViewById(R.id.startIdCard);
//        resizeIdCard = findViewById(R.id.resizeIdCard);

//        grayIdCard   = findViewById(R.id.grayIdCard);
//        thresholdIdCard = findViewById(R.id.thresholdIdCard);
        subImage     = findViewById(R.id.subImage);

        matchImage   = findViewById(R.id.matchImage);

        textureView  = findViewById(R.id.textureView);

        textureView.getViewTreeObserver().addOnGlobalLayoutListener(this);

        findViewById(R.id.startPhoto).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if(registerStatus == REGISTER_STATUS_DONE){
                    registerStatus = REGISTER_STATUS_READY;
                }
            }
        });

        requestPermission();
    }

    private String[] permissions = new String[]{
            Manifest.permission.CAMERA,
            Manifest.permission.READ_PHONE_STATE,
            Manifest.permission.WRITE_EXTERNAL_STORAGE
    };

    private final int REQUEST_PERMISSIONS = 2;

    private void requestPermission(){
        if(!EasyPermission.hasPermissions(this, permissions)){
            EasyPermission.with(this)
                    .rationale("否则无法打开相机")
                    .addRequestCode(REQUEST_PERMISSIONS)
                    .permissions(permissions)
                    .request();
        }
    }

    @Override
    public void onPermissionGranted(int requestCode, List<String> perms) {
        initCamera();
    }

    @Override
    public void onPermissionDenied(int requestCode, List<String> perms) {
        Toast.makeText(this,"权限申请失败!",Toast.LENGTH_SHORT).show();
    }

    @Override
    public void onGlobalLayout() {
        textureView.getViewTreeObserver().removeOnGlobalLayoutListener(this);
        initCamera();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        EasyPermission.onRequestPermissionsResult(this,REQUEST_PERMISSIONS,permissions,grantResults);
    }

    @Override
    protected void onResume() {
        super.onResume();
    }

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i(TAG, "OpenCV loaded successfully");
                    initOriginalImage();

                    tempOpenCVFlag = true;
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    private void initCamera() {
        DisplayMetrics metrics = new DisplayMetrics();
        getWindowManager().getDefaultDisplay().getMetrics(metrics);

        CameraListener cameraListener = new CameraListener() {
            @Override
            public void onCameraOpened(Camera camera, int cameraId, int displayOrientation, boolean isMirror) {
            }

            @Override
            public void onPreview(byte[] nv21, Camera camera) {
                if(tempOpenCVFlag && registerStatus == REGISTER_STATUS_DONE){
//                    registerStatus = REGISTER_STATUS_PROCESSING;
                    matchImage(nv21);
                }
            }

            @Override
            public void onCameraClosed(){
                Log.i(TAG, "onCameraClosed: ");
            }

            @Override
            public void onCameraError(Exception e) {
                Log.i(TAG, "onCameraError: " + e.getMessage());
            }

            @Override
            public void onCameraConfigurationChanged(int cameraID, int displayOrientation) {
                Log.i(TAG, "onCameraConfigurationChanged: " + cameraID + "  " + displayOrientation);
            }
        };

        cameraHelper = new CameraHelper.Builder()
                .previewViewSize(new android.graphics.Point(textureView.getMeasuredWidth(), textureView.getMeasuredHeight()))
                .rotation(getWindowManager().getDefaultDisplay().getRotation())
                .specificCameraId(rgbCameraId != null ? rgbCameraId : Camera.CameraInfo.CAMERA_FACING_FRONT)
                .isMirror(false)
                .previewOn(textureView)
                .cameraListener(cameraListener)
                .build();
        cameraHelper.init();
        cameraHelper.start();
    }

    private void initOriginalImage(){
        try{
            templateImage = Utils.loadResource(this,R.drawable.six);
        }catch (Exception e){
            e.printStackTrace();
        }

        if(templateImage == null){
            Log.i(TAG,"the template image is null ... ");
            return;
        }

        showImage(startIdCard,templateImage);

        templateKeyPoints = new MatOfKeyPoint();
    }

    public void matchImage(byte[] nv21) {
        Camera.Size previewSize = cameraHelper.previewSize;
        BitmapFactory.Options newOpts = new BitmapFactory.Options();
        newOpts.inJustDecodeBounds = true;
        YuvImage yuvimage = new YuvImage(
                nv21,
                ImageFormat.NV21,
                previewSize.width,
                previewSize.height,
                null);
        ByteArrayOutputStream bao = new ByteArrayOutputStream();
        yuvimage.compressToJpeg(new Rect(0, 0, previewSize.width,previewSize.height), 100, bao);// 80--JPG图片的质量[0-100],100最高
        byte[] rawImage = bao.toByteArray();
        //将rawImage转换成bitmap
        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inSampleSize = 2;
        options.inPreferredConfig = Bitmap.Config.RGB_565;
        Bitmap bitmap = BitmapFactory.decodeByteArray(rawImage, 0, rawImage.length, options);

        Bitmap rotationMap = null;
        if(cameraHelper.mCameraId == 1){
            rotationMap = ImageRotateUtil.of().rotateBitmapByDegree(bitmap,-90);
        }else{
            rotationMap = ImageRotateUtil.of().rotateBitmapByDegree(bitmap,-270);
        }

        if(rotationMap == null){
            Log.i(TAG,"the bitmap is null ...");
//            registerStatus = REGISTER_STATUS_DONE;
            return;
        }
        matchImage.setImageBitmap(rotationMap);

        Mat originalImage = new Mat();
        Utils.bitmapToMat(rotationMap,originalImage);

        if(originalImage == null){
            Log.i(TAG,"the original image is null ...");
//            registerStatus = REGISTER_STATUS_DONE;
            return;
        }

        Mat resT = new Mat();
        Mat resO = new Mat();

        //即当detector 又当Detector
        SIFT sift = SIFT.create();

        //展示图片
//        showImage(startIdCard,templateImage);

//        showImage(resizeIdCard,originalImage);

        MatOfKeyPoint originalKeyPoints = new MatOfKeyPoint();

        //获取模板图的特征点
        // 提取所有特征点 分别保存到两个对象当中
        sift.detect(templateImage, templateKeyPoints);
        sift.detect(originalImage, originalKeyPoints);

        // 然后进行比较 并赋值到 resT 与 resO 两个对象当中
        sift.compute(templateImage, templateKeyPoints, resT);
        sift.compute(originalImage, originalKeyPoints, resO);

        List<MatOfDMatch> matches = new LinkedList();
        DescriptorMatcher descriptorMatcher = DescriptorMatcher.create(DescriptorMatcher.FLANNBASED);

        /**
         * knnMatch方法的作用就是在给定特征描述集合中寻找最佳匹配
         * 使用KNN-matching算法，令K=2，则每个match得到两个最接近的descriptor，然后计算最接近距离和次接近距离之间的比值，当比值大于既定值时，才作为最终match。
         */
        try{
            descriptorMatcher.knnMatch(resT, resO, matches, 2);

            LinkedList<DMatch> goodMatchesList = new LinkedList();

            for(int k=0; k<matches.size(); k++){
                DMatch[] dmatcharray = matches.get(k).toArray();
                DMatch m1 = dmatcharray[0];
                DMatch m2 = dmatcharray[1];

                if (m1.distance <= m2.distance * 2.0) {
                    goodMatchesList.addLast(m1);
                }
            }

            matchesPointCount = goodMatchesList.size();
            //当匹配后的特征点大于等于 4 个，则认为模板图在原图中，该值可以自行调整

            Log.i(TAG,"the point px size = " + matchesPointCount);

            if (matchesPointCount >= 40) {
                Log.i(TAG,"the POINT image is > 40 ");

                List<KeyPoint> templateKeyPointList = templateKeyPoints.toList();
                List<KeyPoint> originalKeyPointList = originalKeyPoints.toList();
                LinkedList<Point> objectPoints = new LinkedList();
                LinkedList<Point> scenePoints = new LinkedList();

                for(int i = 0; i<goodMatchesList.size(); i++){
                    objectPoints.addLast(templateKeyPointList.get(goodMatchesList.get(i).queryIdx).pt);
                    scenePoints.addLast(originalKeyPointList.get(goodMatchesList.get(i).trainIdx).pt);
                }

                MatOfPoint2f objMatOfPoint2f = new MatOfPoint2f();
                objMatOfPoint2f.fromList(objectPoints);
                MatOfPoint2f scnMatOfPoint2f = new MatOfPoint2f();
                scnMatOfPoint2f.fromList(scenePoints);
                //使用 findHomography 寻找匹配上的关键点的变换
                Mat homography = Calib3d.findHomography(objMatOfPoint2f, scnMatOfPoint2f, Calib3d.RANSAC, 3);

                /**
                 * 透视变换(Perspective Transformation)是将图片投影到一个新的视平面(Viewing Plane)，也称作投影映射(Projective Mapping)。
                 */
                Mat templateCorners = new Mat(4, 1, CvType.CV_32FC2);
                Mat templateTransformResult = new Mat(4, 1, CvType.CV_32FC2);
                templateCorners.put(0, 0, new double[]{0, 0});
                templateCorners.put(1, 0, new double[]{templateImage.cols(), 0});
                templateCorners.put(2, 0, new double[]{templateImage.cols(), templateImage.rows()});
                templateCorners.put(3, 0, new double[]{0, templateImage.rows()});
                //使用 perspectiveTransform 将模板图进行透视变以矫正图象得到标准图片
                Core.perspectiveTransform(templateCorners, templateTransformResult, homography);

                //矩形四个顶点  匹配的图片经过旋转之后就这个矩形的四个点的位置就不是正常的abcd了
                double[] pointA = templateTransformResult.get(0, 0);
                double[] pointB = templateTransformResult.get(1, 0);
                double[] pointC = templateTransformResult.get(2, 0);
                double[] pointD = templateTransformResult.get(3, 0);

                //指定取得数组子集的范围
                int rowStart = (int) pointA[1];
                int rowEnd = (int) pointC[1];
                int colStart = (int) pointD[0];
                int colEnd = (int) pointB[0];
                //rowStart, rowEnd, colStart, colEnd 好像必须左上右下  没必要从原图扣下来模板图了

                Mat subMat = originalImage.submat(rowStart, rowEnd, colStart, colEnd);

                if(subMat == null || subMat.rows() < 100 || subMat.rows() < 100){
                    Log.i(TAG,"sub mat is empty ... ");
//                    registerStatus = REGISTER_STATUS_DONE;
                }else{
                    Log.i(TAG,"sub mat is not empty ... ");
                    showImage(subImage,subMat);
                }

                //将匹配的图像用用四条线框出来
                Imgproc.rectangle(originalImage, new Point(pointA), new Point(pointC), new Scalar(0, 255, 0));

                Imgproc.line(originalImage, new Point(pointA), new Point(pointB), new Scalar(0, 255, 0), 4);//上 A->B
                Imgproc.line(originalImage, new Point(pointB), new Point(pointC), new Scalar(0, 255, 0), 4);//右 B->C
                Imgproc.line(originalImage, new Point(pointC), new Point(pointD), new Scalar(0, 255, 0), 4);//下 C->D
                Imgproc.line(originalImage, new Point(pointD), new Point(pointA), new Scalar(0, 255, 0), 4);//左 D->A

                MatOfDMatch goodMatches = new MatOfDMatch();
                goodMatches.fromList(goodMatchesList);
                Mat matchOutput = new Mat(originalImage.rows() * 2, originalImage.cols() * 2, Imgcodecs.IMREAD_COLOR);
                Features2d.drawMatches(templateImage, templateKeyPoints, originalImage, originalKeyPoints, goodMatches, matchOutput, new Scalar(0, 255, 0), new Scalar(255, 0, 0), new MatOfByte(), 2);
                Features2d.drawMatches(templateImage, templateKeyPoints, originalImage, originalKeyPoints, goodMatches, matchOutput, new Scalar(0, 255, 0), new Scalar(255, 0, 0), new MatOfByte(), 2);

//                showImage(matchImage,matchOutput);

            } else {
                Log.i(TAG,"the template image is not exit of original image ");
            }
        }catch (Exception e){
//            registerStatus = REGISTER_STATUS_DONE;
            e.printStackTrace();
        }

//        registerStatus = REGISTER_STATUS_DONE;
//            showImage(grayIdCard,matchOutput);
    }

    private void showImage(ImageView imageView,Mat mat){
        if(mat.cols() == 0 || mat.rows() == 0){
            Log.i(TAG,"the mat is not image ... ");
            return;
        }else{
            Log.i(TAG,"mat.cols() = " + mat.cols() + "----- mat.rows() = " + mat.rows());
        }
        Bitmap bitmap = Bitmap.createBitmap(mat.cols(),mat.rows(),Bitmap.Config.ARGB_8888);

        Utils.matToBitmap(mat,bitmap);

        if(bitmap != null){
            imageView.setImageBitmap(bitmap);
        }
    }
}
