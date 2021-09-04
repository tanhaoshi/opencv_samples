package org.opencv.samples.puzzle15;

import android.graphics.Bitmap;
import android.util.Log;
import android.widget.ImageView;

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

import java.util.LinkedList;
import java.util.List;

/**
 * create by ths on 2021/9/3
 */
public class TempSaveCode {

    /**
     * 临时保存 图像识别代码
     */

    private static final String TAG = "TempSaveCode";

    int matchesPointCount = 0;

    public void matchImage() {

        Mat resT = new Mat();
        Mat resO = new Mat();

        //即当detector 又当Detector
        SIFT sift = SIFT.create();

        Mat templateImage = null;
        try{
//            templateImage = Utils.loadResource(this,R.drawable.six);
        }catch (Exception e){
            e.printStackTrace();
        }

        if(templateImage == null){
            Log.i(TAG,"the template image is null ... ");
            return;
        }

        Mat originalImage = null;
        try{
//            originalImage = Utils.loadResource(this,R.drawable.six_empty);
        }catch (Exception e){
            e.printStackTrace();
        }

        if(originalImage == null){
            Log.i(TAG,"the original image is null ...");
            return;
        }

        //展示图片
//        showImage(startIdCard,templateImage);

//        showImage(resizeIdCard,originalImage);

        MatOfKeyPoint templateKeyPoints = new MatOfKeyPoint();
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

            Mat subMat = null;

            try{
                subMat = originalImage.submat(rowStart, rowEnd, colStart, colEnd);
            }catch (Exception e){
                e.printStackTrace();
            }

            if(subMat == null || subMat.rows() < 100 || subMat.rows() < 100){
                Log.i(TAG,"sub mat is empty ... ");
            }else{
//                showImage(thresholdIdCard,subMat);
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

//            showImage(grayIdCard,matchOutput);
        } else {
            Log.i(TAG,"the template image is not exit of original image ");
        }
    }

    private void showImage(ImageView imageView, Mat mat){
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
