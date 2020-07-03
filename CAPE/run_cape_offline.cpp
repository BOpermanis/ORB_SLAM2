/*
 * Copyright 2018 Pedro Proenza <p.proenca@surrey.ac.uk> (University of Surrey)
 *
 */

#include <iostream>
#include <cstdio>
#define _USE_MATH_DEFINES
#include <math.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include "CAPE/capewrap.cpp"
#include <cstdlib>
#include<zconf.h>


using namespace std;


bool done = false;
float COS_ANGLE_MAX = cos(M_PI/12);
float MAX_MERGE_DIST = 50.0f;
bool cylinder_detection= true;
std::vector<cv::Vec3b> color_code;


bool loadCalibParameters1(string filepath, cv:: Mat & intrinsics_rgb, cv::Mat & dist_coeffs_rgb, cv:: Mat & intrinsics_ir, cv::Mat & dist_coeffs_ir, cv::Mat & R, cv::Mat & T){

    cv::FileStorage fs(filepath,cv::FileStorage::READ);
    if (fs.isOpened()){
        fs["RGB_intrinsic_params"]>>intrinsics_rgb;
        fs["RGB_distortion_coefficients"]>>dist_coeffs_rgb;
        fs["IR_intrinsic_params"]>>intrinsics_ir;
        fs["IR_distortion_coefficients"]>>dist_coeffs_ir;
        fs["Rotation"]>>R;
        fs["Translation"]>>T;
        fs.release();
        return true;
    }else{
        cerr<<"Calibration file missing"<<endl;
        return false;
    }
}


std::map<string, float> loadCalibParameters(string filepath){

    std::map<string, float> params;
    cv::Mat K_rgb, K_ir;

    cv::FileStorage fs(filepath,cv::FileStorage::READ);
    if (fs.isOpened()){
        fs["RGB_intrinsic_params"]>>K_rgb;
        fs["IR_intrinsic_params"]>>K_ir;

        float fx_ir = K_ir.at<double>(0,0); float fy_ir = K_ir.at<double>(1,1);
        float cx_ir = K_ir.at<double>(0,2); float cy_ir = K_ir.at<double>(1,2);
        float fx_rgb = K_rgb.at<double>(0,0); float fy_rgb = K_rgb.at<double>(1,1);
        float cx_rgb = K_rgb.at<double>(0,2); float cy_rgb = K_rgb.at<double>(1,2);

        params["Camera.fx"] = fx_ir;
        params["Camera.fy"] = fy_ir;
        params["Camera.cx"] = cx_ir;
        params["Camera.cy"] = cy_ir;
        params["Camera.fx"] = fx_rgb;
        params["Camera.fy"] = fy_rgb;
        params["Camera.cx"] = cx_rgb;
        params["Camera.cy"] = cy_rgb;
        params["Camera.width"] = 640.0;
        params["Camera.height"] = 480.0;

        fs.release();
        return params;
    }else{
        cerr<<"Calibration file missing"<<endl;
        return params;
    }
}

void projectPointCloud(cv::Mat & X, cv::Mat & Y, cv::Mat & Z, cv::Mat & U, cv::Mat & V, float fx_rgb, float fy_rgb, float cx_rgb, float cy_rgb, double z_min, Eigen::MatrixXf & cloud_array){

    int width = X.cols;
    int height = X.rows;

    // Project to image coordinates
    cv::divide(X,Z,U,1);
    cv::divide(Y,Z,V,1);
    U = U*fx_rgb + cx_rgb;
    V = V*fy_rgb + cy_rgb;
    // Reusing U as cloud index
    //U = V*width + U + 0.5;

    float * sz, * sx, * sy, * u_ptr, * v_ptr, * id_ptr;
    float z, u, v;
    int id;
    for(int r=0; r< height; r++){
        sx = X.ptr<float>(r);
        sy = Y.ptr<float>(r);
        sz = Z.ptr<float>(r);
        u_ptr = U.ptr<float>(r);
        v_ptr = V.ptr<float>(r);
        for(int c=0; c< width; c++){
            z = sz[c];
            u = u_ptr[c];
            v = v_ptr[c];
            if(z>z_min && u>0 && v>0 && u<width && v<height){
                id = floor(v)*width + u;
                cloud_array(id,0) = sx[c];
                cloud_array(id,1) = sy[c];
                cloud_array(id,2) = z;
            }
        }
    }
}

void organizePointCloudByCell(Eigen::MatrixXf & cloud_in, Eigen::MatrixXf & cloud_out, cv::Mat & cell_map){

    int width = cell_map.cols;
    int height = cell_map.rows;
    int mxn = width*height;
    int mxn2 = 2*mxn;

    int id, it(0);
    int * cell_map_ptr;
    for(int r=0; r< height; r++){
        cell_map_ptr = cell_map.ptr<int>(r);
        for(int c=0; c< width; c++){
            id = cell_map_ptr[c];
            *(cloud_out.data() + id) = *(cloud_in.data() + it);
            *(cloud_out.data() + mxn + id) = *(cloud_in.data() + mxn + it);
            *(cloud_out.data() + mxn2 + id) = *(cloud_in.data() + mxn2 + it);
            it++;
        }
    }
}

int main(int argc, char ** argv){

    string sequence;
    int PATCH_SIZE;
    if (argc>1){
        PATCH_SIZE = atoi(argv[1]);
        sequence = argv[2];
    }else{
        PATCH_SIZE = 20;
        sequence = "seq_example";
    }

    stringstream string_buff;
    string data_path = "/CAPE/Data/";
    string_buff<<data_path<<sequence;

    // Get intrinsics
    cv::Mat K_rgb, K_ir, dist_coeffs_rgb, dist_coeffs_ir, R_stereo, t_stereo;
    stringstream calib_path;
    calib_path<<string_buff.str()<<"/calib_params.xml";
    auto params = loadCalibParameters(calib_path.str());

    auto plane_detector = capewrap(params);

    // Read frame 1 to allocate and get dimension
    cv::Mat rgb_img, d_img;
    int width, height;
    stringstream image_path;
    stringstream depth_img_path;
    stringstream rgb_img_path;
    rgb_img_path<<string_buff.str()<<"/rgb_0.png";
    depth_img_path<<string_buff.str()<<"/depth_0.png";

    rgb_img = cv::imread(rgb_img_path.str(),cv::IMREAD_COLOR);

    if(rgb_img.data){
        width = rgb_img.cols;
        height = rgb_img.rows;
    }else{
        cout<<"Error loading file";
        return -1;
    }

    cv::namedWindow("Seg");

    // Populate with random color codes
    for(int i=0; i<100;i++){
        cv::Vec3b color;
        color[0]=rand()%255;
        color[1]=rand()%255;
        color[2]=rand()%255;
        color_code.push_back(color);
    }

    // Add specific colors for planes
    color_code[0][0] = 0; color_code[0][1] = 0; color_code[0][2] = 255;
    color_code[1][0] = 255; color_code[1][1] = 0; color_code[1][2] = 204;
    color_code[2][0] = 255; color_code[2][1] = 100; color_code[2][2] = 0;
    color_code[3][0] = 0; color_code[3][1] = 153; color_code[3][2] = 255;
    // Add specific colors for cylinders
    color_code[50][0] = 178; color_code[50][1] = 255; color_code[50][2] = 0;
    color_code[51][0] = 255; color_code[51][1] = 0; color_code[51][2] = 51;
    color_code[52][0] = 0; color_code[52][1] = 255; color_code[52][2] = 51;
    color_code[53][0] = 153; color_code[53][1] = 0; color_code[53][2] = 255;

    int i=0;
    while(true){

        // Read frame i
        rgb_img_path.str("");
        rgb_img_path<<string_buff.str()<<"/rgb_"<<i<<".png";
        rgb_img = cv::imread(rgb_img_path.str(),cv::IMREAD_COLOR);

        if (!rgb_img.data)
            break;

        cout<<"Frame: "<<i<<endl;

        // Read depth image
        depth_img_path.str("");
        depth_img_path<<string_buff.str()<<"/depth_"<<i<<".png";

        d_img = cv::imread(depth_img_path.str(), cv::IMREAD_ANYDEPTH);
        d_img.convertTo(d_img, CV_32F);

        cv::Size s = d_img.size();
        cout << d_img.dims << " " << s.width << " " << s.height << endl;

        cv::Mat_<cv::Vec3b> seg_rz = cv::Mat_<cv::Vec3b>(height,width,cv::Vec3b(0,0,0));
        cv::Mat_<uchar> seg_output = cv::Mat_<uchar>(height,width,uchar(0));

        // Get intrinsics

        loadCalibParameters1("/CAPE/Data/seq_example/calib_params.xml", K_rgb, dist_coeffs_rgb, K_ir, dist_coeffs_ir, R_stereo, t_stereo);

        cout << R_stereo << endl;
        cout << t_stereo << endl;
        auto output = plane_detector.process(d_img, R_stereo, t_stereo);

        // Map segments with color codes and overlap segmented image w/ RGB
        uchar * sCode;
        uchar * dColor;
        uchar * srgb;
        int code;
        for(int r=0; r<  height; r++){
            dColor = seg_rz.ptr<uchar>(r);
            sCode = seg_output.ptr<uchar>(r);
            srgb = rgb_img.ptr<uchar>(r);
            for(int c=0; c< width; c++){
                code = *sCode;
                if (code>0){
                    dColor[c*3] =   color_code[code-1][0]/2 + srgb[0]/2;
                    dColor[c*3+1] = color_code[code-1][1]/2 + srgb[1]/2;
                    dColor[c*3+2] = color_code[code-1][2]/2 + srgb[2]/2;;
                }else{
                    dColor[c*3] =  srgb[0];
                    dColor[c*3+1] = srgb[1];
                    dColor[c*3+2] = srgb[2];
                }
                sCode++; srgb++; srgb++; srgb++;
            }
        }

        // Show frame rate and labels
        cv::rectangle(seg_rz,  cv::Point(0,0),cv::Point(width,20), cv::Scalar(0,0,0),-1);
        std::stringstream fps;
        cv::putText(seg_rz, fps.str(), cv::Point(15,15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,255,255,1));

        cout<<"Nr cylinders:"<<output.nr_cylinders<<endl;
        cout<<"Nr planes:"<<output.nr_planes<<endl;
        int cylinder_code_offset = 50;
        // show cylinder labels
        if (output.nr_cylinders>0){
            std::stringstream text;
            text<<"Cylinders:";
            cv::putText(seg_rz, text.str(), cv::Point(width/2,15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,255,255,1));
            for(int j=0;j<output.nr_cylinders;j++){
                cv::rectangle(seg_rz,  cv::Point(width/2 + 80+15*j,6),cv::Point(width/2 + 90+15*j,16), cv::Scalar(color_code[cylinder_code_offset+j][0],color_code[cylinder_code_offset+j][1],color_code[cylinder_code_offset+j][2]),-1);
            }
        }
//        cv::imshow("Seg", seg_rz);
//        cv::waitKey(1);
//        sleep(10);
        i++;

    }
    return 0;
}

