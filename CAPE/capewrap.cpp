//
// Created by root on 7/1/20.
//

/*
 * Copyright 2018 Pedro Proenza <p.proenca@surrey.ac.uk> (University of Surrey)
 *
 */


#ifndef capewrap_cpp
#define capewrap_cpp

#include <iostream>
#include <cstdio>

#define _USE_MATH_DEFINES

#include <math.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include "CAPE/CAPE.h"

using namespace std;

class capewrap {
public:
    void projectPointCloud(cv::Mat &X, cv::Mat &Y, cv::Mat &Z, cv::Mat &U, cv::Mat &V, float fx_rgb, float fy_rgb, float cx_rgb,
                           float cy_rgb, double z_min, Eigen::MatrixXf &cloud_array) {

        int width = X.cols;
        int height = X.rows;

        // Project to image coordinates
        cv::divide(X, Z, U, 1);
        cv::divide(Y, Z, V, 1);
        U = U * fx_rgb + cx_rgb;
        V = V * fy_rgb + cy_rgb;
        // Reusing U as cloud index
        //U = V*width + U + 0.5;

        float *sz, *sx, *sy, *u_ptr, *v_ptr, *id_ptr;
        float z, u, v;
        int id;
        for (int r = 0; r < height; r++) {
            sx = X.ptr<float>(r);
            sy = Y.ptr<float>(r);
            sz = Z.ptr<float>(r);
            u_ptr = U.ptr<float>(r);
            v_ptr = V.ptr<float>(r);
            for (int c = 0; c < width; c++) {
                z = sz[c];
                u = u_ptr[c];
                v = v_ptr[c];
                if (z > z_min && u > 0 && v > 0 && u < width && v < height) {
                    id = floor(v) * width + u;
                    cloud_array(id, 0) = sx[c];
                    cloud_array(id, 1) = sy[c];
                    cloud_array(id, 2) = z;
                }
            }
        }
    }

    void organizePointCloudByCell(Eigen::MatrixXf &cloud_in, Eigen::MatrixXf &cloud_out, cv::Mat &cell_map) {

        int width = cell_map.cols;
        int height = cell_map.rows;
        int mxn = width * height;
        int mxn2 = 2 * mxn;

        int id, it(0);
        int *cell_map_ptr;
        for (int r = 0; r < height; r++) {
            cell_map_ptr = cell_map.ptr<int>(r);
            for (int c = 0; c < width; c++) {
                id = cell_map_ptr[c];
                *(cloud_out.data() + id) = *(cloud_in.data() + it);
                *(cloud_out.data() + mxn + id) = *(cloud_in.data() + mxn + it);
                *(cloud_out.data() + mxn2 + id) = *(cloud_in.data() + mxn2 + it);
                it++;
            }
        }
    }

    bool done = false;
    float COS_ANGLE_MAX = cos(M_PI / 12);
    float MAX_MERGE_DIST = 50.0f;
    bool cylinder_detection = true;

    std::vector<cv::Vec3b> color_code;

    CAPE *plane_detector;

    int PATCH_SIZE = 20;
    cv::Mat_<int> cell_map;

    float fx_ir, fy_ir, cx_ir, cy_ir, fx_rgb, fy_rgb, cx_rgb, cy_rgb;
    int width, height;
    cv::Mat rgb_img, d_img;
    // d_img.convertTo(d_img, CV_32F);

    cv::Mat_<float> X, Y, X_t, Y_t, X_pre, Y_pre, U, V;
    Eigen::MatrixXf cloud_array, cloud_array_organized;

    capewrap(cv::FileStorage fSettings) {

        for (int i = 0; i < 100; i++) {
            cv::Vec3b color;
            color[0] = rand() % 255;
            color[1] = rand() % 255;
            color[2] = rand() % 255;
            color_code.push_back(color);
        }

        // Add specific colors for planes
        color_code[0][0] = 0;
        color_code[0][1] = 0;
        color_code[0][2] = 255;
        color_code[1][0] = 255;
        color_code[1][1] = 0;
        color_code[1][2] = 204;
        color_code[2][0] = 255;
        color_code[2][1] = 100;
        color_code[2][2] = 0;
        color_code[3][0] = 0;
        color_code[3][1] = 153;
        color_code[3][2] = 255;
        // Add specific colors for cylinders
        color_code[50][0] = 178;
        color_code[50][1] = 255;
        color_code[50][2] = 0;
        color_code[51][0] = 255;
        color_code[51][1] = 0;
        color_code[51][2] = 51;
        color_code[52][0] = 0;
        color_code[52][1] = 255;
        color_code[52][2] = 51;
        color_code[53][0] = 153;
        color_code[53][1] = 0;
        color_code[53][2] = 255;


        // Get intrinsics
        fx_ir = fSettings["Camera.fx"];
        fy_ir = fSettings["Camera.fy"];
        cx_ir = fSettings["Camera.cx"];
        cy_ir = fSettings["Camera.cy"];
        fx_rgb = fSettings["Camera.fx"];
        fy_rgb = fSettings["Camera.fy"];
        cx_rgb = fSettings["Camera.cx"];
        cy_rgb = fSettings["Camera.cy"];
        width  = fSettings["Camera.width"];
        height = fSettings["Camera.height"];



        int nr_horizontal_cells = width / PATCH_SIZE;
        int nr_vertical_cells = height / PATCH_SIZE;

        for (int r = 0; r < height; r++) {
            for (int c = 0; c < width; c++) {
                // Not efficient but at this stage doesn t matter
                X_pre.at<float>(r, c) = (c - cx_ir) / fx_ir;
                Y_pre.at<float>(r, c) = (r - cy_ir) / fy_ir;
            }
        }

        // Pre-computations for maping an image point cloud to a cache-friendly array where cell's local point clouds are contiguous
        cell_map = cv::Mat_<int>(height, width);

        for (int r = 0; r < height; r++) {
            int cell_r = r / PATCH_SIZE;
            int local_r = r % PATCH_SIZE;
            for (int c = 0; c < width; c++) {
                int cell_c = c / PATCH_SIZE;
                int local_c = c % PATCH_SIZE;
                cell_map.at<int>(r, c) =
                        (cell_r * nr_horizontal_cells + cell_c) * PATCH_SIZE * PATCH_SIZE + local_r * PATCH_SIZE + local_c;
            }
        }

        X = cv::Mat_<float>(height, width);
        Y = cv::Mat_<float>(height, width);
        X_pre = cv::Mat_<float>(height, width);
        Y_pre = cv::Mat_<float>(height, width);

        U = cv::Mat_<float>(height, width);
        V = cv::Mat_<float>(height, width);
        X_t = cv::Mat_<float>(height, width);
        Y_t = cv::Mat_<float>(height, width);
        cloud_array = Eigen::MatrixXf(width * height, 3);
        cloud_array_organized = Eigen::MatrixXf(width * height, 3);
        cv::Mat_<int> cell_map(height, width);

        plane_detector = new CAPE(height, width, PATCH_SIZE, PATCH_SIZE, cylinder_detection, COS_ANGLE_MAX, MAX_MERGE_DIST);
    }

    cv::Mat_<cv::Vec3b> process(const cv::Mat &imRGB, const cv::Mat &imD, const cv::Mat &R_stereo, const cv::Mat &t_stereo, bool flag_rotate=false) {
        rgb_img = imRGB;
        d_img = imD;
        // Populate with random color codes

        // Initialize CAPE

        // Backproject to point cloud
        X = X_pre.mul(d_img);
        Y = Y_pre.mul(d_img);
        cloud_array.setZero();

        // The following transformation+projection is only necessary to visualize RGB with overlapped segments
        // Transform point cloud to color reference frame
        if (flag_rotate) {
            X_t = ((float) R_stereo.at<double>(0, 0)) * X + ((float) R_stereo.at<double>(0, 1)) * Y +
                  ((float) R_stereo.at<double>(0, 2)) * d_img + (float) t_stereo.at<double>(0);
            Y_t = ((float) R_stereo.at<double>(1, 0)) * X + ((float) R_stereo.at<double>(1, 1)) * Y +
                  ((float) R_stereo.at<double>(1, 2)) * d_img + (float) t_stereo.at<double>(1);
            d_img = ((float) R_stereo.at<double>(2, 0)) * X + ((float) R_stereo.at<double>(2, 1)) * Y +
                    ((float) R_stereo.at<double>(2, 2)) * d_img + (float) t_stereo.at<double>(2);
        }

        projectPointCloud(X_t, Y_t, d_img, U, V, fx_rgb, fy_rgb, cx_rgb, cy_rgb, t_stereo.at<double>(2), cloud_array);

        cv::Mat_<cv::Vec3b> seg_rz = cv::Mat_<cv::Vec3b>(height, width, cv::Vec3b(0, 0, 0));
        cv::Mat_<uchar> seg_output = cv::Mat_<uchar>(height, width, uchar(0));

        // Run CAPE
        int nr_planes, nr_cylinders;
        vector<PlaneSeg> plane_params;
        vector<CylinderSeg> cylinder_params;
        organizePointCloudByCell(cloud_array, cloud_array_organized, cell_map);
        plane_detector->process(cloud_array_organized, nr_planes, nr_cylinders, seg_output, plane_params,
                                cylinder_params);

        // Map segments with color codes and overlap segmented image w/ RGB
        uchar *sCode;
        uchar *dColor;
        uchar *srgb;
        int code;
        for (int r = 0; r < height; r++) {
            dColor = seg_rz.ptr<uchar>(r);
            sCode = seg_output.ptr<uchar>(r);
            srgb = rgb_img.ptr<uchar>(r);
            for (int c = 0; c < width; c++) {
                code = *sCode;
                if (code > 0) {
                    dColor[c * 3] = color_code[code - 1][0] / 2 + srgb[0] / 2;
                    dColor[c * 3 + 1] = color_code[code - 1][1] / 2 + srgb[1] / 2;
                    dColor[c * 3 + 2] = color_code[code - 1][2] / 2 + srgb[2] / 2;;
                } else {
                    dColor[c * 3] = srgb[0];
                    dColor[c * 3 + 1] = srgb[1];
                    dColor[c * 3 + 2] = srgb[2];
                }
                sCode++;
                srgb++;
                srgb++;
                srgb++;
            }
        }

        // Show frame rate and labels
        int cylinder_code_offset = 50;
        // show cylinder labels
        if (nr_cylinders > 0) {
            std::stringstream text;
            cv::putText(seg_rz, text.str(), cv::Point(width / 2, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                        cv::Scalar(255, 255, 255, 1));
            for (int j = 0; j < nr_cylinders; j++) {
                cv::rectangle(seg_rz, cv::Point(width / 2 + 80 + 15 * j, 6), cv::Point(width / 2 + 90 + 15 * j, 16),
                              cv::Scalar(color_code[cylinder_code_offset + j][0],
                                         color_code[cylinder_code_offset + j][1],
                                         color_code[cylinder_code_offset + j][2]), -1);
            }
        }
        return seg_rz;
    }
};

#endif // capewrap_cpp
