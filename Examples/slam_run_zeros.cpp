//
// Created by root on 7/14/20.
//
#include<unistd.h>

/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/


#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>

#include<opencv2/core/core.hpp>

#include<System.h>

using namespace std;

int main()
{
//    if(argc != 5)
//    {
//        cerr << endl << "Usage: ./rgbd_tum path_to_vocabulary path_to_settings path_to_sequence path_to_association" << endl;
//        return 1;
//    }


    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM2::System SLAM("/ORBvoc.txt","/slamdoom/tmp/orbslam2/Examples/RGB-D/TUM1.yaml",ORB_SLAM2::System::RGBD,true);

    // Vector for tracking time statistics
    vector<float> vTimesTrack;

    // Main loop
    cv::Mat imRGB, imD, imGray;
    int nImages = 1000;
    for(int ni=0; ni<nImages; ni++) {
        cout << "frame " << ni << endl;
        // Read image and depthmap from file
        imGray = cv::Mat::zeros(640, 480, CV_8U);
        cv::cvtColor(imGray, imRGB, CV_GRAY2BGR);
        imD = cv::Mat::zeros(640, 480, CV_32F);
//        imRGB = cv::imread(string(argv[3])+"/"+vstrImageFilenamesRGB[ni],CV_LOAD_IMAGE_UNCHANGED);
//        imD = cv::imread(string(argv[3])+"/"+vstrImageFilenamesD[ni],CV_LOAD_IMAGE_UNCHANGED);

        if (imRGB.empty()) {
            return 1;
        }

        std::chrono::steady_clock::time_point tframe = std::chrono::steady_clock::now();
        // Pass the image to the SLAM system
        SLAM.TrackRGBD(imRGB.clone(), imD.clone(), double(ni));

    }

    // Stop all threads
    SLAM.Shutdown();


    return 0;
}

